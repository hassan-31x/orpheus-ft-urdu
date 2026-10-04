#!/usr/bin/env python3
"""One-epoch Urdu Orpheus adaptation. See README before running on Kaggle."""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import gc
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from orpheus_utils import (
    AUDIO_BASE, CODEC_ID, FORMAT_VERSION, SPECIAL, DriveStore, HuggingFaceStore, SnapshotStore, append_jsonl,
    atomic_json, clean_text, contained, fingerprint, interleave_codes,
    latest_checkpoint, plot_logs, prompt_ids, read_json, safe_unzip,
    seal_checkpoint, sha256, training_row,
)

LOG = logging.getLogger("orpheus_urdu")
DEFAULTS = Path(__file__).parent / "configs/aslp50h.json"


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULTS)
    p.add_argument("--mode", choices=["all", "prepare", "train", "storage-check", "audit"], default="all")
    p.add_argument("--local-only", action="store_true", help="Explicitly opt out of remote checkpoint backups")
    p.add_argument("--checkpoint-backend", choices=["huggingface", "drive", "local"])
    p.add_argument("--hf-repo-id", help="Private Hugging Face model repo: USERNAME/REPOSITORY")
    for name in ["run-id", "work-dir", "data-dir", "train-manifest", "validation-manifest", "test-manifest"]:
        p.add_argument("--" + name)
    for name in ["micro-batch", "grad-accum", "rank", "seed", "max-length", "max-steps"]:
        p.add_argument("--" + name, type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--precision", choices=["4bit", "16bit"])
    p.add_argument("--sampling", choices=["random", "length"])
    p.add_argument("--objective", choices=["all", "audio"])
    args = p.parse_args()
    cfg = read_json(DEFAULTS)
    user = read_json(args.config)
    unknown = set(user) - set(cfg)
    if unknown:
        p.error(f"Unknown config fields: {sorted(unknown)}")
    cfg.update(user)
    for k in cfg:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.local_only:
        cfg["checkpoint_backend"] = "local"
    if cfg["checkpoint_backend"] not in ("huggingface", "drive", "local"):
        p.error("checkpoint_backend must be huggingface, drive or local")
    if cfg["checkpoint_backend"] == "huggingface":
        cfg["hf_repo_id"] = cfg["hf_repo_id"] or os.environ.get("ORPHEUS_HF_REPO")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", cfg["hf_repo_id"] or ""):
            p.error("Set hf_repo_id, --hf-repo-id or ORPHEUS_HF_REPO to USERNAME/REPOSITORY (see README)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", cfg["run_id"]):
        p.error("run_id must be a simple directory name")
    for k in ["micro_batch", "grad_accum", "rank", "max_length", "chunk_size", "save_steps",
              "eval_steps", "logging_steps", "save_total_limit"]:
        if cfg[k] < 1:
            p.error(f"{k} must be positive")
    if cfg["epochs"] != 1:
        p.error("This study is defined as one epoch; use a new study config/code for other schedules")
    if cfg["max_steps"] != -1 and cfg["max_steps"] < 1:
        p.error("max_steps must be -1 (full epoch) or a positive smoke-test step limit")
    if cfg["max_steps"] > 0 and "smoke" not in cfg["run_id"]:
        p.error("A max_steps override requires 'smoke' in run_id to distinguish it from full-epoch results")
    if cfg["sampling"] not in ("random", "length") or cfg["precision"] not in ("4bit", "16bit"):
        p.error("Invalid sampling or precision")
    if cfg["speaker_column"] is not None and not cfg["speaker_column"]:
        p.error("speaker_column must be null or a nonempty column name")
    if cfg["eval_samples"] < 0 or cfg["sample_count"] < 0:
        p.error("eval_samples/sample_count cannot be negative")
    if not 0 < cfg["learning_rate"] < 1 or not 0 <= cfg["warmup_ratio"] < 1:
        p.error("Invalid learning rate/warmup")
    if cfg["invalid_row_policy"] not in ("skip", "strict"):
        p.error("invalid_row_policy must be skip or strict")
    if not 0 <= cfg["maximum_rejected_train_fraction"] < 1:
        p.error("maximum_rejected_train_fraction must be in [0, 1)")
    return args, cfg


@contextlib.contextmanager
def checkpoint_store(cfg):
    if cfg["checkpoint_backend"] == "local":
        LOG.warning("LOCAL ONLY: runtime deletion will lose work unless outputs are saved separately")
        yield SnapshotStore()
        return
    if cfg["checkpoint_backend"] == "huggingface":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN from Kaggle Secrets to a token with write access to your private checkpoint repo (see README)")
        store = HuggingFaceStore(cfg["hf_repo_id"], cfg["run_id"], token)
        store.preflight()
        LOG.info("Checkpoint storage: %s", store.remote)
        yield store
        return
    remote = os.environ.get("ORPHEUS_DRIVE_REMOTE", "gdrive:orpheus_urdu")
    encoded = os.environ.get("RCLONE_CONFIG_B64")
    config = os.environ.get("RCLONE_CONFIG")
    if not encoded and not config:
        raise RuntimeError("Set RCLONE_CONFIG_B64 via Kaggle Secrets (README), or RCLONE_CONFIG path; use --local-only only deliberately")
    with tempfile.TemporaryDirectory(prefix="orpheus-auth-") as tmp:
        if encoded:
            config = Path(tmp) / "rclone.conf"
            config.write_bytes(base64.b64decode(encoded, validate=True))
            config.chmod(0o600)
        store = DriveStore(remote + "/" + cfg["run_id"], config)
        store.preflight()
        LOG.info("Checkpoint storage: %s", store.remote)
        yield store


def find_data(cfg, work):
    if cfg["data_dir"]:
        root = Path(cfg["data_dir"])
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root.resolve()
    import gdown
    archive = work / "dataset.zip"
    if not archive.exists():
        temp = work / "dataset.partial.zip"
        LOG.info("Downloading supplied Drive archive")
        result = gdown.download(url=cfg["download_url"], output=str(temp), fuzzy=True,
                                quiet=False, resume=True)
        if not result or not temp.is_file():
            raise RuntimeError("Drive download failed; check public access/quota or attach archive as Kaggle Dataset")
        import zipfile
        if not zipfile.is_zipfile(temp):
            raise ValueError("Download is not a ZIP (possibly a Drive quota/login page)")
        temp.replace(archive)
    dest = work / "extracted"
    marker = dest / ".extraction.json"
    archive_digest = sha256(archive)
    if marker.exists() and read_json(marker)["archive_sha256"] != archive_digest:
        raise RuntimeError("Archive changed; choose a new work_dir to avoid stale extracted files")
    if not marker.exists():
        LOG.info("Extracting and validating ZIP paths")
        safe_unzip(archive, dest)
        atomic_json(marker, {"archive_sha256": archive_digest})
    candidates = sorted({p.parent for p in dest.rglob("*.csv")
                         if p.name in ("stage1_train.csv", "train.csv", "train_manifest.csv")})
    if cfg["train_manifest"]:
        explicit = contained(dest, cfg["train_manifest"])
        if explicit.is_file():
            return dest.resolve()
    if len(candidates) != 1:
        raise RuntimeError(f"Could not identify a single manifest directory: {candidates}. Set data_dir and manifest names in config")
    return candidates[0].resolve()


def locate_manifest(root, explicit, split):
    if explicit:
        p = contained(root, explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    names = {"train": ["stage1_train.csv", "train.csv", "train_manifest.csv"],
             "validation": ["stage1_validation.csv", "validation.csv", "val.csv", "validation_manifest.csv"],
             "test": ["stage1_test.csv", "test.csv", "test_manifest.csv"]}[split]
    matches = [root / name for name in names if (root / name).is_file()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {split} manifests; set an explicit filename")
    if not matches and split != "test":
        raise FileNotFoundError(f"No {split} manifest under {root}; expected {names}")
    return matches[0] if matches else None


def audit_problem_category(error):
    for prefix, category in (
        ('Duplicate path', 'duplicate_path'), ('Identical audio bytes', 'duplicate_audio'),
        ('Empty text', 'invalid_text'), ('Expected 24kHz', 'audio_format'),
        ('Invalid/overlong duration', 'duration'), ('Nonfinite or entirely silent', 'waveform'),
        ('Empty trusted speaker', 'speaker')):
        if error.startswith(prefix):
            return category
    return 'file_or_other_error'


def report_audit_errors(run, problems, policy="strict"):
    from collections import Counter
    counts = Counter((p['split'], audit_problem_category(p['error'])) for p in problems)
    summary = {'total': len(problems), 'groups': [
        {'split': split, 'reason': reason, 'count': count}
        for (split, reason), count in sorted(counts.items())], 'examples': problems[:10], 'policy': policy}
    atomic_json(run / 'data_error_summary.json', summary)
    if problems:
        LOG.warning('Dataset audit exclusions (%s policy): %s', policy, summary['groups'])
        for problem in summary['examples']:
            LOG.warning('Audit example: split=%s CSV line=%s audio=%s reason=%s',
                      problem['split'], problem['row'] + 2, problem.get('audio'), problem['error'])
    return summary


def audit_retention(splits, manifests, problems, cfg):
    """Decide acceptance using row loss and all measurable rejected durations."""
    summary = {}
    for split, rows in splits.items():
        rejected = [p for p in problems if p['split'] == split]
        kept_seconds = sum(row['duration'] for row in rows)
        rejected_seconds = sum(p['duration_seconds'] for p in rejected
                               if p.get('duration_seconds') is not None)
        known_seconds = kept_seconds + rejected_seconds
        summary[split] = dict(clips=len(rows), hours=kept_seconds / 3600,
            source_rows=manifests[split]['rows'], rejected_rows=len(rejected),
            rejected_row_fraction=len(rejected) / max(1, manifests[split]['rows']),
            rejected_known_hours=rejected_seconds / 3600,
            rejected_unknown_duration_rows=sum(p.get('duration_seconds') is None for p in rejected),
            rejected_known_hour_fraction=rejected_seconds / known_seconds if known_seconds else 0,
            max_seconds=max((row['duration'] for row in rows), default=0))
    errors = []
    if problems and cfg['invalid_row_policy'] == 'strict':
        errors.append(f'{len(problems)} invalid/duplicate rows under strict policy')
    if not splits.get('train') or not splits.get('validation'):
        errors.append('Train and validation must retain at least one valid row')
    train = summary.get('train', {})
    if cfg['invalid_row_policy'] == 'skip':
        limit = cfg['maximum_rejected_train_fraction']
        for key in ('rejected_row_fraction', 'rejected_known_hour_fraction'):
            if train.get(key, 0) > limit:
                errors.append(f'Training {key}={train[key]:.2%} exceeds configured {limit:.2%} limit')
    if train.get('hours', 0) < cfg['minimum_train_hours']:
        errors.append(f"Only {train.get('hours', 0):.2f} retained training hours; "
                      f"minimum_train_hours={cfg['minimum_train_hours']}")
    return summary, errors


def audit_data(root, cfg, run):
    import numpy as np
    import soundfile as sf
    import pandas as pd
    splits, problems, seen_paths, seen_hashes = {}, [], {}, {}
    inventory = []
    manifest_info = {}
    for split in ("train", "validation", "test"):
        path = locate_manifest(root, cfg[f"{split}_manifest"], split)
        if not path:
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {"audio", "text"}.issubset(df.columns):
            raise ValueError(f"{path}: needs audio,text columns; has {list(df.columns)}")
        speaker_col = cfg["speaker_column"]
        if speaker_col and speaker_col not in df:
            raise ValueError(f"Missing trusted speaker column {speaker_col}")
        rows = []
        for i, original in enumerate(df.to_dict("records")):
            duration, audio = None, None
            try:
                text = clean_text(original["text"])
                if not text or not re.search(r"[\u0600-\u06ff]", text):
                    raise ValueError("Empty text or no Arabic/Urdu-script character")
                audio = contained(root, original["audio"])
                relative = audio.relative_to(root).as_posix()
                if relative in seen_paths:
                    raise ValueError(f"Duplicate path across/in splits: {seen_paths[relative]}")
                info = sf.info(audio)
                if info.samplerate != 24000 or info.channels != 1 or info.subtype != "PCM_16":
                    raise ValueError(f"Expected 24kHz mono PCM16; found {info}")
                duration = info.frames / info.samplerate
                if not 0 < duration <= cfg["maximum_clip_seconds"]:
                    raise ValueError(f"Invalid/overlong duration {duration}")
                digest = sha256(audio)
                if digest in seen_hashes:
                    raise ValueError(f"Identical audio bytes across/in splits: {seen_hashes[digest]}")
                # Full signal audit, including test, but no test tokenization or model evaluation.
                wav, sr = sf.read(audio, dtype="float32")
                if not np.isfinite(wav).all() or not np.any(wav):
                    raise ValueError("Nonfinite or entirely silent waveform")
                speaker = clean_text(original[speaker_col]) if speaker_col else None
                if speaker_col and not speaker:
                    raise ValueError("Empty trusted speaker identity")
                seen_paths[relative] = (split, i)
                seen_hashes[digest] = (split, i)
                row = dict(audio=relative, text=text, duration=duration,
                           sha256=digest, speaker=speaker, source_row=i)
                rows.append(row)
                inventory.append(dict(split=split, **row,
                                      peak=float(np.max(np.abs(wav))),
                                      rms=float(np.sqrt(np.mean(wav**2))),
                                      clipped_fraction=float(np.mean(np.abs(wav) >= 0.999)),
                                      text_changed=text != original["text"]))
            except (ValueError, OSError, RuntimeError) as e:
                # Probe duration for rejected rows when the path/header is readable.
                # Missing/corrupt files are recorded with unknown duration, never as zero hours.
                if duration is None:
                    try:
                        probe = sf.info(audio or contained(root, original["audio"]))
                        if probe.samplerate > 0 and probe.frames >= 0:
                            duration = probe.frames / probe.samplerate
                    except (ValueError, OSError, RuntimeError):
                        pass
                problems.append(dict(split=split, row=i, csv_line=i+2,
                                     audio=original.get("audio"), error=str(e),
                                     duration_seconds=duration))
            if (i+1) % 1000 == 0:
                LOG.info("Audited %s %d/%d clips", split, i+1, len(df))
        splits[split] = rows
        manifest_info[split] = dict(file=path.name, sha256=sha256(path), rows=len(df))
    atomic_json(run / "data_errors.json", problems)
    pd.DataFrame(inventory).to_csv(run / "data_inventory.csv", index=False)
    report_audit_errors(run, problems, cfg["invalid_row_policy"])
    summary, failures = audit_retention(splits, manifest_info, problems, cfg)
    report = dict(splits=summary, manifests=manifest_info,
                  invalid_row_policy=cfg["invalid_row_policy"],
                  maximum_rejected_train_fraction=cfg["maximum_rejected_train_fraction"],
                  audit_passed=not failures, failure_reasons=failures,
                  duplicate_policy="First valid occurrence retained: train before validation before test",
                  training_cap=None, trusted_speaker_column=cfg["speaker_column"],
                  split_limitations="Duplicate evaluation rows are excluded; related recordings can still leak. Unknown rejected durations are not estimated")
    atomic_json(run / "dataset_report.json", report)
    atomic_json(run / "normalized_manifests.json", splits)
    LOG.info("Measured retained dataset: %s", summary)
    if failures:
        raise ValueError('; '.join(failures) + f"; details: {run / 'dataset_report.json'}")
    if problems:
        LOG.warning("Continuing with %d documented row exclusions; all retained training rows will be used", len(problems))
    return splits


def environment(run):
    import torch
    env = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
               vram_bytes=torch.cuda.get_device_properties(0).total_memory,
               packages={m: importlib.metadata.version(m) for m in
                         ["unsloth", "unsloth_zoo", "transformers", "peft", "accelerate",
                          "datasets", "snac", "bitsandbytes", "numpy", "pyarrow",
                          "pandas", "soundfile", "tokenizers", "huggingface_hub", "safetensors"]})
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    freeze = result.stdout
    gpu = None
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        gpu = result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return env, freeze, gpu


def encode_data(root, splits, cfg, tokenizer, vocab_size, cache, store, cache_id, run):
    import torch
    import pandas as pd
    import soundfile as sf
    from snac import SNAC
    from datasets import load_dataset
    cache.mkdir(parents=True, exist_ok=True)
    store.restore_files("cache/" + cache_id, cache)
    codec = None
    files, encoding_errors, accepted_audio = {}, [], {}
    try:
        for split in ("train", "validation"):
            rows, paths = splits[split], []
            accepted_audio[split] = []
            folder = cache / split
            folder.mkdir(exist_ok=True)
            for start in range(0, len(rows), cfg["chunk_size"]):
                subset = rows[start:start+cfg["chunk_size"]]
                path = folder / f"{start:07d}.parquet"
                meta = path.with_suffix(".json")
                chunk_identity = fingerprint(subset)
                if path.exists() and meta.exists():
                    info = read_json(meta)
                    if info["row_identity"] != chunk_identity or info["sha256"] != sha256(path):
                        raise RuntimeError(f"Stale/corrupt cache chunk: {path}")
                    if len(pd.read_parquet(path)) != info["rows"] or info["rows"] + len(info.get("excluded", [])) != len(subset):
                        raise RuntimeError(f"Incomplete cache chunk: {path}")
                else:
                    if codec is None:
                        codec = SNAC.from_pretrained(CODEC_ID, revision=cfg["codec_revision"]).eval().cuda()
                    encoded, excluded = [], []
                    for row in subset:
                        wav, sr = sf.read(root / row["audio"], dtype="float32")
                        if sr != 24000:
                            raise ValueError("Audio changed after validation")
                        with torch.inference_mode():
                            codes = codec.encode(torch.from_numpy(wav).cuda().view(1, 1, -1))
                        c0, c1, c2 = [c[0].cpu().tolist() for c in codes]
                        tokens = interleave_codes(c0, c1, c2, cfg["dedup"])
                        try:
                            item = training_row(prompt_ids(tokenizer, row["text"], row["speaker"]),
                                                tokens, cfg["objective"], cfg["max_length"], vocab_size)
                        except ValueError as exc:
                            if cfg["invalid_row_policy"] != "skip" or not str(exc).startswith("Sequence has "):
                                raise
                            excluded.append(dict(split=split, row=row["source_row"], audio=row["audio"],
                                                 error=str(exc), duration_seconds=row["duration"],
                                                 stage="encoding_context_limit"))
                            continue
                        item.update(audio=row["audio"], raw_frames=len(c0), kept_frames=len(tokens)//7)
                        encoded.append(item)
                    temp = path.with_suffix(".tmp.parquet")
                    pd.DataFrame(encoded, columns=["input_ids", "attention_mask", "labels", "length", "audio",
                                                   "raw_frames", "kept_frames"]).to_parquet(temp, index=False)
                    temp.replace(path)
                    info = dict(row_identity=chunk_identity, rows=len(encoded), excluded=excluded, sha256=sha256(path))
                    atomic_json(meta, info)
                    LOG.info("Encoded %s %d/%d", split, start+len(subset), len(rows))
                # Retry upload even for local cached chunks; a previous upload may have failed.
                for local in (path, meta):
                    store.put(local, f"cache/{cache_id}/{split}/{local.name}")
                encoding_errors.extend(info.get("excluded", []))
                accepted = pd.read_parquet(path)["audio"].tolist()
                accepted_audio[split].extend(accepted)
                if info["rows"]:
                    paths.append(str(path))
                atomic_json(run / "encoding_errors.json", encoding_errors)
            files[split] = paths
    finally:
        del codec
        gc.collect()
        torch.cuda.empty_cache()
    for split, audio_paths in accepted_audio.items():
        allowed = set(audio_paths)
        splits[split] = [row for row in splits[split] if row["audio"] in allowed]
    report = read_json(run / "dataset_report.json")
    all_exclusions = read_json(run / "data_errors.json") + encoding_errors
    summary, failures = audit_retention(splits, report["manifests"], all_exclusions, cfg)
    report.update(splits=summary, encoding_excluded_rows=len(encoding_errors),
                  audit_passed=not failures, failure_reasons=failures)
    atomic_json(run / "dataset_report.json", report)
    atomic_json(run / "training_manifests.json", splits)
    for name in ("encoding_errors.json", "dataset_report.json", "training_manifests.json"):
        store.put(run / name, "preparation/" + name)
    if failures:
        raise ValueError('; '.join(failures) + f"; see {run / 'dataset_report.json'}")
    if encoding_errors:
        LOG.warning("Excluded %d context-overflow sequences; retained hours/counts: %s", len(encoding_errors), summary)
    ds = load_dataset("parquet", data_files=files, cache_dir=str(cache / "hf"))
    for split in files:
        if len(ds[split]) != len(splits[split]):
            raise RuntimeError("Cache dataset coverage mismatch")
    return ds


def optional_plots(run):
    try:
        plot_logs(run)
    except Exception as exc:
        LOG.warning('Optional plotting failed (%s); training/checkpoint work continues', type(exc).__name__)
        append_jsonl(run / 'optional_errors.jsonl', dict(stage='plotting', error=str(exc)))


def train(cfg, ds, tokenizer, splits, run, store, identity, session_start):
    # Unsloth was imported by main BEFORE Transformers/Torch.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    import numpy as np
    import psutil
    from transformers import Trainer, TrainingArguments, TrainerCallback
    from synthesize import generate_audio

    model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"],
        use_exact_model_name=True, max_seq_length=cfg["max_length"], dtype=None,
        load_in_4bit=cfg["precision"] == "4bit")
    if loaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("Preprocessing and training tokenizer vocabularies differ")
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"], use_rslora=cfg["rslora"])
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    atomic_json(run / "parameters.json", dict(total=total, trainable=trainable,
                                               trainable_percent=100*trainable/total))
    with (run / "trainable_parameters.txt").open("w") as f:
        for n, p in model.named_parameters():
            if p.requires_grad:
                f.write(f"{n}\t{tuple(p.shape)}\t{p.numel()}\n")

    def collate(examples):
        width = math.ceil(max(len(x["input_ids"]) for x in examples)/8)*8
        return {key: torch.tensor([list(x[key]) + [pad]*(width-len(x[key])) for x in examples], dtype=torch.long)
                for key, pad in [("input_ids", SPECIAL["pad"]), ("attention_mask", 0), ("labels", -100)]}

    eval_ds = ds["validation"]
    if cfg["eval_samples"] and len(eval_ds) > cfg["eval_samples"]:
        eval_ds = eval_ds.shuffle(seed=cfg["seed"]).select(range(cfg["eval_samples"]))
    samples = random.Random(cfg["seed"]).sample(splits["validation"],
                                               min(cfg["sample_count"], len(splits["validation"])))
    atomic_json(run / "monitoring_prompts.json", samples)
    atomic_json(run / "periodic_eval_rows.json", eval_ds["audio"])
    stop_requested = [False]

    def request_stop(signum, frame):
        stop_requested[0] = True
        LOG.warning("Received signal %d; requesting save/stop at next optimizer boundary", signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            scalar = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            clean = {k: v if math.isfinite(v) else None for k, v in scalar.items()}
            record = dict(step=state.global_step, epoch=state.epoch,
                          timestamp=time.time(), session_elapsed_seconds=time.monotonic()-session_start,
                          gpu_allocated_gb=torch.cuda.memory_allocated()/2**30,
                          gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                          ram_gb=psutil.Process().memory_info().rss/2**30)
            record.update(clean)
            record["nonfinite_metrics"] = ",".join(k for k, v in scalar.items() if not math.isfinite(v))
            append_jsonl(run / "metrics.jsonl", record)
            atomic_json(run / "status.json", dict(status="training", **record))
            if "loss" in scalar and not math.isfinite(scalar["loss"]):
                raise FloatingPointError("Nonfinite training loss recorded; stopping before publishing another checkpoint")

        def on_step_end(self, args, state, control, **kwargs):
            remaining = cfg["session_hours"]*3600 - (time.monotonic()-session_start)
            # Budget includes preparation and downloads. Leave time for samples/upload.
            if stop_requested[0] or remaining < 900:
                control.should_save = True
                control.should_training_stop = True
                LOG.warning("Stopping at step %d for session limit/signal", state.global_step)
            if state.global_step >= state.max_steps:
                control.should_save = True
            return control

        def samples(self, state):
            if not samples:
                return
            # Monitoring must not change subsequent dropout/sampler RNG streams.
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[0]):
                    FastLanguageModel.for_inference(model)
                    for i, row in enumerate(samples):
                        dest = run / "samples" / f"step-{state.global_step:07d}" / f"sample-{i:02d}.wav"
                        try:
                            metadata = generate_audio(model, tokenizer, row["text"], dest,
                                                      speaker=row["speaker"], seed=cfg["seed"]+i,
                                                      max_length=cfg["max_length"],
                                                      max_new_tokens=cfg["sample_max_new_tokens"],
                                                      temperature=cfg["sample_temperature"],
                                                      top_p=cfg["sample_top_p"],
                                                      repetition_penalty=cfg["sample_repetition_penalty"],
                                                      codec_revision=cfg["codec_revision"])
                            metadata.update(reference_audio=row["audio"], step=state.global_step)
                            atomic_json(dest.with_suffix(".json"), metadata)
                        except Exception as e:
                            LOG.exception("Monitoring synthesis failed; checkpoint remains the priority")
                            append_jsonl(run / "sample_errors.jsonl",
                                         dict(step=state.global_step, sample=i, error=str(e)))
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                FastLanguageModel.for_training(model)
                model.config.use_cache = False
                model.train()
                gc.collect()
                torch.cuda.empty_cache()

        def on_train_begin(self, args, state, control, **kwargs):
            if cfg["sample_at_start"] and state.global_step == 0:
                self.samples(state)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = run / f"checkpoint-{state.global_step}"
            tokenizer.save_pretrained(checkpoint)
            atomic_json(checkpoint / "run_identity.json", identity)
            seal_checkpoint(checkpoint)
            atomic_json(run / "status.json", dict(status="checkpoint_saved", step=state.global_step,
                                                  epoch=state.epoch, timestamp=time.time()))
            # First secure training state. Optional samples cannot delay the first upload.
            optional_plots(run)
            store.backup(run, checkpoint)
            self.samples(state)
            optional_plots(run)
            # The required training snapshot is already secured. Sample refresh is optional.
            try:
                store.backup(run, checkpoint)
            except Exception as exc:
                LOG.warning("Optional sample snapshot refresh failed (%s); training snapshot remains saved",
                            type(exc).__name__)
                append_jsonl(run / "optional_errors.jsonl", dict(stage="sample_snapshot_refresh",
                                                               error=str(exc)))
            LOG.info("Checkpoint %s safely archived%s", checkpoint.name,
                     " to " + store.remote if store.remote else " locally")

    arguments = TrainingArguments(
        output_dir=str(run), num_train_epochs=1, max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["micro_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type=cfg["scheduler"],
        weight_decay=cfg["weight_decay"], max_grad_norm=1.0,
        optim="adamw_8bit", fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        eval_strategy="steps", eval_steps=cfg["eval_steps"], prediction_loss_only=True,
        save_strategy="steps", save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"], save_only_model=False,
        logging_steps=cfg["logging_steps"], logging_first_step=True, logging_nan_inf_filter=False,
        report_to=["tensorboard"], logging_dir=str(run / "tensorboard"),
        group_by_length=cfg["sampling"] == "length", length_column_name="length",
        dataloader_drop_last=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        remove_unused_columns=False, seed=cfg["seed"], data_seed=cfg["seed"],
        ignore_data_skip=False, disable_tqdm=True, label_names=["labels"],
        load_best_model_at_end=False)
    monitor = Monitor()
    trainer = Trainer(model=model, args=arguments, train_dataset=ds["train"],
                      eval_dataset=eval_ds, data_collator=collate,
                      processing_class=tokenizer, callbacks=[monitor])
    previous = latest_checkpoint(run)
    LOG.info("Resuming: %s; effective batch=%d; train clips=%d", previous,
             cfg["micro_batch"]*cfg["grad_accum"], len(ds["train"]))
    if previous and read_json(previous / "run_identity.json") != identity:
        raise RuntimeError("Checkpoint run identity changed; use a new run_id for an ablation")
    result = trainer.train(resume_from_checkpoint=str(previous) if previous else None)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    reached_schedule = trainer.state.global_step >= trainer.state.max_steps
    if not reached_schedule:
        atomic_json(run / "status.json", dict(status="paused_for_resume", step=trainer.state.global_step,
                                              epoch=trainer.state.epoch))
        cp = latest_checkpoint(run)
        if cp:
            store.backup(run, cp)
        LOG.info("Session ended safely. Rerun identical command/config to continue the SAME epoch")
        return
    metrics = trainer.evaluate(eval_dataset=ds["validation"], metric_key_prefix="full_validation")
    trainer.save_metrics("validation", metrics)
    final = run / "adapter_final"
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    atomic_json(final / "run_identity.json", identity)
    atomic_json(final / "resolved_config.json", cfg)
    atomic_json(run / "status.json", dict(status="smoke_complete" if cfg["max_steps"] > 0 else "complete",
                                          step=trainer.state.global_step, epoch=trainer.state.epoch,
                                          validation=metrics))
    optional_plots(run)
    cp = latest_checkpoint(run)
    if not cp:
        raise RuntimeError("No resumable final checkpoint was created")
    store.backup(run, cp)
    LOG.info("Finished; adapter at %s", final)


def audit_with_reports(root, cfg, run, store):
    try:
        return audit_data(root, cfg, run)
    finally:
        # Keep failed-audit evidence remotely even before the first model checkpoint.
        for name in ('data_errors.json', 'data_error_summary.json', 'data_inventory.csv', 'dataset_report.json', 'normalized_manifests.json'):
            path = run / name
            if path.is_file():
                try:
                    store.put(path, 'audit/' + name)
                except Exception as exc:
                    LOG.warning('Audit report backup failed for %s (%s); local file retained',
                                name, type(exc).__name__)


def main():
    args, cfg = cli()
    # Limit Unsloth to one GPU; T4 x2 is not a single 32GB GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    session_start = time.monotonic()
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    run = work / "runs" / cfg["run_id"]
    run.mkdir(parents=True, exist_ok=True)
    # POSIX lock is released on crashes; stale PID files do not block future sessions.
    import fcntl
    with (run / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is using this run_id") from None
        with checkpoint_store(cfg) as store:
            if args.mode == "storage-check":
                LOG.info("Storage preflight passed; no dataset download or GPU training requested")
                return
            if args.mode == "audit":
                # CPU-only dataset inspection; no model imports, SNAC or tokenization.
                root = find_data(cfg, work)
                audit_with_reports(root, cfg, run, store)
                LOG.info("Dataset audit passed. No model loaded or training started")
                return
            store.restore(run)
            status = run / "status.json"
            if status.exists() and read_json(status).get("status") in ("complete", "smoke_complete"):
                previous_cfg = read_json(run / "resolved_config.json")
                changed = [k for k, v in cfg.items() if k not in
                           ("work_dir", "data_dir", "session_hours", "download_url", "model_revision", "codec_revision", "checkpoint_backend", "hf_repo_id")
                           and previous_cfg.get(k) != v]
                if changed:
                    raise RuntimeError(f"Completed run_id reused with different settings {changed}; choose a new run_id")
                LOG.info("This run is already complete; use synthesize.py or a new run_id")
                return
            handler = logging.FileHandler(run / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOG.addHandler(handler)
            # Import order is essential for Unsloth's patches.
            from unsloth import FastLanguageModel  # noqa: F401
            import torch
            from transformers import AutoTokenizer, AutoConfig, set_seed
            from huggingface_hub import HfApi
            if not torch.cuda.is_available():
                raise RuntimeError("Enable a Kaggle NVIDIA GPU accelerator and Internet")
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=0; this script uses one GPU")
            set_seed(cfg["seed"])
            # Reuse committed revisions on resume, rather than resolving a moving branch anew.
            old = read_json(run / "resolved_config.json") if (run / "resolved_config.json").exists() else None
            if old:
                for k in ("model_revision", "codec_revision"):
                    if cfg[k] in ("main", old[k]):
                        cfg[k] = old[k]
            api = HfApi()
            cfg["model_revision"] = api.model_info(cfg["model_id"], revision=cfg["model_revision"]).sha
            cfg["codec_revision"] = api.model_info(CODEC_ID, revision=cfg["codec_revision"]).sha
            root = find_data(cfg, work)
            splits = audit_with_reports(root, cfg, run, store)
            tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
            model_config = AutoConfig.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
            vocab_hash = fingerprint(tokenizer.get_vocab())
            token_identity = dict(format_version=FORMAT_VERSION, model_id=cfg["model_id"],
                                  model_revision=cfg["model_revision"], tokenizer_sha256=vocab_hash,
                                  encoding_source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                                          ("finetune_aslp_50h.py", "orpheus_utils.py")},
                                  encoding_packages={name: importlib.metadata.version(name) for name in
                                                     ("torch", "snac", "tokenizers", "soundfile", "numpy")},
                                  codec=CODEC_ID, codec_revision=cfg["codec_revision"],
                                  max_length=cfg["max_length"], objective=cfg["objective"], dedup=cfg["dedup"],
                                  special=SPECIAL, audio_base=AUDIO_BASE,
                                  splits={s: fingerprint(rows) for s, rows in splits.items()})
            cache_id = fingerprint(token_identity)
            env, freeze, gpu = environment(run)
            # Same code, dependency versions, sequence order, schedule, objective and batch geometry.
            semantic = {k: v for k, v in cfg.items() if k not in
                        ("work_dir", "data_dir", "download_url", "train_manifest", "validation_manifest", "test_manifest", "session_hours", "checkpoint_backend", "hf_repo_id")}
            identity = dict(config=semantic, token_cache_sha256=cache_id,
                            packages=env["packages"], torch=env["torch"],
                            source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                           ("finetune_aslp_50h.py", "orpheus_utils.py", "synthesize.py")})
            identity_path = run / "run_identity.json"
            if identity_path.exists() and read_json(identity_path) != identity:
                raise RuntimeError("Run data/config/code/dependencies changed. Use a new run_id or restore original versions from requirements-resolved.txt")
            atomic_json(run / "environment.json", env)
            (run / "requirements-resolved.txt").write_text(freeze)
            if gpu:
                (run / "gpu.txt").write_text(gpu)
            import shutil
            sources = run / "source"
            sources.mkdir(exist_ok=True)
            for name in identity["source_sha256"]:
                shutil.copy2(Path(__file__).parent / name, sources / name)
            shutil.copy2(Path(__file__).parent / "evaluate_asr.py", sources / "evaluate_asr.py")
            atomic_json(identity_path, identity)
            atomic_json(run / "resolved_config.json", cfg)
            atomic_json(run / "token_format.json", token_identity)
            for p in (identity_path, run / "resolved_config.json", run / "dataset_report.json",
                      run / "requirements-resolved.txt", run / "environment.json"):
                store.put(p, "preparation/" + p.name)
            # Preparation metadata and immutable chunks also resume before the first checkpoint.
            cache = work / "cache" / cache_id
            ds = encode_data(root, splits, cfg, tokenizer, model_config.vocab_size, cache, store, cache_id, run)
            import numpy as np
            stats = {}
            for split in ("train", "validation"):
                lengths = np.asarray(ds[split]["length"])
                stats[split] = dict(rows=len(lengths), total_tokens=int(lengths.sum()),
                                    length_percentiles={str(q): float(np.percentile(lengths, q)) for q in (0, 50, 90, 95, 99, 100)},
                                    raw_frames=int(sum(ds[split]["raw_frames"])),
                                    kept_frames=int(sum(ds[split]["kept_frames"])))
            atomic_json(run / "token_statistics.json", stats)
            if args.mode == "prepare":
                for p in run.iterdir():
                    if p.is_file() and p.name != ".lock":
                        store.put(p, "preparation/" + p.name)
                LOG.info("Preparation complete; reuse same config/run_id for training")
                return
            train(cfg, ds, tokenizer, splits, run, store, identity, session_start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Run failed; inspect the error above. Resume is available only if a verified checkpoint exists")
        sys.exit(1)
