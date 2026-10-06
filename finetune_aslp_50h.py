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

from pipeline_recovery import Heartbeat, set_activity, DeferredUploads, retry_io, valid_chunk, memory_selection, optional_evaluation, checkpoint_budget, ensure_checkpoint_space, save_with_space_retry, reconcile_run_identity, optional_action, export_checkpoint_adapter, compatible_encoding_identity

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
    for k in ("test_train_clips", "test_validation_clips"):
        if cfg[k] is not None and (not isinstance(cfg[k], int) or cfg[k] < 1):
            p.error(f"{k} must be null or a positive integer")
    if (cfg["test_train_clips"] or cfg["test_validation_clips"]) and "smoke" not in cfg["run_id"]:
        p.error("A test clip subset requires 'smoke' in run_id to distinguish it from full-epoch results")
    if cfg["init_from_run"] is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", str(cfg["init_from_run"])) \
                or cfg["init_from_run"] == cfg["run_id"]:
            p.error("init_from_run must be a DIFFERENT, finished run_id in the same Hugging Face repo")
        if cfg["checkpoint_backend"] != "huggingface":
            p.error("init_from_run needs checkpoint_backend=huggingface")
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
    if cfg["optimizer"] not in ("adamw_torch", "adamw_8bit"):
        p.error("optimizer must be adamw_torch or adamw_8bit")
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


def upload_files(store, items):
    """Batch uploads: a Hub store turns this into one commit instead of one per file."""
    items = list(items)
    put_many = getattr(store, "put_many", None)
    if put_many is not None:
        put_many(items)
    else:
        for local, relative in items:
            store.put(local, relative)


def stage_init_run(cfg, work, run, token, store_factory=None):
    """Seed a NEW run from a finished one: its final LoRA weights and encoded-token cache.

    Only weights are inherited; the optimizer, LR schedule and data order start fresh, so
    this is a genuine second epoch. Once the new run has its own checkpoint it resumes
    from that and this does nothing. Returns the adapter directory, or None.
    """
    source = cfg.get("init_from_run")
    if not source or latest_checkpoint(run) is not None:
        return None
    import shutil
    target, marker = run / "init_adapter", run / "init_source.json"
    if marker.is_file() and (target / "adapter_model.safetensors").is_file():
        return target
    factory = store_factory or (lambda run_id: HuggingFaceStore(cfg["hf_repo_id"], run_id, token))
    prior = factory(source)
    staging = work / "init_runs" / source
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    set_activity(f"fetching previous run {source}")
    prior.restore(staging)
    checkpoint = latest_checkpoint(staging)
    if checkpoint is None:
        raise RuntimeError(f"Run {source!r} has no verified checkpoint in {cfg['hf_repo_id']}; "
                           "check init_from_run (nothing was changed)")
    status = read_json(staging / "status.json") if (staging / "status.json").is_file() else {}
    if status.get("status") != "complete":
        raise RuntimeError(f"Run {source!r} is {status.get('status')!r}, not 'complete'. Finish that run first "
                           "(a TEST/smoke run or an unfinished epoch cannot seed another epoch)")
    target.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(checkpoint / name, target / (name + ".tmp"))
        (target / (name + ".tmp")).replace(target / name)
    atomic_json(marker, dict(init_from_run=source, checkpoint=checkpoint.name, step=status.get("step"),
                             adapter_sha256=sha256(target / "adapter_model.safetensors")))
    LOG.info("Initialising %s from run %s (%s, adapter sha256 %s...)", cfg["run_id"], source,
             checkpoint.name, read_json(marker)["adapter_sha256"][:12])
    # Reuse the previous run's SNAC-encoded tokens when the data and encoding code are unchanged.
    try:
        if (staging / "token_format.json").is_file() and not (run / "token_format.json").exists():
            shutil.copy2(staging / "token_format.json", run / "token_format.json")
            if (staging / "source").is_dir():
                shutil.copytree(staging / "source", run / "source", dirs_exist_ok=True)
            old_cache = fingerprint(read_json(staging / "token_format.json"))
            prior.restore_files("cache/" + old_cache, work / "cache" / old_cache, skip_existing=True)
    except Exception as exc:
        LOG.warning("Previous token cache not reused (%s); audio will be re-encoded", type(exc).__name__)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def load_init_adapter(model, directory):
    """Load previous LoRA weights into the fresh adapter, failing loudly on any mismatch."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    state = load_file(str(Path(directory) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(result, "unexpected_keys", None) or [])
    lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if unexpected or sum("lora_" in k for k in state) != len(lora):
        raise RuntimeError(f"Previous adapter does not match this model (unexpected keys: {unexpected[:3]}, "
                           f"adapter tensors {len(state)}, model LoRA tensors {len(lora)}); "
                           "keep rank/lora_alpha/targets equal to the previous run")
    if not any(float(p.detach().abs().sum()) > 0 for n, p in lora.items() if "lora_B" in n):
        raise RuntimeError("Previous adapter loaded as all zeros; refusing to train from an untrained adapter")
    LOG.info("Loaded %d LoRA tensors from the previous run's adapter", len(lora))


def find_data(cfg, work):
    if cfg["data_dir"]:
        root = Path(cfg["data_dir"])
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root.resolve()
    import gdown
    archive = work / "dataset.zip"
    dest = work / "extracted"
    marker = dest / ".extraction.json"
    if not archive.exists() and not marker.exists():
        temp = work / "dataset.partial.zip"
        LOG.info("Downloading supplied Drive archive")
        result = retry_io(lambda: gdown.download(url=cfg["download_url"], output=str(temp), fuzzy=True,
                                                     quiet=False, resume=True))
        if not result or not temp.is_file():
            raise RuntimeError("Drive download failed; check public access/quota or attach archive as Kaggle Dataset")
        import zipfile
        if not zipfile.is_zipfile(temp):
            raise ValueError("Download is not a ZIP (possibly a Drive quota/login page)")
        temp.replace(archive)
    dest = work / "extracted"
    marker = dest / ".extraction.json"
    archive_digest = sha256(archive) if archive.exists() else read_json(marker)["archive_sha256"]
    if marker.exists() and read_json(marker)["archive_sha256"] != archive_digest:
        raise RuntimeError("Archive changed; choose a new work_dir to avoid stale extracted files")
    if not marker.exists():
        LOG.info("Extracting and validating ZIP paths")
        safe_unzip(archive, dest)
        atomic_json(marker, {"archive_sha256": archive_digest})
    # The successfully extracted WAVs are retained; ZIP is a redundant download.
    archive.unlink(missing_ok=True)
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
    retention_issues = []
    if cfg['invalid_row_policy'] == 'skip':
        limit = cfg['maximum_rejected_train_fraction']
        for key in ('rejected_row_fraction', 'rejected_known_hour_fraction'):
            if train.get(key, 0) > limit:
                retention_issues.append(f'Training {key}={train[key]:.2%} exceeds configured {limit:.2%} limit')
    if train.get('hours', 0) < cfg['minimum_train_hours']:
        retention_issues.append(f"Only {train.get('hours', 0):.2f} retained training hours; "
                      f"minimum_train_hours={cfg['minimum_train_hours']}")
    if cfg.get('enforce_retention_limits', True) or cfg['invalid_row_policy'] == 'strict':
        errors.extend(retention_issues)
    else:
        train['retention_warnings'] = retention_issues
        for issue in retention_issues:
            LOG.warning('Retention warning; continuing under permissive policy: %s', issue)
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
        source_rows = len(df)
        # Test runs: seeded subset BEFORE audit/encoding; original CSV row numbers are kept.
        limit = cfg.get("test_train_clips") if split == "train" else cfg.get("test_validation_clips")
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=cfg["seed"]).sort_index()
            LOG.warning("TEST SUBSET: %s uses %d of %d manifest rows", split, len(df), source_rows)
        rows = []
        for i, original in zip(df.index.tolist(), df.to_dict("records")):
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
        manifest_info[split] = dict(file=path.name, sha256=sha256(path), rows=len(df), source_rows=source_rows)
    atomic_json(run / "data_errors.json", problems)
    pd.DataFrame(inventory).to_csv(run / "data_inventory.csv", index=False)
    report_audit_errors(run, problems, cfg["invalid_row_policy"])
    summary, failures = audit_retention(splits, manifest_info, problems, cfg)
    report = dict(splits=summary, manifests=manifest_info,
                  invalid_row_policy=cfg["invalid_row_policy"],
                  maximum_rejected_train_fraction=cfg["maximum_rejected_train_fraction"],
                  audit_passed=not failures, failure_reasons=failures,
                  duplicate_policy="First valid occurrence retained: train before validation before test",
                  training_cap=dict(train=cfg.get("test_train_clips"), validation_and_test=cfg.get("test_validation_clips"))
                  if cfg.get("test_train_clips") or cfg.get("test_validation_clips") else None,
                  trusted_speaker_column=cfg["speaker_column"],
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
    try:
        store.restore_files("cache/" + cache_id, cache, skip_existing=True)
    except RuntimeError as exc:
        LOG.warning("Remote cache restore unavailable (%s); verified local chunks reused and missing chunks regenerated", type(exc).__name__)
    codec = None
    files, encoding_errors, accepted_audio = {}, [], {}
    # Chunks already stored remotely are never re-committed (previously every restart
    # re-uploaded every chunk, one commit per file, which tripped Hub rate limits).
    listing = getattr(store, "remote_names", None)
    remote = listing() if listing is not None else set()
    pending, last_flush = [], time.monotonic()

    def flush(force=False):
        nonlocal pending, last_flush
        if pending and (force or len(pending) >= 40 or time.monotonic() - last_flush > 900):
            upload_files(store, pending)
            pending, last_flush = [], time.monotonic()
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
                if valid_chunk(path, meta, chunk_identity, len(subset), pd.read_parquet):
                    info = read_json(meta)
                else:
                    if path.exists() or meta.exists():
                        LOG.warning("Rebuilding damaged/incomplete derived cache chunk: %s", path)
                    if codec is None:
                        codec = retry_io(lambda: SNAC.from_pretrained(CODEC_ID, revision=cfg["codec_revision"])).eval().cuda()
                    encoded, excluded = [], []
                    for row in subset:
                        try:
                            wav, sr = sf.read(root / row["audio"], dtype="float32")
                            if sr != 24000:
                                raise ValueError("Audio changed after validation")
                            with torch.inference_mode():
                                codes = codec.encode(torch.from_numpy(wav).cuda().view(1, 1, -1))
                            c0, c1, c2 = [c[0].cpu().tolist() for c in codes]
                            tokens = interleave_codes(c0, c1, c2, cfg["dedup"])
                        except (ValueError, OSError, RuntimeError) as exc:
                            # Kernel/driver failures are global; do not mislabel all data as bad.
                            oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
                            row_error = isinstance(exc, (ValueError, OSError)) or type(exc).__name__.startswith('Libsndfile')
                            if cfg["invalid_row_policy"] != "skip" or not (row_error or oom):
                                raise
                            excluded.append(dict(split=split, row=row["source_row"], audio=row["audio"],
                                                 error=str(exc), duration_seconds=row["duration"], stage="encoding_audio_or_memory"))
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
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
                    set_activity(f"encoding {split} {start+len(subset)}/{len(rows)}")
                # Upload chunks missing remotely (including ones a previous upload lost), in batches.
                for local in (path, meta):
                    relative = f"cache/{cache_id}/{split}/{local.name}"
                    if relative not in remote:
                        pending.append((local, relative))
                flush()
                encoding_errors.extend(info.get("excluded", []))
                accepted = pd.read_parquet(path)["audio"].tolist()
                accepted_audio[split].extend(accepted)
                if info["rows"]:
                    paths.append(str(path))
                atomic_json(run / "encoding_errors.json", encoding_errors)
            files[split] = paths
            flush(force=True)
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
    upload_files(store, [(run / name, "preparation/" + name)
                         for name in ("encoding_errors.json", "dataset_report.json", "training_manifests.json")])
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

    model, loaded_tokenizer = retry_io(lambda: FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"],
        use_exact_model_name=True, max_seq_length=cfg["max_length"], dtype=None,
        load_in_4bit=cfg["precision"] == "4bit"))
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

    save_bytes = checkpoint_budget(model)
    def check_disk():
        pending = getattr(store, 'queue', {}).get('checkpoint')
        return ensure_checkpoint_space(run, save_bytes, remote=bool(store.remote), protected=[pending],
                                       archive_snapshots=getattr(store, "archive_snapshots", True))
    check_disk()  # Before any optimizer steps, including on fresh runs.
    previous = latest_checkpoint(run)
    if previous is None and cfg.get("init_from_run"):
        load_init_adapter(model, run / "init_adapter")
    def probe_memory(row):
        try:
            # AdamW moments and step temporaries are allocated after the first real
            # backward pass. Reserve their budget during the probe as well.
            reserve = torch.empty(trainable * 12 + 64 * 1024**2, dtype=torch.uint8,
                                  device=next(model.parameters()).device)
            with torch.random.fork_rng(devices=[0]):
                model.train()
                batch = {k: v.to(next(model.parameters()).device) for k, v in collate([row] * cfg["micro_batch"]).items()}
                dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
                for _ in range(min(cfg["grad_accum"], 2)):
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite loss in model preflight")
                    loss.backward()
                torch.cuda.synchronize()
        finally:
            batch, loss, reserve = None, None, None
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    ds["train"], memory_exclusions = memory_selection(ds["train"], run, probe_memory,
                                                      resuming=previous is not None)
    retained = set(ds["train"]["audio"])
    splits["train"] = [row for row in splits["train"] if row["audio"] in retained]
    if not splits["train"]:
        raise RuntimeError("No training rows remain after the GPU memory probe")
    report = read_json(run / "dataset_report.json")
    original_problems = read_json(run / "data_errors.json") + read_json(run / "encoding_errors.json")
    inventory = {row['audio']: row for row in read_json(run / 'training_manifests.json')['train']}
    memory_problems = [dict(split='train', duration_seconds=inventory[e['audio']]['duration'], **e)
                       for e in memory_exclusions]
    summary, failures = audit_retention(splits, report["manifests"], original_problems + memory_problems, cfg)
    report.update(splits=summary, memory_excluded_rows=len(memory_problems), failure_reasons=failures)
    atomic_json(run / "dataset_report.json", report)
    if failures:
        upload_files(store, [(run / name, "preparation/" + name) for name in ("training_selection.json", "dataset_report.json")])
        raise ValueError('; '.join(failures))
    atomic_json(run / "effective_training_manifests.json", splits)
    upload_files(store, [(run / name, "preparation/" + name) for name in
                         ("training_selection.json", "dataset_report.json", "effective_training_manifests.json")])
    LOG.info("Model backward preflight passed; training rows=%d, optimizer=%s", len(ds["train"]), cfg["optimizer"])

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
            optional_action(lambda: append_jsonl(run / "metrics.jsonl", record), stage="JSONL metrics")
            optional_action(lambda: atomic_json(run / "status.json", dict(status="training", **record)), stage="training status")
            if tensorboard[0] is not None:
                try:
                    for key, value in clean.items():
                        if value is not None:
                            tensorboard[0].add_scalar(key, value, state.global_step)
                except Exception as exc:
                    LOG.warning("Disabling optional TensorBoard after %s", type(exc).__name__)
                    tensorboard[0] = None
            if "loss" in scalar and not math.isfinite(scalar["loss"]):
                raise FloatingPointError("Nonfinite training loss recorded; stopping before publishing another checkpoint")

        def on_step_end(self, args, state, control, **kwargs):
            set_activity("training", state.global_step)
            remaining = cfg["session_hours"]*3600 - (time.monotonic()-session_start)
            # Budget includes preparation and downloads. Leave time for samples/upload.
            if stop_requested[0] or remaining < 900:
                control.should_save = True
                control.should_training_stop = True
                LOG.warning("Stopping at step %d for session limit/signal", state.global_step)
            if state.global_step == 1 or state.global_step >= state.max_steps:
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
                        set_activity(f"generating sample {i+1}/{len(samples)} after checkpoint-{state.global_step}")
                        try:
                            metadata = generate_audio(model, tokenizer, row["text"], dest,
                                                      speaker=row["speaker"], seed=cfg["seed"]+i,
                                                      max_length=cfg["max_length"],
                                                      max_new_tokens=cfg["sample_max_new_tokens"],
                                                      temperature=cfg["sample_temperature"],
                                                      top_p=cfg["sample_top_p"],
                                                      repetition_penalty=cfg["sample_repetition_penalty"],
                                                      codec_revision=cfg["codec_revision"],
                                                      max_time=cfg["sample_max_seconds"])
                            metadata.update(reference_audio=row["audio"], step=state.global_step)
                            atomic_json(dest.with_suffix(".json"), metadata)
                        except Exception as e:
                            LOG.exception("Monitoring synthesis failed; checkpoint remains the priority")
                            optional_action(lambda: append_jsonl(run / "sample_errors.jsonl",
                                         dict(step=state.global_step, sample=i, error=str(e))), stage="sample error log")
            except Exception as exc:
                # Inference setup is optional too; always restore training mode below.
                LOG.warning("Optional monitoring setup failed (%s); continuing training", type(exc).__name__)
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
            optional_action(lambda: atomic_json(run / "status.json", dict(status="checkpoint_saved", step=state.global_step,
                                                  epoch=state.epoch, timestamp=time.time())), stage="checkpoint status")
            # Upload before plots/synthesis; skip optional work near session cutoff.
            set_activity(f"uploading {checkpoint.name}", state.global_step)
            uploaded = store.backup(run, checkpoint)
            if cfg["session_hours"]*3600 - (time.monotonic()-session_start) < 900:
                LOG.info("Skipping optional samples/plots near session cutoff; checkpoint secured locally")
                return
            self.samples(state)
            optional_plots(run)
            # The required training snapshot is already secured. Upload only the new small
            # monitoring files; a second full snapshot doubled per-checkpoint upload and storage.
            try:
                step_dir = run / "samples" / f"step-{state.global_step:07d}"
                items = [(p, "monitoring/" + p.relative_to(run).as_posix())
                         for p in sorted(step_dir.glob("*")) if p.is_file()]
                items += [(run / name, "monitoring/" + name) for name in
                          ("status.json", "metrics.jsonl", "metrics.csv", "loss_and_lr.png", "sample_errors.jsonl")
                          if (run / name).is_file()]
                upload_files(store, items)
            except Exception as exc:
                LOG.warning("Optional sample snapshot refresh failed (%s); training snapshot remains saved",
                            type(exc).__name__)
                optional_action(lambda: append_jsonl(run / "optional_errors.jsonl", dict(stage="sample_snapshot_refresh",
                                                               error=str(exc))), stage="sample snapshot error log")
            LOG.info("Checkpoint %s verified locally; remote upload status: %s", checkpoint.name,
                     "local only" if not store.remote else ("uploaded" if uploaded else "pending (see backup_status.json)"))

    # Own the optional writer so a runtime TensorBoard error cannot abort Trainer.
    reporting = []
    tensorboard = [None]
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard[0] = SummaryWriter(log_dir=str(run / "tensorboard"))
    except Exception as exc:
        reporting = []
        LOG.warning("Optional TensorBoard unavailable (%s); canonical JSONL metrics remain enabled", type(exc).__name__)
        optional_action(lambda: append_jsonl(run / "optional_errors.jsonl", dict(stage="tensorboard", error=type(exc).__name__)), stage="TensorBoard error log")

    arguments = TrainingArguments(
        output_dir=str(run), num_train_epochs=1, max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["micro_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type=cfg["scheduler"],
        weight_decay=cfg["weight_decay"], max_grad_norm=1.0,
        optim=cfg["optimizer"], fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        eval_strategy="steps", eval_steps=cfg["eval_steps"], prediction_loss_only=True,
        save_strategy="steps", save_steps=cfg["save_steps"],
        save_total_limit=max(2, cfg["save_total_limit"]), save_only_model=False,
        logging_steps=cfg["logging_steps"], logging_first_step=True, logging_nan_inf_filter=False,
        report_to=reporting, logging_dir=str(run / "tensorboard"),
        group_by_length=cfg["sampling"] == "length", length_column_name="length",
        dataloader_drop_last=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        remove_unused_columns=False, seed=cfg["seed"], data_seed=cfg["seed"],
        ignore_data_skip=False, disable_tqdm=True, label_names=["labels"],
        load_best_model_at_end=False)
    monitor = Monitor()
    class ResilientTrainer(Trainer):
        def _save_checkpoint(self, model, trial):
            set_activity(f"saving checkpoint-{self.state.global_step}", self.state.global_step)
            check_disk()
            def recover():
                import shutil
                partial = run / f"checkpoint-{self.state.global_step}"
                # Only the failed current write; never remove a sealed checkpoint.
                if partial.is_dir() and not (partial / "COMPLETE.json").exists():
                    shutil.rmtree(partial)
                check_disk()
            return save_with_space_retry(
                lambda: super(ResilientTrainer, self)._save_checkpoint(model, trial), recover)

        def save_model(self, *args, **kwargs):
            check_disk()
            return super(ResilientTrainer, self).save_model(*args, **kwargs)

        def evaluate(self, *args, **kwargs):
            set_activity("validation", self.state.global_step)
            def cleanup():
                gc.collect()
                torch.cuda.empty_cache()
            return optional_evaluation(lambda: super(ResilientTrainer, self).evaluate(*args, **kwargs),
                                       run, self.state.global_step, cleanup)

    trainer = ResilientTrainer(model=model, args=arguments, train_dataset=ds["train"],
                      eval_dataset=eval_ds, data_collator=collate,
                      processing_class=tokenizer, callbacks=[monitor])
    previous = latest_checkpoint(run)
    LOG.info("Resuming: %s; effective batch=%d; train clips=%d", previous,
             cfg["micro_batch"]*cfg["grad_accum"], len(ds["train"]))
    if previous and read_json(previous / "run_identity.json") != identity:
        raise RuntimeError("Checkpoint run identity changed; use a new run_id for an ablation")
    result = trainer.train(resume_from_checkpoint=str(previous) if previous else None)
    optional_action(lambda: trainer.save_metrics("train", result.metrics), stage="train metrics export")
    optional_action(lambda: trainer.save_state(), stage="root Trainer state export")
    if tensorboard[0] is not None:
        optional_action(lambda: tensorboard[0].close(), stage="TensorBoard close")
    reached_schedule = trainer.state.global_step >= trainer.state.max_steps
    if not reached_schedule:
        atomic_json(run / "status.json", dict(status="paused_for_resume", step=trainer.state.global_step,
                                              epoch=trainer.state.epoch))
        cp = latest_checkpoint(run)
        if cp:
            store.backup(run, cp)
        LOG.info("Session ended safely. Rerun identical command/config to continue the SAME epoch")
        return
    cp = latest_checkpoint(run)
    if cp is None or cp.name != f"checkpoint-{trainer.state.global_step}":
        trainer._save_checkpoint(model, None)
        monitor.on_save(arguments, trainer.state, trainer.control)
        cp = latest_checkpoint(run)
    set_activity("exporting final adapter")
    final = export_checkpoint_adapter(cp, run / "adapter_final")
    tokenizer.save_pretrained(final)
    atomic_json(final / "run_identity.json", identity)
    atomic_json(final / "resolved_config.json", cfg)
    metrics = trainer.evaluate(eval_dataset=ds["validation"], metric_key_prefix="full_validation")
    optional_action(lambda: trainer.save_metrics("validation", metrics), stage="validation metrics export")
    smoke = cfg["max_steps"] > 0 or bool(cfg.get("test_train_clips") or cfg.get("test_validation_clips"))
    atomic_json(run / "status.json", dict(status="smoke_complete" if smoke else "complete",
                                          step=trainer.state.global_step, epoch=trainer.state.epoch,
                                          validation=metrics))
    optional_plots(run)
    cp = latest_checkpoint(run)
    if not cp:
        raise RuntimeError("No resumable final checkpoint was created")
    set_activity("final upload")
    store.backup(run, cp, force=True)
    set_activity("finished")
    LOG.info("Finished; adapter at %s", final)


def audit_with_reports(root, cfg, run, store):
    try:
        return audit_data(root, cfg, run)
    finally:
        # Keep failed-audit evidence remotely even before the first model checkpoint.
        names = ('data_errors.json', 'data_error_summary.json', 'data_inventory.csv', 'dataset_report.json', 'normalized_manifests.json')
        items = [(run / name, 'audit/' + name) for name in names if (run / name).is_file()]
        try:
            upload_files(store, items)
        except Exception as exc:
            LOG.warning('Audit report backup failed (%s); local files retained', type(exc).__name__)


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
            store = DeferredUploads(store, run)
            if args.mode == "storage-check":
                LOG.info("Storage preflight passed; no dataset download or GPU training requested")
                return
            # Stage/stack/log evidence on the remote every 10 min, independent of Kaggle logs.
            import atexit
            heartbeat = Heartbeat(run, store.store).start()
            atexit.register(heartbeat.stop)
            if args.mode == "audit":
                # CPU-only dataset inspection; no model imports, SNAC or tokenization.
                root = find_data(cfg, work)
                audit_with_reports(root, cfg, run, store)
                LOG.info("Dataset audit passed. No model loaded or training started")
                return
            store.restore(run)
            if cfg.get("init_from_run"):
                stage_init_run(cfg, work, run, os.environ.get("HF_TOKEN"))
            existing = latest_checkpoint(run)
            if existing is not None:
                previous_identity = read_json(existing / "run_identity.json")
                changed_sources = [name for name, digest in previous_identity.get("source_sha256", {}).items()
                                   if not (Path(__file__).parent / name).is_file()
                                   or sha256(Path(__file__).parent / name) != digest]
                if changed_sources:
                    raise RuntimeError(f"Verified {existing.name} of run_id {cfg['run_id']} was made by different code "
                                       f"({changed_sources}). To continue it, rerun with the commit that created it (REPO_REF); "
                                       "to start fresh on this code, set a new RUN_ID_OVERRIDE. Nothing was deleted; "
                                       "no audio processing was started")
            status = run / "status.json"
            if status.exists() and read_json(status).get("status") in ("complete", "smoke_complete"):
                previous_cfg = read_json(run / "resolved_config.json")
                changed = [k for k, v in cfg.items() if k not in
                           ("work_dir", "data_dir", "session_hours", "download_url", "model_revision", "codec_revision", "checkpoint_backend", "hf_repo_id")
                           and previous_cfg.get(k) != v]
                if changed:
                    raise RuntimeError(f"Completed run_id reused with different settings {changed}; choose a new run_id")
                completed_checkpoint = latest_checkpoint(run)
                if completed_checkpoint:
                    store.backup(run, completed_checkpoint, force=True)
                LOG.info("This run is already complete; pending remote backup retried. Use synthesize.py or a new run_id")
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
            cfg["model_revision"] = retry_io(lambda: api.model_info(cfg["model_id"], revision=cfg["model_revision"])).sha
            cfg["codec_revision"] = retry_io(lambda: api.model_info(CODEC_ID, revision=cfg["codec_revision"])).sha
            tokenizer = retry_io(lambda: AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"]))
            model_config = retry_io(lambda: AutoConfig.from_pretrained(cfg["model_id"], revision=cfg["model_revision"]))
            required_vocab = max(max(SPECIAL.values()), AUDIO_BASE + 7 * 4096 - 1) + 1
            if model_config.vocab_size < required_vocab:
                raise RuntimeError(f"Base model vocabulary {model_config.vocab_size} cannot represent Orpheus audio tokens; expected >= {required_vocab}")
            root = find_data(cfg, work)
            splits = audit_with_reports(root, cfg, run, store)
            vocab_hash = fingerprint(tokenizer.get_vocab())
            token_identity = dict(format_version=FORMAT_VERSION, model_id=cfg["model_id"],
                                  model_revision=cfg["model_revision"], tokenizer_sha256=vocab_hash,
                                  encoding_source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                                          ("finetune_aslp_50h.py", "orpheus_utils.py", "pipeline_recovery.py")},
                                  encoding_packages={name: importlib.metadata.version(name) for name in
                                                     ("torch", "snac", "tokenizers", "soundfile", "numpy")},
                                  codec=CODEC_ID, codec_revision=cfg["codec_revision"],
                                  max_length=cfg["max_length"], objective=cfg["objective"], dedup=cfg["dedup"],
                                  special=SPECIAL, audio_base=AUDIO_BASE,
                                  splits={s: fingerprint(rows) for s, rows in splits.items()})
            old_format = run / "token_format.json"
            if old_format.exists():
                token_identity = compatible_encoding_identity(
                    read_json(old_format), token_identity, run / "source", Path(__file__).parent)
            cache_id = fingerprint(token_identity)
            env, freeze, gpu = environment(run)
            # Same code, dependency versions, sequence order, schedule, objective and batch geometry.
            semantic = {k: v for k, v in cfg.items() if k not in
                        ("work_dir", "data_dir", "download_url", "train_manifest", "validation_manifest", "test_manifest", "session_hours", "checkpoint_backend", "hf_repo_id")}
            identity = dict(config=semantic, token_cache_sha256=cache_id,
                            packages=env["packages"], torch=env["torch"],
                            source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                           ("finetune_aslp_50h.py", "orpheus_utils.py", "synthesize.py", "pipeline_recovery.py")})
            identity_path = run / "run_identity.json"
            reconcile_run_identity(run, identity)
            atomic_json(run / "environment.json", env)
            (run / "requirements-resolved.txt").write_text(freeze)
            if gpu:
                (run / "gpu.txt").write_text(gpu)
            import shutil
            sources = run / "source"
            sources.mkdir(exist_ok=True)
            for name in (*identity["source_sha256"], "evaluate_asr.py"):
                original, captured = Path(__file__).parent / name, sources / name
                if original.resolve() != captured.resolve():
                    shutil.copy2(original, captured)
            (sources / "configs").mkdir(exist_ok=True)
            if DEFAULTS.resolve() != (sources / "configs/aslp50h.json").resolve():
                shutil.copy2(DEFAULTS, sources / "configs/aslp50h.json")
            atomic_json(identity_path, identity)
            atomic_json(run / "resolved_config.json", cfg)
            atomic_json(run / "token_format.json", token_identity)
            items = [(p, "preparation/" + p.name) for p in
                     (identity_path, run / "resolved_config.json", run / "dataset_report.json",
                      run / "requirements-resolved.txt", run / "environment.json", run / "token_format.json")]
            items += [(source, "preparation/source/" + str(source.relative_to(sources)))
                      for source in sorted(sources.rglob("*")) if source.is_file() and source.suffix in (".py", ".json")]
            upload_files(store, items)
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
                upload_files(store, [(p, "preparation/" + p.name) for p in run.iterdir()
                                     if p.is_file() and p.name != ".lock"])
                LOG.info("Preparation complete; reuse same config/run_id for training")
                return
            train(cfg, ds, tokenizer, splits, run, store, identity, session_start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Run failed; inspect the error above. Resume is available only if a verified checkpoint exists")
        sys.exit(1)
