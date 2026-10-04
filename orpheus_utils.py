"""CPU-only data/token/artifact utilities; no CUDA imports at module scope."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path

SPECIAL = dict(end_text=128009, start_speech=128257, end_speech=128258,
               start_human=128259, end_human=128260, start_ai=128261,
               end_ai=128262, pad=128263)
AUDIO_BASE = 128266
OFFSETS = [4096 * i for i in range(7)]
CODEC_ID = "hubertsiuzdak/snac_24khz"
FORMAT_VERSION = 1


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\t\n")
    return " ".join(text.split())


def contained(root, relative):
    root = Path(root).resolve()
    # Reject backslashes too: archives may be made on Windows.
    p = Path(str(relative).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", str(p)):
        raise ValueError(f"Unsafe relative path: {relative}")
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes dataset: {relative}")
    return resolved


def safe_unzip(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            contained(destination, member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are unsupported")
        z.extractall(destination)


def safe_untar(archive, destination):
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as t:
        for m in t.getmembers():
            contained(destination, m.name)
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"Unsupported TAR member: {m.name}")
        # Members validated above; filter also protects modern Python runtimes.
        t.extractall(destination, filter="data")


def prompt_ids(tokenizer, text, speaker=None):
    text = clean_text(text)
    if not text:
        raise ValueError("Empty synthesis text")
    prompt = f"{speaker}: {text}" if speaker else text
    return ([SPECIAL["start_human"]] + tokenizer.encode(prompt, add_special_tokens=True)
            + [SPECIAL["end_text"], SPECIAL["end_human"], SPECIAL["start_ai"],
               SPECIAL["start_speech"]])


def interleave_codes(c0, c1, c2, dedup="none"):
    if not c0 or len(c1) != 2 * len(c0) or len(c2) != 4 * len(c0):
        raise ValueError("Unexpected SNAC codebook lengths")
    if any(not 0 <= v < 4096 for codes in (c0, c1, c2) for v in codes):
        raise ValueError("SNAC code outside codebook")
    if dedup not in ("none", "coarse"):
        raise ValueError("Unknown frame deduplication policy")
    result, previous = [], None
    for i, v in enumerate(c0):
        frame = [v, c1[2*i], c2[4*i], c2[4*i+1], c1[2*i+1], c2[4*i+2], c2[4*i+3]]
        if dedup == "none" or previous is None or v != previous:
            result.extend(AUDIO_BASE + v + offset for v, offset in zip(frame, OFFSETS))
            previous = frame[0]
    return result


def training_row(prefix, audio_ids, objective, max_length, vocab_size):
    ids = prefix + audio_ids + [SPECIAL["end_speech"], SPECIAL["end_ai"]]
    if len(ids) > max_length:
        raise ValueError(f"Sequence has {len(ids)} tokens > {max_length}; increase max_length, rebuild cache")
    if min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError("Token ID outside model vocabulary")
    if objective not in ("all", "audio"):
        raise ValueError("Unknown training objective")
    labels = ids.copy()
    if objective == "audio":
        # Learn start_speech, all audio, and both stop delimiters.
        labels[:len(prefix)-1] = [-100] * (len(prefix)-1)
    return dict(input_ids=ids, labels=labels, attention_mask=[1]*len(ids), length=len(ids))


def decode_frames(ids):
    stopped = SPECIAL["end_speech"] in ids
    if stopped:
        ids = ids[:ids.index(SPECIAL["end_speech"])]
    frames, invalid = [], None
    for start in range(0, len(ids)-6, 7):
        f = [v - AUDIO_BASE - offset for v, offset in zip(ids[start:start+7], OFFSETS)]
        if any(not 0 <= v < 4096 for v in f):
            invalid = start
            break
        frames.append(f)
    status = dict(ended_with_speech_stop=stopped, generated_tokens=len(ids),
                  complete_frames=len(frames), invalid_frame_token_offset=invalid,
                  trailing_tokens=len(ids) % 7)
    if not frames:
        raise ValueError(f"No valid SNAC frames; first tokens: {ids[:14]}")
    codes = [[f[0] for f in frames], [v for f in frames for v in (f[1], f[4])],
             [v for f in frames for v in (f[2], f[3], f[5], f[6])]]
    return codes, status


def seal_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    required = ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"]
    missing = [n for n in required if not (checkpoint / n).is_file()]
    if missing:
        raise RuntimeError(f"Checkpoint is not resumable: missing {missing}")
    files = {str(p.relative_to(checkpoint)): sha256(p) for p in sorted(checkpoint.rglob("*"))
             if p.is_file() and p.name != "COMPLETE.json" and not p.name.endswith(".tmp")}
    atomic_json(checkpoint / "COMPLETE.json", dict(files=files))


def verify_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    marker = checkpoint / "COMPLETE.json"
    if not marker.is_file():
        return False
    for relative, digest in read_json(marker)["files"].items():
        p = contained(checkpoint, relative)
        if not p.is_file() or sha256(p) != digest:
            raise RuntimeError(f"Checkpoint checksum failed: {p}")
    return True


def latest_checkpoint(run):
    candidates = sorted(Path(run).glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for p in candidates:
        if verify_checkpoint(p):
            return p
    return None


class DriveStore:
    """rclone OAuth credentials live outside experiment artifacts. No remote deletes."""
    def __init__(self, remote=None, config=None):
        self.remote = remote.rstrip("/") if remote else None
        self.config = config

    def run(self, *args):
        cmd = ["rclone"]
        if self.config:
            cmd += ["--config", str(self.config)]
        cmd += ["--retries", "3", "--low-level-retries", "10", "--contimeout", "30s",
                "--timeout", "5m", "--log-level", "ERROR", *map(str, args)]
        # Do not print stderr: a backend error could include sensitive token/config fields.
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"Drive operation {args[0]} failed (code {r.returncode}); check access/quota/network")
        return r.stdout

    def preflight(self):
        if self.remote:
            # Write AND read a probe before allocating GPU time.
            with tempfile.TemporaryDirectory() as d:
                probe = Path(d) / "probe.json"
                atomic_json(probe, {"time": time.time()})
                self.put(probe, "connection_probe.json")
                out = Path(d) / "readback.json"
                self.get("connection_probe.json", out)
                if sha256(probe) != sha256(out):
                    raise RuntimeError("Drive probe checksum mismatch")

    def put(self, local, relative):
        if self.remote:
            self.run("copyto", local, self.remote + "/" + relative)

    def get(self, relative, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.run("copyto", self.remote + "/" + relative, local)

    def names(self):
        if not self.remote:
            return []
        return self.run("lsf", self.remote, "--files-only").splitlines()

    def restore(self, run):
        if not self.remote:
            return
        if "latest.json" not in self.names():
            folders = self.run("lsf", self.remote, "--dirs-only").splitlines()
            if "preparation/" in folders:
                # Encoding may have been interrupted BEFORE any Trainer checkpoint existed.
                self.run("copy", self.remote + "/preparation", run, "--ignore-existing")
            return
        with tempfile.TemporaryDirectory(dir=Path(run).parent) as d:
            pointer = Path(d) / "latest.json"
            self.get("latest.json", pointer)
            info = read_json(pointer)
            archive = Path(d) / "snapshot.tar.gz"
            self.get(info["archive"], archive)
            if sha256(archive) != info["sha256"]:
                raise RuntimeError("Drive snapshot checksum mismatch")
            staging = Path(d) / "staging"
            staging.mkdir()
            safe_untar(archive, staging)
            cp = staging / info["checkpoint"]
            if not verify_checkpoint(cp):
                raise RuntimeError("Remote checkpoint missing completion marker")
            # Refuse to overwrite newer local progress.
            local = latest_checkpoint(run)
            if local and int(local.name.split("-")[-1]) >= info["step"]:
                return
            import shutil
            shutil.copytree(staging, run, dirs_exist_ok=True)

    def backup(self, run, checkpoint):
        if not self.remote:
            return
        run, checkpoint = Path(run), Path(checkpoint)
        if not verify_checkpoint(checkpoint):
            raise RuntimeError("Refusing to upload incomplete checkpoint")
        with tempfile.TemporaryDirectory(dir=run.parent) as d:
            archive = Path(d) / "snapshot.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=1) as t:
                for p in sorted(run.iterdir()):
                    if p.name.startswith("checkpoint-") and p != checkpoint:
                        continue
                    if p.name in (".lock",) or p.name.endswith(".tmp"):
                        continue
                    t.add(p, arcname=p.name)
            digest = sha256(archive)
            remote_name = f"snapshots/{checkpoint.name}-{digest[:12]}.tar.gz"
            self.put(archive, remote_name)
            # rclone verifies transfers; pointer is published only AFTER archive succeeds.
            pointer = Path(d) / "latest.json"
            atomic_json(pointer, dict(archive=remote_name, sha256=digest,
                                      checkpoint=checkpoint.name,
                                      step=int(checkpoint.name.split("-")[-1])))
            self.put(pointer, "latest.json")


def plot_logs(run):
    """Keep logs canonical; plots are derived and can be recreated after interruption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    path = run / "metrics.jsonl"
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    import csv
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(records)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for key in ("loss", "eval_loss", "full_validation_loss"):
        # A resumed run may repeat steps; retain latest measurement per step/key.
        points = {r["step"]: r[key] for r in records if isinstance(r.get(key), (int, float))}
        if points:
            xy = sorted(points.items())
            axes[0].plot([x for x, _ in xy], [y for _, y in xy], label=key)
    axes[0].set(xlabel="Optimizer step", ylabel="Token cross entropy")
    if axes[0].lines:
        axes[0].legend()
    points = {r["step"]: r["learning_rate"] for r in records if isinstance(r.get("learning_rate"), (int, float))}
    xy = sorted(points.items())
    axes[1].plot([x for x, _ in xy], [y for _, y in xy])
    axes[1].set(xlabel="Optimizer step", ylabel="Learning rate")
    fig.tight_layout()
    fig.savefig(run / "loss_and_lr.png", dpi=160)
    plt.close(fig)
