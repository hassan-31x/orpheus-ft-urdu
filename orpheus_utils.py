"""CPU-only data/token/artifact utilities; no CUDA imports at module scope."""
from __future__ import annotations

import hashlib
import json
import logging
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
    files = read_json(marker)["files"]
    required = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"}
    if not isinstance(files, dict) or not required <= set(files):
        raise RuntimeError(f"Incomplete checkpoint manifest: {checkpoint}")
    for relative, digest in files.items():
        p = contained(checkpoint, relative)
        if not p.is_file() or sha256(p) != digest:
            raise RuntimeError(f"Checkpoint checksum failed: {p}")
    return True


def latest_checkpoint(run):
    candidates = sorted((p for p in Path(run).glob("checkpoint-*")
                         if p.is_dir() and p.name.split("-")[-1].isdigit()),
                        key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    damaged = []
    for p in candidates:
        try:
            if verify_checkpoint(p):
                if damaged:
                    logging.getLogger("orpheus_urdu").warning(
                        "Ignoring damaged checkpoints %s; resuming verified %s", damaged, p.name)
                return p
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            damaged.append(p.name)
    if damaged:
        raise RuntimeError(f"No verified checkpoint remains; damaged candidates: {damaged}. "
                           "Restore a remote copy before resuming")
    return None


class SnapshotStore:
    """Verified artifacts shared by local, Drive and Hugging Face storage."""
    def __init__(self, remote=None):
        self.remote = remote

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
                    raise RuntimeError("Remote storage probe checksum mismatch")

    def put(self, local, relative):
        if self.remote:
            raise NotImplementedError

    def put_many(self, items, lane="files"):
        """Upload [(local, relative), ...]; Hub overrides this with one commit."""
        for local, relative in items:
            self.put(local, relative)

    def get(self, relative, local):
        raise NotImplementedError

    def names(self):
        return []

    def restore_files(self, prefix, destination, skip_existing=False):
        """Restore a preparation folder or immutable cache without backend-specific calls."""
        if not self.remote:
            return
        prefix = prefix.strip("/") + "/"
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(self.names()):
            if not name.startswith(prefix) or ".tmp" in Path(name).name:
                continue
            local = contained(destination, name[len(prefix):])
            if skip_existing and local.exists():
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            temp = local.with_name(local.name + ".download.tmp")
            try:
                self.get(name, temp)
                temp.replace(local)
            finally:
                temp.unlink(missing_ok=True)

    def restore(self, run):
        if not self.remote:
            return
        if "latest.json" not in self.names():
            # Encoding may have been interrupted BEFORE any Trainer checkpoint existed.
            self.restore_files("preparation", run, skip_existing=True)
            return
        with tempfile.TemporaryDirectory(dir=Path(run).parent) as d:
            pointer = Path(d) / "latest.json"
            self.get("latest.json", pointer)
            info = read_json(pointer)
            archive = Path(d) / "snapshot.tar.gz"
            self.get(info["archive"], archive)
            if sha256(archive) != info["sha256"]:
                raise RuntimeError("Remote snapshot checksum mismatch")
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
            with tarfile.open(archive, "w:gz", compresslevel=1, dereference=True) as t:
                for p in sorted(run.iterdir()):
                    if p.name.startswith("checkpoint-") and p != checkpoint:
                        continue
                    if p.name in (".lock",) or p.name.endswith(".tmp"):
                        continue
                    t.add(p, arcname=p.name)
            digest = sha256(archive)
            remote_name = f"snapshots/{checkpoint.name}-{digest[:12]}.tar.gz"
            pointer = Path(d) / "latest.json"
            atomic_json(pointer, dict(archive=remote_name, sha256=digest,
                                      checkpoint=checkpoint.name,
                                      step=int(checkpoint.name.split("-")[-1])))
            self.publish_snapshot(archive, remote_name, pointer)

    def publish_snapshot(self, archive, remote_name, pointer):
        # Drive uploads the pointer only AFTER the immutable archive succeeds.
        self.put(archive, remote_name)
        self.put(pointer, "latest.json")


class DriveStore(SnapshotStore):
    """rclone OAuth credentials live outside experiment artifacts. No remote deletes."""
    def __init__(self, remote=None, config=None):
        super().__init__(remote.rstrip("/") if remote else None)
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

    def put(self, local, relative):
        if self.remote:
            self.run("copyto", local, self.remote + "/" + relative)

    def get(self, relative, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.run("copyto", self.remote + "/" + relative, local)

    def names(self):
        if not self.remote:
            return []
        return self.run("lsf", self.remote, "--files-only", "--recursive").splitlines()

    def restore_files(self, prefix, destination, skip_existing=False):
        if self.remote and any(n.startswith(prefix.rstrip("/") + "/") for n in self.names()):
            options = ["--ignore-existing"] if skip_existing else []
            self.run("copy", self.remote + "/" + prefix.strip("/"), destination,
                     "--exclude", "*.tmp*", *options)


_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{6,}"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(token|access_token|password|authorization)([=:]\s*)[^\s&\"',]+"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
)


def redact(text, *secrets, limit=800):
    """Remove Hub tokens and auth headers before text reaches logs or reports."""
    text = str(text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "***")
    text = _SECRET_PATTERNS[0].sub("hf_***", text)
    text = _SECRET_PATTERNS[1].sub(lambda m: m.group(1) + " ***", text)
    text = _SECRET_PATTERNS[2].sub(lambda m: m.group(1) + m.group(2) + "***", text)
    text = _SECRET_PATTERNS[3].sub("://***@", text)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


HUB_HINTS = {
    "authentication": "HF_TOKEN was rejected (invalid, expired, revoked or not sent). Create a new token and update the Kaggle Secret.",
    "permission": "HF_TOKEN is valid but may not write to this repository. Use a 'write' token, or a fine-grained token with 'Write access to contents/settings of all repos' or this repo selected.",
    "not_found": "Repository not visible to this token: check hf_repo_id spelling/owner and that a fine-grained token includes this repository.",
    "quota": "Hub storage or usage quota appears exceeded; free space in the account or use a different repository.",
    "rate_limit": "Hub rate limit reached; retry later or reduce commit frequency.",
    "payload_too_large": "File exceeds a Hub request size limit.",
    "conflict": "Concurrent Hub commit conflict; normally transient.",
    "server": "Hugging Face server error; normally transient (see status.huggingface.co).",
    "connectivity": "Network failure reaching huggingface.co: check Kaggle Internet setting, proxy/DNS and HF_ENDPOINT.",
    "offline_mode": "HF_HUB_OFFLINE is set; unset it to allow uploads.",
    "api_request": "Hub rejected the request; see server_message.",
    "client_or_local": "Error raised locally by the Hub client before/without an HTTP response; see exception_type and message.",
    "timeout": "Upload exceeded its time limit (stalled connection or very slow transfer); it stays queued locally and is retried later.",
}
_RETRYABLE = {"rate_limit", "conflict", "server", "connectivity", "client_or_local"}


class HubOperationError(RuntimeError):
    """Redacted Hub failure; .diagnostics holds JSON-safe evidence for reports."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _header(response, name):
    try:
        return (getattr(response, "headers", None) or {}).get(name)
    except Exception:
        return None


def describe_hub_error(exc, operation, *secrets):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    status = status if isinstance(status, int) else None
    server_message = getattr(exc, "server_message", None) or _header(response, "x-error-message")
    if not server_message and response is not None:
        try:
            body = response.json()
            server_message = body.get("error") if isinstance(body, dict) else None
        except Exception:
            server_message = None
    chain, seen = [], exc
    while seen is not None and len(chain) < 6:
        chain.append(seen)
        seen = seen.__cause__ or seen.__context__
    names = [c.__name__ for e in chain for c in type(e).__mro__]
    message = redact(exc, *secrets)
    text = f"{server_message or ''} {message}".lower()
    if status is None:
        if "OfflineModeIsEnabled" in names:
            category = "offline_mode"
        elif any(k in n for n in names for k in ("ConnectionError", "ConnectError", "Timeout", "SSLError",
                                                   "ProxyError", "NameResolution", "socket")):
            category = "connectivity"
        else:
            category = "client_or_local"
    elif status == 507 or any(k in text for k in ("quota", "storage limit", "insufficient storage",
                                                    "exceeded your", "storage space")):
        category = "quota"
    else:
        category = {401: "authentication", 403: "permission", 404: "not_found", 408: "connectivity",
                    409: "conflict", 412: "conflict", 413: "payload_too_large",
                    429: "rate_limit"}.get(status, "server" if status >= 500 else "api_request")
    retry_after = _header(response, "retry-after")
    rate_limit = _header(response, "ratelimit")
    if retry_after is None and rate_limit:
        # Hub format (IETF draft): "api";r=0;t=<seconds until reset>
        match = re.search(r"\bt=(\d+)", str(rate_limit))
        retry_after = match.group(1) if match else None
    return dict(operation=operation, category=category, http_status=status,
                exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exception_chain=[f"{type(e).__module__}.{type(e).__qualname__}" for e in chain[1:]],
                server_message=redact(server_message, *secrets) if server_message else None,
                request_id=_header(response, "x-request-id"), error_code=_header(response, "x-error-code"),
                retry_after=retry_after, rate_limit=rate_limit, message=message, hint=HUB_HINTS[category])


def hub_token_report(whoami, repo_id):
    """Summarise token scope from HfApi.whoami(); can_write None means undetermined."""
    namespace = repo_id.split("/")[0]
    access = ((whoami or {}).get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    orgs = [o.get("name") for o in (whoami or {}).get("orgs") or [] if isinstance(o, dict)]
    report = dict(user=(whoami or {}).get("name"), token_role=role, token_name=access.get("displayName"),
                  namespace_is_user_or_org=namespace in [(whoami or {}).get("name"), *orgs], can_write=None)
    if role == "write":
        report["can_write"] = True
    elif role == "read":
        report["can_write"] = False
    elif role == "fineGrained":
        scoped = (access.get("fineGrained") or {}).get("scoped") or []
        grants = []
        for entry in scoped:
            entity = entry.get("entity") or {}
            if entity.get("name") in (namespace, repo_id):
                grants += entry.get("permissions") or []
        report["fine_grained_permissions_for_repo"] = sorted(set(grants))
        report["can_write"] = "repo.write" in grants
    return report


class HuggingFaceStore(SnapshotStore):
    """Private Hub model repo; synchronous commits and bounded transfer retries.

    Training snapshots and caches are stored under runs/<run_id>/. This is an
    artifact repository, not a directly loadable push_to_hub model directory.
    Dependencies are imported lazily so CPU tests/local/Drive do not need Hub auth.
    """
    def __init__(self, repo_id, run_id, token=None, *, api=None, download=None, commit_add=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id or ""):
            raise ValueError("hf_repo_id must be USERNAME/REPOSITORY")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id):
            raise ValueError("Invalid run_id")
        if api is None or download is None or commit_add is None:
            from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
            api = api if api is not None else HfApi(token=token)
            download = download if download is not None else hf_hub_download
            commit_add = commit_add if commit_add is not None else CommitOperationAdd
        self.repo_id, self.prefix = repo_id, f"runs/{run_id}"
        super().__init__(f"hf://{repo_id}/{self.prefix}")
        self.api, self._download, self._commit_add = api, download, commit_add
        self._token, self._files = token, None
        self.token_report, self.last_error = None, None
        self._inflight = {}

    # Hub/Xet transfers have no overall deadline; a stalled one must not freeze training.
    UPLOAD_TIMEOUTS = {"checkpoint": 1800, "files": 600, "heartbeat": 300}

    def _bounded(self, lane, operation, function):
        import threading
        previous = self._inflight.get(lane)
        if previous is not None and previous.is_alive():
            raise HubOperationError(f"Hugging Face {operation} skipped: the previous {lane} upload is still "
                                    "running in the background", self._timeout_info(operation, still_running=True))
        result = {}

        def work():
            try:
                result["value"] = function()
            except BaseException as exc:
                result["error"] = exc
        thread = threading.Thread(target=work, name=f"hub-{lane}", daemon=True)
        self._inflight[lane] = thread
        thread.start()
        thread.join(self.UPLOAD_TIMEOUTS[lane])
        if thread.is_alive():
            info = self._timeout_info(operation)
            self.last_error = info
            logging.getLogger("orpheus_urdu").warning("Hugging Face %s timed out: %s", operation, json.dumps(info))
            raise HubOperationError(f"Hugging Face {operation} timed out after {self.UPLOAD_TIMEOUTS[lane]}s. "
                                    f"{HUB_HINTS['timeout']}", info)
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _timeout_info(self, operation, still_running=False):
        return dict(operation=operation, category="timeout", http_status=None, exception_type="TimeoutError",
                    exception_chain=[], server_message=None, request_id=None, error_code=None,
                    retry_after=None, rate_limit=None, still_running=still_running,
                    message="previous upload still running" if still_running else "upload deadline exceeded",
                    hint=HUB_HINTS["timeout"])

    def _call(self, operation, function, *, missing_ok=False, attempts=3, **kwargs):
        log = logging.getLogger("orpheus_urdu")
        for attempt in range(1, attempts + 1):
            try:
                return function(**kwargs)
            except Exception as exc:
                info = describe_hub_error(exc, operation, self._token)
                if missing_ok and info["http_status"] == 404:
                    return None
                info["attempt"] = f"{attempt}/{attempts}"
                if self.token_report and info["category"] in ("authentication", "permission", "not_found"):
                    info["token_report"] = self.token_report
                self.last_error = info
                log.warning("Hugging Face %s attempt %d/%d failed: %s", operation, attempt, attempts,
                            json.dumps(info, ensure_ascii=False))
                if info["category"] not in _RETRYABLE or attempt == attempts:
                    # Original traceback suppressed: request objects may carry auth headers.
                    # The redacted evidence above and in .diagnostics replaces it.
                    raise HubOperationError(
                        f"Hugging Face {operation} failed [{info['category']}] HTTP {info['http_status']} "
                        f"{info['exception_type']}: {info['server_message'] or info['message']} "
                        f"(request id {info['request_id']}). {info['hint']}"
                        + (f" Token scope: {json.dumps(self.token_report)}" if "token_report" in info else ""),
                        info) from None
                delay = 2 ** attempt
                if info["category"] == "rate_limit":
                    # Waiting out a long limit here would stall encoding/training and further
                    # retries keep the limit tripped; callers' deferred queue backs off instead.
                    try:
                        delay = max(float(info["retry_after"]), 5)
                    except (TypeError, ValueError):
                        delay = None
                    if delay is None or delay > 60:
                        raise HubOperationError(
                            f"Hugging Face {operation} rate limited (HTTP 429, retry-after "
                            f"{info['retry_after']}): {info['server_message'] or info['message']}. {info['hint']}",
                            info) from None
                time.sleep(delay)

    def inspect_token(self):
        """Record token role/scope as evidence; never fatal except a rejected token."""
        whoami = getattr(self.api, "whoami", None)
        if whoami is None:
            return None
        try:
            # One attempt: whoami is rate limited and this check is diagnostic only.
            self.token_report = hub_token_report(self._call("token inspection", whoami, attempts=1), self.repo_id)
        except HubOperationError as exc:
            if exc.diagnostics["category"] == "authentication":
                raise
            return None
        log = logging.getLogger("orpheus_urdu")
        log.info("HF token: %s", json.dumps(self.token_report))
        if self.token_report["can_write"] is False:
            log.warning("HF_TOKEN appears unable to write to %s; the probe upload will confirm", self.repo_id)
        return self.token_report

    def _path(self, relative):
        # Validate the remote artifact path as well as local extraction paths.
        contained(Path("/tmp/orpheus-hub-paths"), relative)
        return self.prefix + "/" + relative

    def preflight(self):
        log = logging.getLogger("orpheus_urdu")
        try:
            import importlib.metadata as md
            versions = {}
            for name in ("huggingface_hub", "hf_xet", "hf_transfer", "requests", "httpx"):
                try:
                    versions[name] = md.version(name)
                except md.PackageNotFoundError:
                    versions[name] = None
            env = {k: redact(os.environ[k]) for k in ("HF_ENDPOINT", "HF_HUB_OFFLINE", "HF_HUB_ENABLE_HF_TRANSFER",
                                                     "HF_HUB_DISABLE_XET", "HTTPS_PROXY", "HTTP_PROXY")
                   if k in os.environ}
            log.info("Hub client: %s env=%s", versions, env)
        except Exception:
            pass
        self.inspect_token()
        info = self._call("repository inspection", self.api.repo_info,
                          repo_id=self.repo_id, repo_type="model", missing_ok=True)
        if info is None:
            self._call("repository setup", self.api.create_repo, repo_id=self.repo_id,
                       repo_type="model", private=True, exist_ok=True)
            info = self._call("repository inspection", self.api.repo_info,
                              repo_id=self.repo_id, repo_type="model")
        if not info.private:
            raise RuntimeError("Checkpoint repository must be private; choose a private repo or change its visibility on Hugging Face")
        super().preflight()

    def put(self, local, relative):
        self._call("upload", self.api.upload_file, repo_id=self.repo_id, repo_type="model",
                   path_or_fileobj=str(local), path_in_repo=self._path(relative),
                   commit_message=f"Save {self.prefix}/{relative}", run_as_future=False)
        if self._files is not None:
            self._files.add(relative)

    def put_many(self, items, lane="files"):
        # One commit for many files: Hub limits commits separately from API requests.
        items = list(items)
        if not items:
            return
        operations = [self._commit_add(path_in_repo=self._path(relative), path_or_fileobj=str(local))
                      for local, relative in items]
        self._bounded(lane, "batch upload", lambda: self._call(
            "batch upload", self.api.create_commit, repo_id=self.repo_id, repo_type="model",
            operations=operations, commit_message=f"Save {len(items)} files under {self.prefix}",
            run_as_future=False))
        if self._files is not None:
            self._files.update(relative for _, relative in items)

    def get(self, relative, local):
        import shutil
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        # local_dir avoids retaining another whole archive in the global Hub cache.
        # Download sidecar metadata and any credentials remain outside run artifacts.
        with tempfile.TemporaryDirectory(dir=local.parent, prefix="hub-download-") as d:
            downloaded = self._call("download", self._download, repo_id=self.repo_id,
                                    repo_type="model", filename=self._path(relative),
                                    token=self._token, local_dir=d)
            Path(downloaded).replace(local)

    def names(self):
        if self._files is None:
            files = self._call("file listing", self.api.list_repo_files,
                               repo_id=self.repo_id, repo_type="model")
            prefix = self.prefix + "/"
            self._files = {p[len(prefix):] for p in files if p.startswith(prefix)}
        return sorted(self._files)

    # Hub commits can publish files and the pointer atomically without a tar copy.
    archive_snapshots = False

    def backup(self, run, checkpoint):
        import shutil
        run, checkpoint = Path(run), Path(checkpoint)
        if not verify_checkpoint(checkpoint):
            raise RuntimeError("Refusing to upload incomplete checkpoint")
        with tempfile.TemporaryDirectory(dir=run.parent) as d:
            # Model/optimizer files are immutable while training waits here.
            # Log handlers and TensorBoard may still write in background threads.
            paths, files = {}, {}
            for path in sorted(run.rglob("*")):
                relative = path.relative_to(run)
                if not path.is_file() or relative.parts[0] == ".lock" or path.name.endswith(".tmp"):
                    continue
                if relative.parts[0].startswith("checkpoint-") and relative.parts[0] != checkpoint.name:
                    continue
                source = path
                if relative.parts[0] in ("tensorboard", "monitoring") or str(relative) in ("run.log", "stacks.log"):
                    source = contained(Path(d) / "evidence", str(relative))
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, source)
                paths[str(relative)] = source
                files[str(relative)] = sha256(source)
            snapshot = f"snapshots/{checkpoint.name}-{fingerprint(files)[:20]}"
            info = dict(format="files-v1", checkpoint=checkpoint.name,
                        step=int(checkpoint.name.split("-")[-1]),
                        files={name: dict(path=f"{snapshot}/{name}", sha256=digest)
                               for name, digest in files.items()})
            pointer = Path(d) / "latest.json"
            atomic_json(pointer, info)
            operations = [self._commit_add(path_in_repo=self._path(entry["path"]),
                          path_or_fileobj=str(paths[name])) for name, entry in info["files"].items()]
            operations.append(self._commit_add(path_in_repo=self._path("latest.json"),
                                               path_or_fileobj=str(pointer)))
            # If this times out, the commit may still land later (atomically, pointer included);
            # the lane guard stops a newer checkpoint racing it until it finishes.
            self._bounded("checkpoint", "checkpoint commit", lambda: self._call(
                "checkpoint commit", self.api.create_commit, repo_id=self.repo_id,
                repo_type="model", operations=operations,
                commit_message=f"Checkpoint {self.prefix}: {checkpoint.name}", run_as_future=False))
        if self._files is not None:
            self._files.update(entry["path"] for entry in info["files"].values())
            self._files.add("latest.json")

    def restore(self, run):
        import shutil
        run = Path(run)
        if "latest.json" not in self.names():
            return super().restore(run)
        with tempfile.TemporaryDirectory(dir=run.parent) as d:
            pointer = Path(d) / "latest.json"
            self.get("latest.json", pointer)
            info = read_json(pointer)
            if info.get("format") != "files-v1":
                return super().restore(run)  # Existing tar snapshots remain supported.
            local = latest_checkpoint(run)
            if local and int(local.name.split("-")[-1]) >= info["step"]:
                return
            staging = Path(d) / "staging"
            staging.mkdir()
            for name, entry in info["files"].items():
                target = contained(staging, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                self.get(entry["path"], target)
                if sha256(target) != entry["sha256"]:
                    raise RuntimeError(f"Remote snapshot checksum mismatch: {name}")
            checkpoint = contained(staging, info["checkpoint"])
            if not verify_checkpoint(checkpoint):
                raise RuntimeError("Remote checkpoint missing completion marker")
            # Rename downloaded files on the same filesystem; no second full copy.
            for path in sorted(staging.rglob("*"), key=lambda p: (p.name == "COMPLETE.json", str(p))):
                if path.is_file():
                    destination = contained(run, str(path.relative_to(staging)))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)

    def publish_snapshot(self, archive, remote_name, pointer):
        # The archive and pointer become visible in one atomic Hub commit.
        # create_commit returns only after uploading blobs and committing succeeds.
        operations = [self._commit_add(path_in_repo=self._path(remote_name), path_or_fileobj=str(archive)),
                      self._commit_add(path_in_repo=self._path("latest.json"), path_or_fileobj=str(pointer))]
        self._call("checkpoint commit", self.api.create_commit, repo_id=self.repo_id,
                   repo_type="model", operations=operations,
                   commit_message=f"Checkpoint {self.prefix}: {remote_name}", run_as_future=False)
        if self._files is not None:
            self._files.update((remote_name, "latest.json"))


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
