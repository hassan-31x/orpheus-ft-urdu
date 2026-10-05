"""Bounded recovery for transient IO and optional remote uploads."""
import errno
import logging
import time
from pathlib import Path

from orpheus_utils import atomic_json, read_json, redact, sha256, verify_checkpoint

LOG = logging.getLogger('orpheus_urdu')


def retry_io(operation, attempts=3, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            transient = status in (408, 429, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError))
            transient |= isinstance(exc, OSError) and getattr(exc, 'errno', None) in (
                errno.ECONNRESET, errno.ECONNREFUSED, errno.ETIMEDOUT, errno.ENETUNREACH)
            # requests/httpx use their own exception hierarchies.
            transient |= type(exc).__name__ in ('ReadTimeout', 'ConnectTimeout', 'ConnectionError',
                                                'ConnectError', 'RemoteProtocolError', 'ChunkedEncodingError')
            if not transient or attempt == attempts - 1:
                raise
            LOG.warning('Transient IO failure (%s); retry %d/%d', type(exc).__name__, attempt + 2, attempts)
            sleep(2 ** attempt)


def valid_chunk(path, meta, identity, expected_rows, read_parquet):
    """Cache is derived data: validate it or permit deterministic regeneration."""
    if not path.is_file() or not meta.is_file():
        return False
    try:
        info = read_json(meta)
        return (info['row_identity'] == identity and info['sha256'] == sha256(path)
                and len(read_parquet(path)) == info['rows']
                and info['rows'] + len(info.get('excluded', [])) == expected_rows)
    except (ValueError, KeyError, OSError, RuntimeError):
        return False


class DeferredUploads:
    """Keep local progress while remote uploads retry at subsequent checkpoints.

    Authentication and restore use the real store and remain strict. The queue
    never claims remote durability, changes a checkpoint pointer or deletes data.
    """
    def __init__(self, store, run, clock=time.monotonic):
        self.store, self.run, self.clock = store, Path(run), clock
        self.remote = store.remote
        self.path = self.run / 'pending_uploads.json'
        self.queue = self.load_queue()
        self.next_attempt = 0
        self.last_upload_error = None
        self.failures = 0

    def _failed(self, stage, exc):
        # Redacted reason is kept so backup_status.json explains why state is LOCAL only.
        # Exponential backoff: repeated attempts during an outage/rate limit waste
        # training time and keep a Hub rate limit tripped.
        self.failures += 1
        diagnostics = getattr(exc, 'diagnostics', {}) or {}
        delay = min(60 * 2 ** (self.failures - 1), 1800)
        if diagnostics.get('category') == 'rate_limit':
            try:
                delay = max(delay, float(diagnostics.get('retry_after')) + 5)
            except (TypeError, ValueError):
                delay = max(delay, 600)
        self.next_attempt = self.clock() + delay
        self.last_upload_error = dict(stage=stage, exception_type=type(exc).__name__, message=redact(exc, limit=600),
                                      category=diagnostics.get('category'), consecutive_failures=self.failures,
                                      next_retry_seconds=round(delay), time=time.time())
        return f"{self.last_upload_error['message']} [next retry in {round(delay)}s]"

    def _succeeded(self):
        self.failures = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def load_queue(self):
        empty = {'files': {}, 'checkpoint': None, 'last_uploaded_checkpoint': None}
        if not self.path.exists():
            return empty
        try:
            value = read_json(self.path)
            if (not isinstance(value.get('files'), dict) or 'checkpoint' not in value
                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value['files'].items())
                    or not isinstance(value['checkpoint'], (str, type(None)))
                    or not isinstance(value.get('last_uploaded_checkpoint'), (str, type(None)))):
                raise ValueError('Malformed upload queue')
            return value
        except (ValueError, AttributeError, OSError):
            LOG.warning('Unreadable optional upload queue; rebuilding from local artifacts')
            return empty

    def restore(self, run):
        try:
            self.store.restore(run)
        except Exception as exc:
            from orpheus_utils import latest_checkpoint
            local = latest_checkpoint(run)
            if local is None:
                raise
            LOG.warning('Remote restore unavailable (%s); using verified local checkpoint %s',
                        type(exc).__name__, local.name)
        self.queue = self.load_queue()

    def save(self):
        optional_action(lambda: self._save_queue(), stage='upload status persistence')

    def _save_queue(self):
        atomic_json(self.path, self.queue)
        atomic_json(self.run / 'backup_status.json', dict(
            remote_configured=bool(self.remote), pending_files=len(self.queue['files']),
            pending_checkpoint=self.queue['checkpoint'], last_upload_error=self.last_upload_error,
            remote_snapshot_current=bool(self.remote and self.queue.get('last_uploaded_checkpoint') and self.queue['checkpoint'] is None)))

    def put(self, local, relative):
        self.put_many([(local, relative)])

    def put_many(self, items):
        """Queue files and attempt them as ONE remote commit when not backing off."""
        items = [(str(Path(local).resolve()), relative) for local, relative in items]
        if not self.remote or not items:
            return
        for local, relative in items:
            self.queue['files'][relative] = local
        if self.clock() >= self.next_attempt:
            self._flush_queue('file upload')
        self.save()

    def _flush_queue(self, stage, limit=500):
        pending = []
        for relative, local in list(self.queue['files'].items()):
            if not Path(local).is_file():
                self.queue['files'].pop(relative, None)
                LOG.warning('Discarding stale optional upload queue entry: %s', relative)
            elif len(pending) < limit:
                pending.append((local, relative))
        if not pending:
            return True
        try:
            put_many = getattr(self.store, 'put_many', None)
            if put_many is not None:
                put_many(pending)
            else:
                for local, relative in pending:
                    self.store.put(local, relative)
        except Exception as exc:
            reason = self._failed(stage, exc)
            LOG.warning('Upload of %d files deferred (%s: %s); local artifacts retained, e.g. %s',
                        len(pending), type(exc).__name__, reason, pending[0][1])
            return False
        self._succeeded()
        for _, relative in pending:
            self.queue['files'].pop(relative, None)
        return True

    def remote_names(self):
        """Files already stored remotely; empty if listing fails (then uploads are retried)."""
        try:
            return set(self.store.names())
        except Exception as exc:
            LOG.warning('Remote file listing unavailable (%s); existing remote files may be re-uploaded',
                        type(exc).__name__)
            return set()

    def backup(self, run, checkpoint, *, force=False):
        if not verify_checkpoint(checkpoint):
            raise RuntimeError('Incomplete checkpoint cannot be queued as resumable')
        if not self.remote:
            return True
        self.queue['checkpoint'] = str(Path(checkpoint).resolve())
        if force:
            self.next_attempt = 0
        if self.clock() < self.next_attempt:
            self.save()
            return False
        try:
            self.store.backup(run, checkpoint)
        except Exception as exc:
            reason = self._failed('checkpoint backup', exc)
            LOG.warning('Remote checkpoint upload deferred (%s: %s). Verified checkpoint remains LOCAL at %s',
                        type(exc).__name__, reason, checkpoint)
            self.save()
            return False
        self.queue['checkpoint'] = None
        self.queue['last_uploaded_checkpoint'] = Path(checkpoint).name
        self.last_upload_error = None
        self._succeeded()
        # Queued optional files go up together in one bounded commit.
        if self.queue['files']:
            self._flush_queue('queued file upload')
        self.save()
        return True


def memory_selection(dataset, run, probe, *, resuming=False, attempts=8):
    """Before optimizer steps, drop lengths that fail a real backward-memory probe."""
    import math
    selection = Path(run) / 'training_selection.json'
    original = dataset
    saved = read_json(selection) if selection.exists() else None
    if saved:
        wanted = set(saved['audio'])
        dataset = dataset.filter(lambda row: row['audio'] in wanted)
        if set(dataset['audio']) != wanted:
            raise RuntimeError('Saved training selection no longer matches the token cache')
    exclusions = list(saved.get('excluded', [])) if saved else []
    for _ in range(attempts):
        if not len(dataset):
            raise RuntimeError('No training sequences fit the available GPU')
        lengths = dataset['length']
        longest = max(range(len(dataset)), key=lambda i: lengths[i])
        row = dataset[longest]
        try:
            probe(row)
        except Exception as exc:
            if type(exc).__name__ != 'OutOfMemoryError' and 'out of memory' not in str(exc).lower():
                raise
            if resuming:
                raise RuntimeError('Current GPU cannot fit the saved training selection; dataset cannot change during checkpoint resume') from None
            padded = math.ceil(row['length'] / 8) * 8
            limit = max(8, math.floor(padded * 0.8 / 8) * 8)
            removed = dataset.filter(lambda r: math.ceil(r['length'] / 8) * 8 >= limit)
            exclusions.extend(dict(audio=r['audio'], length=r['length'], stage='training_memory_probe') for r in removed)
            LOG.warning('Backward memory probe exceeded GPU capacity; excluding %d sequences padded to >=%d tokens', len(removed), limit)
            dataset = dataset.filter(lambda r: math.ceil(r['length'] / 8) * 8 < limit)
            continue
        atomic_json(selection, dict(audio=list(dataset['audio']), excluded=exclusions,
                                    source_rows=len(original), retained_rows=len(dataset)))
        return dataset, exclusions
    raise RuntimeError('GPU memory probe exhausted its bounded attempts without a usable training sequence')


def optional_evaluation(operation, run, step, cleanup):
    """Optional metrics must not prevent saving optimization progress."""
    from orpheus_utils import append_jsonl
    try:
        metrics = operation()
        import math
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite optional evaluation metric')
        optional_action(lambda: atomic_json(Path(run) / 'evaluation_status.json', dict(status='passed', step=step)), stage='evaluation status')
        return metrics
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        cleanup()
        LOG.warning('Optional validation failed (%s); training/adapter saving continues', type(exc).__name__)
        record = dict(stage='validation', step=step, error_type=type(exc).__name__,
                      evaluation_available=False)
        optional_action(lambda: append_jsonl(Path(run) / 'optional_errors.jsonl', record), stage='evaluation error log')
        optional_action(lambda: atomic_json(Path(run) / 'evaluation_status.json', dict(status='failed', **record)), stage='evaluation status')
        return {}


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def checkpoint_budget(model):
    """Budget full embeddings: PEFT may include frozen resized vocabulary weights."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding = 0
    for getter in ('get_input_embeddings', 'get_output_embeddings'):
        layer = getattr(model, getter, lambda: None)()
        if layer is not None:
            embedding += sum(p.numel() * max(p.element_size(), 4) for p in layer.parameters())
    # FP32 adapter, AdamW state, gradients/serialization slack, metadata.
    return trainable * 16 + embedding + 256 * 1024**2


def ensure_checkpoint_space(run, checkpoint_bytes, *, remote, protected=(), archive_snapshots=True):
    """Keep newest verified state; prune only older verified checkpoints if needed."""
    import shutil
    from orpheus_utils import latest_checkpoint
    run = Path(run)
    latest = latest_checkpoint(run)
    protected = {Path(p).resolve() for p in protected if p}
    if latest:
        protected.add(latest.resolve())
    # Reserve the next write. Only archive backends need a second temporary copy.
    extras = sum(directory_bytes(p) if p.is_dir() else p.stat().st_size
                 for p in run.iterdir() if not p.name.startswith('checkpoint-'))
    required = checkpoint_bytes + 512 * 1024**2
    if remote and archive_snapshots:
        required += checkpoint_bytes + extras
    removed = []
    candidates = sorted(run.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1])
                        if p.name.split('-')[-1].isdigit() else -1)
    for candidate in candidates:
        if shutil.disk_usage(run).free >= required:
            break
        if candidate.resolve() in protected:
            continue
        try:
            verified = verify_checkpoint(candidate)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            verified = False
        if verified:
            shutil.rmtree(candidate)
            removed.append(candidate.name)
    free = shutil.disk_usage(run).free
    report = dict(free_bytes=free, required_bytes=required, checkpoint_estimate_bytes=checkpoint_bytes,
                  snapshot_format="archive" if remote and archive_snapshots else "direct_files",
                  removed_old_checkpoints=removed, enough=free >= required)
    atomic_json(run / 'disk_budget.json', report)
    if free < required:
        raise OSError(errno.ENOSPC, f'Checkpoint disk budget needs {required / 2**30:.2f} GiB free; '
                      f'only {free / 2**30:.2f} GiB available. See disk_budget.json')
    return report


def save_with_space_retry(operation, recover):
    """Retry ENOSPC once in the SAME process, retaining in-memory training state."""
    try:
        return operation()
    except Exception as exc:
        if (getattr(exc, 'errno', None) != errno.ENOSPC
                and not any(message in str(exc) for message in (
                    'No space left on device', 'file write failed', 'failed writing file'))):
            raise
        LOG.warning('Checkpoint write exhausted disk; reclaiming space and retrying in memory')
        recover()
        return operation()


def reconcile_run_identity(run, identity):
    """Refresh preparation-only attempts; never reinterpret saved optimizer state."""
    import shutil
    from orpheus_utils import latest_checkpoint, append_jsonl
    run = Path(run)
    path = run / 'run_identity.json'
    if not path.exists() or read_json(path) == identity:
        return False
    previous = latest_checkpoint(run)
    if previous is not None:
        raise RuntimeError(f'Verified checkpoint {previous.name} belongs to different data/config/code/dependencies. '
                           'Restore its original versions or use a new run_id; saved training state was preserved')
    # No verified optimizer state exists. Keep the earlier attempt's evidence,
    # then allow a fresh optimizer schedule under the current identity.
    history = run / 'attempt_history'
    history.mkdir(exist_ok=True)
    attempt = history / f'{time.time_ns()}'
    attempt.mkdir()
    for name in ('run_identity.json', 'resolved_config.json', 'environment.json',
                 'requirements-resolved.txt', 'token_format.json', 'run.log',
                 'metrics.jsonl', 'optional_errors.jsonl', 'parameters.json',
                 'trainable_parameters.txt', 'status.json'):
        source = run / name
        if source.is_file():
            shutil.copy2(source, attempt / name)
    if (run / 'source').is_dir():
        shutil.copytree(run / 'source', attempt / 'source')
    # Saved memory selections/metrics belong to the previous attempt. Audio and
    # content-addressed encoded caches are untouched and verified on reuse.
    for name in ('training_selection.json', 'effective_training_manifests.json',
                 'memory_preflight.json', 'metrics.jsonl', 'optional_errors.jsonl',
                 'status.json', 'train_results.json', 'validation_results.json',
                 'trainer_state.json'):
        source = run / name
        if source.is_file():
            shutil.move(str(source), str(attempt / name))
    append_jsonl(run / 'attempt_recovery.jsonl', dict(
        timestamp=time.time(), reason='identity_changed_without_verified_checkpoint',
        evidence=str(attempt.relative_to(run)), optimizer_restarts_at_step=0))
    LOG.warning('Earlier attempt has NO verified checkpoint. Archived its evidence at %s; '
                'continuing automatically with a fresh optimizer schedule. Audio/cache files retained', attempt)
    return True


def optional_action(operation, *, stage):
    """Diagnostics may fail independently of optimizer/checkpoint state."""
    try:
        return operation()
    except Exception as exc:
        LOG.warning('Optional %s failed (%s); continuing', stage, type(exc).__name__)
        return None


def export_checkpoint_adapter(checkpoint, destination):
    """Reuse the final sealed adapter instead of serializing multi-GB weights twice."""
    import os
    import shutil
    checkpoint, destination = Path(checkpoint), Path(destination)
    if not verify_checkpoint(checkpoint):
        raise RuntimeError('Cannot export an unverified checkpoint')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        source, target = checkpoint / name, destination / name
        temp = destination / (name + '.tmp')
        temp.unlink(missing_ok=True)
        try:
            os.link(source, temp)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            shutil.copy2(source, temp)
        temp.replace(target)
    return destination


def compatible_encoding_identity(previous, current, previous_source, current_source):
    """Reuse existing tokens only when data/settings AND token-producing code match."""
    import ast
    previous_source, current_source = Path(previous_source), Path(current_source)
    if {k: v for k, v in previous.items() if k != 'encoding_source_sha256'} != {
            k: v for k, v in current.items() if k != 'encoding_source_sha256'}:
        return current
    selected = {
        'finetune_aslp_50h.py': {'encode_data'},
        'orpheus_utils.py': {'clean_text', 'prompt_ids', 'interleave_codes', 'training_row',
                            'fingerprint', 'canonical', 'sha256', 'read_json', 'atomic_json',
                            'SPECIAL', 'AUDIO_BASE', 'OFFSETS', 'CODEC_ID', 'FORMAT_VERSION'},
        'pipeline_recovery.py': {'valid_chunk', 'retry_io'},
    }
    def signature(path, wanted):
        tree = ast.parse(path.read_text())
        found = {}
        for node in tree.body:
            names = {node.name} if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else set()
            if isinstance(node, ast.Assign):
                names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            for name in names & wanted:
                found[name] = ast.dump(node, include_attributes=False)
        if set(found) != wanted:
            raise ValueError('Incomplete encoding source evidence')
        return found
    try:
        for filename, wanted in selected.items():
            if signature(previous_source / filename, wanted) != signature(current_source / filename, wanted):
                return current
    except (OSError, SyntaxError, ValueError):
        return current
    LOG.info('Token-producing code, data and settings match; reusing the existing encoded-cache identity')
    return previous
