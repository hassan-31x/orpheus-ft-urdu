"""Bounded recovery for transient IO and optional remote uploads."""
import errno
import logging
import time
from pathlib import Path

from orpheus_utils import atomic_json, read_json, sha256, verify_checkpoint

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
        self.queue = read_json(self.path) if self.path.exists() else {'files': {}, 'checkpoint': None, 'last_uploaded_checkpoint': None}
        self.next_attempt = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def restore(self, run):
        self.store.restore(run)
        if self.path.exists():
            self.queue = read_json(self.path)

    def save(self):
        atomic_json(self.path, self.queue)
        atomic_json(self.run / 'backup_status.json', dict(
            remote_configured=bool(self.remote), pending_files=len(self.queue['files']),
            pending_checkpoint=self.queue['checkpoint'],
            remote_snapshot_current=bool(self.remote and self.queue.get('last_uploaded_checkpoint') and self.queue['checkpoint'] is None)))

    def put(self, local, relative):
        if not self.remote:
            return
        self.queue['files'][relative] = str(Path(local).resolve())
        if self.clock() >= self.next_attempt:
            try:
                self.store.put(local, relative)
                self.queue['files'].pop(relative, None)
            except Exception as exc:
                self.next_attempt = self.clock() + 60
                LOG.warning('Upload deferred (%s); local artifact retained: %s', type(exc).__name__, relative)
        self.save()

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
            self.next_attempt = self.clock() + 60
            LOG.warning('Remote checkpoint upload deferred (%s). Verified checkpoint remains LOCAL at %s',
                        type(exc).__name__, checkpoint)
            self.save()
            return False
        self.queue['checkpoint'] = None
        self.queue['last_uploaded_checkpoint'] = Path(checkpoint).name
        # Bound optional queue draining so an outage cannot consume the session.
        for relative, local in list(self.queue['files'].items())[:8]:
            self.put(local, relative)
            if self.clock() < self.next_attempt:
                break
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
        atomic_json(Path(run) / 'evaluation_status.json', dict(status='passed', step=step))
        return metrics
    except (RuntimeError, ValueError, ImportError) as exc:
        cleanup()
        LOG.warning('Optional validation failed (%s); training/adapter saving continues', type(exc).__name__)
        record = dict(stage='validation', step=step, error_type=type(exc).__name__,
                      evaluation_available=False)
        append_jsonl(Path(run) / 'optional_errors.jsonl', record)
        atomic_json(Path(run) / 'evaluation_status.json', dict(status='failed', **record))
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


def ensure_checkpoint_space(run, checkpoint_bytes, *, remote, protected=()):
    """Keep newest verified state; prune only older verified checkpoints if needed."""
    import shutil
    from orpheus_utils import latest_checkpoint
    run = Path(run)
    latest = latest_checkpoint(run)
    protected = {Path(p).resolve() for p in protected if p}
    if latest:
        protected.add(latest.resolve())
    # New full checkpoint plus uncompressed upper bound for upload tar and final adapter.
    extras = sum(directory_bytes(p) if p.is_dir() else p.stat().st_size
                 for p in run.iterdir() if not p.name.startswith('checkpoint-'))
    required = checkpoint_bytes * (3 if remote else 2) + extras + 512 * 1024**2
    removed = []
    candidates = sorted(run.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1])
                        if p.name.split('-')[-1].isdigit() else -1)
    for candidate in candidates:
        if shutil.disk_usage(run).free >= required:
            break
        if candidate.resolve() not in protected and verify_checkpoint(candidate):
            shutil.rmtree(candidate)
            removed.append(candidate.name)
    free = shutil.disk_usage(run).free
    report = dict(free_bytes=free, required_bytes=required, checkpoint_estimate_bytes=checkpoint_bytes,
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
        if getattr(exc, 'errno', None) != errno.ENOSPC and 'No space left on device' not in str(exc):
            raise
        LOG.warning('Checkpoint write exhausted disk; reclaiming space and retrying in memory')
        recover()
        return operation()
