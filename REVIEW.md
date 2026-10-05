# Pipeline recovery review

This review covers the checked-in notebook and scripts after reading the supplied `logs.txt`. The log establishes that the archive's retained training partition is 37.41h and the whole retained corpus is 46.74h. The former 40h training minimum was incorrect. It does not establish that model optimization has run successfully.

## Stage-by-stage outcomes

| Stage | Checked behavior and recovery | Stops that remain necessary |
|---|---|---|
| Notebook | Fresh-runtime installation always precedes imports; one training interpreter; current requested Git ref fetched; missing recovery module detected before installation | Missing credentials, unavailable repository, installation failure |
| Dependencies | Project graph checks replace global Kaggle checks; CUDA wheels constrained; venv/virtualenv fallback | Incompatible required packages or CUDA stack |
| Download/model metadata | Bounded retries for transient HTTP/timeouts; partial Drive downloads resume; vocabulary checked before dataset download/audit | Access/quota denial, invalid ZIP, model lacking audio vocabulary |
| Audit | Bad/duplicate rows logged and excluded; train-first duplicate precedence; original and retained counts/hours recorded; retention thresholds warn by default | Missing required manifests/columns, no usable train or validation rows |
| Encoding | Decode/format/context failures and per-clip codec OOM recorded; damaged derived chunks rebuilt; existing verified chunks reused; missing remote chunks can be regenerated | Vocabulary mismatch, invalid global codec/model behavior, local IO exhaustion |
| Model memory | Real longest-sequence forward/backward preflight with configured microbatch, up to two accumulated backward passes and estimated optimizer memory reserve; conservative shorter-length retries before optimizer steps | No usable sequence fits; kernel/driver errors; nonfinite loss |
| Optimization | Standard PyTorch AdamW default; LR/epoch/batch geometry unchanged mid-run; required JSONL metrics remain canonical | Corrupt/changed optimizer checkpoints, nonfinite training loss; later training CUDA failures not caught blindly |
| Evaluation | Optional runtime/value/import failures recorded as unavailable; no fabricated loss; final adapter exported before final evaluation | Local export/write failures |
| Checkpoints/uploads | Local full-state checkpoints sealed and hashed; remote writes deferred with cooldown; newest pending checkpoint retried; final attempt forced; atomic remote pointer retained | Incomplete/corrupt optimizer state; initial authentication/restore failures; local disk exhaustion |
| Samples/plots | Sample generation and plotting errors logged; optional TensorBoard failure disables only TensorBoard | Broken core training state cannot be repaired by ignoring errors |
| Final notebook | Optional synthesis failure reports saved adapter rather than failing completed training; playback IO/value failures tolerated; backup status displayed | Training subprocess failures remain visible |

## Research evidence retained

- `data_errors.json`, `data_error_summary.json`: audit exclusions.
- `encoding_errors.json`: codec/decode/context exclusions, also stored in chunk metadata.
- `training_selection.json`: memory-filtered IDs/order and exclusions.
- `training_manifests.json`: encoded rows before the backward-memory filter.
- `effective_training_manifests.json`: actual training/validation rows after that filter.
- `dataset_report.json`: final counts, measured/unknown-duration losses and warnings.
- `evaluation_status.json`, `optional_errors.jsonl`, `sample_errors.jsonl`: optional failures.
- `pending_uploads.json`, `backup_status.json`: local queue and remote durability status.
- `metrics.jsonl`, resolved configuration/packages, source hashes: optimization evidence.

## Verification and limits

CPU fault-injection tests exercise transient retry versus permanent failure, corrupt cache recognition, cached exclusion replay, upload outage/cooldown/recovery, forced final backup, optional evaluation failure without fake metrics, padded-length memory filtering, saved selection reuse, refusal to change selection on optimizer resume, and the observed corpus-size regression. Notebook code cells are syntax-checked and checked for interpreter consistency. The repository-name guard is tested against the actual owner `hassan-31x`, so replacing a placeholder cannot accidentally blacklist that username. Full mocked training-flow tests cover backward preflight through sealed checkpoint, final adapter export, remote upload and optional evaluation failure. All 62 CPU tests pass. Existing codec, artifact, remote atomicity and credential-sanitization tests are retained.

Tests using fake tensors, datasets and Hub APIs verify decisions and persistence, not CUDA kernels, actual Parquet binaries, real network access or final speech quality. No Kaggle GPU is available in the local workspace. The automatic real backward preflight will exercise the model on the actual runtime; even that cannot prove every future optimizer step or network operation succeeds. The memory reserve is an estimate, not a measured guarantee.

The permissive policy can exclude much more data than planned. Counts, hours, reasons and evaluation denominators must be reported; a successful process is not evidence of a good Urdu voice. A pending local upload is not a remote backup. If a runtime is deleted during an outage, unuploaded updates/cache chunks can be lost. Neither scenario is concealed as a fully recovered run.

This version changes source/config identities and the default optimizer. Existing trained checkpoints require their original version/config for exact resume. The supplied log failed before encoding/optimization, so that particular run has no optimizer progress affected by this change.

## October 5: disk exhaustion during PEFT checkpoint save

Added full embedding-aware storage estimates before optimizer steps and checkpoint saves, redundant ZIP removal with extraction reuse, no pip download cache for new installs, step-1 checkpointing, protected checkpoint pruning, one in-process ENOSPC checkpoint retry, and notebook disk diagnostics. Resized embeddings are retained. Existing incomplete files can be cleaned with the documented recovery cell after the failed subprocess exits. Strict source identity is retained; the notebook exposes a run-ID override for fresh attempts with no recoverable checkpoint.

Validation: 66 CPU tests passed, including frozen embedding sizing, Rust-style ENOSPC retry, preservation of latest/pending checkpoints, and propagation of unrelated errors. Real Kaggle disk capacity and CUDA execution are unverified locally.
