# Unattended pipeline review — October 5

Reviewed setup, dependency checks, dataset download/audit, encoding, memory probing, training callbacks, checkpoint serialization, Hub/Drive storage, resume, final export, synthesis, and notebook cells. The user's latest job stopped before optimizer work at an oversized disk reservation. Earlier evidence shows actual optimization reached about 10% of an epoch before checkpoint serialization exhausted disk. Neither establishes a successful complete epoch.

## Current behavior

| Stage | Recovery implemented | Remaining hard stop |
|---|---|---|
| Setup | Dedicated venv, inherited constrained CUDA wheels, virtualenv fallback, project-scoped dependency checks; repeated setup preserves already-satisfying packages | Missing/incompatible required packages or GPU stack |
| Preparation | Download retries, extraction reuse, redundant ZIP removal; bad audio/duplicates/context overflow recorded and skipped | No usable train/validation rows, wrong model vocabulary, inaccessible source data |
| Cache | Verified chunks reused; damaged derived chunks rebuilt. Data/settings and parsed token-producing code must match before reusing an earlier encoding identity across storage-only changes | Actual encoding behavior/data/package change requires rebuilding |
| Metadata recovery | An attempt with no verified checkpoint can refresh its identity automatically, with prior evidence archived | Changed source/config/optimizer state with a verified checkpoint cannot silently become a different experiment |
| GPU preflight | Longest-sequence forward/backward with an optimizer-memory estimate; shorter retained sequence limits can be selected before optimization | No sequence fits, invalid kernels, nonfinite training loss; later CUDA failures are not swallowed |
| Training metrics | JSONL is primary; runtime TensorBoard errors disable that writer. Optional metric/status export IO errors warn and continue | Metrics can have gaps during IO failures; checkpoint state remains required |
| Checkpoint save | Step 1 plus configured interval/final step; budgets upcoming write, protects latest/pending copies, retries supported disk-write failures once in the same process after removing the failed unsealed write | No space for any safe checkpoint after recovery; invalid optimizer/model serialization |
| Hub upload | Direct files plus pointer in one synchronous commit; stable copies of live log/event files prevent changing-file checksum failures; upload outages defer without discarding local training | Initial access failure; a pending upload is not remote durability |
| Drive upload | Immutable tar first, pointer after successful transfer; separate archive disk allowance | Archive needs extra local space |
| Queue/restore | Unreadable optional queue rebuilds; missing stale optional queue files dropped. Remote restore outage can use verified local state. Corrupt newest local checkpoint can fall back to an older verified checkpoint | No valid local/remote state; corruption is never treated as a valid checkpoint |
| Monitoring | Checkpoint upload precedes samples/plots; default samples start after first save; optional work skipped near session cutoff | Failure to restore a valid training mode cannot be ignored |
| Validation | Runtime/value/import/IO errors and nonfinite metrics marked unavailable; no fabricated loss | Actual training nonfinite loss remains fatal |
| Final export | Reuses final sealed adapter weights using hardlinks where supported, copying otherwise; final export precedes full validation | Unsupported hardlinks may require copy space; model artifacts must remain immutable |
| Notebook | Fresh-runtime install precedes training; optional status parsing, playback, synthesis and synthesis-report failures do not invalidate saved training | Real training subprocess failure remains visible |

At least two checkpoint directories are retained by Trainer during saving so its rotation cannot remove the last predecessor before sealing the new save. Disk-pressure cleanup protects the newest verified checkpoint and a pending upload. Corrupt files are preserved for inspection. Rollback to an older verified checkpoint repeats work since that checkpoint and is logged.

The Hugging Face budget for the supplied estimate is 5.07 GiB versus the reported 10.57 GiB free. This is a capacity estimate, not a reservation against other processes or a measurement of every third-party transfer buffer.

## Evidence and reproducibility

- Audit/encoding/memory exclusions retain row identifiers and measured duration losses.
- `metrics.jsonl`, `evaluation_status.json`, optional error logs and notebook stdout show available measurements and failures. IO problems can prevent a local diagnostic file being written; stdout still warns.
- `pending_uploads.json` and `backup_status.json` distinguish local progress from uploaded progress.
- `disk_budget.json` records measured free space, estimated requirement and pruning.
- `attempt_history/` and `attempt_recovery.jsonl` preserve earlier unsaved attempts rather than mixing their active loss logs with a new optimizer schedule.
- Preparation backups include source and token-format evidence; captured sources include defaults needed to run them. Training source identity remains strict for optimizer resume and mismatches are checked before GPU imports/audio auditing.
- Final adapter weights can share filesystem storage with a checkpoint; do not edit either in place.

## Verification

89 CPU tests pass. New failure-injection cases cover the actual mocked Trainer call sequence through checkpoint saving, retry, callback sealing/upload and final export; disk-write retry; runtime TensorBoard and report failures; nonfinite/failed optional validation; upload queue IO/corruption; offline restore with/without verified local state; corrupt newest checkpoint fallback; adapter hardlink/copy export; cache compatibility; and live log mutation during a Hub commit. Existing tests cover artifact roundtrips, atomic pointer publication, legacy archive restore, the reported disk capacity and later-save pruning, dependency scoping, notebook interpreter consistency, codec representation, and data retention.

Notebook cells and Python files are syntax checked; README notebook cells are synchronized. Tests use filesystem artifacts and mocked GPU/framework/Hub objects. They do not prove CUDA kernels, real Hub authentication/transfers, or end-to-end speech quality. No Kaggle GPU is available in this workspace. The training run itself performs a real backward probe and saves at optimizer step 1; there is no requirement to spend GPU time on a separate smoke run first.

A GPU reset, nonfinite optimization, unrecoverable checkpoint corruption, or inability to save after cleanup cannot safely be ignored. This review adds recovery for identified failures; it does not claim that all future runtime failures are predictable.
