# Urdu Orpheus: ASLP language adaptation on Kaggle

This repository adapts `unsloth/orpheus-3b-0.1-pretrained` to Urdu using every row of your supplied **training manifest**, for **one epoch**. It downloads your Drive ZIP, validates the paired audio/transcripts, encodes speech with SNAC, trains LoRA adapters, measures held-out loss, generates intermediate Urdu samples, and backs up resumable training state to a **private Hugging Face repository**. Google Drive/rclone remains an optional storage backend.

The supplied URL is included in `configs/aslp50h.json`:

<https://drive.google.com/file/d/1Xslte2s7PLzLCqBxDope2LQi4cmnfDJ7/view>

Drive displays the filename `Urdu_Orpheus_US_Std_Training_47h.zip`. The script measures actual hours; “50h” is the experiment label, not a fabricated duration. All valid training clips are used; validation/test manifests retain their evaluation roles. There is no ten-hour sampling limit, and no automatic truncation or silent skipping of clips. If you want a different train/validation split, prepare and document it before starting a new experiment. Training on the test set would invalidate its use as a held-out test.

This is **stage 1: Urdu language adaptation**. Arbitrary labels such as “happy” or “sad” have not been taught by this run. Stage 2 needs aligned Urdu speech, accurate transcripts, and verified emotion labels. A speech emotion recognition dataset alone is insufficient if its speech has no usable transcripts.

## 1. Files and prerequisites

| File | Purpose |
|---|---|
| `finetune_aslp_50h.py` | Download, audit, cache, train, monitor, resume |
| `configs/aslp50h.json` | Full initial experiment settings |
| `orpheus_utils.py` | Token format, artifact verification, Hugging Face/Drive persistence |
| `synthesize.py` | Single/batch inference from final or intermediate adapters |
| `evaluate_asr.py` | Optional Urdu ASR WER/CER with bootstrap intervals |
| `kaggle_run.ipynb` | Ready-to-import notebook; edit checkpoint repository and session budget |
| `setup_kaggle.py` | Dedicated training environment; preserves the installed CUDA stack |
| `check_environment.py` | Project dependency validation and CPU/CUDA smoke checks |
| `METHODOLOGY.md` | Research review, exact choices, ablations, evaluation limitations |
| `tests/test_pipeline.py`, `tests/test_environment.py` | CPU tests for tokens, persistence, dependency checks and setup |

For Kaggle shell commands elsewhere in this guide, replace `python` with `/kaggle/working/orpheus-env/bin/python` after running setup. Notebook subprocess cells already use `TRAIN_PYTHON`.

Your original `finetune_aslp_10h.py`, paper, and `resources.md` are unchanged. The original Python file contains notebook shell syntax (`!nvidia-smi`); the new scripts are normal Python programs.

Use a Linux NVIDIA GPU environment. Kaggle T4 with 4-bit loading is the conservative starting point. The script exposes one GPU with `CUDA_VISIBLE_DEVICES=0`; **two T4s do not combine into one 32GB device**. This is not a distributed-training script. 16-bit LoRA is a separate experiment for a device with adequate VRAM. A GPU label is not a guarantee that the chosen sequence length/microbatch fits.

Enable **Internet**, choose an **NVIDIA GPU accelerator**, and check the runtime limit and remaining GPU allowance shown in your Kaggle account. Set `session_hours` to a budget **shorter than that displayed limit**, allowing preparation, notebook setup and uploads. The configured value is a user-selected budget, not a claim about Kaggle's session limit; the notebook uses `10.5` as an example to adjust. A full epoch may require several sessions.

Check free disk space. 50 hours of 24kHz mono PCM16 audio is about 8.64GB before file overhead. You also need space for the compressed download, extracted audio, base weights, Arrow caches, retained resumable checkpoints and upload buffers (a full temporary snapshot archive is needed only for Drive); final adapter weights share the final checkpoint’s local storage where hardlinks are supported. Actual use depends on clip lengths, model precision and optimizer state; provision several tens of GB and inspect `df -h /kaggle/working`.

The complete GPU workflow has **not been run in this local macOS workspace**. CPU tests exercise both storage backends, including mocked Hub upload/restore and failed-commit recovery; authenticated remote transfers, Unsloth/CUDA installation, ZIP contents and GPU training must be verified in your Kaggle smoke run. The public Drive page was inspected; the multi-GB ZIP was not downloaded here.

## 2. Put the code on GitHub

Create a repository and upload the new scripts, `configs/`, `tests/`, both Markdown guides, requirements and notebook. Use a repository you control. No GitHub repository has been created or published automatically by this work.

Do not upload audio, adapter weights, training outputs, OAuth configuration or access tokens. `.gitignore` excludes the common generated files; inspect what you commit. Preserve dataset/model attribution and check redistribution terms before publishing derived assets. The attached IEEE paper can remain a private reference; it is not needed at runtime.

For reproducible runs, clone a fixed commit or tag rather than a moving branch. Record the commit in your research notes. The pipeline also hashes and saves the actual training source files.

## 3. Set up checkpoint storage: Hugging Face recommended

No rclone installation, Google OAuth client or Drive mount is needed for the default workflow. The **input dataset still downloads from your supplied Google Drive link**; checkpoint storage is independent of that download.

1. Create/sign in to your [Hugging Face account](https://huggingface.co/join).
2. Create a **private model repository**, for example `YOUR_USERNAME/orpheus-urdu-checkpoints`. This is an artifact store, so choose **Model**, not Dataset or Space. If it does not exist, the script can create it privately when your token permits creation. An existing public repository is rejected before any artifacts are uploaded.
3. Create a [user access token](https://huggingface.co/docs/hub/security-tokens) with write access to that repository. A fine-grained token scoped to an existing repository is suitable; if you want the script to create the repository, the token must also allow that creation.
4. In Kaggle **Add-ons → Secrets**, add **`HF_TOKEN`**, paste the token and grant the notebook access. Never put the token in a JSON configuration, GitHub, notebook output or shell command.
5. Load the token into the environment and set the repository ID:

   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   os.environ["ORPHEUS_HF_REPO"] = "YOUR_USERNAME/orpheus-urdu-checkpoints"
   ```

6. Use the default config (`checkpoint_backend="huggingface"`). Set `hf_repo_id` in your JSON, use `--hf-repo-id`, or leave it null and use `ORPHEUS_HF_REPO`. An explicit config/CLI repo ID takes precedence over the environment variable.
7. Test authentication and read/write access **without downloading audio or loading a GPU model**:

   ```bash
   python finetune_aslp_50h.py --mode storage-check \
     --hf-repo-id YOUR_USERNAME/orpheus-urdu-checkpoints
   ```

The script uploads and reads back a small probe, then exits for `storage-check`. Normal runs perform the same preflight before processing data. Private artifacts appear under `runs/<run_id>/` inside the repository. Neither the token nor a Hub login file is written into experiment artifacts. There is no need to run `huggingface-cli login` or enable Trainer's separate `push_to_hub` feature.

Checkpoint uploads use synchronous [Hub upload/commit APIs](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/upload). A single commit publishes the complete snapshot archive and `latest.json` together. Restarts read the pointer, download the archive and verify its SHA-256 and checkpoint file hashes before resuming. Token-cache chunks and preparation metadata use the same private repository. Transient API failures are retried up to three attempts; initial authentication/permission failures stop before processing data. After preflight succeeds, failed writes are queued locally and retried at later checkpoints; the previous committed remote checkpoint remains recoverable. `backup_status.json` distinguishes local progress from remote durability.

Check your Hugging Face private-storage allowance before a long run. Snapshots contain **optimizer state as well as adapter weights**, and remote history is retained. Removing files from the visible tree may not immediately reclaim historical storage; consult [Hub storage guidance](https://huggingface.co/docs/hub/storage-limits). No particular free storage capacity is assumed by this project.

### Optional: keep Google Drive/rclone storage

Choose `--checkpoint-backend drive`, or set `checkpoint_backend="drive"` in your JSON, to use the original backend. Only this option needs the rclone setup below. The public download link grants no write access to your Drive. Kaggle does not support Colab's `drive.mount()` API. This backend uses **rclone + Google OAuth**, with the configuration stored as a Kaggle Secret.

1. Install rclone on your own computer using [rclone's installation instructions](https://rclone.org/install/).
2. Follow [Google Drive configuration](https://rclone.org/drive/) and [create your own OAuth client](https://rclone.org/drive/#making-your-own-client-id). Enable the Google Drive API, configure the consent screen and authorized account, and create an OAuth desktop client. Rclone currently recommends your own client because its shared client is being retired during 2026.
3. Run `rclone config`. Create a remote named **`gdrive`**, choose the Google Drive backend, enter your client ID/secret, select a read/write scope appropriate to your experiment folder, and complete browser authorization. Scope `drive.file` can work for files created by this rclone client; scope `drive` gives broader Drive access. Read-only scope cannot save checkpoints.
4. If the OAuth app remains in external **Testing** status, check Google's refresh-token expiry rules. Configure an appropriate long-lived authorization for repeated sessions; reauthorize when required. Never place the OAuth refresh token in GitHub.
5. Test the connection locally: `rclone lsd gdrive:`. Run `rclone config file` to locate the configuration. Use an **unencrypted** configuration containing the authorized `gdrive` remote for this setup; encrypted configurations require additional password setup and are not covered by the notebook.
6. Base64 encode that file locally:

   ```python
   import base64
   from pathlib import Path
   # Replace with the path printed by `rclone config file`.
   value = base64.b64encode(Path("/YOUR/PATH/rclone.conf").read_bytes()).decode()
   # Copy value to the secret UI privately. Do not display it in a saved notebook.
   ```

7. In Kaggle, open **Add-ons → Secrets**, add **`RCLONE_CONFIG_B64`**, paste that value and grant the notebook access. Base64 is encoding, not encryption; keep the value private.
8. For this backend, the default storage root is `gdrive:orpheus_urdu/<run_id>`. You may set `ORPHEUS_DRIVE_REMOTE` to a different root in the notebook. Do not append `<run_id>` yourself; the script does that. Install rclone in Kaggle with `apt-get update -qq` and `apt-get install -y -qq rclone` before running the script.

The script writes and reads back a small probe before doing GPU work. Credentials are decoded into a temporary directory with restricted permissions and are not included in snapshots. Token refresh can update the temporary rclone config during a session; future sessions use the refresh token in your Kaggle Secret. If you revoke it, replace the secret after reauthorization.

For deliberate testing without either remote backend, add `--local-only` or select `--checkpoint-backend local`. That explicitly disables remote protection: local `/kaggle/working` files alone are not a reliable resume strategy across deleted runtimes.

## 4. Expected data layout

The trial's format is supported directly:

```text
processed_dataset/
  stage1_train.csv
  stage1_validation.csv
  stage1_test.csv          # optional; audited but not trained/evaluated
  audio/
    clip_0001.wav
    ...
```

```csv
audio,text
audio/clip_0001.wav,آج موسم بہت خوشگوار ہے۔
```

CSV files must be UTF-8, with `audio` and `text` columns. Extra metadata is allowed. Paths are relative to the manifest directory; absolute paths and path traversal are rejected. Audio must match the prior trial: **24,000Hz, mono, PCM16 WAV**. The script validates this instead of resampling or changing audio without recording it. If you supply a newly preprocessed variant, document it as a separate dataset version.

The auto-discovery supports `stage1_train.csv`/`stage1_validation.csv` and `train.csv`/`validation.csv` (also `val.csv`), inside a single matching archive folder. If the archive has multiple candidate datasets or different manifest names, set `data_dir`, `train_manifest`, `validation_manifest`, and optionally `test_manifest` in a copy of the config. Explicit manifest paths are relative to `data_dir`.

The audit records every audio file's SHA-256, duration, peak, RMS, clipped fraction and transcript normalization changes. It checks nonempty Urdu-script text, finite/non-silent signals, duplicate audio paths and identical files across splits. The default `invalid_row_policy="skip"` excludes unusable and duplicate rows and records them in `data_errors.json`. The default `enforce_retention_limits=false` treats the 5% row/hour thresholds and `minimum_train_hours` as recorded warnings, so usable data continues automatically. Set `enforce_retention_limits=true` to enforce those thresholds. Set `invalid_row_policy="strict"` to reject any exclusion. Evaluation exclusions are reported separately and do not stop training unless validation becomes empty. Similar excerpts of the same source recording can still leak across splits: preserve source/session-disjoint splits whenever possible.

Default speaker prompting is disabled because the trial identified unreliable speaker IDs. Do not use segment-local `SPEAKER_0001` as a global identity. If you have genuinely verified global speaker labels, set `speaker_column` and use the matching speaker string at inference. This is a separate conditioning experiment.

## 5. Configure and verify the Kaggle environment

Use the notebook cells at the end of this guide, or import `kaggle_run.ipynb`. The notebook already contains this project’s GitHub URL. Edit `HF_REPO_ID` and `SESSION_HOURS` in Cell 1 first.

Start a **fresh Kaggle session** after updating the repository, since the earlier setup changed the shared notebook packages. Replace the old installation cell with Cell 2 below (or import the updated notebook). Do not keep the old global `pip check` cell.

`setup_kaggle.py` creates `/kaggle/working/orpheus-env` with its own pip and project packages. It inherits Kaggle's existing CUDA wheels using [venv's system-site-packages option](https://docs.python.org/3/library/venv.html). Installation goes into the venv and does not change the notebook kernel, Papermill, or Jupyter. This is isolation of installations, with shared base package visibility; it does not hide every preinstalled library.

If the image lacks stdlib venv/ensurepip, setup falls back to virtualenv installed in a bootstrap directory. It does not install virtualenv into the notebook kernel.

Torch, torchvision, torchaudio, Triton and xformers are constrained to their installed versions. An incompatible resolver result fails instead of replacing the CUDA stack or downloading a second Torch version. Requirements pin Unsloth, Transformers, TRL and Datasets; the remaining resolved versions are recorded. The project reads local Parquet and audio files and does not need `gcsfs`, `s3fs`, MoviePy, BigFrames, Pathos, TPOT, Colab, Dopamine or Gradio.

A global `pip check` reports conflicts for all installed tools, including unrelated Kaggle packages. `check_environment.py` instead checks **every active dependency reachable from this project's requirements**, including transitive version constraints and activated extras. Missing or incompatible project dependencies still fail the cell. CPU smoke checks exercise Urdu Parquet and WAV round trips; `--gpu` checks Unsloth, Trainer, SNAC and bitsandbytes imports plus an xformers CUDA attention backward pass. This does not replace the two-step model/checkpoint test below.

To check installation without a GPU, run the Cell 2 setup command with `--gpu` omitted. Restore `--gpu` before a saved training run. A restarted Kaggle session may discard the venv; rerun setup on the actual GPU image, since CPU and GPU images can differ. `TRAIN_PYTHON` must be used for storage checks, training and inference; the notebook kernel remains on its original interpreter. No kernel restart is needed just to use the venv. **Saved runs start fresh: keep installation in Cell 2 every time. An import-only replacement fails with `No module named unsloth`.** Updating the GitHub repository does not replace cells already copied into Kaggle; import the updated notebook itself.

Setup evidence is saved alongside the venv in `/kaggle/working/orpheus-env-setup/`: `installation.log`, `installation.json`, `cuda-constraints.txt`, `resolved-requirements.txt`, `cpu-check.json`, and `gpu-check.json`. Keep these with your experiment output. A failed check reports its actual project dependencies or import/kernel error; inspect that report before starting training. The local CPU regression tests do not prove compatibility with every Kaggle GPU image. Use Torch 2.6+ for resumable optimizer/RNG loading.

Before a long run, create a **separate smoke config**:

```python
import json
from pathlib import Path
cfg = json.loads(Path("/kaggle/working/urdu-orpheus/configs/aslp50h.json").read_text())
cfg["hf_repo_id"] = "YOUR_USERNAME/orpheus-urdu-checkpoints"
cfg.update(run_id="aslp50h-smoke", max_steps=2, save_steps=1,
           eval_steps=1, eval_samples=4, sample_count=1)
Path("/kaggle/working/smoke.json").write_text(json.dumps(cfg, indent=2))
```

Run with `--config /kaggle/working/smoke.json`, supplying your repository ID in the config, CLI or environment. This intentionally prepares all supplied training data and then trains only two steps. It tests the real maximum-length data/cache, a training backward pass, checkpoint optimizer state, validation, sample decoding, and remote upload. `smoke_complete` must not be reported as a full epoch.

To test resume, pause a longer smoke configuration before its step limit, confirm `latest.json` and the snapshot exist under `runs/<run_id>/` in your private Hub repo (or your chosen Drive folder), and rerun the same command in a fresh runtime. Do not change `max_steps` for that resume. The restored step should advance from the saved value rather than start at zero. A completed two-step smoke run simply exits on rerun; it does not exercise continuation by itself.

### First complete experiment

After verifying the smoke run, use `configs/aslp50h.json` with `max_steps=-1`. Default settings:

| Setting | Initial value |
|---|---|
| Base | English pretrained Orpheus 3B, revision resolved and saved |
| Epochs | 1 |
| Weight loading | 4-bit QLoRA |
| Microbatch / accumulation | 1 / 8; effective batch 8 on one GPU |
| LoRA rank / alpha | 32 / 64; seven attention/MLP projections |
| Learning rate | `1e-4` |
| Schedule | cosine; 3% warmup |
| Optimizer / decay / grad clip | PyTorch AdamW / `0.001` / `1.0` |
| Seed / sampling | 3407 / seeded random without replacement |
| Checkpoint storage | Private Hugging Face model repository; `HF_TOKEN` + repository ID |
| Sequence limit | 2048; record context-overflow exclusions without truncating |
| Loss | all sequence tokens; padding ignored |
| Frame removal | none |
| Checkpoint / periodic validation | every 100 optimizer steps |
| Periodic validation | fixed 256-row subset; full validation at completion |
| Samples | 3 fixed validation prompts at initialization and saves |
| Local checkpoint retention | 3 |

The lower learning rate and rank are initial engineering choices, not empirically proven Urdu optima. See `METHODOLOGY.md` for the proposed ablations. Set a **new `run_id` for every changed configuration**.

## 6. Automatic persistence and continuation

Local layout:

```text
/kaggle/working/orpheus/
  dataset.zip
  extracted/
  cache/<content-fingerprint>/train/*.parquet
  cache/<content-fingerprint>/validation/*.parquet
  runs/<run_id>/
    resolved_config.json
    run_identity.json
    checkpoint-100/
    checkpoint-200/
    adapter_final/         # only after the complete schedule
    samples/step-0000100/
    metrics.jsonl
    metrics.csv
    loss_and_lr.png
    tensorboard/
    ...
```

Hugging Face layout (default):

```text
YOUR_USERNAME/orpheus-urdu-checkpoints  [private model repository]
  runs/<run_id>/
    preparation/
    cache/<fingerprint>/
    snapshots/checkpoint-100-<hash>/<checkpoint and evidence files>
    snapshots/checkpoint-200-<hash>/<checkpoint and evidence files>
    latest.json
    connection_probe.json
```

Optional Drive layout:

```text
orpheus_urdu/<run_id>/
  preparation/             # configuration/audit/environment; even before first save
  cache/<fingerprint>/     # Parquet chunks + checksums
  snapshots/checkpoint-100-<hash>.tar.gz
  snapshots/checkpoint-200-<hash>.tar.gz
  latest.json              # latest successful remote checkpoint pointer
  connection_probe.json
```

1. Encoding commits each 100-row chunk locally, computes its checksum and copies it to the selected private remote store with its metadata. A restart validates and reuses completed chunks. A chunk whose metadata/upload was interrupted is recomputed or uploaded again.
2. Trainer checkpoints include adapter weights, optimizer, scheduler, mixed-precision state when applicable, RNG state, Trainer state and tokenizer. The script verifies required files, hashes the checkpoint files and writes `COMPLETE.json` only after saving finishes.
3. Hugging Face uploads checkpoint and evidence files directly, without building a local tar archive. Drive uses a tar snapshot. Both exclude other checkpoint directories, the local lock and the separate token cache.
4. Hugging Face publishes the snapshot and `latest.json` in **one synchronous commit**. Drive uploads the immutable snapshot first and updates the pointer only after success. If upload/commit fails, the verified checkpoint stays local, training continues, and a queue records the pending upload. The previous remote checkpoint remains the remote resume target until a newer upload succeeds. An outage cooldown prevents every optional file from repeatedly blocking on network retries.
5. Direct snapshot file hashes (or legacy/Drive archive hashes) and checkpoint file hashes are verified on restore. A newer verified local checkpoint wins over an older remote checkpoint. Incomplete local checkpoint directories without a completion marker are skipped. A damaged newest local checkpoint is skipped in favor of an older verified checkpoint, with a warning; damaged files are preserved. If none verifies, training stops rather than silently restarting.
6. Periodic samples are generated after the first checkpoint upload, then an updated snapshot is uploaded with the samples. Training RNG state is restored after monitoring. Each checkpoint may therefore have two immutable remote snapshots.
7. Local retention is bounded; remote snapshots are retained for research/recovery and are never automatically deleted. Budget Hub/Drive space for adapter **and optimizer** state at each saved step. Choose a larger `save_steps` after the smoke run if upload/storage cost is excessive, and archive/remove obsolete snapshots manually after verifying your retained recovery copies. Hub repository history has its own storage implications.

The script catches SIGTERM/SIGINT by asking Trainer to save and stop at the next optimizer boundary. The `session_hours` guard reserves 15 minutes within its budget for saving/samples/upload. This is best effort: a hard kill, out-of-memory error, session deletion or lost network cannot guarantee an emergency save. Recovery is bounded by the **last successfully uploaded** checkpoint and committed cache chunks. Set a shorter budget if uploads take longer than the reserve.

On graceful session cutoff, `status.json` says `paused_for_resume`. Rerun the identical command/config in a fresh session, with the same repo commit, environment and secret. The process restores the latest remote snapshot, downloads/revalidates audio if necessary, restores cache, and passes the checkpoint to `Trainer.train(resume_from_checkpoint=...)`. It retains `num_train_epochs=1`; this continues the original epoch, not a new epoch. `ignore_data_skip=False` preserves the resume position.

The cache identity includes normalized row ordering/audio hashes, model/codec revisions, token format, encoding source/package versions, objective, sequence limit and deduplication. The training identity additionally checks batch geometry, schedule, settings, key package versions and training source hashes. Storage backend/repository selection is recorded in `resolved_config.json` but excluded from the mathematical training identity. If you reinstall a newer stack, resume can be rejected. Restore the original package versions from `requirements-resolved.txt`, preferably the original Kaggle image as well. Do not change the identity file to bypass this check.

Switching the storage setting alone does **not** copy old artifacts into the new remote. Use one repository/backend consistently for a run, or transfer its snapshots, pointer and cache before switching. Existing checkpoints created by an earlier project commit still require that original code/environment: this update changes source hashes, so it is intended for new runs. Use the earlier commit to finish any already-started study rather than bypassing its identity checks.

Do not run two separate Kaggle sessions with the same `run_id` concurrently; the local lock cannot coordinate different machines writing the same remote pointer. Hub commits are atomic, but two writers can still overwrite each other's logical progress.

### Manual recovery and inspection

If `latest.json` is inaccessible or points to a damaged archive, inspect `runs/<run_id>/snapshots/` in the private Hub repository (or `snapshots/` in Drive). Download a previous immutable archive and restore it **to a separate run working directory**. Hub commit history also retains earlier pointer versions. Inspect `status.json`, `resolved_config.json`, `COMPLETE.json` and checksums before resuming. Preserve the corrupt copy for investigation. Never infer resumability from `adapter_model.safetensors` alone.

For ordinary Hub inspection:

```python
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
files = api.list_repo_files("YOUR_USERNAME/orpheus-urdu-checkpoints", repo_type="model")
print("\n".join(p for p in files if p.startswith("runs/YOUR_RUN_ID/snapshots/")))
```

You can restore the latest run artifacts without loading a GPU model:

```python
from pathlib import Path
from orpheus_utils import HuggingFaceStore
store = HuggingFaceStore("YOUR_USERNAME/orpheus-urdu-checkpoints", "YOUR_RUN_ID",
                        token=os.environ["HF_TOKEN"])
store.preflight()
run = Path("/kaggle/working/orpheus/runs/YOUR_RUN_ID")
run.mkdir(parents=True, exist_ok=True)
store.restore(run)
```

Snapshots are versioned files in the Hub artifact repository, or archives for Drive/older runs. `PeftModel.from_pretrained("YOUR_USERNAME/orpheus-urdu-checkpoints")` will not load an adapter directly from its root. Restore the snapshot and pass the resulting local `adapter_final` or `checkpoint-N` directory to `synthesize.py`.

For optional Drive inspection:

```bash
rclone --config /YOUR/PRIVATE/rclone.conf lsf gdrive:orpheus_urdu/YOUR_RUN_ID
```

A final adapter is an inference artifact. A complete Trainer checkpoint is the artifact needed to resume optimization. Keep both.

## 7. Listen during and after training

Automatic monitoring writes WAVs under `samples/step-XXXXXXX/`. Each sample has prompt, reference path, seed, generation settings, duration, stop-token status, invalid-frame offset and raw generated tokens. These are the same prompts and seeds across checkpoints. Baseline generation at step zero can fail to produce Urdu; this is logged without preventing checkpoint storage.

Listen without allocating another model while training:

```python
from pathlib import Path
from IPython.display import Audio, display
root = Path("/kaggle/working/orpheus/runs/aslp50h-r32-lr1e4-seed3407/samples")
paths = sorted(root.glob("step-*/*.wav"))
for path in paths[-3:]:
    print(path)
    display(Audio(filename=str(path)))
```

After training releases the GPU:

```bash
python synthesize.py \
  --adapter /kaggle/working/orpheus/runs/aslp50h-r32-lr1e4-seed3407/adapter_final \
  --text 'آج موسم بہت خوشگوار ہے اور ہم سب باہر سیر کے لیے جا رہے ہیں۔' \
  --output /kaggle/working/urdu_test.wav
```

An intermediate checkpoint works with the same command: replace `adapter_final` with `checkpoint-100`. Run this in a separate idle session or after pausing training; loading another 3B model in the training runtime can cause an OOM.

`--temperature`, `--top-p`, `--repetition-penalty`, `--seed` and `--max-new-tokens` are exposed. The output budget is capped to remaining model context. Seven audio tokens form one SNAC frame; token budgets are not seconds. If the speech stop token is absent or invalid frames occur, metadata marks the problem. Longer sentences may need shorter sentence-level prompts or a separately validated larger training context.

### Batch intelligibility evaluation

Prepare a fixed CSV `heldout_prompts.csv` with a `text` column containing at least 100 manually checked, held-out Urdu sentences. Add `speaker` only for a speaker-conditioned experiment. Generate all prompts with one loaded model:

```bash
python synthesize.py --adapter /PATH/adapter_final \
  --manifest /PATH/heldout_prompts.csv --output /kaggle/working/final_eval
python synthesize.py --adapter /PATH/adapter_final --base-only \
  --manifest /PATH/heldout_prompts.csv --output /kaggle/working/base_eval
python evaluate_asr.py --manifest /kaggle/working/final_eval/generated.csv \
  --output /kaggle/working/final_eval/asr
```

The evaluator transcribes with a pinned resolved revision of Whisper-large-v3, saves raw/normalized reference and hypothesis per utterance, corpus and mean-utterance WER/CER, and 1,000 bootstrap resamples for 95% intervals. It loads the ASR model only after TTS training/inference has exited. Whisper mistakes in Urdu can inflate the scores; evaluate the original held-out human speech with the same ASR as a calibration baseline. Manually inspect substitutions and severe outliers. Measure human naturalness separately with blinded native-Urdu listening.

These offline evaluation outputs are outside the training run and are **not automatically uploaded** by the Trainer callback. Upload them explicitly to the same private Hub repository:

```python
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id="YOUR_USERNAME/orpheus-urdu-checkpoints", repo_type="model",
    folder_path="/kaggle/working/final_eval",
    path_in_repo="research/YOUR_RUN_ID/final_eval")
```

Alternatively, copy them to Drive with the optional rclone backend or save them as Kaggle outputs.

## 8. Research artifacts

| Artifact | Interpretation |
|---|---|
| `metrics.jsonl` / `metrics.csv` | Step, epoch, loss/LR/gradient logs, validation loss, timestamps, GPU/RAM stats |
| `loss_and_lr.png` | Rebuilt loss and learning-rate curves |
| `tensorboard/` | Standard event logs; use `%tensorboard --logdir ...` |
| `dataset_report.json` | Actual split hours/counts, manifests, audit limitations |
| `data_inventory.csv` | Every clip's normalized transcript, hash, duration and signal stats |
| `data_errors.json` | Audit failures; normally empty |
| `normalized_manifests.json` | Exact cleaned text and order used |
| `token_format.json` / `token_statistics.json` | Codec/token identity, sequence percentiles, token/frame counts |
| `resolved_config.json` | Full settings and resolved model/codec commits |
| `environment.json` / `requirements-resolved.txt` / `gpu.txt` | Software and hardware evidence |
| `source/` / `run_identity.json` | Saved training code and hash-based resume identity |
| `parameters.json` / `trainable_parameters.txt` | Trainable parameter count and names; measured rather than assumed |
| `monitoring_prompts.json` / `periodic_eval_rows.json` | Exact monitoring/evaluation subsets |
| `samples/` / `sample_errors.jsonl` | Generated speech, decoding metadata and failures |
| `train_results.json` / `validation_results.json` | Trainer summary metrics |
| `status.json` / `run.log` | Last state and pipeline diagnostics |
| `checkpoint-*/COMPLETE.json` | Resumable checkpoint file checksums |

Interrupted sessions may leave JSONL records after the last remotely committed step. Resuming from that step can repeat measurements; timestamps preserve both attempts and plots use the latest measurement for a given step/metric. Logging is every ten optimizer steps by default, not every microbatch. `session_elapsed_seconds` is per process; use timestamps and session boundaries to account for total study time.

Final validation loss is token cross entropy for the configured objective. It is not a MOS, pronunciation accuracy or emotion score. Do not compare an audio-only loss to an all-token loss as if they were the same metric.

## 9. Troubleshooting

| Problem | Action |
|---|---|
| Global `pip check` lists conflicts for Kaggle tools | Start a fresh session and use the venv setup in Cell 2; project dependency checks replace the global check. See Section 5. |
| Hub preflight fails | Check `HF_TOKEN` Secret access, token write permission, repository ID/type, private visibility, storage allowance and Internet. Try `--mode storage-check` first. |
| Missing Hub repo ID | Set `hf_repo_id`, `--hf-repo-id` or `ORPHEUS_HF_REPO` to `USERNAME/REPOSITORY`. |
| Optional Drive preflight fails | Check notebook Secret access, OAuth expiry, remote name, Drive scope, quota and Internet. No GPU training begins before the probe succeeds. |
| ZIP download fails | Check sharing and Drive download quota. Upload the extracted paired dataset as a private Kaggle Dataset and set `data_dir` to its `/kaggle/input/...` path. |
| No manifests found | Inspect archive folder names; set explicit `data_dir` and filenames. |
| Training hours below 35 | Inspect `dataset_report.json`. The supplied ~47h archive includes validation/test; the logged retained training split is 37.41h, so the former 40h threshold was incorrect. |
| Clip exceeds 2048 tokens | Review duration/text, then increase `max_length` for a new run/cache if it fits VRAM. Do not clip the audio while leaving the full transcript. |
| CUDA OOM | Microbatch already defaults to 1. Use a smaller rank/new run or a larger GPU. Reduce context only after reviewing the data. Avoid concurrent model loading. |
| NaN/Inf training logs | Inspect data and gradients; reduce LR in a new run. Preserve evidence; do not relabel the resumed experiment. |
| Resume identity differs | Restore the original code/config/environment or start a new `run_id`. Changing batch size during resume is intentionally rejected. |
| Runtime ends unexpectedly | Restart with the same repo/backend and run ID; only the last successfully committed remote checkpoint is recoverable. |
| Voice changes between samples | This stage learns language from diverse speakers without speaker conditioning. Stable voice identity requires verified conditioning data. |
| Emotion words are spoken aloud | Stage 1 does not train an arbitrary emotion-control interface. Use grounded emotion annotations in stage 2. |

CPU validation of this repository:

```bash
python -m unittest discover -s tests -v
python finetune_aslp_50h.py --help
python synthesize.py --help
```

## 10. Kaggle cells: clone, background run, status and playback

These are the runnable setup cells requested for the workflow. You can also import `kaggle_run.ipynb`, which contains the recommended saved-run path.

### Cell 1 — clone a fixed version

```python
import os, sys, subprocess, re
from pathlib import Path

REPO_URL = "https://github.com/hassan-31x/orpheus-ft-urdu.git"
REPO_REF = "main"  # Use the same fixed commit for an experiment and its resumes.
HF_REPO_ID = "hassan-31x/orpheus-urdu-checkpoints"  # Your private model repository.
RUN_ID_OVERRIDE = None  # Optional separate experiment; unsaved failed attempts recover automatically.
AUDIT_ONLY = False  # Set True with GPU OFF to inspect data before training.
MINIMUM_TRAIN_HOURS = 35  # The 47h archive contains ~38h train plus validation/test.
SESSION_HOURS = 10.5  # Set below the limit displayed in your Kaggle account.
REPO = Path("/kaggle/working/urdu-orpheus")

if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", HF_REPO_ID):
    raise ValueError("HF_REPO_ID must have the format username/repository")
from kaggle_secrets import UserSecretsClient
try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    raise RuntimeError("Add the HF_TOKEN Kaggle Secret and enable access for this notebook") from None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

if not REPO.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
# Fetch even when the directory already exists; checkout alone leaves old code.
subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO, check=True)
for name in ("setup_kaggle.py", "check_environment.py", "requirements-kaggle.txt", "pipeline_recovery.py"):
    if not (REPO / name).is_file():
        raise RuntimeError(f"Repository version lacks {name}. Upload the updated project before running.")
print("Project commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
```

### Cell 2 — install and preflight

```python
# Always install in this runtime: saved Kaggle jobs start with a fresh environment.
ENV = Path("/kaggle/working/orpheus-env")
TRAIN_PYTHON = str(ENV / "bin" / "python")
setup_command = [sys.executable, "-u", str(REPO / "setup_kaggle.py"), "--venv", str(ENV)]
if not AUDIT_ONLY:
    setup_command.append("--gpu")
subprocess.run(setup_command, cwd=REPO, check=True)
assert Path(TRAIN_PYTHON).is_file(), "Training environment was not created"
print("All training subprocesses will use:", TRAIN_PYTHON)
subprocess.run(["df", "-h", "/kaggle/working"], check=True)
```

### Cell 3 — private credentials and config

```python
import json
cfg = json.loads((REPO / "configs/aslp50h.json").read_text())
if RUN_ID_OVERRIDE:
    cfg["run_id"] = RUN_ID_OVERRIDE
cfg["checkpoint_backend"] = "huggingface"
cfg["hf_repo_id"] = HF_REPO_ID
cfg["session_hours"] = SESSION_HOURS
cfg["minimum_train_hours"] = MINIMUM_TRAIN_HOURS
print("Minimum retained TRAIN split hours:", cfg["minimum_train_hours"])
print("Retention thresholds:", "enforced" if cfg["enforce_retention_limits"] else "warnings only")
print("Data policy:", cfg["invalid_row_policy"], "maximum training exclusion fraction:", cfg["maximum_rejected_train_fraction"])
CONFIG = Path("/kaggle/working/run-config.json")
CONFIG.write_text(json.dumps(cfg, indent=2))
RUN = Path(cfg["work_dir"]) / "runs" / cfg["run_id"]
subprocess.run([TRAIN_PYTHON, str(REPO / "finetune_aslp_50h.py"),
                "--config", str(CONFIG), "--mode", "storage-check"], cwd=REPO, check=True)
```

### Cell 4A — recommended unattended saved run

```python
# Blocks until the audit/training process completes. Failures print diagnostics.
command = [TRAIN_PYTHON, "-u", str(REPO / "finetune_aslp_50h.py"), "--config", str(CONFIG)]
if AUDIT_ONLY:
    command += ["--mode", "audit"]
try:
    subprocess.run(command, cwd=REPO, check=True)
except subprocess.CalledProcessError:
    import shutil
    print("Free disk GiB:", round(shutil.disk_usage(RUN).free / 2**30, 2))
    path = RUN / "disk_budget.json"
    if path.exists():
        print("Disk budget:", path.read_text())
    print("Training process exited. The traceback above is the failure; audit exclusions are separate.")
    raise
```

Use Kaggle **Save Version → Save & Run All** (wording may vary with the UI). The committed notebook executes the cells as a background Kaggle job; the blocking training cell keeps the job alive while training. You can close your laptop after confirming the saved run has started. The job still has Kaggle runtime/quota limits. Enable the `HF_TOKEN` Secret for the notebook before saving the version. Turn off an unused interactive GPU session to avoid consuming a second allocation.

Do **not** use a detached Popen cell in a committed notebook and let the notebook immediately finish: the environment may be torn down with its child process.

### Cell 4B — optional interactive detached process

Use this instead of 4A only when you want to keep the interactive notebook open for status/sample playback:

```python
LOG_PATH = Path("/kaggle/working/orpheus-background.log")
with LOG_PATH.open("a") as log:
    proc = subprocess.Popen([TRAIN_PYTHON, "-u", str(REPO / "finetune_aslp_50h.py"),
                             "--config", str(CONFIG)], cwd=REPO,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path("/kaggle/working/orpheus.pid").write_text(str(proc.pid))
print("PID", proc.pid, "log", LOG_PATH)
```

This detaches the process from the cell; it does not remove session inactivity/runtime limits or create an independent scheduled Kaggle job. Never launch 4A and 4B for the same run simultaneously. For a saved version containing 4B, add a blocking `proc.wait()` cell and check its return code; using 4A is simpler.

### Cell 5 — status and latest samples

```python
from IPython.display import Audio, display

def optional_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as exc:
        print("Optional report unavailable:", path.name, type(exc).__name__)
        return {}

status_value = optional_json(RUN / "status.json")
print(status_value or ("Dataset audit only" if AUDIT_ONLY else "No training status recorded"))
for path in sorted((RUN / "samples").glob("step-*/*.wav"))[-3:]:
    print(path)
    try:
        display(Audio(filename=str(path)))
    except (OSError, ValueError) as exc:
        print("Optional playback unavailable:", type(exc).__name__)
backup = optional_json(RUN / "backup_status.json")
if backup:
    print("Remote backup:", backup)
```

In the saved-run path this cell executes after 4A returns. During an interactive detached run you can execute it repeatedly.

### Cell 6 — test the final adapter after completion

```python
if AUDIT_ONLY:
    print("Dataset audit completed; set AUDIT_ONLY=False and enable GPU for training.")
else:
    status_value = optional_json(RUN / "status.json")
    if status_value.get("status") in ("complete", "smoke_complete"):
        output = Path("/kaggle/working/urdu_test.wav")
        try:
            subprocess.run([TRAIN_PYTHON, str(REPO / "synthesize.py"),
                            "--adapter", str(RUN / "adapter_final"),
                            "--text", "آج موسم بہت خوشگوار ہے اور ہم سب باہر سیر کے لیے جا رہے ہیں۔",
                            "--output", str(output)], cwd=REPO, check=True)
            display(Audio(filename=str(output)))
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            print("Training adapter is saved at", RUN / "adapter_final")
            print("Optional final synthesis failed; training remains complete. See synthesis_status.json.")
            try:
                (RUN / "synthesis_status.json").write_text(json.dumps({"status": "failed", "returncode": getattr(exc, "returncode", None), "error_type": type(exc).__name__}))
            except OSError:
                print("Could not write the optional synthesis report.")

    else:
        print("Saved for continuation; rerun the same config to finish the epoch.")
```

For a later Kaggle session, rerun Cells 1–4A with the **same commit, private repository, run ID, config and package versions**. Automatic remote restore handles continuation. The default cells need only the `HF_TOKEN` secret; rclone and its OAuth setup are unnecessary unless you deliberately choose the Drive backend.

## When the dataset audit stops a run

A traceback with `invalid/duplicate rows` means the audit rejected rows before encoding or model training. Read `data_error_summary.json` and `data_errors.json` in your run directory. The summary groups reasons by split and prints example paths and CSV lines in the notebook; full details remain in the adjacent file. The reports and valid-row inventory are also uploaded under `runs/<run_id>/audit/` when the selected backend permits it. Rejected rows are automatically excluded under the default skip policy, with every reason logged; audio is not automatically converted. Duplicates keep the first valid occurrence in train → validation → test order, so duplicate evaluation rows are excluded without adding evaluation audio to training.

The training subprocess exits after this error. If the saved notebook still says Running, it may be finishing output conversion; stopping that failed job does not interrupt active training. Save/download the diagnostic JSON before ending an interactive session if remote backup is unavailable. Updating GitHub will not update a process or notebook already running.

Before using more GPU time, import the updated notebook, set `AUDIT_ONLY=True` in Cell 1, turn GPU off, and run it. This downloads/checks the data without model imports or tokenization. Exclusions now pass automatically under permissive defaults while retention warnings are recorded; train and validation must still remain nonempty. After it passes, set `AUDIT_ONLY=False` and enable GPU. The actual run exercises checkpointing at optimizer step 1; a separate smoke experiment is optional. A CPU audit does not prove the CUDA training, VRAM budget or live checkpoint restore will succeed. Same-data/source/config requirements for checkpoint resume still apply; do not switch an existing trained run to modified code. An audit failure before encoding has no training checkpoint to preserve.

## Automatic fallbacks and their limits

Defaults: `invalid_row_policy="skip"`, `enforce_retention_limits=false`, `maximum_rejected_train_fraction=0.05`, and `minimum_train_hours=35`. Count and measurable-duration thresholds produce warnings by default. Set `enforce_retention_limits=true` to make them stopping conditions. Missing/corrupt files may have unknown duration; the report identifies these and still counts them toward the row limit. A small row fraction alone cannot establish a small hour fraction. Validation and test have separate original/retained counts and hours; discarded evaluation duplicates change the benchmark denominator and must be reported.

Sequences exceeding `max_length` are excluded during SNAC encoding, without truncating their speech/text pair. Their reasons/durations are in `encoding_errors.json`, their cache chunks record accepted and excluded rows, and cached restarts restore the same decision. The retention report includes **audit plus encoding plus GPU-memory** exclusions. `training_manifests.json` records the final retained rows; the updated `dataset_report.json` records final hours/counts. One epoch now means one pass over that retained partition.

Monitoring synthesis and plotting failures are logged without stopping checkpoint work. Temporary downloads receive bounded retries; failed remote writes are queued while verified local state is retained. Broken model/token formats, changed/corrupt optimizer checkpoints, empty train/validation partitions, and unusable CUDA kernels still stop. Local disk exhaustion cannot be hidden because the required training artifacts cannot then be saved. Automatic changes to LR, context length, optimizer or batch geometry are not applied mid-run. No fallback can guarantee every Kaggle run succeeds.

For this update, upload all changed files and re-import the notebook, then leave `AUDIT_ONLY=False` to audit, skip small exclusions and proceed automatically. Do not update a live process. Since the reported failure occurred before encoding/training, that failed run has no optimizer progress to lose. Existing trained runs require their original code/config/environment for exact resume.

### Measured split sizes from the supplied run log

The failed run retained **17,392 training clips / 37.4096h**, **2,129 validation clips / 4.6381h**, and **2,142 test clips / 4.6911h**: **46.7388h total**. The 423 exclusions were exact audio-byte duplicates: 277 train, 80 validation, 66 test. Training exclusions removed 0.4075h (about 24.45 minutes), or 1.0776% of measurable source training hours. The failure was the old `minimum_train_hours=40` check, not excessive filtering or a model/GPU error. The archive's total hours cannot be used as the minimum for its training partition.

The default and notebook override now use `minimum_train_hours=35`, which accepts the measured training partition. Retention floors are now warnings under the permissive default; strict enforcement remains configurable. A regression test reproduces the logged split counts/hours and verifies the default accepts them. If an existing `/kaggle/working/run-config.json` still contains 40, set it to 35 before rerunning; changing the repository JSON alone does not rewrite a previously generated run config. This failure happened before tokenization and training, so there is no optimizer progress to preserve.

## Unattended recovery review

Read `REVIEW.md` for the complete stage-by-stage review and test boundaries. The notebook now performs model forward/backward memory checks automatically, before any optimizer updates. It tries the longest retained training sequence at the configured microbatch, exercises up to two accumulated backward passes, and reserves estimated AdamW-state/temporary memory. If a fresh run gets CUDA OOM, it lowers the allowed padded length by roughly 20%, records the discarded rows, and retries up to eight times. It never truncates audio against a full transcript. `training_selection.json` freezes the accepted training order/IDs for resume; `effective_training_manifests.json` records the final rows. A resumed optimizer checkpoint cannot silently switch to a different selection.

Default `optimizer="adamw_torch"` uses standard PyTorch AdamW for LoRA parameters, avoiding an additional bitsandbytes optimizer kernel dependency. Base loading remains 4-bit and still requires compatible bitsandbytes. This choice increases optimizer memory relative to 8-bit AdamW; the backward preflight reserves estimated state memory. Keep `adamw_8bit` only for a separately measured experiment. A memory probe is an early compatibility check, not a guarantee against all later allocator or kernel failures.

Unreadable/invalid audio encountered during encoding, codec OOM on an individual clip, and context overflow produce documented row exclusions. Verified chunks are reused without downloading over existing files; damaged/incomplete derived chunks are rebuilt. A remote cache download failure permits re-encoding missing chunks. Optimizer checkpoint restore and checksum verification remain strict.

`pending_uploads.json` and `backup_status.json` track remote writes. Failed uploads use a 60-second cooldown, then retry on subsequent writes/checkpoints; at most eight queued optional files drain per successful checkpoint. Final and completed-run backup attempts bypass cooldown once. Training can finish locally with **remote backup pending**. If Kaggle deletes that runtime before uploads succeed, only the previously uploaded checkpoint is recoverable; queued files are not remote backups. Check the displayed backup status before discarding outputs. No credential values are put in the queue.

Optional TensorBoard, plotting, sample generation, validation and final notebook synthesis failures are recorded rather than turning usable training progress into a failed notebook. `evaluation_status.json` records unavailable evaluation; no placeholder loss is fabricated. The final adapter is exported before whole-validation evaluation. Canonical `metrics.jsonl` remains the primary metric record; a reporting IO failure is warned in stdout and may leave gaps. Checkpoint files remain required for safe resume.

### Disk-full checkpoint failure (October 5 fix)

`SafetensorError ... No space left on device` means the model was training but its checkpoint could not be written. PEFT can include frozen, resized embedding weights, so adapter saves can be much larger than the LoRA matrices. This version keeps those weights for correctness, budgets them plus optimizer state (and temporary upload archives only for the Drive backend), checks free disk before training and each save, saves at optimizer step 1, and prunes older verified checkpoints under pressure while preserving the newest and any pending upload. A checkpoint write that runs out of space removes only its unsealed partial directory and retries once in the same process. Insufficient space after cleanup remains a hard failure; skipping every save would leave training unprotected.

The dataset ZIP is removed after successful extraction, and a completed extraction is reused without downloading it again. WAVs and encoded data remain available. Dependency installation now uses `--no-cache-dir`. The notebook prints disk diagnostics instead of misleading old audit exclusions. `disk_budget.json` records estimated requirements and free space. Budgets are conservative estimates, not a guarantee against other processes consuming disk.

For the already-failed job:

1. Keep the existing Kaggle session/files if available. The exited training subprocess has lost unsaved in-memory weights; an incomplete checkpoint cannot restore that progress.
2. Upload the updated project to GitHub and import the updated `kaggle_run.ipynb` (a Git fetch does not replace notebook cells).
3. Keep your existing run ID. If the earlier attempt has **no verified checkpoint**, the updated script automatically archives its metadata and metrics under `attempt_history/`, resets stale training selections, and starts a fresh optimizer schedule. No run-ID override or metadata deletion is required. If a verified checkpoint exists, retain its files and original source; the strict source-identity resume check still applies.
4. In the existing runtime, you can reclaim only the known redundant archive and incomplete checkpoint files with the cell below, **after the old training subprocess has stopped**. Preserve complete checkpoints. Then run the updated notebook cells. Existing extracted audio is reused; token caches may be regenerated because source fingerprints changed.

```python
from pathlib import Path
import shutil
work = Path("/kaggle/working/orpheus")
if (work / "extracted/.extraction.json").is_file():
    (work / "dataset.zip").unlink(missing_ok=True)
old_run = work / "runs/aslp50h-r32-lr1e4-seed3407"
for cp in old_run.glob("checkpoint-*"):
    if cp.is_dir() and not (cp / "COMPLETE.json").exists():
        shutil.rmtree(cp)  # Failed, unsealed writes only; training must be stopped.
print("Free GiB:", shutil.disk_usage(work).free / 2**30)
```

Do not delete Hugging Face model caches, extracted WAVs, or sealed checkpoints to make a run appear resumable. If the remaining disk cannot fit the budget, place the source data on a mounted Kaggle Dataset and set `data_dir` accordingly before starting a new experiment.

### Updated code after an attempt without a saved checkpoint

A changed source fingerprint used to stop even when only preparation metadata existed. The script now checks for a verified checkpoint first. With no verified checkpoint, it preserves earlier evidence in `attempt_history/`, records `attempt_recovery.jsonl`, clears stale memory selections and per-attempt metrics, and continues automatically from optimizer step zero. Audio and content-addressed caches remain; cache reuse still requires the current fingerprint. With a verified checkpoint, a mismatched identity still stops rather than altering saved training semantics. An empty Hugging Face commit warning is informational and is unrelated to this check.

### Reduced local checkpoint storage for Hugging Face

Hugging Face snapshots now use `files-v1`: the complete checkpoint, experiment evidence and `latest.json` are published in one synchronous commit, directly from existing files. No full tar copy is written locally. Restores verify every downloaded file and the sealed checkpoint before moving files into the run directory; older archive snapshots remain readable.

The free-space requirement is the next checkpoint estimate plus 512 MiB of margin. Existing files are already reflected in measured free space. Older verified checkpoints are removed under pressure while the newest and pending-upload checkpoint are protected. Final adapter export is budgeted when it occurs, rather than being reserved throughout training. Drive still reserves an additional checkpoint/archive copy and experiment evidence.

For the supplied failure (checkpoint estimate 4,903,436,288 bytes), the Hugging Face check now requires **5.07 GiB**, within the logged **10.57 GiB** free. This is an estimate; external writes and transfer-library buffers can still consume storage. Update the GitHub project, rerun the notebook fetch cell and training cell using the existing config/run ID. No new secret, dependency install, manual disk-limit override, or checkpoint-format setting is needed.

### Full review: preventing late failures and repeated preparation

The run now starts monitoring after its first saved optimizer step by default (`sample_at_start=false`). It uploads a checkpoint before plotting or synthesis and skips that optional work near session cutoff. TensorBoard is managed by the monitor and disabled if its writer fails; JSONL continues. Optional metrics/status reports, validation IO failures, nonfinite validation results, plotting and notebook playback errors cannot by themselves abort the optimization process. Missing metrics are reported as unavailable, never invented.

The final adapter reuses the verified final checkpoint weights through hardlinks on the same filesystem, avoiding another multi-GB serialization. Filesystems without hardlinks fall back to copying. Editing exported adapter weights in place would also change their linked checkpoint; treat both as immutable model artifacts.

Hub backups take stable copies of live logs and TensorBoard events before hashing/uploading; model and optimizer files are uploaded directly. Unreadable optional upload queues are rebuilt. If a remote restore fails but a verified local checkpoint exists, the local checkpoint is used. No local checkpoint means a failed remote restore still stops, since silently starting over could overwrite recoverable remote progress.

A damaged newest local checkpoint falls back to an older checksum-verified copy. A corrupt-only checkpoint set stops. Trainer retains at least two checkpoint directories during saving so rotation cannot remove the last good predecessor before the new save is sealed; disk-pressure pruning continues to protect the newest verified checkpoint.

Storage/reporting edits no longer necessarily force SNAC encoding again. Reuse requires identical data/model/tokenizer/codec/settings/package fingerprints AND matching parsed implementations of the token-producing functions/constants in captured source. Changes to tokenization or data rebuild the cache. Preparation backups now include source and token-format evidence so this comparison is also possible across sessions. The encoding source fingerprint is retained when compatibility is proven, while training source hashes record the current code.

Re-running setup installs missing/incompatible requirements without upgrading already-satisfying packages. Captured run/source now includes default config files, and executing captured sources does not fail by copying a file onto itself. Incompatible saved training sources are detected before GPU imports and audio audit.

Use the updated notebook cells from this README or `kaggle_run.ipynb`. Keep the existing run ID for your attempt with no verified checkpoint. Once training has a checkpoint, keep its recorded source/environment for resumes. The remaining hard failures are invalid optimization state, unusable CUDA/kernel state, no usable data, and inability to write any safe checkpoint after recovery. These cannot be treated as successful training.
