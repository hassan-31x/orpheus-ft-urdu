# Urdu Orpheus: ASLP language adaptation on Kaggle

This repository adapts `unsloth/orpheus-3b-0.1-pretrained` to Urdu using every row of your supplied **training manifest**, for **one epoch**. It downloads your Drive ZIP, validates the paired audio/transcripts, encodes speech with SNAC, trains LoRA adapters, measures held-out loss, generates intermediate Urdu samples, and backs up resumable training state to your Google Drive.

The supplied URL is included in `configs/aslp50h.json`:

<https://drive.google.com/file/d/1Xslte2s7PLzLCqBxDope2LQi4cmnfDJ7/view>

Drive displays the filename `Urdu_Orpheus_US_Std_Training_47h.zip`. The script measures actual hours; “50h” is the experiment label, not a fabricated duration. All valid training clips are used; validation/test manifests retain their evaluation roles. There is no ten-hour sampling limit, and no automatic truncation or silent skipping of clips. If you want a different train/validation split, prepare and document it before starting a new experiment. Training on the test set would invalidate its use as a held-out test.

This is **stage 1: Urdu language adaptation**. Arbitrary labels such as “happy” or “sad” have not been taught by this run. Stage 2 needs aligned Urdu speech, accurate transcripts, and verified emotion labels. A speech emotion recognition dataset alone is insufficient if its speech has no usable transcripts.

## 1. Files and prerequisites

| File | Purpose |
|---|---|
| `finetune_aslp_50h.py` | Download, audit, cache, train, monitor, resume |
| `configs/aslp50h.json` | Full initial experiment settings |
| `orpheus_utils.py` | Token format, artifact verification, Drive persistence |
| `synthesize.py` | Single/batch inference from final or intermediate adapters |
| `evaluate_asr.py` | Optional Urdu ASR WER/CER with bootstrap intervals |
| `kaggle_run.ipynb` | Ready-to-import notebook; replace repository URL |
| `METHODOLOGY.md` | Research review, exact choices, ablations, evaluation limitations |
| `tests/test_pipeline.py` | CPU tests for token and persistence correctness |

Your original `finetune_aslp_10h.py`, paper, and `resources.md` are unchanged. The original Python file contains notebook shell syntax (`!nvidia-smi`); the new scripts are normal Python programs.

Use a Linux NVIDIA GPU environment. Kaggle T4 with 4-bit loading is the conservative starting point. The script exposes one GPU with `CUDA_VISIBLE_DEVICES=0`; **two T4s do not combine into one 32GB device**. This is not a distributed-training script. 16-bit LoRA is a separate experiment for a device with adequate VRAM. A GPU label is not a guarantee that the chosen sequence length/microbatch fits.

Enable **Internet**, choose an **NVIDIA GPU accelerator**, and check the runtime limit and remaining GPU allowance shown in your Kaggle account. Set `session_hours` to a budget **shorter than that displayed limit**, allowing preparation, notebook setup and uploads. The shipped `10.5` is an example budget, not a claim that every Kaggle session lasts that long. A full epoch may require several sessions.

Check free disk space. 50 hours of 24kHz mono PCM16 audio is about 8.64GB before file overhead. You also need space for the compressed download, extracted audio, base weights, Arrow caches, three local resumable checkpoints, final adapter and a temporary snapshot archive. Actual use depends on clip lengths, model precision and optimizer state; provision several tens of GB and inspect `df -h /kaggle/working`.

The complete GPU workflow has **not been run in this local macOS workspace**. The CPU tests pass, but Unsloth/CUDA installation, Drive authentication, ZIP contents and GPU training must be verified in your Kaggle smoke run. The public Drive page was inspected; the multi-GB ZIP was not downloaded here.

## 2. Put the code on GitHub

Create a repository and upload the new scripts, `configs/`, `tests/`, both Markdown guides, requirements and notebook. Use a repository you control. No GitHub repository has been created or published automatically by this work.

Do not upload audio, adapter weights, training outputs, OAuth configuration or access tokens. `.gitignore` excludes the common generated files; inspect what you commit. Preserve dataset/model attribution and check redistribution terms before publishing derived assets. The attached IEEE paper can remain a private reference; it is not needed at runtime.

For reproducible runs, clone a fixed commit or tag rather than a moving branch. Record the commit in your research notes. The pipeline also hashes and saves the actual training source files.

## 3. Authorize checkpoint storage in your own Google Drive

The public download link grants no write access to your Drive. Kaggle does not support Colab's `drive.mount()` API. The pipeline uses **rclone + Google OAuth**, with the configuration stored as a Kaggle Secret.

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
8. The default storage root is `gdrive:orpheus_urdu/<run_id>`. You may set `ORPHEUS_DRIVE_REMOTE` to a different root in the notebook. Do not append `<run_id>` yourself; the script does that.

The script writes and reads back a small probe before doing GPU work. Credentials are decoded into a temporary directory with restricted permissions and are not included in snapshots. Token refresh can update the temporary rclone config during a session; future sessions use the refresh token in your Kaggle Secret. If you revoke it, replace the secret after reauthorization.

For deliberate testing without Drive, add `--local-only`. That explicitly disables remote protection: local `/kaggle/working` files alone are not a reliable resume strategy across deleted runtimes.

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

The audit records every audio file's SHA-256, duration, peak, RMS, clipped fraction and transcript normalization changes. It checks nonempty Urdu-script text, finite/non-silent signals, duplicate audio paths and identical files across splits. Invalid rows stop the run and are written to `data_errors.json`; review and repair the dataset rather than silently discarding data. Similar excerpts of the same source recording can still leak across splits: preserve source/session-disjoint splits whenever possible.

Default speaker prompting is disabled because the trial identified unreliable speaker IDs. Do not use segment-local `SPEAKER_0001` as a global identity. If you have genuinely verified global speaker labels, set `speaker_column` and use the matching speaker string at inference. This is a separate conditioning experiment.

## 5. Configure and verify the Kaggle environment

Use the notebook cells at the end of this guide, or import `kaggle_run.ipynb`. Replace the repository URL first.

`requirements-kaggle.txt` pins Transformers/TRL to the versions used by the inspected upstream Orpheus notebook and bounds the dataset/codec dependencies. Unsloth and the CUDA stack are not fully locked across all possible Kaggle images. Pip resolves a compatible stack, and each run records `pip freeze`, CUDA, Torch and package versions. **A resolved requirements file is evidence of the environment, not proof it was tested on every GPU.**

Run `pip check` and the import preflight. Restart the kernel if installing packages has affected modules already imported in the notebook. The training subprocess imports Unsloth before Transformers as required. Do not manually install a random xformers/Torch wheel; match the [Unsloth installation guide](https://unsloth.ai/docs/get-started/install) to the runtime's Torch/CUDA versions if the preflight fails. Use Torch 2.6+ because resumable Trainer checkpoints load optimizer/RNG state through Torch serialization.

Before a long run, create a **separate smoke config**:

```python
import json
from pathlib import Path
cfg = json.loads(Path("/kaggle/working/urdu-orpheus/configs/aslp50h.json").read_text())
cfg.update(run_id="aslp50h-smoke", max_steps=2, save_steps=1,
           eval_steps=1, eval_samples=4, sample_count=1)
Path("/kaggle/working/smoke.json").write_text(json.dumps(cfg, indent=2))
```

Run with `--config /kaggle/working/smoke.json`. This intentionally prepares all supplied training data and then trains only two steps. It tests the real maximum-length data/cache, a training backward pass, checkpoint optimizer state, validation, sample decoding, and Drive upload. `smoke_complete` must not be reported as a full epoch.

To test resume, pause a longer smoke configuration before its step limit, confirm `latest.json` and snapshot exist in Drive, and rerun the same command in a fresh runtime. Do not change `max_steps` for that resume. The restored step should advance from the saved value rather than start at zero. A completed two-step smoke run simply exits on rerun; it does not exercise continuation by itself.

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
| Optimizer / decay / grad clip | 8-bit AdamW / `0.001` / `1.0` |
| Seed / sampling | 3407 / seeded random without replacement |
| Sequence limit | 2048; fail rather than truncate |
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

Drive layout:

```text
orpheus_urdu/<run_id>/
  preparation/             # configuration/audit/environment; even before first save
  cache/<fingerprint>/     # Parquet chunks + checksums
  snapshots/checkpoint-100-<hash>.tar.gz
  snapshots/checkpoint-200-<hash>.tar.gz
  latest.json              # latest successful remote checkpoint pointer
  connection_probe.json
```

1. Encoding commits each 100-row chunk locally, computes its checksum and copies it to Drive with its metadata. A restart validates and reuses completed chunks. A chunk whose metadata/upload was interrupted is recomputed or uploaded again.
2. Trainer checkpoints include adapter weights, optimizer, scheduler, mixed-precision state when applicable, RNG state, Trainer state and tokenizer. The script verifies required files, hashes the checkpoint files and writes `COMPLETE.json` only after saving finishes.
3. A tar snapshot includes the current complete checkpoint and experiment evidence. It excludes other checkpoint directories, the local lock and the separate token cache.
4. The snapshot is uploaded first. **Only after success** is `latest.json` updated. If upload fails, the process stops and the previous remote checkpoint stays the resume target. Retrying with the same local runtime can upload the newer complete local checkpoint after continuing.
5. Snapshot archive hashes and checkpoint file hashes are verified on restore. A newer verified local checkpoint wins over an older remote checkpoint. Incomplete local checkpoint directories without a completion marker are skipped. A checksum mismatch raises an error so corruption is investigated rather than quietly used.
6. Periodic samples are generated after the first checkpoint upload, then an updated snapshot is uploaded with the samples. Training RNG state is restored after monitoring. Each checkpoint may therefore have two immutable remote snapshots.
7. Local retention is bounded; remote snapshots are retained for research/recovery and are never automatically deleted. Budget Drive space for adapter **and optimizer** state at each saved step. Choose a larger `save_steps` after the smoke run if upload/storage cost is excessive, and archive/remove obsolete snapshots manually after verifying your retained recovery copies.

The script catches SIGTERM/SIGINT by asking Trainer to save and stop at the next optimizer boundary. The `session_hours` guard reserves 15 minutes within its budget for saving/samples/upload. This is best effort: a hard kill, out-of-memory error, session deletion or lost network cannot guarantee an emergency save. Recovery is bounded by the **last successfully uploaded** checkpoint and committed cache chunks. Set a shorter budget if uploads take longer than the reserve.

On graceful session cutoff, `status.json` says `paused_for_resume`. Rerun the identical command/config in a fresh session, with the same repo commit, environment and secret. The process restores the latest remote snapshot, downloads/revalidates audio if necessary, restores cache, and passes the checkpoint to `Trainer.train(resume_from_checkpoint=...)`. It retains `num_train_epochs=1`; this continues the original epoch, not a new epoch. `ignore_data_skip=False` preserves the resume position.

The cache identity includes normalized row ordering/audio hashes, model/codec revisions, token format, encoding source/package versions, objective, sequence limit and deduplication. The training identity additionally checks batch geometry, schedule, settings, key package versions and training source hashes. If you reinstall a newer stack, resume can be rejected. Restore the original package versions from `requirements-resolved.txt`, preferably the original Kaggle image as well. Do not change the identity file to bypass this check.

Do not run two separate Kaggle sessions with the same `run_id` concurrently; the local lock cannot coordinate different machines writing the same Drive pointer.

### Manual recovery and inspection

If `latest.json` is inaccessible or points to a damaged archive, use rclone to inspect `snapshots/`, download a previous immutable archive and restore it **to a separate run working directory**. Inspect its `status.json`, `resolved_config.json`, `COMPLETE.json` and checksums before resuming. Preserve the corrupt copy for investigation. Never infer resumability from `adapter_model.safetensors` alone.

For ordinary inspection:

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

These offline evaluation outputs are outside the training run and are **not automatically uploaded** by the Trainer callback. Copy the directories into your own research folder on Drive using your authorized rclone configuration or save them as Kaggle outputs.

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
| Drive preflight fails | Check notebook Secret access, OAuth expiry, remote name, Drive scope, quota and Internet. No GPU training begins before the probe succeeds. |
| ZIP download fails | Check sharing and Drive download quota. Upload the extracted paired dataset as a private Kaggle Dataset and set `data_dir` to its `/kaggle/input/...` path. |
| No manifests found | Inspect archive folder names; set explicit `data_dir` and filenames. |
| Training hours below 40 | Inspect `dataset_report.json`; you may have selected the 10h archive. Change `minimum_train_hours` only for a deliberately smaller new experiment. |
| Clip exceeds 2048 tokens | Review duration/text, then increase `max_length` for a new run/cache if it fits VRAM. Do not clip the audio while leaving the full transcript. |
| CUDA OOM | Microbatch already defaults to 1. Use a smaller rank/new run or a larger GPU. Reduce context only after reviewing the data. Avoid concurrent model loading. |
| NaN/Inf training logs | Inspect data and gradients; reduce LR in a new run. Preserve evidence; do not relabel the resumed experiment. |
| Resume identity differs | Restore the original code/config/environment or start a new `run_id`. Changing batch size during resume is intentionally rejected. |
| Runtime ends unexpectedly | Restart with the same run ID; only the last successful Drive save is guaranteed. |
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
import subprocess
from pathlib import Path
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
REPO_REF = "main"  # Prefer a fixed tag/commit before starting the real study.
REPO = Path("/kaggle/working/urdu-orpheus")
if not REPO.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
subprocess.run(["git", "checkout", REPO_REF], cwd=REPO, check=True)
```

### Cell 2 — install and preflight

```python
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO / "requirements-kaggle.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "rclone"], check=True)
subprocess.run([sys.executable, "-c", "from unsloth import FastLanguageModel; import torch; from snac import SNAC; assert torch.cuda.is_available(); assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2,6); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"], check=True)
subprocess.run(["df", "-h", "/kaggle/working"], check=True)
```

### Cell 3 — private credentials and config

```python
import os, json
from kaggle_secrets import UserSecretsClient
os.environ["RCLONE_CONFIG_B64"] = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
os.environ["ORPHEUS_DRIVE_REMOTE"] = "gdrive:orpheus_urdu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cfg = json.loads((REPO / "configs/aslp50h.json").read_text())
cfg["session_hours"] = 10.5  # CHANGE to below the runtime limit displayed in your account.
CONFIG = Path("/kaggle/working/run-config.json")
CONFIG.write_text(json.dumps(cfg, indent=2))
RUN = Path(cfg["work_dir"]) / "runs" / cfg["run_id"]
```

### Cell 4A — recommended unattended saved run

```python
# Blocks the notebook until training finishes or gracefully checkpoints for resume.
subprocess.run([sys.executable, "-u", str(REPO / "finetune_aslp_50h.py"),
                "--config", str(CONFIG)], cwd=REPO, check=True)
```

Use Kaggle **Save Version → Save & Run All** (wording may vary with the UI). The committed notebook executes the cells as a background Kaggle job; the blocking training cell keeps the job alive while training. You can close your laptop after confirming the saved run has started. The job still has Kaggle runtime/quota limits. Enable the Secret for the notebook before saving the version. Turn off an unused interactive GPU session to avoid consuming a second allocation.

Do **not** use a detached Popen cell in a committed notebook and let the notebook immediately finish: the environment may be torn down with its child process.

### Cell 4B — optional interactive detached process

Use this instead of 4A only when you want to keep the interactive notebook open for status/sample playback:

```python
LOG_PATH = Path("/kaggle/working/orpheus-background.log")
with LOG_PATH.open("a") as log:
    proc = subprocess.Popen([sys.executable, "-u", str(REPO / "finetune_aslp_50h.py"),
                             "--config", str(CONFIG)], cwd=REPO,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path("/kaggle/working/orpheus.pid").write_text(str(proc.pid))
print("PID", proc.pid, "log", LOG_PATH)
```

This detaches the process from the cell; it does not remove session inactivity/runtime limits or create an independent scheduled Kaggle job. Never launch 4A and 4B for the same run simultaneously. For a saved version containing 4B, add a blocking `proc.wait()` cell and check its return code; using 4A is simpler.

### Cell 5 — status and latest samples

```python
from IPython.display import Audio, display
status = RUN / "status.json"
print(json.loads(status.read_text()) if status.exists() else "Preparing dataset/cache")
for path in sorted((RUN / "samples").glob("step-*/*.wav"))[-3:]:
    print(path)
    display(Audio(filename=str(path)))
```

In the saved-run path this cell executes after 4A returns. During an interactive detached run you can execute it repeatedly.

### Cell 6 — test the final adapter after completion

```python
status_value = json.loads((RUN / "status.json").read_text())
if status_value["status"] in ("complete", "smoke_complete"):
    output = Path("/kaggle/working/urdu_test.wav")
    subprocess.run([sys.executable, str(REPO / "synthesize.py"),
                    "--adapter", str(RUN / "adapter_final"),
                    "--text", "آج موسم بہت خوشگوار ہے اور ہم سب باہر سیر کے لیے جا رہے ہیں۔",
                    "--output", str(output)], cwd=REPO, check=True)
    display(Audio(filename=str(output)))
else:
    print("Saved for continuation; rerun the same config to finish the epoch.")
```

For a later Kaggle session, rerun Cells 1–4A with the **same commit, run ID, config and package versions**. Automatic Drive restore handles continuation.
