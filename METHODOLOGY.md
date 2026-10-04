# Methodology and research notes

Research and implementation date: **5 October 2026**. This document distinguishes reported results, upstream examples, proposed engineering choices and experiments still to be run. No Urdu quality scores are invented; the Kaggle run must supply the empirical results.

## 1. Study question and scope

Does one epoch of parameter-efficient adaptation on the supplied ASLP/UrduSpeech Standard Urdu training subset improve the intelligibility and perceived naturalness of an English-pretrained Orpheus model on held-out Urdu prompts?

The eventual project is emotion-controlled Urdu TTS. The present study establishes language adaptation and a reproducible training/evaluation pipeline. It does not demonstrate arbitrary emotion control, speaker cloning or production readiness.

Every training-manifest example is used once in a completed epoch, subject to the underlying Trainer's single-GPU sampler. No ten-hour cap, oversampling, weighted resampling or silent data filtering is implemented. Validation is excluded from gradient updates; test audio is audited but not encoded for this training stage. “Entire dataset” here means the complete training partition rather than merging the evaluation partitions into training.

Report actual training duration from `dataset_report.json`, not “50h” merely because the experiment was named that way. The supplied Drive page identifies the ZIP as `Urdu_Orpheus_US_Std_Training_47h.zip`; the contents and the partition durations require runtime inspection.

## 2. Detailed review of the supplied ten-hour trial

The attached trial is a Colab-exported Python file. Its stated inputs are `stage1_train.csv` with 6,633 clips, `stage1_validation.csv` with 806 clips and a separate test CSV. The file documents one epoch over prepared 24kHz mono PCM16 clips, nominally around two to ten seconds. It contains Colab-specific mounting and notebook shell syntax, so it cannot be cloned and executed unchanged as a standard Kaggle Python program.

### Data handling

The trial applies NFC normalization, removes control characters and collapses whitespace. It requires paired `audio,text`, validates relative paths, Urdu-script presence and audio format/duration, checks repeated paths and train/validation path overlap, and avoids speaker conditioning because the existing IDs were judged unreliable. These are valuable safeguards and are retained or strengthened.

Its ZIP extraction rejects absolute paths and `..` but does not cover all archive link/Windows-path cases. The new utilities additionally reject symlinks and unsafe TAR entries and enforce path containment after resolution.

### Speech tokens

The trial loads `hubertsiuzdak/snac_24khz`, encodes each clip on GPU and maps three codebooks into seven interleaved Orpheus token bands. It builds human/text/AI/speech delimiters correctly, uses text-plus-audio causal labels, and rejects sequences exceeding 2048 tokens rather than silently truncating them.

It follows the upstream notebook's coarse-frame deduplication: if the first/coarsest code repeats, it removes the next whole seven-token frame even when finer codes differ. This is **not** an exact duplicate check on all seven codes. It changes timing and can remove acoustic detail. The new default retains every frame; `dedup="coarse"` is available as a documented ablation. This change is an engineering hypothesis about preserving audio, not proof that it outperforms the upstream recipe for Urdu.

### Caching

The trial commits 100-row Parquet chunks in Drive, giving a useful preprocessing restart point. It considers a chunk reusable if its row count matches, which cannot detect changed transcripts, audio content, tokenizer revision, objective or encoding policy. The new cache fingerprints those inputs and verifies per-chunk hashes and row identity.

### Optimization and saving

The trial uses the English pretrained base, 4-bit loading, LoRA rank 16/alpha 32 on seven attention/MLP projections, dropout zero, Unsloth gradient checkpointing, microbatch 1, accumulation 8, learning rate `2e-4`, cosine schedule, 3% warmup, 8-bit AdamW, seed 3407 and a full epoch. It saves every 250 optimizer steps and retains three local/Drive-mounted checkpoints. It uses `get_last_checkpoint` for resume.

The validation dataset is built but not passed to Trainer during training. `eval_strategy="no"` means no periodic held-out loss is recorded; validation occurs only after training. Its final adapter is useful for inference but must not be confused with optimizer/scheduler/RNG state needed for exact continuation.

### Inference

The trial uses the same prompt delimiters, samples at temperature 0.6/top-p 0.95/repetition penalty 1.1, stops at the speech delimiter, reverses seven-token codebook offsets and decodes SNAC into 24kHz audio. The new implementation keeps this structure, writes raw generated IDs and explicitly reports invalid frames, partial frames and missing stop delimiters.

## 3. Detailed review of the Bangla paper

Supplied paper: Utshob Sutradhar, Naimul Hasan Nahid, Elahi Md Toufiq and Priyankar Biswas, **“Human-like Bangla Text-to-Speech Synthesis using the Orpheus TTS Model Fine-tuning”**, ICECTE 2026, DOI [10.1109/ICECTE69292.2026.11429426](https://doi.org/10.1109/ICECTE69292.2026.11429426). All five pages were read, including the methodology, result discussion, references and extracted evaluation figures.

### What is reported

| Aspect | Reported detail |
|---|---|
| Source corpus | [OpenSLR 37](https://www.openslr.org/37/), with processed data on `utshobs/bangla-tts` |
| Speakers | Six native male Bangladeshi speakers |
| Paired clips | 1,891 |
| Duration | 2h 56m 18s |
| Original audio | 48kHz, mono, 16-bit WAV |
| Training sample rate | Resampled to 24kHz |
| Environment | Google Colab, Unsloth; T4 and A100-SXM4-40GB explored |
| Main reported final schedule | 15 epochs, 7,095 optimizer steps |
| Trainable parameters | 97,255,424, described as 2.86% of approximately 3.39B |
| Reported final run time | 153.25 minutes on A100 |
| Training loss | From 5.667 down to 0.045 |
| Reported mean CER / WER | 5.8% / 11.1% |
| Reported speaker similarity | 99.96% |

The paper's workflow is paired text/audio preparation → speech/text tokenization → efficient adaptation → checkpoints and loss visualization → generated speech → linguistic and acoustic evaluation. It describes repeated listening and testing after successful tuning, and TensorBoard monitoring. These are useful methodological elements to adopt.

### What is not reproducibly specified

The paper does not provide a complete run configuration: the exact learning rate, optimizer settings, LoRA rank/alpha, dropout, quantization, scheduler, accumulation, random seed, split protocol and exact base checkpoint/revision are not sufficiently specified. Its narrative refers to a multilingual base, while the architecture section identifies Llama 3B. Do not infer a particular downloadable checkpoint merely from that description.

The upstream Unsloth notebook uses rank 64 and reports exactly 97,255,424 trainable parameters. The matching count suggests a related adaptation configuration, but **does not prove** the paper's precise settings. The experiment should measure its own parameter counts and identify its actual base revision.

The ratio `7,095 / 15 = 473` updates per epoch is consistent with an effective batch near four for 1,891 clips, but rounding, Trainer version and accumulation behavior can affect this inference. It is not a reported batch setting.

### Evaluation limitations

Figures 6 and 7 explicitly show **two** generated sentences. Figure 6 shows CER values 0 and 0.117, whose mean is approximately 0.058; the surrounding prose says 11.1% for the second CER, an internal inconsistency. Figure 7 shows WER values 0 and 0.222, mean 0.111. These figures do not establish a corpus-wide generalization score or support confidence intervals.

Figure 5 prints “SNAC Speaker Similarity Scores” and a single 0.9996 value, but the paper does not specify a reproducible speaker embedding extractor, normalization, comparison protocol or test-set aggregation. SNAC is a speech codec, not by itself a validated speaker-verification measure. Do not equate code/spectral similarity with speaker identity accuracy.

The mel-cepstral/spectral figure is a qualitative visual comparison. It is not a reported aligned corpus-level mel-cepstral distortion statistic. A low training loss, similar-looking spectrograms or two successful sentences do not by themselves demonstrate human-level naturalness.

The paper does not disclose enough information about held-out partitioning, human MOS testing or common evaluation conditions for its cross-model comparisons to be replicated exactly. Report its numbers as the authors' claims; do not use them as expected Urdu results or as a stopping threshold.

The methodology prose also describes transformer/diffusion and encoder/decoder adjustment. The [upstream Orpheus implementation](https://github.com/canopyai/Orpheus-TTS) identifies a Llama-based autoregressive speech-token model. The pipeline implemented here follows that code rather than treating the paper's architectural shorthand as an executable specification.

### What this Urdu study adopts

Adopt paired multilingual adaptation, 24kHz codec input, parameter-efficient tuning, GPU-efficient execution, loss tracking, repeated auditory checks, intermediate checkpoints, and ASR-based intelligibility evaluation with stronger sample coverage.

Do not copy the 15-epoch schedule: the user requested one epoch on a much larger training corpus. Do not promise the paper's loss, speaker score, WER or time on a Kaggle T4.

## 4. Upstream evidence and dataset context

Primary resources consulted:

1. [Unsloth TTS fine-tuning guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning): paired transcripts and actual speech tokens are essential; 24kHz audio is used for Orpheus; its examples prefer 16-bit LoRA when memory allows. Some explanatory examples on this page are simplified, so the executable notebook is the token-format reference.
2. [Unsloth Orpheus notebook](https://github.com/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb): exact delimiter IDs, SNAC interleaving, causal labels, LoRA, padding and inference. The inspected notebook uses the English **fine-tuned** base, rank 64/alpha 64, 16-bit loading, microbatch 1/accumulation 4 and a demonstration step limit. This project intentionally preserves the trial's **pretrained** base and uses one full epoch.
3. [Orpheus upstream repository](https://github.com/canopyai/Orpheus-TTS): Llama 3B speech-token architecture, pretrained versus production voices, prompting and expression tags. Its suggested repetition penalty is at least 1.1 for stable generation.
4. [Orpheus fine-tuning config](https://github.com/canopyai/Orpheus-TTS/blob/main/finetune/config.yaml) and [LoRA implementation](https://github.com/canopyai/Orpheus-TTS/blob/main/finetune/lora.py): a one-epoch configuration at `5e-5`; a rank-32/alpha-64 rsLoRA example with optional embedding/head training. Those choices come from upstream, not from the Bangla paper.
5. [SNAC implementation](https://github.com/hubertsiuzdak/snac): encoder/decoder, hierarchical codebooks, preprocessing and model loading. Decoder noise and codec padding are relevant to reproducibility and waveform duration.
6. [ASLP-lab UrduSpeech dataset card](https://huggingface.co/datasets/ASLP-lab/UrduSpeech) and [UrduSpeech paper](https://arxiv.org/abs/2605.17846): 156h released corpus across Standard Urdu, Urdu-English code switching and Pakistani-accented English; in-the-wild media content, transcripts and paralinguistic metadata. The Standard Urdu subset is reported as 59.2h. Our supplied prepared archive is a different subset/representation and must be measured separately.
7. [Transformers 4.57.6 Trainer documentation](https://huggingface.co/docs/transformers/v4.57.6/en/main_classes/trainer): sampler, accumulation, checkpointing and resume semantics.
8. [Hugging Face Hub upload/commit APIs](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/upload), [download APIs](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/download) and [token permissions](https://huggingface.co/docs/hub/security-tokens): authenticated private artifact storage with synchronous commits. The project bounds `huggingface_hub` to `>=0.34,<1` for the Transformers 4.x stack, so the corresponding versioned documentation is the API reference.
9. [rclone Google Drive documentation](https://rclone.org/drive/) and [copyto](https://rclone.org/commands/rclone_copyto/): optional personal Drive storage, retries and file transfers.
10. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks): notebook setup and GPU execution. Limits depend on the account/runtime; check the live UI rather than relying on a fixed historical limit.

The UrduSpeech corpus includes emotion-related metadata, but broad model-generated paralinguistic descriptions require verification before use as TTS emotion supervision. The project should not assume the prepared CSVs preserve those annotations or that a neutral-language run has learned explicit control.

The UrduSpeech paper itself reports difficulties with Whisper on code switching and Urdu script. This supports treating ASR-based TTS evaluation as an imperfect proxy and calibrating it on held-out human audio.

## 5. Exact input representation

For each clip:

```text
start_human
  tokenizer.encode(Urdu text, add_special_tokens=True)
  end_text
end_human
start_ai
start_speech
  seven interleaved audio tokens per SNAC frame
end_speech
end_ai
```

| Symbol | ID |
|---|---:|
| End text | 128009 |
| Start / end speech | 128257 / 128258 |
| Start / end human | 128259 / 128260 |
| Start / end AI | 128261 / 128262 |
| Padding | 128263 |
| Audio offset | 128266 |

For codec frame index `i`, the seven positions are:

```text
c0[i]       + 128266 + 0*4096
c1[2i]      + 128266 + 1*4096
c2[4i]      + 128266 + 2*4096
c2[4i+1]    + 128266 + 3*4096
c1[2i+1]    + 128266 + 4*4096
c2[4i+2]    + 128266 + 5*4096
c2[4i+3]    + 128266 + 6*4096
```

Each code must be within `[0,4095]`. The script checks codebook lengths and vocabulary bounds. The CPU regression test reverses this interleaving and compares every original code, guarding against a plausible-looking but incorrect acoustic token layout.

Text normalization preserves Urdu letter choices, diacritics, digits and punctuation. It does not transliterate, map Urdu characters to Arabic alternatives, expand numerals, add phonemes or prepend unverified emotion/speaker tags. Those changes could require paired-data review and separate ablations.

No augmentation, denoising, silence trimming or amplitude normalization is applied to the already prepared clips. Pitch/speed modifications change prosody and could create an unintended emotion/voice target. Signal statistics are recorded for review rather than used as unvalidated automatic exclusion thresholds.

The default uses all-token causal cross entropy, matching the trial and executable Unsloth representation. Right padding uses the Orpheus pad token; padding labels are `-100`. An optional audio objective masks the human/text context while learning `start_speech`, audio and stop delimiters. It is a separate hypothesis, not an automatically superior replacement.

## 6. Optimization rationale and controls

**Initial configuration:** rank 32, alpha 64, standard LoRA (not rsLoRA), zero LoRA dropout, seven attention/MLP projection targets, 4-bit base loading, microbatch 1, eight accumulated microbatches, learning rate `1e-4`, 3% warmup, cosine decay, weight decay `0.001`, max gradient norm 1.0, seed 3407, one epoch.

These defaults balance a feasible single-T4 run against more adapter capacity than the ten-hour rank-16 trial. The learning rate sits between the upstream fine-tuning configuration and the demonstration notebook. This is a defensible starting point, **not a validated optimum**.

Unsloth gradient checkpointing reduces activation storage. 8-bit optimizer state reduces memory. Dynamic right padding rounds batch width to a multiple of eight; microbatch one already limits padding waste. The SNAC model is released before loading the language model. Parquet/Arrow chunks keep encoding resumable and training I/O local. Validation stores only losses, not huge vocabulary logits. Monitoring decodes on CPU to avoid GPU contention with the training model.

The script deliberately avoids sequence packing: each audio/text pair remains a separate supervised sequence, with no accidental attention between unrelated utterances. It also avoids full-model updates and embedding/head training on the initial Kaggle budget. Training those modules adds memory and changes adapter behavior and requires its own experiment.

The one-epoch duration means total optimizer updates depend on clip count and batch geometry. Record Trainer's actual `global_step` and `epoch`; do not infer exact step count from nominal hours. A `max_steps` smoke override takes precedence over the epoch schedule and must be identified as a smoke run.

### Sampling and batch size

`sampling="random"` uses Trainer's seeded random sampling without replacement. `sampling="length"` uses Trainer's seeded length grouping; it reduces padding waste when microbatch exceeds one. Length grouping is not oversampling, and it should preserve full training coverage. Do not claim it stratifies speaker, content domain or emotion: the script implements none of those samplers.

Keep batch size constant within a run. Randomly changing batch geometry can confound gradient scale, LR schedule and resume ordering; randomization of example order is the meaningful initial control. A study of variable batch schedules would need its own reproducible sampler, weighting scheme and resume state.

### Proposed ablation sequence

Run comparisons with the same corpus, held-out prompts, decoding settings and one-epoch budget. Use a unique `run_id` for every row. Prioritize a few meaningful runs within GPU allowance rather than starting an unaffordable Cartesian sweep.

| Question | Suggested variants | Control |
|---|---|---|
| Learning rate | `5e-5`, `1e-4`, `2e-4` | r32, effective batch 8 |
| Adapter capacity | r16/a32, r32/a64, r64/a128 | LR `1e-4`; alpha/r ratio fixed |
| Loading precision | 4-bit versus 16-bit LoRA | Same rank, effective batch, data/order where feasible |
| Batch geometry | 1×8 versus 2×4 versus 4×2 | Effective batch 8; check OOM first |
| Optimization batch size | effective 4, 8, 16 | Microbatch fixed; report changed steps/schedule |
| Order | random versus length-grouped | Especially relevant at microbatch ≥2 |
| Loss target | all-token versus audio-target-only | Compare intelligibility/MOS; loss magnitudes differ |
| Frame policy | none versus coarse-frame removal | Measure timing and audible artifacts, not just loss |
| Rank stabilization | standard LoRA versus rsLoRA | Account for changed scaling; do not assume same alpha is equivalent |
| Base checkpoint | English pretrained versus English production-finetuned or a justified multilingual base | Separate identities, same evaluation; verify token format |
| Reproducibility | seeds 3407, 42, 1234 | Repeat shortlisted configs; report variance |
| Generation sampling | temperatures 0.4/0.6/0.8; top-p 0.9/0.95 | Same trained adapter and prompts; report speed/stop failures |

Changing `precision`, rank, LR, batch size, ordering or objective under an existing checkpoint ID is intentionally rejected. A new configuration is a new optimization trajectory. Warm-starting a new study from an existing adapter is different from exactly resuming that study; the current pipeline provides exact-config checkpoint resume, not an adapter warm-start experiment API.

## 7. Persistence and reproducibility

Cache identity covers every normalized row in order, audio byte hashes, token IDs, tokenizer vocabulary hash, base/codec revisions, encoding code/package versions, objective, maximum context and frame policy. Key library versions and actual source hashes are also included in the training identity.

Checkpoints store full Trainer optimization state, not only inference weights. A completion marker hashes all checkpoint files. **Private Hugging Face storage is the default**; Google Drive/rclone and explicit local-only storage are alternatives. A lost runtime restores the latest committed snapshot, verifies it, then lets Trainer recover model/optimizer/scheduler/RNG and skip previously consumed data. Local checkpoints are retained in a bounded set; remote history is retained independently.

For Hugging Face, artifacts live in a private **model** repository under `runs/<run_id>/`. Token-cache chunks and preparation metadata are committed as they are completed. A single synchronous `create_commit` publishes a content-named snapshot archive and `latest.json` together, so readers cannot see a pointer to an uncommitted checkpoint. The archive carries training state, logs, configuration, source code and samples; token caches are stored separately. This repository layout is an artifact archive, not a directly loadable root-level PEFT model. Restore the snapshot before inference.

The Hub backend takes `HF_TOKEN` from the process environment and the repository ID from the config/CLI or `ORPHEUS_HF_REPO`. It checks private visibility and performs a read/write probe before audio processing. Missing repositories can be created privately if the token allows it; an existing public repository is rejected. Credentials are excluded from configuration and snapshot serialization. Transient transfer errors receive up to three attempts, while authentication/permission errors stop immediately. Failed commits leave the previous committed pointer intact.

For Drive, the same verified archive format is uploaded with rclone; its pointer is written only after the immutable archive transfer succeeds. The shared storage interface lets training/cache logic use either backend without changing the token representation or optimization schedule. Backend/repository fields are recorded as infrastructure settings but excluded from the mathematical training identity. Moving a run between stores still requires transferring its artifacts, and earlier project commits still require their original source/environment hashes for continuation.

Storage allowance and commit history are part of the experiment's operational budget. Remote retention is not bounded by Trainer's local `save_total_limit`, and the script never deletes remote history automatically. A `--mode storage-check` command tests authorization and read/write access without downloading the corpus or allocating a GPU model.

This gives resumability with explicit conditions: unchanged data/order, compatible software, available base weights and successful authenticated Hub/Drive transfers. GPU kernels and floating-point reductions can still differ across hardware/software. This study is **seeded and auditable**, not claimed to be bitwise identical on every accelerator. Avoid switching GPUs mid-comparison where practical, or document the change.

The Kaggle image contained incompatible requirements among unrelated preinstalled packages (Colab, Gradio, MoviePy, cloud filesystem drivers and others). A global `pip check` therefore stopped before training. Setup now uses a dedicated [venv with inherited system packages](https://docs.python.org/3/library/venv.html), allowing reuse of CUDA wheels while project installs shadow host packages without changing the notebook kernel. Torch/torchvision/torchaudio/Triton/xformers versions are constrained to the image's installed values. No GCS/S3 driver is installed for this local Parquet/WAV workflow.

`check_environment.py` walks the active requirement graph from the project roots, checking missing distributions, transitive version bounds and extras under the current Python/platform markers. It excludes unrelated distributions, but fails on conflicts in the training graph. CPU smoke tests verify Urdu Parquet and audio IO; the GPU check imports the training stack and executes xformers attention forward/backward. Model loading, codec decoding and checkpoint restore still require the separate end-to-end smoke experiment. Installation logs, pip's installation report, CUDA constraints, resolved versions and CPU/GPU reports are retained under `orpheus-env-setup`. The fixed Datasets 3.6.0/fsspec 2025.3.0 combination preserves the prior dataset API; shared GCS/S3 incompatibilities do not belong to the active graph. These checks have not been executed on a Kaggle GPU locally.

CPU regression tests cover full checkpoint roundtrips, atomic Hub snapshot/pointer publication, failed commits preserving the old pointer, corrupt archive rejection, interrupted-preparation/cache restoration, private visibility checks, retries and backend selection. Hub tests use an in-memory API double; they do not establish live authentication, storage capacity or GPU training compatibility. Run the documented Kaggle storage check and separate smoke experiment before the complete epoch.

Training updates since the last uploaded checkpoint may be lost after a hard kill. Graceful signal/session stopping reduces that risk but cannot make the runtime or network infallible. Checkpoint cadence should be tuned based on measured save/upload time and storage budget.

## 8. Evaluation plan stronger than the reference paper

### During training

Log training loss, learning rate, available gradient norm, epoch/step, timing and GPU/RAM use every ten optimizer steps. Evaluate a fixed seeded validation subset every 100 steps and the whole validation partition after the completed schedule. Save fixed-prompt Urdu samples at initialization and after checkpoints. Listen for missing words, incorrect letters/phonemes, repeated syllables, unstable voice, clipped endings, unnatural timing and background noise reproduction.

Monitoring restores RNG states and training mode after generation. Generated token sequences and stop/invalid-frame information are retained so a broken decoder cannot masquerade as a linguistic quality issue. Monitoring failures are logged; a previously saved complete checkpoint remains recoverable.

### After training

1. Freeze at least 100 manually checked held-out Urdu prompts, ideally several hundred covering short/long utterances, question/statement intonation, Urdu digits, rare words and varied phonetic patterns. Include additional unseen-authored prompts and label them separately from corpus-held-out prompts.
2. Generate the same prompts from the untuned base, the final Urdu adapter and the earlier ten-hour adapter if you retained its compatible model identity. Use the same temperature, top-p, repetition penalty and per-prompt seeds.
3. Transcribe generated audio with a fixed Urdu-capable ASR revision. Save raw hypotheses, score normalization and counts. Report corpus WER/CER and mean-utterance scores, not just the mean of two convenient successes.
4. Run the same ASR on held-out human speech to quantify evaluator error. Inspect script errors and code-switching separately. Do not translate the reference to make the score look lower.
5. Use the optional evaluator's utterance bootstrap intervals as an initial uncertainty estimate. If related segments share a source/speaker, utterance independence is optimistic: group/source bootstrap and split-level analysis should replace it in the final paper when those identities are available.
6. Conduct blinded native-Urdu listening with randomized system order and a predefined rubric. Rate naturalness, pronunciation/intelligibility and prosody separately on a 1–5 scale; retain rater IDs, prompt IDs, system condition and rating records. Report listener/sample counts and confidence intervals. Human MOS is measured from ratings, never inferred from training loss.
7. Report generation runtime/real-time factor separately from training runtime and include stop-token failures, invalid-frame rates and budget-limited utterances. Save failed examples; do not quietly remove them from the evaluation denominator.

For a speaker-similarity study, first establish genuine speaker identities, reference utterances and a validated speaker embedding model/protocol for Urdu. The present unconditioned mixture cannot reasonably claim deterministic voice identity. No speaker score is implemented using SNAC code overlap.

For spectral comparisons, compare matched prompts, align generated/reference audio appropriately and report the exact feature extraction and DTW protocol. A visual mel plot is descriptive, not a validated naturalness measurement. These optional speaker/spectral methods are proposed study additions; they are not included as automatic scores in this repository.

## 9. Emotion stage prerequisites and future design

Keep the Urdu adapter and its base identity as the stage-1 artifact. Before stage 2, audit the Urdu SER dataset for transcripts, usable audio quality, globally meaningful speaker IDs, label taxonomy, balanced class coverage and source-disjoint partitions. Some SER datasets provide emotion labels only or repeated elicited phrases; obtain/check transcripts and ensure the content diversity is suitable for TTS.

An eventual prompt might use a documented prefix such as `[emotion=happy]` followed by Urdu text, but that exact prefix must appear in aligned stage-2 training examples and in inference. Adding a new special tokenizer token also requires updating/trainable embeddings and vocabulary checks; the current stage-1 code does not do this. Existing Orpheus expressions such as `<sigh>` and `<laugh>` represent specific expressions, not a universal emotion taxonomy.

Once reliable emotion supervision exists, define the taxonomy in Urdu/English, map source labels consistently and evaluate controllability on held-out text across emotions. Distinguish a requested label from an audible change; native listeners and a separately calibrated SER model can assess emotion realization. Compare pronunciation retention before and after stage 2, and consider a documented mixture of neutral and emotion-labeled Urdu to reduce forgetting. This is future work, not an implemented training capability in this stage.

## 10. Paper reporting checklist and citation material

Report the measured dataset hours/counts, corpus provenance/version, split construction/leakage limitations, audio audit, text normalization, codebook representation, exact base/codec commits, adapter targets/rank/scaling, loading precision, actual trainable parameter count, batch geometry, LR schedule, one-epoch updates, seeds, hardware/software, total session/training/encoding/upload time, checkpoint strategy, decoding settings and evaluation failures.

Provide loss curves and held-out metrics with actual listening outcomes. State whether a result is from a complete epoch, a smoke run, or a paused intermediate checkpoint. Explain ASR limitations and avoid comparing unmatched datasets/decoding protocols as an architectural improvement.

Useful citation starters; verify final formatting against the venue's requirements:

```bibtex
@inproceedings{sutradhar2026banglaorpheus,
  title={Human-like Bangla Text-to-Speech Synthesis using the Orpheus TTS Model Fine-tuning},
  author={Sutradhar, Utshob and Nahid, Naimul Hasan and Toufiq, Elahi Md and Biswas, Priyankar},
  booktitle={2026 5th International Conference on Electrical, Computer \& Telecommunication Engineering (ICECTE)},
  year={2026},
  doi={10.1109/ICECTE69292.2026.11429426}
}

@misc{haq2026urduspeech,
  title={UrduSpeech: A 156-Hour Urdu Speech Corpus with 12-Dimension Paralinguistic Annotations},
  author={Haq, Attia Nafees ul and Zhu, Zeyu and Hu, Jingbin and He, ChunJiang and Xie, Lei},
  year={2026},
  eprint={2605.17846},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2605.17846}
}
```

Also cite Orpheus, Unsloth, SNAC, LoRA/QLoRA and the ASR evaluator using their original papers or repository citation metadata. Dataset licenses and original corpus attributions apply independently of this training code.

The saved Kaggle notebook always executes installation before GPU imports, since a committed job cannot rely on packages installed in a prior interactive session. All training/storage/inference subprocesses use the generated venv interpreter. Repository checkout fetches the requested ref even when the clone exists; record the printed commit and pin it for resume. A stdlib venv failure falls back to virtualenv in a separate bootstrap directory.
