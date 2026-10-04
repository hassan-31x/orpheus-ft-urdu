#!/usr/bin/env python3
"""Optional offline intelligibility evaluation; run after training releases the GPU.

Input CSV: audio,text; audio paths are relative to the CSV's directory.
This measures ASR agreement, not naturalness or emotional controllability.
"""
import argparse
import csv
import json
import re
from pathlib import Path

from orpheus_utils import atomic_json, clean_text, contained


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def normalize(text):
    # Explicit, conservative scoring policy. Never maps Urdu letters to Arabic variants.
    return clean_text(re.sub(r"[^\w\s]", "", clean_text(text)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("asr_evaluation"))
    p.add_argument("--asr-model", default="openai/whisper-large-v3")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", type=int, default=0, help="-1 CPU; 0 GPU (after training)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    import torch
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from transformers import pipeline
    from huggingface_hub import HfApi
    revision = HfApi().model_info(args.asr_model, revision=args.revision).sha
    asr = pipeline("automatic-speech-recognition", model=args.asr_model, revision=revision,
                   torch_dtype=torch.float16 if args.device >= 0 else torch.float32, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with args.manifest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            audio_path = contained(args.manifest.parent, row["audio"])
            signal, sr = sf.read(audio_path, dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            if sr != 16000:
                divisor = gcd(sr, 16000)
                signal = resample_poly(signal, 16000//divisor, sr//divisor).astype(np.float32)
            hypothesis = asr(dict(raw=signal, sampling_rate=16000),
                             generate_kwargs=dict(language="urdu", task="transcribe"))["text"]
            reference = normalize(row["text"])
            hyp = normalize(hypothesis)
            if not reference:
                raise ValueError(f"Empty normalized reference: {row['audio']}")
            char_errors = distance(reference, hyp)
            word_errors = distance(reference.split(), hyp.split())
            records.append(dict(audio=row["audio"], reference=row["text"], hypothesis=hypothesis,
                                normalized_reference=reference, normalized_hypothesis=hyp,
                                character_errors=char_errors, reference_characters=len(reference),
                                word_errors=word_errors, reference_words=len(reference.split()),
                                cer=char_errors/len(reference), wer=word_errors/len(reference.split())))
            atomic_json(args.output / "per_utterance.json", records)
    if not records:
        raise ValueError("Empty evaluation manifest")
    summary = dict(utterances=len(records), asr_model=args.asr_model, asr_revision=revision,
                   normalization="NFC; remove controls; collapse whitespace; remove punctuation; preserve letters/digits; CER includes spaces",
                   corpus_wer=sum(r["word_errors"] for r in records)/sum(r["reference_words"] for r in records),
                   corpus_cer=sum(r["character_errors"] for r in records)/sum(r["reference_characters"] for r in records),
                   mean_utterance_wer=sum(r["wer"] for r in records)/len(records),
                   mean_utterance_cer=sum(r["cer"] for r in records)/len(records))
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        boot_wer, boot_cer = [], []
        for _ in range(args.bootstrap):
            sample = [records[i] for i in rng.integers(0, len(records), size=len(records))]
            boot_wer.append(sum(r["word_errors"] for r in sample)/sum(r["reference_words"] for r in sample))
            boot_cer.append(sum(r["character_errors"] for r in sample)/sum(r["reference_characters"] for r in sample))
        summary["bootstrap"] = dict(resamples=args.bootstrap, seed=args.seed, unit="utterance",
                                    corpus_wer_95ci=np.percentile(boot_wer, [2.5, 97.5]).tolist(),
                                    corpus_cer_95ci=np.percentile(boot_cer, [2.5, 97.5]).tolist())
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
