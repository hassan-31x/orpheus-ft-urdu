#!/usr/bin/env python3
"""Generate Urdu speech from a final adapter or a saved intermediate checkpoint."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

from orpheus_utils import CODEC_ID, SPECIAL, atomic_json, decode_frames, prompt_ids, read_json


def generate_audio(model, tokenizer, text, output, *, speaker=None, seed=3407,
                   max_length=2048, max_new_tokens=1400, temperature=0.6,
                   top_p=0.95, repetition_penalty=1.1, codec_revision="main", max_time=None):
    import torch
    import soundfile as sf
    from snac import SNAC
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = prompt_ids(tokenizer, text, speaker)
    budget = min(max_new_tokens, max_length-len(prefix))
    if budget < 8:
        raise ValueError("Text leaves no usable audio-token context; use shorter text")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
    started = time.monotonic()
    with torch.inference_mode():
        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=budget, do_sample=True, temperature=temperature,
                                top_p=top_p, repetition_penalty=repetition_penalty,
                                eos_token_id=SPECIAL["end_speech"], pad_token_id=SPECIAL["pad"],
                                use_cache=True, **({"max_time": max_time} if max_time else {}))
    ids = result[0, len(prefix):].cpu().tolist()
    atomic_json(output.with_suffix(".tokens.json"), ids)
    codes, status = decode_frames(ids)
    # CPU decoder avoids stealing training GPU memory during periodic samples.
    decoder = SNAC.from_pretrained(CODEC_ID, revision=codec_revision).eval().cpu()
    with torch.inference_mode():
        signal = decoder.decode([torch.tensor([c], dtype=torch.long) for c in codes]).squeeze().cpu().numpy()
    sf.write(output, signal, 24000, subtype="PCM_16")
    metadata = dict(text=text, speaker=speaker, seed=seed, sample_rate=24000,
                    duration_seconds=len(signal)/24000, elapsed_seconds=time.monotonic()-started,
                    decoding=dict(temperature=temperature, top_p=top_p,
                                  repetition_penalty=repetition_penalty, token_budget=budget),
                    reached_token_budget=len(ids) >= budget, **status)
    atomic_json(output.with_suffix(".json"), metadata)
    del decoder, inputs, result
    gc.collect()
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--text", default="آج موسم بہت خوشگوار ہے اور ہم سب باہر سیر کے لیے جا رہے ہیں۔")
    p.add_argument("--output", type=Path, default=Path("urdu_test.wav"))
    p.add_argument("--manifest", type=Path, help="Batch generation CSV with text and optional speaker; --output becomes a directory")
    p.add_argument("--speaker", help="Only for an adapter trained with a trusted speaker column")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--max-new-tokens", type=int, default=1400)
    p.add_argument("--base-only", action="store_true", help="Use run identity for an untuned baseline; do not load adapter")
    args = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from unsloth import FastLanguageModel
    from peft import PeftModel
    identity = read_json(args.adapter / "run_identity.json")
    cfg = identity["config"]
    if not args.manifest and bool(args.speaker) != bool(cfg["speaker_column"]):
        p.error("Speaker prompting must match training: provide --speaker only for a speaker-conditioned adapter")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 8:
        p.error("Invalid generation parameters")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"], use_exact_model_name=True,
        max_seq_length=cfg["max_length"], dtype=None, load_in_4bit=cfg["precision"] == "4bit")
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    rows = [dict(text=args.text, speaker=args.speaker)]
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(not r.get("text") for r in rows):
            p.error("Batch manifest must contain nonempty text rows")
        args.output.mkdir(parents=True, exist_ok=True)
    generated = []
    for i, row in enumerate(rows):
        speaker = row.get("speaker") or args.speaker
        if bool(speaker) != bool(cfg["speaker_column"]):
            p.error(f"Row {i}: speaker conditioning does not match training")
        dest = args.output / f"utterance-{i:05d}.wav" if args.manifest else args.output
        metadata = generate_audio(model, tokenizer, row["text"], dest, speaker=speaker,
                                  seed=args.seed+i, max_length=cfg["max_length"],
                                  max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                  top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                  codec_revision=cfg["codec_revision"])
        generated.append(dict(audio=dest.name, text=row["text"], speaker=speaker or ""))
        print(f"Saved {dest}: {metadata['duration_seconds']:.2f}s; stop={metadata['ended_with_speech_stop']}; invalid_frame={metadata['invalid_frame_token_offset']}")
        if args.manifest:
            with (args.output / "generated.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, ["audio", "text", "speaker"])
                w.writeheader()
                w.writerows(generated)


if __name__ == "__main__":
    main()
