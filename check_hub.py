#!/usr/bin/env python3
"""Seconds-long Hugging Face checkpoint-storage check; needs only huggingface_hub.

Runs the same preflight as training (token scope, private repo, probe upload and
read-back) and prints redacted diagnostics. Use it before the slow Kaggle install,
or on any machine:  HF_TOKEN=... python check_hub.py --repo USER/REPO
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from orpheus_utils import HubOperationError, HuggingFaceStore


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="private model repo USER/REPO")
    p.add_argument("--run-id", help="defaults to run_id in configs/aslp50h.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = args.run_id or json.loads((Path(__file__).parent / "configs/aslp50h.json").read_text())["run_id"]
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    try:
        store = HuggingFaceStore(args.repo, run_id, token)
    except ImportError as exc:
        print(f"huggingface_hub unavailable in this Python: {exc}", file=sys.stderr)
        sys.exit(3)
    try:
        store.preflight()
    except HubOperationError as exc:
        print("\nHUB CHECK FAILED\n" + json.dumps(exc.diagnostics, indent=2, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
    except (RuntimeError, ValueError) as exc:
        print(f"\nHUB CHECK FAILED\n{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"HUB CHECK PASSED: wrote and read back runs/{run_id}/connection_probe.json in {args.repo}")


if __name__ == "__main__":
    main()
