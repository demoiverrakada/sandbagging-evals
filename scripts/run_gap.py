#!/usr/bin/env python3
"""Week 1 deliverable: measure the elicitation gap.

Runs the same benchmark under a neutral prompt and under each sandbagging prompt,
and reports the accuracy drop. A large drop = we successfully induced sandbagging,
which is the precondition for everything downstream.

Usage:
    python scripts/run_gap.py --dataset wmdp-bio --n 200
    python scripts/run_gap.py --dataset mmlu --n 200 --modes neutral blatant weak target
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# make `import sbe` work when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.mcq_eval import evaluate
from sbe.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="wmdp-bio")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["neutral", "blatant", "weak", "target"])
    ap.add_argument("--out", default=None, help="default: results/gap_<dataset>.json")
    args = ap.parse_args()
    out = args.out or f"results/gap_{args.dataset}.json"

    items = load_mcq(args.dataset, n=args.n, seed=args.seed)
    print(f"[sbe] loaded {len(items)} items from {args.dataset}")

    model, tok, device = load_model(args.model)

    results = {}
    for mode in args.modes:
        r = evaluate(model, tok, items, mode, device)
        results[mode] = r
        print(f"  {mode:10s} accuracy = {r['accuracy']:.3f}")

    # summary: gap vs neutral
    print("\n=== ELICITATION GAP (neutral - sandbag) ===")
    base = results.get("neutral", {}).get("accuracy")
    if base is not None:
        for mode in args.modes:
            if mode == "neutral":
                continue
            print(f"  {mode:10s} gap = {base - results[mode]['accuracy']:+.3f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"model": args.model, "dataset": args.dataset, "n": len(items), "results": results},
            f,
            indent=2,
        )
    print(f"\n[sbe] wrote {out}")


if __name__ == "__main__":
    main()