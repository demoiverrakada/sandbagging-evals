#!/usr/bin/env python3
"""Week 2: noise-injection sweep.

For each noise level sigma (and a few seeds), we perturb the weights ONCE and then
evaluate every prompt mode under that same perturbed model. Honest models should
degrade monotonically; sandbaggers should show an anomalous accuracy rise.

Usage:
    # quick sanity check
    python scripts/run_noise.py --dataset wmdp-bio --n 50 --seeds 1 --sigmas 0 0.02 0.05
    # the real sweep
    python scripts/run_noise.py --dataset wmdp-bio --n 100 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, pstdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.mcq_eval import evaluate
from sbe.model import load_model
from sbe.noise import apply_noise, select_noise_params, snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="wmdp-bio")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="number of noise draws to average")
    ap.add_argument("--modes", nargs="+", default=["neutral", "blatant", "target"])
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/noise_{args.dataset}.json"

    items = load_mcq(args.dataset, n=args.n, seed=args.seed)
    print(f"[sbe] loaded {len(items)} items from {args.dataset}")
    model, tok, device = load_model(args.model)

    names = select_noise_params(model)
    print(f"[sbe] noising {len(names)} weight matrices")
    snap = snapshot(model, names)

    # curve[mode][sigma] = list of accuracies over seeds
    curve = {m: {s: [] for s in args.sigmas} for m in args.modes}
    for sigma in args.sigmas:
        for k in range(args.seeds):
            apply_noise(model, snap, sigma, seed=1000 * k + 1, device=device)
            for mode in args.modes:
                r = evaluate(model, tok, items, mode, device, show_progress=False)
                curve[mode][sigma].append(r["accuracy"])
        line = "  ".join(f"{m}={mean(curve[m][sigma]):.3f}" for m in args.modes)
        print(f"  sigma={sigma:<6} {line}")

    # restore clean weights
    apply_noise(model, snap, 0.0, seed=0, device=device)

    summary = {
        "model": args.model, "dataset": args.dataset, "n": len(items),
        "sigmas": args.sigmas, "modes": args.modes, "seeds": args.seeds,
        "curve": {m: {str(s): {"mean": mean(v), "std": pstdev(v) if len(v) > 1 else 0.0,
                               "raw": v}
                      for s, v in curve[m].items()} for m in args.modes},
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sbe] wrote {out}")


if __name__ == "__main__":
    main()