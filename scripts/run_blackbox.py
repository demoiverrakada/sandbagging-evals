#!/usr/bin/env python3
"""Week 2.5: black-box self-consistency detector.

Higher inconsistency -> more likely sandbagging. We report AUROC separating neutral
from each sandbagging strategy, to compare against the probe and noise detectors.

Usage:
    python scripts/run_blackbox.py --dataset wmdp-bio --n 100 --perms 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbe.blackbox import consistency_scores
from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.metrics import detection_summary
from sbe.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="wmdp-bio")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=6)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/blackbox_{args.dataset}.json"

    items = load_mcq(args.dataset, n=args.n, seed=args.seed)
    print(f"[sbe] loaded {len(items)} items from {args.dataset}")
    model, tok, device = load_model(args.model)

    scores = {}
    for mode in ["neutral", "blatant", "target"]:
        scores[mode] = consistency_scores(model, tok, items, mode, device,
                                           n_perms=args.perms, seed=args.seed)
        print(f"  {mode:8s} mean inconsistency = {scores[mode].mean():.3f}")

    result = {
        "analysis_status": "exploratory_pilot",
        "model": args.model,
        "dataset": args.dataset,
        "n": len(items),
        "perms": args.perms,
        "mean_inconsistency": {m: float(scores[m].mean()) for m in scores},
        "metrics": {},
    }
    for sb in ["blatant", "target"]:
        metrics = detection_summary(
            scores["neutral"],
            scores[sb],
            target_fpr=0.01,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        result["metrics"][sb] = metrics
        lo, hi = metrics["auroc_paired_bootstrap_95_ci"]
        print(
            f"  AUROC neutral-vs-{sb:8s} = {metrics['auroc']:.3f} "
            f"(paired-bootstrap 95% CI {lo:.3f}-{hi:.3f}; "
            f"empirical TPR@1%FPR={metrics['empirical_tpr_at_target_fpr']:.2f})"
        )
        if not metrics["sample_size_supports_target_fpr_at_zero_fp"]:
            print(
                "    LOW-FPR WARNING: even zero false positives would leave a "
                f"one-sided 95% FPR upper bound of "
                f"{metrics['zero_false_positive_rate_one_sided_95_upper']:.3f}."
            )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sbe] wrote {out}")


if __name__ == "__main__":
    main()
