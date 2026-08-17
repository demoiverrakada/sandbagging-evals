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

from sklearn.metrics import roc_auc_score, roc_curve

from sbe.blackbox import consistency_scores
from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.model import load_model


def tpr_at_fpr(y, s, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="wmdp-bio")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=6)
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

    result = {"model": args.model, "dataset": args.dataset, "n": len(items),
              "perms": args.perms, "mean_inconsistency": {m: float(scores[m].mean())
                                                          for m in scores},
              "auroc": {}, "tpr@1fpr": {}}
    y_neu = np.zeros(len(items))
    for sb in ["blatant", "target"]:
        y = np.concatenate([y_neu, np.ones(len(items))])
        s = np.concatenate([scores["neutral"], scores[sb]])
        result["auroc"][sb] = float(roc_auc_score(y, s))
        result["tpr@1fpr"][sb] = tpr_at_fpr(y, s)
        print(f"  AUROC neutral-vs-{sb:8s} = {result['auroc'][sb]:.3f} "
              f"(TPR@1%FPR={result['tpr@1fpr'][sb]:.2f})")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sbe] wrote {out}")


if __name__ == "__main__":
    main()