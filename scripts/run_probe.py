#!/usr/bin/env python3
"""Week 3: linear activation probe for sandbagging.

Caches last-token residuals under neutral / blatant / target, then per layer trains
a logistic-regression probe to separate honest (neutral) from sandbagging (blatant),
and reports:
  - in-distribution AUROC (test on held-out blatant vs neutral)
  - cross-strategy AUROC (SAME probe, tested on target vs neutral)  <- the H2 test

Usage:
    python scripts/run_probe.py --dataset wmdp-bio --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sbe.activations import cache_last_token_residuals
from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.model import load_model


def tpr_at_fpr(y, s, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def fit_probe(Xtr, ytr):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default="wmdp-bio")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/probe_{args.dataset}.json"

    items = load_mcq(args.dataset, n=args.n, seed=args.seed)
    print(f"[sbe] loaded {len(items)} items from {args.dataset}")
    model, tok, device = load_model(args.model)

    X = {}
    for mode in ["neutral", "blatant", "target"]:
        feats, _ = cache_last_token_residuals(model, tok, items, mode, device)
        X[mode] = feats                                  # (n, L+1, d)
    n, n_layers, _ = X["neutral"].shape

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    cut = int(args.train_frac * n)
    tr, te = idx[:cut], idx[cut:]

    in_dist, cross = [], []
    for layer in range(n_layers):
        # train: neutral(0) vs blatant(1) on train split
        Xtr = np.concatenate([X["neutral"][tr, layer], X["blatant"][tr, layer]])
        ytr = np.array([0] * len(tr) + [1] * len(tr))
        clf = fit_probe(Xtr, ytr)

        # in-distribution test: neutral vs blatant on test split
        Xte = np.concatenate([X["neutral"][te, layer], X["blatant"][te, layer]])
        yte = np.array([0] * len(te) + [1] * len(te))
        s = clf.decision_function(Xte)
        in_dist.append({"layer": layer, "auroc": float(roc_auc_score(yte, s)),
                        "tpr@1fpr": tpr_at_fpr(yte, s)})

        # cross-strategy test: SAME probe, neutral vs target on test split
        Xc = np.concatenate([X["neutral"][te, layer], X["target"][te, layer]])
        yc = np.array([0] * len(te) + [1] * len(te))
        sc = clf.decision_function(Xc)
        cross.append({"layer": layer, "auroc": float(roc_auc_score(yc, sc)),
                      "tpr@1fpr": tpr_at_fpr(yc, sc)})

    best = max(in_dist, key=lambda r: r["auroc"])
    best_cross = next(r for r in cross if r["layer"] == best["layer"])
    print(f"\n  best layer = {best['layer']}  "
          f"in-dist AUROC={best['auroc']:.3f} (TPR@1%FPR={best['tpr@1fpr']:.2f})  |  "
          f"cross-strategy AUROC={best_cross['auroc']:.3f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "n": n,
                   "n_layers": n_layers, "best_layer": best["layer"],
                   "in_dist": in_dist, "cross": cross}, f, indent=2)
    print(f"[sbe] wrote {out}")


if __name__ == "__main__":
    main()