#!/usr/bin/env python3
"""Week 3: linear activation probe for sandbagging.

Caches last-token residuals under neutral / blatant / target. For every layer it
trains on a training split, selects the layer on a separate validation split,
and reports exactly once on an untouched test split.

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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sbe.activations import cache_last_token_residuals
from sbe.config import DEFAULT_MODEL
from sbe.data import load_mcq
from sbe.metrics import detection_summary, paired_split_indices
from sbe.model import load_model


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
    ap.add_argument("--validation-frac", type=float, default=0.25)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
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

    split = paired_split_indices(
        n,
        train_fraction=args.train_frac,
        validation_fraction=args.validation_frac,
        seed=args.seed,
    )

    layer_results = []
    fitted_probes = []
    for layer in range(n_layers):
        # train: neutral(0) vs blatant(1) on train split
        Xtr = np.concatenate(
            [
                X["neutral"][split.train, layer],
                X["blatant"][split.train, layer],
            ]
        )
        ytr = np.array([0] * len(split.train) + [1] * len(split.train))
        clf = fit_probe(Xtr, ytr)
        fitted_probes.append(clf)

        def scores(mode, indices):
            return clf.decision_function(X[mode][indices, layer])

        validation_neutral = scores("neutral", split.validation)
        validation_blatant = scores("blatant", split.validation)
        validation_target = scores("target", split.validation)
        layer_results.append(
            {
                "layer": layer,
                "validation_in_distribution_auroc": float(
                    roc_auc_score(
                        np.concatenate(
                            [
                                np.zeros(len(split.validation)),
                                np.ones(len(split.validation)),
                            ]
                        ),
                        np.concatenate([validation_neutral, validation_blatant]),
                    )
                ),
                "validation_cross_strategy_auroc": float(
                    roc_auc_score(
                        np.concatenate(
                            [
                                np.zeros(len(split.validation)),
                                np.ones(len(split.validation)),
                            ]
                        ),
                        np.concatenate([validation_neutral, validation_target]),
                    )
                ),
            }
        )

    selected = max(
        layer_results,
        key=lambda result: result["validation_in_distribution_auroc"],
    )
    best_layer = selected["layer"]
    best_probe = fitted_probes[best_layer]
    test_neutral = best_probe.decision_function(X["neutral"][split.test, best_layer])
    test_blatant = best_probe.decision_function(X["blatant"][split.test, best_layer])
    test_target = best_probe.decision_function(X["target"][split.test, best_layer])
    test_in_distribution = detection_summary(
        test_neutral,
        test_blatant,
        target_fpr=0.01,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    test_cross_strategy = detection_summary(
        test_neutral,
        test_target,
        target_fpr=0.01,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    print(
        f"\n  validation-selected layer = {best_layer}  "
        f"validation AUROC={selected['validation_in_distribution_auroc']:.3f}  |  "
        f"test in-dist AUROC={test_in_distribution['auroc']:.3f}  |  "
        f"test cross-strategy AUROC={test_cross_strategy['auroc']:.3f}"
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "analysis_status": "corrected_split_exploratory_pilot",
                "model": args.model,
                "dataset": args.dataset,
                "n": n,
                "n_layers": n_layers,
                "split": {
                    "seed": args.seed,
                    "train_fraction": args.train_frac,
                    "validation_fraction": args.validation_frac,
                    "train_items": len(split.train),
                    "validation_items": len(split.validation),
                    "test_items": len(split.test),
                },
                "selection_metric": "validation_in_distribution_auroc",
                "selected_layer": best_layer,
                "layer_validation_results": layer_results,
                "test_in_distribution": test_in_distribution,
                "test_cross_strategy": test_cross_strategy,
                "confound_warning": (
                    "Prompt condition is perfectly correlated with the class label. "
                    "High performance may identify prompt identity rather than strategic deception."
                ),
            },
            f,
            indent=2,
        )
    print(f"[sbe] wrote {out}")


if __name__ == "__main__":
    main()
