#!/usr/bin/env python3
"""Figure 3: probe AUROC vs layer, in-distribution vs cross-strategy."""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/probe_wmdp-bio.json")
    ap.add_argument("--out", default="results/fig3_probe.png")
    args = ap.parse_args()
    try:
        d = json.load(open(args.input))
    except FileNotFoundError:
        sys.exit(f"{args.input} not found — run run_probe.py first")

    if "layer_validation_results" not in d:
        sys.exit(
            "Legacy probe result detected. Rerun scripts/run_probe.py with the "
            "train/validation/test implementation before plotting."
        )

    layer_results = d["layer_validation_results"]
    layers = [r["layer"] for r in layer_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        layers,
        [r["validation_in_distribution_auroc"] for r in layer_results],
        marker="o",
        label="validation: neutral vs blatant",
    )
    ax.plot(
        layers,
        [r["validation_cross_strategy_auroc"] for r in layer_results],
        marker="s",
        label="validation: neutral vs target",
    )
    selected_layer = d["selected_layer"]
    ax.axvline(selected_layer, ls=":", c="tab:red", lw=1, label="selected layer")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(layers[-1], 0.51, "chance", color="gray", ha="right", fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(
        f"Validation-only probe layer selection — {d['dataset']} "
        f"({d['model'].split('/')[-1]})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
