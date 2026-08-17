#!/usr/bin/env python3
"""Figure 1: the elicitation gap.

Grouped bar chart of accuracy per prompt mode, one group of bars per dataset,
with a dashed line at chance (0.25).

Usage:
    python scripts/plot_gap.py                       # auto-globs results/gap_*.json
    python scripts/plot_gap.py --inputs results/gap_wmdp-bio.json results/gap_mmlu.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["neutral", "blatant", "weak", "target"]
CHANCE = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=sorted(glob.glob("results/gap_*.json")))
    ap.add_argument("--out", default="results/fig1_gap.png")
    args = ap.parse_args()
    if not args.inputs:
        sys.exit("no results/gap_*.json found — run run_gap.py first")

    data = {}
    for path in args.inputs:
        d = json.load(open(path))
        data[d["dataset"]] = {m: d["results"][m]["accuracy"] for m in d["results"]}

    datasets = list(data.keys())
    modes = [m for m in MODE_ORDER if any(m in data[ds] for ds in datasets)]
    x = np.arange(len(modes))
    w = 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ds in enumerate(datasets):
        vals = [data[ds].get(m, np.nan) for m in modes]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=ds)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.axhline(CHANCE, ls="--", c="gray", lw=1)
    ax.text(len(modes) - 0.5, CHANCE + 0.01, "chance", color="gray", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Elicitation gap: accuracy by prompt mode (Qwen2.5-3B-Instruct)")
    ax.legend(title="dataset")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()