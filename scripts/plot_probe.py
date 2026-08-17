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

    layers = [r["layer"] for r in d["in_dist"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, [r["auroc"] for r in d["in_dist"]], marker="o",
            label="in-distribution (train blatant, test blatant)")
    ax.plot(layers, [r["auroc"] for r in d["cross"]], marker="s",
            label="cross-strategy (train blatant, test target)")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(layers[-1], 0.51, "chance", color="gray", ha="right", fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Linear-probe sandbagging detection — {d['dataset']} "
                 f"({d['model'].split('/')[-1]})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()