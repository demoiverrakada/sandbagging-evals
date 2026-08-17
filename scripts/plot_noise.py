#!/usr/bin/env python3
"""Figure 2: accuracy vs weight-noise, per prompt mode.

Honest (neutral) should slope down monotonically; a sandbagger (blatant/target)
should show an anomalous hump — accuracy rising above its sigma=0 value.
"""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/noise_wmdp-bio.json")
    ap.add_argument("--out", default="results/fig2_noise.png")
    args = ap.parse_args()

    try:
        d = json.load(open(args.input))
    except FileNotFoundError:
        sys.exit(f"{args.input} not found — run run_noise.py first")

    sigmas = d["sigmas"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in d["modes"]:
        means = [d["curve"][mode][str(s)]["mean"] for s in sigmas]
        stds = [d["curve"][mode][str(s)]["std"] for s in sigmas]
        ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=3, label=mode)

    ax.axhline(0.25, ls="--", c="gray", lw=1)
    ax.text(sigmas[-1], 0.26, "chance", color="gray", ha="right", fontsize=8)
    ax.set_xlabel("noise level  sigma  (fraction of per-tensor std)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Noise-injection signature — {d['dataset']} ({d['model'].split('/')[-1]})")
    ax.legend(title="prompt mode")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()