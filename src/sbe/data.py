"""Load WMDP / MMLU as a unified list of multiple-choice items:
    {"question": str, "choices": [4 strings], "answer_idx": int}
"""
from __future__ import annotations

import random
from datasets import load_dataset


def _sample(items, n, seed):
    if n and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def load_mcq(dataset: str, n: int = 200, seed: int = 0):
    dataset = dataset.lower()
    if dataset.startswith("wmdp"):
        # dataset ids: cais/wmdp with configs wmdp-bio / wmdp-chem / wmdp-cyber
        ds = load_dataset("cais/wmdp", dataset, split="test")
        items = [
            {"question": r["question"], "choices": list(r["choices"]), "answer_idx": int(r["answer"])}
            for r in ds
        ]
    elif dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        items = [
            {"question": r["question"], "choices": list(r["choices"]), "answer_idx": int(r["answer"])}
            for r in ds
        ]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    items = [it for it in items if len(it["choices"]) == 4]  # keep it 4-way
    return _sample(items, n, seed)
