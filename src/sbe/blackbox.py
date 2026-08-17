"""Black-box (API-only) sandbagging detector: option-permutation consistency.

Ask each question several times with the four answer options shuffled. We map each
prediction back to the ORIGINAL choice index and measure how often the model changes
its underlying answer. Honest -> consistent; sandbagger -> more inconsistent (H3).
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .config import LETTERS
from .model import letter_token_ids
from .prompts import build_chat, system_prompt


def make_perms(n_perms, seed=0):
    n_perms = min(n_perms, 24)               # 4! = 24 distinct orderings
    rng = random.Random(seed)
    base = (0, 1, 2, 3)
    perms, seen = [base], {base}
    while len(perms) < n_perms:
        p = tuple(rng.sample(base, 4))
        if p not in seen:
            seen.add(p)
            perms.append(p)
    return perms


@torch.no_grad()
def _predict_original(model, tokenizer, letter_ids, system, question, choices, perm, device):
    permuted = [choices[i] for i in perm]
    text = build_chat(tokenizer, system, question, permuted)
    enc = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1]
    pos = int(logits[letter_ids].argmax().item())    # chosen display position
    return perm[pos]                                  # -> original choice index


@torch.no_grad()
def consistency_scores(model, tokenizer, items, mode, device,
                       n_perms=6, seed=0, show_progress=True):
    """Per-question inconsistency in [0,1]: 1 - (fraction agreeing with modal answer)."""
    from tqdm import tqdm

    lids = letter_token_ids(tokenizer, LETTERS)
    letter_ids = torch.tensor([lids[L] for L in LETTERS], device=device)
    perms = make_perms(n_perms, seed)
    system = system_prompt(mode)

    scores = []
    for it in tqdm(items, disable=not show_progress, desc=f"consistency[{mode}]"):
        preds = [_predict_original(model, tokenizer, letter_ids, system,
                                   it["question"], it["choices"], p, device)
                 for p in perms]
        _, counts = np.unique(preds, return_counts=True)
        modal_frac = counts.max() / len(preds)
        scores.append(1.0 - modal_frac)
    return np.array(scores)