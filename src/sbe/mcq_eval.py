"""Logit-based multiple-choice evaluation.

For each question we take the model's next-token logits at the answer position and
restrict to the four option-letter tokens (A/B/C/D). This is deterministic, fast, and
gives us exactly the forward pass we'll reuse for activation probing in Week 3.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .config import LETTERS
from .model import letter_token_ids
from .prompts import build_chat, system_prompt


@torch.no_grad()
def evaluate(model, tokenizer, items, mode: str, device: str, show_progress: bool = True):
    """Run the MCQ eval under a given prompt `mode`. Returns a results dict."""
    lids = letter_token_ids(tokenizer, LETTERS)
    letter_ids = torch.tensor([lids[L] for L in LETTERS], device=device)

    per_item = []
    correct = 0
    iterator = tqdm(items, disable=not show_progress, desc=f"eval[{mode}]")
    for i, it in enumerate(iterator):
        text = build_chat(tokenizer, system_prompt(mode), it["question"], it["choices"])
        enc = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1]              # (vocab,)
        letter_logits = logits[letter_ids].float()        # (4,)
        probs = torch.softmax(letter_logits, dim=-1)
        pred = int(letter_logits.argmax().item())
        is_correct = int(pred == it["answer_idx"])
        correct += is_correct
        per_item.append(
            {
                "idx": i,
                "answer_idx": it["answer_idx"],
                "pred": pred,
                "correct": is_correct,
                "letter_probs": probs.tolist(),
            }
        )

    n = len(items)
    return {
        "mode": mode,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "per_item": per_item,
    }
