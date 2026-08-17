"""Cache last-token residual-stream activations for linear probing.

For each question under a given prompt mode we run one forward pass and grab the
hidden state at the final (answer) position for every layer. This is the same
object your refusal-direction work used — here we probe it for 'am I sandbagging?'
"""
from __future__ import annotations

import numpy as np
import torch

from .config import LETTERS
from .model import letter_token_ids
from .prompts import build_chat, system_prompt


@torch.no_grad()
def cache_last_token_residuals(model, tokenizer, items, mode, device, show_progress=True):
    """Returns (feats, correct):
        feats   : float32 array (n_items, n_layers+1, d_model)
        correct : int array (n_items,)  — was the model's MC answer correct
    """
    from tqdm import tqdm

    lids = letter_token_ids(tokenizer, LETTERS)
    letter_ids = torch.tensor([lids[L] for L in LETTERS], device=device)

    feats, correct = [], []
    for it in tqdm(items, disable=not show_progress, desc=f"cache[{mode}]"):
        text = build_chat(tokenizer, system_prompt(mode), it["question"], it["choices"])
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states                      # tuple(len=n_layers+1) of (1, seq, d)
        last = torch.stack([h[0, -1, :] for h in hs], dim=0)   # (n_layers+1, d)
        feats.append(last.float().cpu().numpy())
        pred = int(out.logits[0, -1][letter_ids].argmax().item())
        correct.append(int(pred == it["answer_idx"]))

    return np.stack(feats, axis=0), np.array(correct)