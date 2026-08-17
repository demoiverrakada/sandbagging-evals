"""Model + tokenizer loading. Kept framework-plain (HF transformers) on purpose:
lighter on memory than TransformerLens and matches the reference repos."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import pick_device, pick_dtype


def load_model(model_name: str):
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[sbe] loading {model_name} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device


def letter_token_ids(tokenizer, letters):
    """Token id the model emits for each option letter as the first response token."""
    ids = {}
    for L in letters:
        toks = tokenizer.encode(L, add_special_tokens=False)
        ids[L] = toks[-1]
    return ids
