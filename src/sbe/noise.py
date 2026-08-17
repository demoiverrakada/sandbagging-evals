"""Weight-space Gaussian noise injection (Tice et al., arXiv 2412.01784).

We snapshot the original weights on CPU, then for each noise level sigma we restore
and add fresh Gaussian noise scaled per-tensor by that tensor's std (so sigma is a
relative fraction that means the same thing across differently-scaled layers).
Noise is drawn on CPU with a seeded generator for reproducibility + MPS safety.
"""
from __future__ import annotations

import torch


def select_noise_params(model):
    """Noise the 2-D weight matrices inside transformer blocks (attn + mlp);
    skip embeddings, lm_head, and 1-D norm params."""
    names = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and ".layers." in name and name.endswith(".weight"):
            names.append(name)
    return names


def snapshot(model, names):
    params = dict(model.named_parameters())
    return {n: params[n].detach().to("cpu").clone() for n in names}


@torch.no_grad()
def apply_noise(model, snap, sigma, seed, device):
    """Restore each snapshotted weight and add N(0, (sigma*std)^2) noise."""
    gen = torch.Generator().manual_seed(seed)
    params = dict(model.named_parameters())
    for n, orig in snap.items():          # orig lives on CPU
        if sigma > 0:
            std = orig.float().std()
            noise = torch.randn(orig.shape, generator=gen) * (sigma * std)
            new = (orig.float() + noise).to(dtype=orig.dtype)
        else:
            new = orig
        params[n].data.copy_(new.to(device))