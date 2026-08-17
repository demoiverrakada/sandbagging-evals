"""Central config + device/dtype selection so the same code runs on Mac (MPS),
Kaggle/Colab (CUDA), and CPU."""
from __future__ import annotations
from dataclasses import dataclass

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> "torch.dtype":
    # bf16 is cleanest for weight-noise injection but MPS support is patchy,
    # so we use fp16 on MPS and bf16 on CUDA.
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # open weights, no gating

LETTERS = ["A", "B", "C", "D"]


@dataclass
class RunConfig:
    model_name: str = DEFAULT_MODEL
    dataset: str = "wmdp-bio"   # wmdp-bio | wmdp-chem | wmdp-cyber | mmlu
    n: int = 200                # number of questions to sample
    seed: int = 0
    max_new_context: int = 2048
