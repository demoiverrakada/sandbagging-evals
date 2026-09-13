# Sandbagging Detection: Black-box vs White-box

Comparing three sandbagging detectors across the access ladder — from
API-only black-box to full-internals white-box — on prompted
sandbagging in Qwen2.5-3B-Instruct.

Sandbagging is when a model strategically underperforms on a capability
evaluation. It breaks the core assumption of evals: that a score
reflects true capability. This project asks which detection methods
actually work, and where each one breaks.

See [PROJECT.md](PROJECT.md) for the full specification, hypotheses,
and references.

## Key results

| Detector | Access level | AUROC (blatant) | TPR @ 1% FPR | Notes |
|---|---|---:|---:|---|
| Self-consistency | Black-box (API only) | 0.79 | 0.00 | Detects blatant but useless at strict thresholds |
| Noise injection | Grey-box (weights) | — | — | Anomalous accuracy rise under noise confirms sandbagging signal |
| Linear probe | White-box (activations) | ~1.00 | ~1.00 | Near-perfect, but likely a prompt-identity confound (see below) |

**Elicitation gap:** neutral accuracy 66%, blatant sandbagging 28% on
WMDP-Bio (Qwen2.5-3B-Instruct). The gap confirms the model follows
the sandbagging instruction.

**Probe caveat:** the linear probe achieves near-perfect separation, but
the training labels (neutral vs sandbagging) are perfectly correlated
with the system prompt. The probe may be detecting which prompt was
used rather than an internal "I am sandbagging" state. This is flagged
as a likely prompt-identity confound — the same issue noted in recent
literature on deception probes.

## Detectors

### 1. Elicitation gap (`scripts/run_gap.py`)

Measures accuracy under neutral, blatant, weak-persona, and
score-targeted sandbagging prompts on WMDP-Bio and MMLU.

### 2. Noise injection (`scripts/run_noise.py`)

Adds calibrated Gaussian noise to transformer weight matrices (Tice et
al., arXiv 2412.01784). An honest model degrades monotonically; a
sandbagger shows anomalous accuracy recovery as noise disrupts the
sandbagging strategy.

### 3. Black-box self-consistency (`scripts/run_blackbox.py`)

Shuffles answer options across multiple presentations of the same
question. A consistent model picks the same underlying answer
regardless of option order; a sandbagger is more inconsistent. Reports
AUROC and TPR@1%FPR.

### 4. Linear activation probe (`scripts/run_probe.py`)

Caches last-token residual-stream activations per layer, trains a
logistic-regression probe to separate honest from sandbagging
activations. Reports in-distribution AUROC and cross-strategy
generalization.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads Qwen2.5-3B-Instruct (~6 GB) from HuggingFace.

## Reproduce

```bash
# 1. Elicitation gap
python scripts/run_gap.py --dataset wmdp-bio --n 200
python scripts/run_gap.py --dataset mmlu     --n 200

# 2. Noise injection sweep
python scripts/run_noise.py --dataset wmdp-bio --n 100 --seeds 3

# 3. Black-box consistency
python scripts/run_blackbox.py --dataset wmdp-bio --n 100 --perms 6

# 4. Linear probe
python scripts/run_probe.py --dataset wmdp-bio --n 200
```

Plotting scripts (`plot_gap.py`, `plot_noise.py`, `plot_probe.py`) are
in `scripts/`.

## Layout

```
src/sbe/
  config.py        device/dtype selection, defaults
  model.py         load model + tokenizer, letter-token ids
  data.py          WMDP / MMLU -> unified MCQ items
  prompts.py       neutral + sandbagging system prompts, MCQ formatting
  mcq_eval.py      logit-based multiple-choice eval
  blackbox.py      option-permutation consistency detector
  noise.py         Gaussian weight-noise injection
  activations.py   residual-stream activation caching for probes
scripts/
  run_gap.py       elicitation gap measurement
  run_noise.py     noise injection sweep
  run_blackbox.py  black-box consistency detector
  run_probe.py     linear activation probe
  plot_gap.py      gap visualization
  plot_noise.py    noise sweep visualization
  plot_probe.py    probe AUROC visualization
```

## Running on Kaggle / Colab

Clone the repo, `pip install -r requirements.txt`, and run any script.
CUDA is picked up automatically. Use Kaggle for the heavy runs
(activation caching, noise sweeps).

## References

- van der Weij et al., *AI Sandbagging*, arXiv 2406.07358
- Tice et al., *Noise Injection Reveals Hidden Capabilities*, arXiv 2412.01784
- Goldowsky-Dill et al. (Apollo), *Detecting Strategic Deception Using Linear Probes*, arXiv 2502.03407
- Greenblatt et al., *Stress-Testing Capability Elicitation with Password-Locked Models*, arXiv 2405.19550
