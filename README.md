# sandbagging-evals

Black-box vs white-box detection of **sandbagging** (LLMs strategically underperforming
on capability evals). See [`PROJECT.md`](PROJECT.md) for the full spec, hypotheses, and
milestones.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

First run downloads Qwen2.5-3B-Instruct (~6 GB) from HuggingFace.

## Week 1 — measure the elicitation gap

Confirm we can induce sandbagging: accuracy should drop sharply under the sandbagging
prompts relative to the neutral prompt.

```bash
# small smoke test first (fast, ~2 min on Mac MPS)
python scripts/run_gap.py --dataset wmdp-bio --n 20

# the real run
python scripts/run_gap.py --dataset wmdp-bio --n 200
python scripts/run_gap.py --dataset mmlu     --n 200   # control
```

Output: per-mode accuracy, the gap vs neutral, and `results/gap.json`.

**What "success" looks like this week:** neutral accuracy clearly above chance (0.25),
and `blatant` sandbagging near/below chance — i.e. a large positive gap. `subtle` and
`target` should land in between. Plot comes next.

## Layout

```
src/sbe/
  config.py     device/dtype selection, defaults
  model.py      load model + tokenizer, letter-token ids
  data.py       WMDP / MMLU -> unified MCQ items
  prompts.py    neutral + sandbagging system prompts, MCQ formatting
  mcq_eval.py   logit-based multiple-choice eval
scripts/
  run_gap.py    Week 1: elicitation gap
```

## Running on Kaggle / Colab

Clone the repo (or upload `src/`), `pip install -r requirements.txt`, and call
`scripts/run_gap.py`. CUDA is picked up automatically. Use Kaggle for the heavy runs
(activation caching, noise sweeps) in later weeks.
