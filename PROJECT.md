# Sandbagging Detection: Black-box vs White-box

**One-line thesis:** On the sandbagging-detection problem, how do detectors compare
across the *access ladder* — and where does each break?

Sandbagging = a model strategically underperforming on a capability evaluation.
It breaks the core assumption of evals: that a score is a true measure of capability.

## The access ladder (the intellectual hook)

| Detector | Access required | Needs labeled data? | Reference |
|---|---|---|---|
| Self-consistency / behavioral | API only (black-box) | no | our baseline |
| Noise injection | weights + forward passes | no | Tice et al. 2024 (arXiv 2412.01784) |
| Linear activation probe | full internals | yes (honest vs deceptive) | Apollo 2025 (arXiv 2502.03407) |

Story: the strongest detector (probe) needs the most access + labeled examples — exactly
what a third-party auditor is least likely to have. Does the cheap black-box method close
the gap?

## Original hypotheses

- **H1** Probe wins on AUROC when tested in-distribution.
- **H2** Probe advantage collapses on *unseen* sandbagging strategies; noise injection,
  being mechanism-based, degrades more gracefully.
- **H3** Self-consistency is near-useless against a competent sandbagger.

## Initial scope

- **Model:** Qwen2.5-3B-Instruct, bf16/fp16 (open weights, no gating). Stretch: Qwen2.5-7B.
- **Induction:** prompted sandbagging (v1). Stretch: LoRA password-lock.
- **Benchmark:** WMDP subset (dangerous-capability signal) + MMLU control.
- **Detectors:** self-consistency (black-box), noise injection (grey-box), linear probe (white-box).
- **Metrics:** AUROC plus an exploratory empirical TPR at 1% FPR. The initial
  sample is not large enough to validate a population 1% FPR operating point.
- **Novel deliverable:** a failure-mode grid — each detector x {blatant, subtle, score-targeted, unseen-strategy}.

## Success criteria (honest)

Succeeds even if every detector fails: "here is the regime where the field's best
sandbagging detectors break" is a real result. Not gambling on a positive outcome.

## Status and next milestone

The initial elicitation-gap, noise, option-permutation, and probe runs are
complete as an exploratory pilot. They exposed two validity problems:

1. the legacy probe selected its layer using test AUROC;
2. the 100-negative black-box sample cannot substantiate a 1% population FPR.

The code now uses train/validation/test separation, paired-bootstrap AUROC
intervals, and explicit low-FPR sample-size diagnostics. The next milestone is
to run those corrected protocols at the larger sample sizes documented in
`results/README.md`, then freeze a matched-prompt design that breaks the
correlation between system-prompt identity and detector label.

## Compute plan

- **Mac M5 (24GB):** dev, prototyping (3B local via MPS), analysis, plotting, writeup.
- **Kaggle (T4x2, 30h/wk):** the workhorse — activation caching, noise sweeps, probe training.
- **Colab free:** overflow.

## Key references

- van der Weij et al., *AI Sandbagging*, arXiv 2406.07358 (induction recipe)
- Greenblatt et al., *Stress-Testing Capability Elicitation with Password-Locked Models*, arXiv 2405.19550
- Tice et al., *Noise Injection Reveals Hidden Capabilities of Sandbagging LMs*, arXiv 2412.01784 (repo: SandbagDetect)
- Goldowsky-Dill et al. (Apollo), *Detecting Strategic Deception Using Linear Probes*, arXiv 2502.03407 (repo: ApolloResearch/deception-detection)
