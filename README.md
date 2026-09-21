# Sandbagging Detection: Black-box vs White-box

An exploratory comparison of detector families across the access ladder:
API-only behavioral inconsistency, weight-noise perturbation, and
activation-based linear probes.

**Status:** exploratory pilot. The initial runs establish useful failure modes,
but they do not validate a deployment detector or a population 1% false-positive
operating point. The original probe result used test-set layer selection and is
retained only as a labelled legacy artifact; corrected code now selects the
layer on a separate validation split.

## Research question

How does sandbagging detection change as an auditor moves from black-box access
to model weights and finally to activations—and which apparent signals survive
confounds, distribution shift, and strict operating points?

The current study uses prompted underperformance in Qwen2.5-3B-Instruct on a
WMDP-Bio slice. Prompted underperformance is a controllable pilot condition, not
a realistic autonomous sandbagging organism.

## Initial results

| Detector | Access | Initial result | Interpretation |
|---|---|---|---|
| Option-permutation inconsistency | API only | AUROC 0.762 blatant; 0.791 target; empirical TPR 0 at 1% FPR | Aggregate discrimination in an `n=100` pilot; low-FPR performance is not validated |
| Weight-noise sweep | Weights | Sandbagging-condition accuracy rose at high noise | Consistent with partial disruption, but based on three seeds with substantial high-noise variance |
| Linear activation probe | Activations and labels | Legacy AUROC near 1.0 | Insufficient evidence: prompt identity determines the label and the legacy layer was selected on test AUROC |

Neutral WMDP-Bio accuracy was 66%, blatant prompted sandbagging 23%, and
target-strategy prompted sandbagging 36%. This establishes that the model
followed the suppression instructions in this setup.

## Validity limits

### Low false-positive rates

The black-box run contains 100 neutral scores. Even zero observed false
positives in an independently evaluated sample of that size leaves a one-sided
95% upper bound of approximately 2.95% on the false-positive rate. The reported
empirical `TPR@1%FPR` is therefore not evidence of a validated population 1%
FPR operating point.

### Probe confounding

The neutral and sandbagging labels are perfectly correlated with different
system prompts. A linear probe can therefore succeed by detecting prompt
identity rather than an internal strategic-deception state.

The legacy implementation also selected its best layer using test AUROC. The
corrected implementation trains on one split, selects the layer on validation
data, and evaluates exactly once on untouched test data.

### Noise interpretation

The initial noise curve is consistent with disruption of prompted
underperformance, but it is not independently diagnostic of sandbagging. More
seeds, uncertainty on contrasts, and non-sandbagging controls are required.

## Reproduce local checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The first model run downloads Qwen2.5-3B-Instruct.

## Initial pilot commands

```bash
python scripts/run_gap.py --dataset wmdp-bio --n 200
python scripts/run_gap.py --dataset mmlu --n 200
python scripts/run_noise.py --dataset wmdp-bio --n 100 --seeds 3
```

## Recommended corrected reruns

```bash
# Paired-bootstrap AUROC interval and explicit low-FPR diagnostics.
python scripts/run_blackbox.py --dataset wmdp-bio --n 1000 --perms 6

# Train/validation/test separation for probe training and layer selection.
python scripts/run_probe.py --dataset wmdp-bio --n 1200

# Only accepts the corrected result schema.
python scripts/plot_probe.py
```

These larger commands are recommended next experiments, not completed results.
See [`results/README.md`](results/README.md) before using an existing artifact.

## Repository layout

```text
src/sbe/
  blackbox.py     option-permutation detector
  metrics.py      splits, uncertainty, and low-FPR diagnostics
  activations.py  residual-stream extraction
  noise.py        weight perturbations
  prompts.py      neutral, prompted, and adaptive conditions
scripts/
  run_gap.py
  run_noise.py
  run_blackbox.py
  run_probe.py
  run_adaptive.py
tests/
  test_metrics.py
results/
  README.md
```

## Supported claims

- Prompted conditions produced a large capability-suppression gap in this model
  and benchmark slice.
- Option-permutation inconsistency showed aggregate discrimination in the
  initial 100-question pilot but no empirical sensitivity at the requested 1%
  FPR threshold.
- The near-perfect legacy probe result is insufficient evidence of strategic
  deception detection.

## Not supported

- A deployment-ready sandbagging detector.
- A validated 1% population false-positive operating point.
- Detection of autonomous or covert strategic deception.
- Generalization beyond one prompted model, one dangerous-capability benchmark
  slice, and the tested strategies.

## References

- van der Weij et al., *AI Sandbagging*, arXiv:2406.07358.
- Tice et al., *Noise Injection Reveals Hidden Capabilities of Sandbagging
  Language Models*, arXiv:2412.01784.
- Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes*,
  arXiv:2502.03407.
- Greenblatt et al., *Stress-Testing Capability Elicitation with
  Password-Locked Models*, arXiv:2405.19550.
