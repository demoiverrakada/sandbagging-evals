# Result status

The JSON and figures in this directory are included for auditability.

## Legacy exploratory run

The existing `probe_wmdp-bio.json` was produced by an earlier implementation
that selected the best layer using test AUROC. It must not be used as a
confirmatory performance estimate. The near-perfect result is additionally
confounded because prompt condition is perfectly correlated with the label.

The existing black-box run used 100 neutral and 100 sandbagging scores. Its
AUROC is an exploratory point estimate. At this sample size, even zero observed
false positives would leave a one-sided 95% upper bound of approximately 2.95%
on the false-positive rate; the reported empirical TPR at 1% FPR does not
validate a population 1% operating point.

## Required corrected reruns

```bash
# Adds paired-bootstrap AUROC intervals and low-FPR sample-size diagnostics.
python scripts/run_blackbox.py --dataset wmdp-bio --n 1000 --perms 6

# Selects the layer on validation data and evaluates once on untouched test data.
python scripts/run_probe.py --dataset wmdp-bio --n 1200

# Plot only after the corrected probe JSON has replaced the legacy file.
python scripts/plot_probe.py
```

The larger runs are recommendations, not completed results. Keep the project
labelled **exploratory pilot** until they are executed and reviewed.
