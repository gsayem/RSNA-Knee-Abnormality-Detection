# RSNA W2.4 — High-Fidelity Fold-Safe Report Supervision

## Controlled validation

- Exact outer-fold SHA256: `1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a`
- W2.3 macro AUROC: `0.771243`
- W2.4 macro AUROC: `0.849857`
- Delta: `+0.078614`

Bootstrap W2.4 − W2.3:

```json
{
  "mean_delta": 0.07853331542652126,
  "ci_low": 0.043022013851949245,
  "ci_high": 0.11192344669242779,
  "p_gt_0": 1.0,
  "n_bootstrap": 2000
}
```

## Source macro metrics

```json
{
  "W2.3": {
    "macro_AUROC": 0.7712431042776814,
    "macro_AP": 0.6832265162015955,
    "macro_Brier": 0.15907222465897464
  },
  "W2.4": {
    "macro_AUROC": 0.8498568811504255,
    "macro_AP": 0.7719323753170148,
    "macro_Brier": 0.1512659803280024
  },
  "primary_teacher": {
    "macro_AUROC": 0.8700395897324982,
    "macro_AP": 0.7587425911821705,
    "macro_Brier": 0.14624853801169588
  }
}
```

## Methodological boundary

Our calibration/fusion layer is fold-safe: held-out gold labels are never used to fit calibrators, reliability weights, or thresholds. The external public teacher is a fixed upstream artifact; W2.4 cannot verify whether its original development process used the same 58-study research cohort.

## W6 inputs

Each fold contains `soft_probabilities_wide.csv`, `soft_label_weights_wide.csv`, 
`soft_label_availability_wide.csv`, and compatibility 
`candidate_selection_scores_wide.csv`.
