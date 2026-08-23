# W3.1 Fold-Safe Conservative Weak Supervision

## Final endpoint

- W3.1 macro OOF AUROC: 0.524301
- W3.1 macro OOF AP: 0.426462
- W3.1 macro OOF F1: 0.341625
- W3.0 reference AUROC: 0.545918
- Delta AUROC: -0.021617
- Practical result bucket: NEGATIVE

## Fixed intervention

- Pseudo lambda: 0.25
- Selection threshold: 0.5
- One pseudo study per gold optimizer step.
- Soft targets remain continuous probabilities.
- Pseudo BCE is masked to selected cells only.
- CandidateSelectionScore is NOT a loss weight.
- No pseudo positive-class weighting is applied.
- Epoch length is defined by the gold loader.

## Leakage controls

- Outer image fold k consumes only W2.3 fold_k pseudo outputs.
- W2.3 fold assignment file is required to match V4/W3.0 exactly.
- No cross-fold probability stability file is used.
- No all-58 W2 soft probabilities/calibrators are used.

## Interpretation caution

- n=58 gold studies makes OOF deltas noisy.
- Epoch 12 is the pre-specified endpoint; no validation-driven epoch selection.
- Bootstrap output, when present, is descriptive rather than an independent significance test.

## Paired descriptive bootstrap

- 95% CI delta macro AUROC: [-0.081607, +0.038623]
- Bootstrap P(delta>0): 0.232
