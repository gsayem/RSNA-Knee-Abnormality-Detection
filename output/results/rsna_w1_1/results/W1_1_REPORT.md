# RSNA W1.1 — Report ↔ Gold Label Alignment Audit

## Scope

- Gold studies: 58
- Labels: 12
- Report/label pairs audited: 696
- NLP model trained: **No**

## Automatic conservative extraction

- Explicitly decidable report/label pairs: 279 / 696 (40.1%)
- Explicit report ↔ gold disagreements detected: 61
- Gold-positive pairs not cleanly classified as report-positive: 95

Important: these are lexicon-audit statistics, not final clinical NLP performance estimates.

## Per-label summary

| Label | Gold + | Decidable coverage | Positive PPV | Negative NPV | Explicit +/− disagreements |
|---|---:|---:|---:|---:|---:|
| ACL | 24 | 67.2% | 84.6% | 100.0% | 4 |
| MCL | 9 | 50.0% | 57.1% | 100.0% | 6 |
| Medial Meniscus | 26 | 44.8% | 81.8% | 100.0% | 4 |
| Lateral Meniscus | 23 | 62.1% | 56.5% | 92.3% | 11 |
| Medial OA | 15 | 13.8% | 85.7% | 100.0% | 1 |
| Lateral OA | 11 | 5.2% | 50.0% | 100.0% | 1 |
| PF OA | 21 | 17.2% | 90.0% | NA | 1 |
| Effusion | 35 | 82.8% | 67.5% | 75.0% | 15 |
| Synovitis | 27 | 25.9% | 73.3% | NA | 4 |
| Baker's | 12 | 37.9% | 66.7% | 100.0% | 5 |
| Contusion | 19 | 39.7% | 66.7% | 100.0% | 5 |
| Fracture | 18 | 34.5% | 71.4% | 100.0% | 4 |

## Interpretation rules

- `not_mentioned` is **not** treated as negative.
- `related_abnormality` is preserved separately (for example bone-marrow edema without an explicit contusion term, or meniscal degeneration without an explicit tear).
- `mixed` preserves contradictory statements instead of forcing a binary label.
- Severity terms are exported separately to help identify expert-label ontology thresholds.

## Highest-priority files

1. `06_explicit_report_gold_disagreements.csv`
2. `07_gold_positive_not_cleanly_extracted.csv`
3. `08_severity_ontology_cases.csv`
4. `10_manual_review_queue.csv`
5. `04_per_label_alignment_summary.csv`

## W2 gate

Do not assign pseudo-label trust weights until the manual-review queue has been inspected. W1.1 is designed to distinguish ontology/severity differences from extraction failures.
