# RSNA W1.2 — Gold Report Assertion Benchmark Preparation

## Scope

- Gold studies: 58
- Labels: 12
- Report/label pairs re-audited: 696
- NLP model trained: **No**
- Pseudo-labels generated: **No**

## W1.2 parser changes

- Sentence/clause splitting now handles punctuation without following whitespace.
- Injury negation is scoped around the injury mention rather than the whole sentence.
- Direct-condition negation is concept-local.
- `not_mentioned` remains distinct from negative.
- W1.1 → W1.2 parser-changed pairs found: 19

## Automatic conservative extraction

- Explicitly decidable report/label pairs: 277 / 696 (39.8%)
- Explicit report ↔ gold disagreements detected: 51
- Gold-positive pairs not cleanly classified as report-positive: 91

These are parser-audit statistics, not clinical NLP performance estimates.

## Adjudication benchmark seed

- Full high-information adjudication pool: 387 pairs
- Compact benchmark seed: 120 pairs (target 120)
- Seed intentionally includes explicit disagreements, missed gold positives, ambiguous/related findings, and concordant controls.

## Per-label summary

| Label | Gold + | Decidable coverage | Positive PPV | Negative NPV | Explicit +/− disagreements |
|---|---:|---:|---:|---:|---:|
| ACL | 24 | 67.2% | 84.6% | 100.0% | 4 |
| MCL | 9 | 50.0% | 53.3% | 100.0% | 7 |
| Medial Meniscus | 26 | 46.6% | 86.4% | 100.0% | 3 |
| Lateral Meniscus | 23 | 56.9% | 86.7% | 94.4% | 3 |
| Medial OA | 15 | 12.1% | 83.3% | 100.0% | 1 |
| Lateral OA | 11 | 3.4% | 100.0% | 100.0% | 0 |
| PF OA | 21 | 17.2% | 90.0% | NA | 1 |
| Effusion | 35 | 86.2% | 68.9% | 100.0% | 14 |
| Synovitis | 27 | 25.9% | 73.3% | NA | 4 |
| Baker's | 12 | 37.9% | 66.7% | 100.0% | 5 |
| Contusion | 19 | 39.7% | 66.7% | 100.0% | 5 |
| Fracture | 18 | 34.5% | 71.4% | 100.0% | 4 |

## Interpretation rules

- `not_mentioned` is **not** treated as negative.
- `related_abnormality` remains separate from a direct target-positive assertion.
- `mixed` preserves contradictory report evidence.
- Severity is recorded as written and is not silently converted into the challenge ontology.

## Highest-priority files

1. `12_adjudication_benchmark_seed.csv`
2. `REVIEW_GUIDE.md`
3. `07_explicit_report_gold_disagreements.csv`
4. `08_gold_positive_not_cleanly_extracted.csv`
5. `04_w11_vs_w12_parser_changes.csv`
6. `05_per_label_alignment_summary.csv`

## W2 gate

The 120-pair seed is a review artifact, not a finished benchmark. W2 trust weights should not be set from automatic W1.2 PPV/NPV alone. First review the seed and distinguish extraction errors from true report ↔ challenge-ontology disagreement.
