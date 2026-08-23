# RSNA W1.3 Final Adjudication Report

## Completion

- Benchmark pairs: 120
- Completed blinded adjudications: 120
- Needs translation: 0
- Unable to adjudicate: 0

## W1.2 parser benchmark

- Exact reviewer ↔ parser assertion agreement: 72.5%

The reviewer assertion is the blinded report-side reference. Challenge gold was revealed only after report interpretation.

## Label-specific report ↔ challenge mapping

| Label | Report-positive N | Positive PPV | Report-negative N | Negative NPV |
|---|---:|---:|---:|---:|
| ACL | 7 | 57.1% | 0 | NA |
| MCL | 6 | 16.7% | 3 | 100.0% |
| Medial Meniscus | 8 | 75.0% | 1 | 100.0% |
| Lateral Meniscus | 3 | 66.7% | 3 | 66.7% |
| Medial OA | 5 | 100.0% | 1 | 100.0% |
| Lateral OA | 4 | 75.0% | 1 | 100.0% |
| PF OA | 3 | 100.0% | 1 | 0.0% |
| Effusion | 10 | 50.0% | 1 | 100.0% |
| Synovitis | 5 | 20.0% | 0 | NA |
| Baker's | 8 | 37.5% | 1 | 100.0% |
| Contusion | 7 | 28.6% | 1 | 100.0% |
| Fracture | 7 | 28.6% | 1 | 100.0% |

## W2 gate

W1.3 still generates no pseudo-labels. Use the adjudicated label-specific PPV/NPV, severity mapping, and parser benchmark to choose and calibrate W2.

The 120-pair benchmark is intentionally high-information and not prevalence-representative. Raw overall accuracy must not be interpreted as expected full-corpus performance.
