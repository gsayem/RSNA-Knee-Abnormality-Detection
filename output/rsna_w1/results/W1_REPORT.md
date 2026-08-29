# RSNA W1 Report Investigation

## Dataset inventory

- Total studies: 4,407
- Gold-labeled studies: 58
- Fully unlabeled studies: 4,349
- Reports present: 4,407
- Reports missing: 0
- Language backend: `lingua`

## Report size

- Median characters: 974.0
- Median words: 126.0
- Median sentences: 15.0

## Dominant scripts

- Latin: 3,866 (87.72%)
- Greek: 321 (7.28%)
- Cyrillic: 220 (4.99%)

## Detected languages / language families

- en: 1,672 (37.94%)
- es: 682 (15.48%)
- tr: 546 (12.39%)
- hr: 401 (9.10%)
- el: 321 (7.28%)
- de: 262 (5.95%)
- bg: 220 (4.99%)
- nl: 153 (3.47%)
- fr: 92 (2.09%)
- la: 53 (1.20%)
- bs: 5 (0.11%)

## Quality / duplication

- Reports with one or more quality flags: 0
- Exact/template duplicate groups: 109
- Near-duplicate pairs above threshold 0.96: 521

## W1 interpretation checklist

1. Compare gold vs unlabeled language/script distribution.
2. Inspect duplicate/template reports before pseudo-labeling.
3. Inspect `13_gold_label_term_associations.csv` and concordance sentences.
4. Determine which languages need dedicated negation/uncertainty handling.
5. Do not build W2 until report terminology and gold representativeness are understood.

## Important limitation

W1 term associations are descriptive because the gold cohort contains only 58 studies. They are not a report classifier and must not be treated as validated pseudo-labeling rules.
