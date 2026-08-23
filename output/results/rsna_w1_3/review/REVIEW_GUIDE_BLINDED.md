# W1.3 Blinded Report Adjudication Guide

## Blinding rule

During adjudication, open only:

- `01_W1_3_BLINDED_REVIEW.csv`
- this guide

Do not open the `private/` directory until the review is
finished. The private file contains the challenge gold labels
and the W1.2 parser outputs.

The task is to annotate what the **report itself asserts**.
Do not try to infer or reproduce the challenge gold label.

---

## ReviewerStatus

Use exactly one:

- `complete`
- `needs_translation`
- `unable_to_adjudicate`

If the report cannot be interpreted reliably, use
`needs_translation`. Do not guess.

---

## ReviewerReportAssertion

Use exactly one:

- `positive`
- `negative`
- `uncertain`
- `mixed`
- `related_abnormality`
- `mentioned_neutral`
- `not_mentioned`

### positive
The report explicitly asserts the target pathology/finding.

### negative
The report explicitly says the target is absent, intact,
normal, or otherwise negative.

### uncertain
The report says possible, suspicious, cannot exclude, etc.

### mixed
Materially conflicting positive and negative statements are
present and cannot be resolved from the report.

### related_abnormality
A related finding is present, but the report does not directly
assert the target.

Examples:
- meniscal degeneration without explicit tear
- bone marrow edema without explicit contusion/bruise
- synovial thickening without explicit synovitis
- compartmental chondral/cartilage abnormality without
  explicit OA/osteoarthritis/arthrosis

### mentioned_neutral
The target anatomy is mentioned without a clear positive,
negative, uncertain, or related finding.

### not_mentioned
No relevant target statement appears.

`not_mentioned` must never be converted to `negative`.

---

## Conservative target interpretation

### ACL
Direct ACL injury/tear/rupture/sprain.
Record grade/partial/complete separately.

### MCL
Direct MCL injury/tear/rupture/sprain.

### Medial Meniscus / Lateral Meniscus
Direct tear/rupture of the specified meniscus.
Degenerative signal without tear -> `related_abnormality`.

### Medial OA / Lateral OA / PF OA
For blinded adjudication, use a conservative definition:
explicit OA / osteoarthritis / arthrosis / gonarthrosis in the
target compartment -> `positive`.

Isolated chondromalacia, cartilage loss, chondral erosion,
osteophytes, or similar degeneration without explicit
OA/arthrosis wording -> `related_abnormality`.

This preserves terminology/ontology uncertainty for later
comparison with challenge gold.

### Effusion
Direct joint effusion or explicit increased intra-articular
fluid/equivalent fluid collection.

### Synovitis
Explicit synovitis -> `positive`.
Synovial hypertrophy/thickening alone -> `related_abnormality`.

### Baker's
Explicit Baker/popliteal cyst.

### Contusion
Explicit bone contusion / bone bruise -> `positive`.
Marrow edema alone -> `related_abnormality`.

### Fracture
Any explicit fracture statement is report-positive regardless
of subtype.

---

## ReviewerEvidenceSpan

Copy the shortest phrase that supports the assertion.

Leave blank only for `not_mentioned`.

---

## ReviewerSeverity

Use one or more values separated by `|`:

- `not_stated`
- `minimal`
- `mild`
- `grade_1`
- `grade_2`
- `grade_3`
- `grade_4`
- `partial`
- `moderate`
- `severe`
- `complete`
- `degenerative`
- `chronic`
- `acute`
- `other`

Record only explicitly stated severity.

---

## ReviewerPhenotypeCategory

Use one:

- `direct_target`
- `mild_or_partial_target`
- `moderate_target`
- `severe_or_complete_target`
- `degenerative_target`
- `related_finding_not_target`
- `historical_or_postoperative`
- `negated_target`
- `uncertain_target`
- `mixed_target`
- `neutral_anatomy_mention`
- `not_mentioned`
- `other`

This describes the report phenotype only. It must not refer to
whether challenge gold is 0 or 1.

---

## ReviewerConfidence

Use:
- `high`
- `medium`
- `low`

This is confidence in the report-side interpretation.

---

## ReviewerTranslationUsed

Optional:
- `yes`
- `no`

If translation or language assistance was used, mention the
method briefly in `ReviewerNotes`.

---

## Methodological rule

Do not repair a report interpretation to make it agree with the
challenge gold label. Disagreement is a primary W1.3 outcome.
