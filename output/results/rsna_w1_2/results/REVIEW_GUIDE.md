# W1.2 Adjudication Guide

## Purpose

Review the report text independently of the automatic parser.
The 12 expert binary labels remain the challenge gold labels;
the reviewer is annotating what the *report itself* asserts.

## ReviewerReportAssertion

Use exactly one:

- positive
- negative
- uncertain
- mixed
- related_abnormality
- mentioned_neutral
- not_mentioned

`not_mentioned` must never be converted automatically to
negative.

## ReviewerEvidenceSpan

Copy the shortest report phrase that justifies the assertion.

## ReviewerSeverity

Record explicit wording only, for example:

- complete
- partial
- grade_1
- grade_2
- mild
- moderate
- severe
- degenerative
- not_stated

Do not infer severity that is not written.

## ReviewerGoldAgreement

Use:

- agrees
- disagrees
- indeterminate

This asks whether the report-side assertion maps cleanly to the
challenge gold binary label.

## ReviewerOntologyCategory

Useful values include:

- direct_match
- mild_or_partial_below_possible_threshold
- severity_threshold_possible
- related_finding_not_target
- historical_or_postoperative
- negated
- uncertain
- report_gold_disagreement_other
- extraction_error
- terminology_gap
- not_applicable

## ReviewerConfidence

Use:

- high
- medium
- low

## Principle

Do not repair disagreement by forcing the report to match the
gold label. W1.2 exists specifically to expose differences
between report wording and challenge-label ontology.
