RSNA Knee Abnormality Detection / Multi-Model Fusion for MRI
High-Fidelity Project Continuation Package

This document reconstructs the technical state of the project so a new ChatGPT instance can continue immediately without access to the original thread.

Status vocabulary

CONFIRMED — explicitly established in this conversation or experiment output.
IMPLEMENTED — code was written.
TESTED — code/experiment was executed.
INFERRED — strongly implied, but not established as a formal claim.
PROPOSED — future work discussed/recommended, not completed.
REJECTED — deliberately stopped.
UNKNOWN — not reliably recoverable; verify source code/output.
1. EXECUTIVE PROJECT STATE

[CONFIRMED] Research problem. The project is for the Kaggle RSNA Knee Abnormality Detection competition. It is a 12-label knee MRI multilabel problem with extreme label scarcity: only 58 fully labeled studies and 4,349 unlabeled studies, while all 4,407 training studies have radiology reports.

[CONFIRMED] Core constraint. Hidden test studies have no reports. Therefore reports are being used as privileged training-time supervision / teacher information, not as hidden-test inputs.

[CONFIRMED] Current working hypothesis. Image representation quality matters, but experiments increasingly showed that weak-supervision quality is the larger bottleneck. The strongest confirmed image representation so far is frozen Curia-2. The strongest confirmed own report teacher is W2.6, a challenge-aware four-label Qwen few-shot mapper layered over W2.3.

[CONFIRMED] Current report result.

W2.3 OOF macro AUROC: 0.771243
W2.6 fs4_replace: approximately 0.81342
W2.6 predeclared fs4_fixed50: approximately 0.80929
Fixed50 was selected for production because it gave better calibration/Brier behavior.

[CONFIRMED] Current image anchor.
Frozen Curia-2 CLS series features + hierarchical study aggregation:

gold OOF: 0.620640
weak-all OOF: 0.641077
gated weak OOF: 0.652317
public LB roughly 0.721 weak-all, 0.717 gated

[CONFIRMED] Current project stage. W2.6 gold validation is complete and successful. The immediate blocker is W2.6-P production pseudo-label generation for the 4,349 unlabeled studies.

[CONFIRMED] Current production failure. The current rsna_w2_6p_batched_production_mapper.py requires 17,396 Qwen predictions. On Kaggle 2×T4 it reached 6,768/17,396 after roughly 12 hours, with throughput degraded to about 9.4 cells/min and ~18.8 h ETA remaining, then hit Kaggle’s 43,200-second notebook timeout. The log also contains repeated OOM context-budget retries.

[CONFIRMED] Available compute.

Desktop: RTX 4060 Ti, 16 GB VRAM, 48 GB RAM
Laptop: RTX 3060, 6 GB VRAM, 64 GB RAM
Apple Silicon: 16 GB unified memory
Kaggle: 2×T4, quota nearly exhausted
TPU v5e access available for roughly 20 hours; exact topology should be reconfirmed
Apple Silicon support is not yet implemented
TPU Qwen inference is not implemented in current scripts

[CONFIRMED] Remaining project path.

Replace/optimize W2.6-P production inference.
Finish final 4,349×12 report-teacher probabilities/weights.
Run W6.0: same Curia architecture, new supervision only.
Run W6.1: Curia patch-token spatialization.
Consider W6.2 partial Curia fine-tuning only if justified.
Final model selection / ensemble / submission.
2. RESEARCH OBJECTIVE AND RESEARCH QUESTIONS
Primary objective

[CONFIRMED] Build the strongest possible knee MRI multilabel classifier under 58-gold supervision by combining:

strong MRI representations;
report-derived weak supervision;
fold-safe calibration;
challenge-specific report mapping.
Secondary objectives

[CONFIRMED]

Determine whether medical-pretrained MRI representations materially outperform generic backbones.
Determine whether 3D medical representations add complementary information.
Improve report supervision beyond rules/NLI.
Separate clinical report semantics from competition-label semantics.
Prevent leakage while using the 58 gold labels efficiently.
Keep hidden-test inference MRI-only.
Current hypotheses

[CONFIRMED]

Curia-2 is materially stronger than ResNet18 and tested OrthoDiffusion features.
Curia CLS-only representation probably leaves useful spatial information unused.
Report supervision quality is currently a larger bottleneck than another encoder swap.
Literal report extraction does not equal the challenge target.
Few-shot challenge mapping improves labels where W2.3 is weak.
Soft/weighted teacher probabilities are preferable to pretending report labels are ground truth.
Intended contribution

[INFERRED, not yet established as publication-ready]
A defensible research contribution could be:

A fold-safe, challenge-aware report-supervision framework for multilabel MRI classification under extreme label scarcity, coupled with a medical-pretrained image representation and later spatial token aggregation.

Research questions
Representation vs supervision: which is currently limiting?
Which MRI backbone works best with only 58 gold studies?
Can report semantics outperform deterministic rules and NLI?
Can the challenge annotation convention be learned fold-safely from very few gold reports?
Which labels benefit most from LLM challenge mapping?
Does improved report supervision improve Curia image training without architecture changes?
Does Curia patch-token spatial modeling improve localized pathology prediction?
Does partial encoder fine-tuning become useful after supervision improves?
3. DATASET AND TASK DEFINITION
Dataset

[CONFIRMED]
Kaggle: RSNA Knee Abnormality Detection

Training counts
Total studies: 4,407
Fully gold-labeled: 58
Fully unlabeled: 4,349
Partially labeled: 0
All training studies have reports
Train CSV

[CONFIRMED]
train.csv includes:

StudyInstanceUID
PatientSex
Report
12 target columns
Labels
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
Gold positive counts
Label	Positive / 58
ACL	24
MCL	9
Medial Meniscus	26
Lateral Meniscus	23
Medial OA	15
Lateral OA	11
PF OA	21
Effusion	35
Synovitis	27
Baker's	12
Contusion	19
Fracture	18
DICOM hierarchy
train_series/
└── <StudyInstanceUID>/
    └── <SeriesInstanceUID>/
        └── <SOPInstanceUID>.dcm
Series metadata

train_series.csv contains:

StudyInstanceUID
SeriesInstanceUID
Fluid_Sensitive
Fat_Suppression
Anatomical_Plane

Counts:

total series: 24,371
sagittal: 9,864
coronal: 8,609
axial: 5,898
Fluid/FatSupp yes: 14,010
no: 10,361

Gold subset:

336 series
10,528 slices
mean 5.793 series/study
all 58 gold studies contain all 3 anatomical planes
structured plane agreed with geometry 336/336
Important data observations

[CONFIRMED]

SeriesDescription is unreliable.
Repeated acquisitions should not automatically be treated as duplicates.
Pixel spacing varies.
Variable spacing discouraged naive isotropic 3D resampling as the first solution.
All DICOMs are available.
Test structure

[CONFIRMED]

visible test: 3 studies
hidden test: approximately 1,300
hidden test has no reports
Kaggle research-code competition
hidden notebook rerun is offline
model assets/weights must be attached
4. LABEL TAXONOMY
Label	Meaning	Image target	Report target	Challenge mapping	Notes
ACL	Anterior cruciate ligament abnormality	ACL MRI pathology	ACL injury/normality	challenge binary	W2.3 already strong
MCL	Medial collateral ligament abnormality	MCL pathology	MCL injury/normality	challenge binary	only 9 gold positives
Medial Meniscus	Medial meniscus abnormality	tear/morphology	tear vs degeneration/extrusion	challenge-specific	degeneration alone should not imply definite tear
Lateral Meniscus	Lateral meniscus abnormality	tear/morphology	tear vs postop/truncation	challenge-specific	similar issue
Medial OA	Medial tibiofemoral OA	degenerative medial compartment	OA/arthrosis/cartilage loss	challenge-specific	W2.6 major gain
Lateral OA	Lateral tibiofemoral OA	lateral degeneration	OA/arthrosis	challenge-specific	W2.6 gain
PF OA	Patellofemoral OA	patella/trochlea degeneration	PF chondropathy/arthrosis	challenge-specific	effusion is not PF OA
Effusion	Joint effusion	fluid	explicit effusion	challenge may disagree with report	W2.3 reasonably useful
Synovitis	Synovial inflammation	synovial abnormality	synovitis/synovial findings	challenge-specific	effusion alone is not synovitis
Baker's	Baker/popliteal cyst	popliteal cyst	cyst evidence	challenge binary	W2.3 strong
Contusion	Bone contusion/bruise	marrow signal	traumatic bruise vs degenerative edema	challenge-specific	degeneration alone insufficient
Fracture	Fracture	fracture MRI	fracture evidence	challenge-specific	report can disagree with gold
W2.5 state ontology

[IMPLEMENTED]

P = PRESENT
A = ABSENT
U = UNCERTAIN
N = NOT_ADDRESSED

Additional fields:

confidence c
uncertain lean l
related evidence r
severity v
chronicity h
literal evidence e
Crucial distinction

[CONFIRMED]

What the radiology report says
             ≠
What the Kaggle target label is

This was empirically observed and drove W2.6.

5. DATA DIRECTORY / FILE STRUCTURE
Kaggle
/kaggle/input/competitions/rsna-knee-abnormality-detection/
├── train.csv
├── train_series.csv
└── ...

/kaggle/input/datasets/isayem/rsna-w2/rsna_w2/
└── results/
    ├── 04_gold_structured_report_features.csv
    └── 08_full_structured_report_labels.csv

/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3/
└── results/
    ├── 06_fold_safe_unlabeled_soft_labels_long.csv
    └── 10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv

/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct/
└── model files

/kaggle/working/
├── rsna_w2_5_gold_gate/
├── rsna_w2_6_v2/
└── rsna_w2_6p/
    ├── cache/
    │   └── w26p_fs4_production_v1.jsonl
    └── results/
Desired local portable layout
PROJECT_ROOT/
├── input/
│   └── train.csv
├── models/
│   └── qwen2-5-7b-instruct/
└── output/
    └── results/
        ├── rsna_w2/
        ├── rsna_w2_3/
        ├── rsna_w2_6_v2/
        └── rsna_w2_6p/

User explicitly requested code compatible with constructs like:

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "input")
6. COMPLETE FILE REGISTRY
File	Stage	Purpose	Status	Superseded by
train.csv	dataset	studies/reports/labels	current	—
train_series.csv	dataset	series metadata	current	—
labeled_dataset_summary.json	audit	gold statistics	tested	—
rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py	W3.0	cached ResNet baseline	tested	W4 image anchor
rsna_w2_3_fold_safe_pseudo_labels.py	W2.3	fold-safe pseudo labels	tested/current dependency	W2.6 for 4 weak labels
rsna_w2_4_high_fidelity_fold_safe_consensus_v2.py	W2.4	Pilkwang external audit	tested/rejected for production	—
rsna_w5_0_orthodiffusion_spatial_fold_safe_train_v2.py	W5	OrthoDiffusion image branch	tested/stopped	—
rsna_w2_5_own_report_teacher_gold_gate.py	W2.5 V1	own Qwen teacher gate	failed	V2
rsna_w2_5_own_report_teacher_gold_gate_v2.py	W2.5 V2	dual-T4 fallback	failed schema	later versions
rsna_w2_5_own_report_teacher_gold_gate_v3.py	W2.5 V3	auto fallback	failed schema/evidence	V4
rsna_w2_5_own_report_teacher_gold_gate_v4.py	W2.5 V4	strict evidence repair	too brittle	V5
rsna_w2_5_own_report_teacher_gold_gate_v5.py	W2.5 V5	one label/call	still brittle	V6.1
rsna_w2_5_own_report_teacher_gold_gate_v6_1.py	W2.5	grounded nonfatal teacher	tested	diagnostic endpoint
rsna_w2_6_fs4_challenge_mapper.py	W2.6 V1	4-label challenge mapper	OOM	V2 portable
rsna_w2_6_fs4_challenge_mapper_v2_portable.py	W2.6 V2	successful fold-safe FS4	tested/success	W2.6-P
rsna_w2_6p_batched_production_mapper.py	W2.6-P	full production mapper	current but too slow	needs optimized successor
04_gold_structured_report_features.csv	W2	gold report feature rows	current dependency	—
08_full_structured_report_labels.csv	W2	all-study report features	current dependency	—
06_fold_safe_unlabeled_soft_labels_long.csv	W2.3	fold-safe unlabeled labels	current dependency	—
10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv	W2.3	cross-fold mean/std	current dependency	—
stage_b_inner_gold_oof_predictions.csv	W2.3	inner OOF predictions	tested	—
heldout_gold_stage_b_predictions.csv	W2.3	held-out predictions	tested	—
soft_labels_long.csv	W2.3	fold-specific soft labels	tested	—
rsna_w2_full.zip	W2	archive	important	—
rsna_w2_3.zip	W2.3	archive	essential	—
report_labels_v2.csv	external	Pilkwang teacher	audit only	—
w2_4_status_console_log.json	W2.4	audit log	reference	—
w2_4_status_and_inspect_public_console_log.txt	W2.4	public teacher inspection	reference	—
rsna_w2_3_Console_Output.txt	W2.3	metrics/log	reference	—
rsna_w3_1_Console_Output.txt	W3.1	one W3.1 run	reference	—
rsna_w3_1_Console_Output(1).txt	W3.1	second W3.1 run	reference	—
rsna_w2_6_v2.zip	W2.6	successful gold outputs	essential	—
rsna_w2_6p_console_log.txt	W2.6-P	current runtime evidence	essential	—
w26p_fs4_production_v1.jsonl	W2.6-P	append-only partial cache	essential if exported	optimized script should reuse
W2.6-P planned outputs
01_w23_production_base_long.csv
02_fs4_exemplar_audit.csv
03_fs4_prompt_metadata.csv
04_fs4_production_long.csv
05_fs4_probabilities_wide.csv
06_final_hybrid_teacher_long.csv
07_final_hybrid_probabilities_wide.csv
08_recommended_teacher_weights_wide.csv
09_recommended_teacher_mask_wide.csv
10_production_summary.json
11_validation_summary.json
w26p_manifest.json
Important hashes
W2.3 fold checksum:
1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a
W2.4 V2 script SHA:
338a10b07b4cc3843c5455a9dd4020d008c3b702deebc8251d3ff9dae98723e7
W5 V2 script SHA:
d9511fc99ce90d8c4df857da778e59e40b7c2a16848ff90ce4c04466075b591b
W2.5 original SHA:
abfc01cce0d6780776a8dc469935e54d04162ba376b5e81cc19cdfbbc9fe2451
W2.5 V2 SHA:
0148653e5512ea2d7ea00042d8a5aa8d5716cb4c0ce114e9e198bfef3a24adf7
W2.5 V3 SHA:
a78210f6cbcbe9a020f21592c13de03a5788587c2cfe22f444718e46730a2cf4
W2.5 V4 SHA:
ca9b78bfd96506b907aac6b6315a7ace7deae48ac53966c1d2bbe336357247a4
W2.5 V5 SHA:
f3ec74a511082b2579461f0a75272026a8d267ae856d5e756f809358867fdaa6
W2.5 V6.1 SHA:
aec1998b55fad252dc396f0815e061d68e5c6ded84ef80ef2d1d391df5b4a5d4
W2.6 initial script SHA:
df3bd382d29424e3633c9a567abb3afaae6b46fd8a8e84f8e6a4dd36086be4df
W2.6 V2 portable SHA:
5ed1e2ba5c26537c17fb5e52e3eba2c1a85eb306e054ef385aaba45cdcf0bc07
W2.6-P SHA:
eb9190e36bf30e95b4e5b383bdc6757245c588e2cebda268a179aff78a9fb535
7. IMAGE PIPELINE
General hierarchy
Study
  ↓
multiple series
  ↓
slices / 3D windows
  ↓
series encoder
  ↓
series representation
  ↓
study aggregation
  ↓
12-label output
DICOM preprocessing

[CONFIRMED]

DICOM per series/study
plane metadata available
plane labels validated against geometry
SeriesDescription unreliable
spacing variable
repeated acquisitions not guaranteed duplicate
variable series count

UNKNOWN — verify exact image scripts

exact DICOM intensity normalization
photometric inversion handling
exact slice sorting
resize interpolation
corrupted slice policy
channel normalization

Do not invent these in a new chat.

ResNet image branch

General structure:

Study
  └─ Series
      └─ sampled slices
          ↓
        ResNet18
          ↓
      attention / aggregation
  ↓
study aggregation
  ↓
12 labels
V1.1

Frozen ResNet18 + metadata:

AUROC .54655
AP .42145
F1 .45195
V2.1

Fine-tune layer4:

.4800/.3781/.3978
worse
V3.1

Position feature:

.4973/.3699/.3843
worse
V4

No metadata:

.54577/.43250/.40716
W3.0

rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py

AUROC .5459179
AP .433128
F1 .407160
LB ~.608
W3.1

Weak supervision:

retained project result .523402
another console log showed .535247
discrepancy is UNKNOWN
both worse than W3.0
hybrid LB ~.610
Public DINOv2 baseline evidence

External/public strong notebook:

DINOv2-S family
20 members
12 transformer blocks
last 6 trainable
10.7M trainable parameters
feature dim 768
336 px
12 slices
6 slots
rank mean
rank loss weight .05
public ~.891
enhanced ~.899
another documented pipeline ~.903

Used as evidence, not target architecture.

Curia W4

Model:
raidium/curia-2

Known:

~86.1M params
DINOv2-family medical CT/MRI
512 input
patch16
12 layers
hidden768
CLS = last_hidden_state[:,0]

W4:

frozen encoder
cached one CLS embedding per series
hierarchical series→study transformer

Metrics:

gold .6206400750
weak .6410767023
gated .6523168374

Per-label gated:

Label	AUROC
ACL	.72794
MCL	.59184
Medial Meniscus	.57212
Lateral Meniscus	.69068
Medial OA	.73798
Lateral OA	.62669
PF OA	.58044
Effusion	.80621
Synovitis	.60335
Baker's	.59601
Contusion	.56815
Fracture	.72639

LB:

gated ~.717
weak ~.721

Critical limitation: only CLS used; patch tokens discarded. This motivates W6.1.

OrthoDiffusion W5

Model:
lt-0123/OrthoDiffusion

3 orientation-specific 3D diffusion U-Nets
frozen
timestep t100
mid2 feature

Cache:

final 4,407 studies
24,094 series encoded
71,277 windows
8.823 GB
short-series repair implemented

Metrics:

gold .5775082
weak .5985844750
W4+W5 50/50 = .642307
gain +.001231 only

REJECTED: stop Ortho branch.

8. REPORT / NLP PIPELINE
W1/W2

W1.2:

exact rule / 7-state
Stage-A accuracy .35

W2 NLI model:
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

NLI label mapping:

entailment 0
neutral 1
contradiction 2

Sanity:

fracture positive ~.9945
“No fracture” negative ~.9975

Stage-A:

semantic only .241667
hybrid .408333
binary polarity hybrid .766667

W2 all58 Stage-B:

macro .759289
W2 Stage-B exact features
EvidencePositiveScore
EvidenceNegativeScore
RelatedScore
UncertaintyFlag
FusedAssertionConfidence
RuleDecidableFlag
SemanticDirectStrength
SemanticDirectMargin
SeverityLow
SeverityModerate
SeverityHigh
SeverityDegenerative
W2.3

Script:
rsna_w2_3_fold_safe_pseudo_labels.py

Outer-fold procedure:

outer-train gold only
inner repeated stratified OOF Stage-B calibration
gates learned from outer-train only
fit mapper
predict all 4,349 unlabeled
held-out gold diagnostic only

Results:

AUROC .7712431043
AP .6832265162
Brier .1590722247

Strong:
ACL, MCL, both menisci, Effusion, Baker's, Fracture; Contusion partial.

Weak:
Medial OA, Lateral OA, PF OA, Synovitis.

W2.4 external audit

rsna_w2_4_high_fidelity_fold_safe_consensus_v2.py

Used Pilkwang:
report_labels_v2.csv

Results:

W2.3 .771243
W2.4 .8498568812
Pilkwang .8700395897

W2.4:

AP .771932
Brier .151266

Bootstrap delta:

+.07853
CI [.04302,.11192]
P>0 = 1

Decision: benchmark only. Not production.

W2.5 own Qwen teacher

Model:
Qwen2.5-7B-Instruct

State schema:

s P/A/U/N
c confidence
l lean
r related
v severity
h chronicity
e evidence
Evolution

V1:

4-bit failed because Kaggle bitsandbytes incompatible
FP16 fallback loaded one T4 and OOMed

V2/V3:

dual-T4 balanced FP16 solved model-loading memory
schema errors surfaced

V4:

strict evidence
repair prompt
still brittle
exposed cross-target contamination

V5:

one target per generation
still evidence failure

V6.1:

literal-target relevance checks
deterministic evidence recovery
unsupported P/A/U downgraded to N
semantic guards for meniscus/OA/synovitis
completed 58-gold gate

W2.5 final:

W2.3 .771243
Qwen raw .713218
Qwen-only .664891
Qwen+W2 .789624
delta +.018380
verdict DO_NOT_FULL_EXTRACT_YET

Reason:
Qwen report interpretation had signal, but not enough calibration/coverage.

W2.6 FS4

Labels:

Medial OA
Lateral OA
PF OA
Synovitis

Each gold held-out query:

outer-train only
TF-IDF retrieval
2 positive exemplars
2 negative exemplars
Qwen challenge probability
query gold never supplied

232 calls.

Results:

W2.3 .77124
direct replace .81342
fixed50 .80929

Fixed50:

delta +.03804
CI ~[.0120,.0673]
P>0 ~.9987

Brier:

W2.3 .15907
replace .15245
fixed50 .15008

Production selection:

8 labels → W2.3
4 weak labels → 0.50 W2.3 + 0.50 FS4
W2.6-P

All 58 gold used as final production exemplar pool.

For each unlabeled × weak label:

TF-IDF query-specific retrieval
2 pos + 2 neg gold examples
W2 structured summary
Qwen probability

Cells:
4,349 × 4 = 17,396

Current implementation is too slow.

9. MULTIMODAL / MULTI-MODEL FUSION
Implemented
Image + metadata

V1.1 included metadata. Removing metadata did not materially hurt AUROC.

Rule + NLI

W2 combines deterministic and semantic report signals.

Qwen + W2

W2.5 logistic fusion:

best own result .789624
W2.6 report fusion

Four weak labels:

$$ P = 0.5P_{W2.3}+0.5P_{FS4} $$

Eight labels:

$$ P=P_{W2.3} $$
W4 + W5 image blend

50/50:

.642307
only +.001231
Planned
Report teacher → MRI student

Reports supervise image training but disappear at test.

Curia spatial fusion

Patch tokens + plane/series structure + study transformer.

Partial fine-tuning

Only after spatial frozen model.

Full image/report test-time fusion

Not appropriate because hidden reports do not exist.

10. MODEL ARCHITECTURE EVOLUTION
Version	Architecture/change	Result	Status
V1.1	frozen ResNet18 + metadata	.54655	superseded
V2.1	layer4 FT	.4800	rejected
V3.1	positional info	.4973	rejected
V4	no metadata	.54577	baseline evidence
W3.0	cached ResNet reproduction	.545918	baseline
W3.1	report weak supervision	.523402 retained	negative
W2	rule+NLI Stage-B	.759289	report baseline
W2.3	fold-safe report Stage-B	.771243	key baseline
W4	frozen Curia CLS	.641 weak/.652 gated	image anchor
W5	OrthoDiffusion	.5986	stopped
W2.4	external Pilkwang consensus	.8499	audit only
W2.5	own Qwen extraction	.7896 best	diagnostic
W2.6	four-label challenge mapper	.8093 fixed50	successful
W2.6-P	full production teacher	incomplete	current blocker
W6.0	same Curia + W2.6 teacher	UNKNOWN	next
W6.1	Curia patch spatial	UNKNOWN	planned
W6.2	partial Curia FT	UNKNOWN	deferred
11. TRAINING STRATEGY
Folds

LOCKED

5-fold greedy multilabel
UID sorted
seed 42
hash:
1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a
Report calibration

Known:

StandardScaler
logistic regression
W2.5 C=.03
outer-fold training only
W2.6 retrieval
2 positive examples
2 negative examples
TF-IDF similarity
fold-safe in validation
all 58 only in production
Image training

UNKNOWN — verify canonical image scripts
Exact:

optimizer
LR
weight decay
epochs
augmentation
batch size
gradient accumulation
scheduler

Do not reconstruct these from generic assumptions.

12. LEAKAGE PREVENTION
What counts as leakage
held-out gold labels used in calibration
held-out gold examples used as few-shot labeled exemplars
fitting retrieval/mapper on validation gold
using all-58 production labels and then claiming pristine 58 OOF
using external Pilkwang teacher as own leakage-free truth
repeatedly tuning prompts directly to all 58 gold outcomes
Fixed folds

All controlled experiments reuse the exact 5-fold assignment.

W2.3

Outer-train only for:

Stage-B calibration
gate thresholds
mapper fitting

Held-out:

diagnostic only
W2.5
Qwen sees report only
gold target never in prompt
Stage-B outer-train only
first three prompt-development gold cases were later excluded in a 55-study sensitivity check
W2.6
query held out
exemplar pool outer-train only
retrieval outer-train only
query gold unknown to prompt
fixed blend predeclared
W2.6-P

Using all 58 is acceptable only for final unlabeled training-resource generation.

After W2.6-P:

Do not claim downstream 58-study OOF as pristine held-out evidence if the image model trained on pseudo labels derived from all 58.

External teacher

Pilkwang stays outside production.

13. OUT-OF-FOLD PIPELINE
Fix 5 folds.
For each outer fold:
train/calibrate on outer-train gold
build retrieval/index only from outer-train
predict outer-held-out gold
Concatenate held-out predictions.
Compute macro/per-label AUROC, AP, Brier.
Bootstrap paired deltas.
Only after method selection use all 58 to generate final unlabeled training resource.

OOF is necessary because 58 examples are too few to tolerate optimistic self-calibration.

14. EXPERIMENT REGISTRY
Experiment	Change	Result	Interpretation
V1.1	frozen ResNet + metadata	.54655	weak baseline
V2.1	FT layer4	.4800	overfit/degraded
V3.1	position	.4973	no gain
V4	no metadata	.54577	metadata low value
W3.0	cached ResNet	.545918	stable baseline
W3.1	weak labels	.523402	negative
W2	report Stage-B	.759289	report signal strong
W2.3	fold-safe report	.771243	stronger
W4	Curia	.652317 gated	best image CV
W5	Ortho	.598584 weak	inferior
W4+W5	blend	.642307	negligible gain
W2.4	external teacher consensus	.849857	report ceiling evidence
W2.5 raw	Qwen extraction	.713218	insufficient
W2.5 Qwen+W2	hybrid	.789624	complementary
W2.6 replace	FS4	.81342	strong
W2.6 fixed50	FS4 blend	.80929	production choice
W2.6-P	full unlabeled inference	incomplete	runtime blocker
15. RESULTS
Image
Experiment	AUROC	AP	F1
V1.1	.54655	.42145	.45195
V2.1	.4800	.3781	.3978
V3.1	.4973	.3699	.3843
V4	.54577	.43250	.40716
W3.0	.5459179	.433128	.407160
W3.1 retained	.523402	.426204	.341625
W4 gold	.620640	UNKNOWN	UNKNOWN
W4 weak	.641077	UNKNOWN	UNKNOWN
W4 gated	.652317	UNKNOWN	UNKNOWN
W5 gold	.577508	UNKNOWN	UNKNOWN
W5 weak	.598584	UNKNOWN	UNKNOWN
W4+W5	.642307	UNKNOWN	UNKNOWN
Report
Model	AUROC	AP	Brier
W2 full	.759289	UNKNOWN	UNKNOWN
W2.3	.771243	.683227	.159072
W2.4 external	.849857	.771932	.151266
Pilkwang external	.870040	.758743	.146249
W2.5 raw Qwen	.713218	.545824	.219601
W2.5 Qwen-only	.664891	.543236	.202691
W2.5 Qwen+W2	.789624	.689152	.173704
W2.6 replace	.81342	.71643 approx	.15245 approx
W2.6 fixed50	.80929	.71923 approx	.15008 approx
W2.3 per label
ACL .926471
MCL .950113
Medial Meniscus .844351
Lateral Meniscus .815528
Medial OA .707752
Lateral OA .659574
PF OA .749678
Effusion .693168
Synovitis .549582
Baker's .830616
Contusion .703779
Fracture .824306
W2.6 weak-label gains
Label	W2.3	Replace	Fixed50
Medial OA	.7078	.8597	.8434
Lateral OA	.6596	.7282	.7244
PF OA	.7497	.8578	.8468
Synovitis	.5496	.7270	.7085
LB/public context
W3.0 ~.608
W3 hybrid ~.610
W4 gated ~.717
W4 weak ~.721
public strong notebooks ~.891–.903
top observed around ~.952
16. ERROR ANALYSIS
Report/gold mismatch

Literal report findings sometimes disagree with challenge targets.

Observed examples:

explicit joint effusion but gold Effusion = 0
osteochondral fracture language but Fracture = 0

Hence Stage A ≠ challenge target.

Cross-label contamination

All-12 Qwen output incorrectly used:

effusion as PF OA evidence
effusion as synovitis evidence
unsupported ACL/MCL absence
meniscal truncation as definite tear

Resolved by label-wise prompting.

Grounding coverage

Strict grounding reduced hallucinations but created many N states. Many N were challenge-positive.

Lesson:
N != challenge-negative.

Image side
ResNet fine-tuning overfit
simple position feature did not help
Curia materially stronger
Ortho complement weak
CLS-only Curia likely spatially incomplete
17. FAILED APPROACHES AND LESSONS
ResNet fine-tuning

Why: adaptation
Result: worse
Lesson: tiny gold insufficient
Revisit: only with stronger supervision, likely Curia not ResNet

Simple positional feature

Why: slice order
Result: worse
Lesson: position needs richer spatial architecture

W3.1 weak labels

Why: report supervision
Result: worse
Lesson: noisy teacher + weak representation can hurt

OrthoDiffusion

Why: 3D spatial complement
Result: tiny blend gain
Lesson: stop branch

W2.4 production

Why: very high report quality
Result: externally strong
Rejected: provenance/user requirement

W2.5 Kaggle 4-bit

bitsandbytes incompatibility.

W2.5 single-T4 FP16

7B weights fill T4.

W2.5 strict retry

same deterministic output repeated.

W2.5 all-label prompt

cross-target contamination.

W2.5 generic Qwen teacher

only +.018 over W2.3 and worse calibration.

W2.6 V1

6144-token prompts OOMed.

W2.6-P

faithful scaling of W2.6 is computationally impractical.

18. DECISION LOG
Decision	Final choice	Reason
Folds	exact fixed 5-fold	attribution
Image anchor	Curia	strongest
Ortho	stop	negligible gain
Report baseline	W2.3	fold-safe
Pilkwang	audit only	own-teacher policy
W2.5 prompt	label-wise	avoid contamination
W2.5 full corpus	stop	insufficient gain
W2.6 labels	4 weak labels only	W2.3 weakness
W2.6 production	fixed50	better calibration
Hidden reports	no test usage	unavailable
Current Qwen machine	prefer local 4060Ti	Kaggle quota
TPU Qwen	defer	implementation cost
Next image run	W6.0 before spatial	isolate supervision
19. REJECTED / DEFERRED IDEAS
Rejected
Generic ResNet FT
current Ortho branch
Pilkwang production
W2.5 full extraction
more broad report prompt experimentation
test-time report fusion
Deferred
Curia spatial patch tokens
partial Curia fine-tuning
final Ortho ensemble
TPU image acceleration
advanced teacher consistency
Current
optimized W2.6-P
Apple Silicon support
partial-cache reuse
possible report-teacher distillation
faster local inference runtime
20. KNOWN BUGS / UNRESOLVED ISSUES
Issue	Symptoms	Current status	Next step
W2.6-P too slow	12h only 6768 cells	blocker	redesign
OOM retry waste	repeated 3584→1536	inefficient	token binning/lower budget
Kaggle timeout	43200s	confirmed	move off Kaggle/reuse cache
4060Ti ETA	~48h	unacceptable	reduce prefill/work
3060 stability	VRAM failure	not useful	deprioritize
Apple support	absent	requested	implement MPS/MLX
TPU Qwen	absent	deferred	only if worthwhile
Partial cache	may contain 6768 cells	availability unknown	export/merge
W6	not started	downstream blocker	after production
W3.1 duplicate values	.535 vs .523	unresolved	low priority
21. CURRENT CODE STATE
Obsolete

W2.5 V1–V5, W2.6 V1, older ResNet variants.

Baselines
W3.0
W2.3
W4 Curia
Completed experimental
W5
W2.4
W2.5 V6.1
W2.6 V2
Current

rsna_w2_6p_batched_production_mapper.py

Important: current, but not recommended to rerun unchanged.

Dependency graph
train.csv
   │
   ├── reports
   │     ↓
   │    W2
   │     ↓
   │    W2.3
   │     ↓
   │    W2.6
   │     ↓
   │    W2.6-P  ← CURRENT BLOCKER
   │     ↓
   │  final pseudo labels
   │
   └── DICOM
         ↓
       Curia W4
         ↓
       W6.0
         ↓
       W6.1
         ↓
      optional W6.2
22. CURRENT ARCHITECTURE — END TO END
Training
58 gold challenge labels
          │
          ├─────────────────────────────────┐
          │                                 │
          ▼                                 │
W2 rule + multilingual NLI                  │
          │                                 │
          ▼                                 │
structured report features                  │
          │                                 │
          ▼                                 │
        W2.3                                │
          │                                 │
          ├── strong 8 labels               │
          │                                 │
          └──────► W2.6 FS4 Qwen ◄─────────┘
                    │
                    │ weak 4 labels
                    ▼
       0.5 W2.3 + 0.5 FS4
                    │
                    ▼
          report teacher targets
                    │
                    ▼
MRI DICOM → Curia → study aggregator → 12 outputs
                    ▲
                    │
               weighted loss
Hidden inference
MRI only
  ↓
series preprocessing
  ↓
Curia-based encoder
  ↓
study aggregation
  ↓
12 probabilities
23. DATA FLOW
Training
Read train CSV.
Split 58 gold / 4,349 unlabeled.
Generate/read W2 structured report features.
Read W2.3 fold-safe probabilities.
Validate W2.6 fold-safely.
Freeze production rule.
Generate final W2.6-P unlabeled FS4 probabilities.
Assemble 4,349×12 teacher matrix.
Train Curia image student with gold + weighted soft targets.
Compare W6 variants.
Inference
Load DICOM.
build series features.
aggregate study.
predict 12 labels.
no reports.
24. CURRENT RESEARCH HYPOTHESIS
Curia is the correct current image foundation.
Report challenge mapping contains useful supervision unavailable to W2.3.
Better weak labels should improve Curia more than another immediate backbone swap.
OA and synovitis need challenge mapping, not literal NLP.
Reports should train the MRI model, not be required at test.
Spatial Curia tokens are the next image-side opportunity.
Controlled one-variable-at-a-time experiments are more valuable than model shopping.
25. SCIENTIFIC VALIDITY CONCERNS
Leakage

Must preserve exact folds.

Privileged information

Reports are downstream expert interpretations of the image. They are acceptable as training privileged information only if this is disclosed.

Circularity

All-58 production pseudo labels invalidate pristine reuse of the same 58 as a clean downstream test.

External contamination

Pilkwang is benchmark only.

Prompt development

Three W2.5 sample gold cases were inspected. A 55-study sensitivity analysis supported robustness.

Reproducibility

Preserve:

fold checksum
prompt versions
model config hashes
W2 hashes
cache prompt SHA
exact production rule
26. PUBLICATION / THESIS IMPLICATIONS

[INFERRED/PROPOSED]

Potential claims:

challenge-aware report supervision improves low-label MRI training;
Curia medical representations outperform generic tested image models;
challenge mapping is distinct from literal finding extraction;
targeted LLM supervision helps the labels where rule/NLI methods fail.

Do not yet claim:

SOTA
spatial Curia improvement
fine-tuning improvement
general superiority beyond this dataset

Required ablations:

W4 old teacher vs W6.0 new teacher
W6.0 CLS vs W6.1 spatial
frozen vs partial FT
W2.3 vs W2.6 report mapping
per-label + bootstrap uncertainty
27. NEXT EXPERIMENTS
Priority 0 — W2.6-P production redesign
Blocker

17,396 long Qwen calls are too expensive.

Requirements for successor

Create something like:

rsna_w2_6p_fast_production_mapper.py

It should:

reuse w26p_fs4_production_v1.jsonl
merge multiple partial caches
support CUDA and Apple Silicon
avoid repeated OOM retries
benchmark first 200–500 cells
print projected full runtime
abort if projected time is unacceptable
preserve W2.6 semantics as much as possible
Proposed optimization directions
A. Compress prompts aggressively

Keep:

target
challenge definition
2 pos / 2 neg information
query report
compact W2 features

Remove:

verbose narrative
unnecessary example details
long reason/state output

Potential output:

{"p":0.731}
B. Use one fixed context budget

Repeated failed token budgets waste minutes.

Bucket prompts by token length and choose a viable budget directly.

C. Faster inference backend

Benchmark:

HF Transformers NF4
optimized CUDA backend if compatible
Apple MLX quantized inference
D. Distill Qwen production mapper

If partial 6,768-cell Kaggle cache is recoverable:

train lightweight student from W2/text/retrieval features to Qwen probability
validate against existing 232 W2.6 gold OOF
use student for remaining cells

This may be the fastest path.

E. Fixed exemplar / prefix-cache experiment

Current query-specific exemplars prevent prefix reuse.

Test one short fold-safe experiment:

fixed 2+/2− exemplars per label
compare to W2.6 gold
if performance remains close, cache prefix KV and production becomes far cheaper
Stopping rule

If optimized production still projects >single-digit hours, stop direct full Qwen scaling and use distillation/approximation.

Priority 1 — W6.0

Same Curia architecture as W4.

Only change:

old weak teacher → final W2.6 teacher.

Purpose:
isolate supervision gain.

Baseline:

W4 weak .641077
W4 gated .652317
Priority 1 — W6.1

Curia patch-token spatial model.

Hypothesis:
CLS discards localized pathology.

Priority 2 — W6.2

Partial last-block Curia fine-tuning.

Only after W6.1.

Priority 2 — final ensemble

Only blend models with actual residual complementarity.

28. IMMEDIATE NEXT ACTION

In the fresh chat, do this first:

Inspect:
rsna_w2_6p_batched_production_mapper.py
rsna_w2_6p_console_log.txt
rsna_w2_6_fs4_challenge_mapper_v2_portable.py
rsna_w2_6_v2.zip
any w26p_fs4_production_v1.jsonl
Do not rerun current production script unchanged.
Determine:
prompt token distribution
OOM retry frequency
prefill vs decode time
per-label runtime
exact reusable cache count
Build one optimized production script with:
local_gpu
kaggle_t4
apple_mps and/or MLX
cache merge/resume
compact prompt
minimal output
token-length buckets
runtime projection
Validate the optimized mapper on the existing 232 W2.6 gold cells before committing full production.
Finish production labels.
Go directly to W6.0.
29. QUESTIONS STILL UNANSWERED
Can W2.6 be compressed without losing its .809–.813 gold-gate advantage?
Is the Kaggle 6,768-cell production cache available for export?
Which backend is fastest for Qwen2.5-7B on the 4060 Ti?
Is Apple MPS practical, or should MLX be used instead?
Can fixed exemplars preserve W2.6 quality?
Can a lightweight student reproduce FS4 probabilities?
How much does new supervision improve unchanged Curia?
How much do Curia patch tokens improve performance?
Does partial Curia FT become beneficial afterward?
What are the exact canonical W4 script/cache filenames?
30. TERMINOLOGY / GLOSSARY

OOF — out-of-fold.

Gold — 58 challenge-labeled studies.

Teacher — report-side soft-label generator.

Student — MRI image model.

P/A/U/N — present / absent / uncertain / not addressed.

Stage A — clinical report interpretation.

Stage B — challenge probability calibration/mapping.

W2 — rule + multilingual NLI report pipeline.

W2.3 — fold-safe report teacher baseline.

W2.4 — Pilkwang external consensus audit.

W2.5 — independent Qwen report reader.

W2.6 — four-label few-shot challenge mapper.

FS4 — Medial OA, Lateral OA, PF OA, Synovitis.

W2.6-P — production mapper for 4,349 unlabeled reports.

Curia/W4 — current image anchor.

Ortho/W5 — stopped OrthoDiffusion branch.

W6.0 — supervision-only Curia update.

W6.1 — spatial Curia.

W6.2 — optional partial Curia fine-tuning.

Fixed50 — 50% W2.3 + 50% FS4.

31. DO NOT FORGET THESE POINTS
Only 58 gold studies.
Exact fold hash is locked.
Hidden test has no reports.
Reports are training-only teacher data.
W2.3 = .771243.
W2.6 fixed50 = ~.80929.
W2.6 improves exactly the four W2.3-weak labels.
Eight other labels stay W2.3 in production.
Pilkwang stays outside own production.
W2.5 generic Qwen work is closed.
Curia is the current image anchor.
Ortho branch is stopped.
Curia currently uses CLS only.
W2.6-P current script is too slow.
Kaggle stopped at ~6,768/17,396 after 12h.
Repeated OOM retries are wasting time.
Export partial production cache if possible.
Desktop 4060Ti is the main local CUDA system.
Laptop 3060 is not attractive for Qwen production.
Apple Silicon support is explicitly requested.
TPU time should probably be preserved for W6.
After production, run W6.0 before W6.1.
Literal report finding != challenge label.
N does not equal challenge-negative.
All-58 production labels are not pristine OOF.
Finish quickly; do not reopen broad model exploration.
32. COPY THIS INTO A NEW CHAT
Project continuation context

We are working on the Kaggle RSNA Knee Abnormality Detection competition. The goal is a high-performing MRI-only hidden-test model under extreme label scarcity. The training set has 4,407 studies, but only 58 are fully labeled. The remaining 4,349 have no challenge labels. All 4,407 training studies have radiology reports. Hidden test has no reports, so reports are being used only as privileged training-time supervision.

The 12 labels are:
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

Gold positives among 58:
ACL 24, MCL 9, Medial Meniscus 26, Lateral Meniscus 23, Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12, Contusion 19, Fracture 18.

DICOM layout:
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm.

train_series.csv has:
StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane.

There are 24,371 series:
9,864 sagittal, 8,609 coronal, 5,898 axial.
The 58 gold studies have 336 series and 10,528 slices, mean 5.793 series/study. All gold studies have all three planes. Structured plane agreed with geometry for 336/336 gold series. SeriesDescription is unreliable. Pixel spacing varies and repeated acquisitions cannot be assumed duplicate.

The exact fold assignment is locked:
5-fold greedy multilabel, StudyUID sorted, seed 42.
Checksum:
1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a

Do not regenerate folds.

Image history

Early ResNet experiments:

V1.1 frozen ResNet18 + metadata: .54655/.42145/.45195 AUROC/AP/F1
V2.1 layer4 fine-tune: .4800/.3781/.3978
V3.1 position: .4973/.3699/.3843
V4 no metadata: .54577/.43250/.40716
W3.0 rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py: .5459179/.433128/.407160
W3.1 weak supervision retained .523402, worse than W3.0
W3.0 public LB ~.608, hybrid ~.610

Curia W4 is the current image anchor:
model raidium/curia-2, medical DINOv2-family, ~86.1M params, 512 input, patch16, 12 layers, hidden768, CLS=last_hidden_state[:,0].

We froze Curia, cached one series CLS embedding, and trained a hierarchical study model.

W4:

gold .6206400750
weak .6410767023
gated .6523168374

Per-label gated:
ACL .72794
MCL .59184
Medial Meniscus .57212
Lateral Meniscus .69068
Medial OA .73798
Lateral OA .62669
PF OA .58044
Effusion .80621
Synovitis .60335
Baker .59601
Contusion .56815
Fracture .72639.

LB roughly .717 gated / .721 weak.

Important limitation: W4 discarded Curia patch tokens and used only CLS. W6.1 should later spatialize Curia.

OrthoDiffusion W5:
lt-0123/OrthoDiffusion, 3 orientation-specific 3D diffusion U-Nets, frozen t100/mid2 features.
Gold .5775082, weak .5985844750.
W4+W5 50/50 = .642307, only +.001231.
Stop Ortho branch.

Report pipeline

W2 semantic model:
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli.

W2 Stage-A:
rule exact7state .35
semantic-only .241667
hybrid .408333
binary polarity hybrid .766667.

W2 full Stage-B all58 macro .759289.

W2 exact Stage-B features:
EvidencePositiveScore
EvidenceNegativeScore
RelatedScore
UncertaintyFlag
FusedAssertionConfidence
RuleDecidableFlag
SemanticDirectStrength
SemanticDirectMargin
SeverityLow
SeverityModerate
SeverityHigh
SeverityDegenerative.

Important W2 files:
04_gold_structured_report_features.csv
08_full_structured_report_labels.csv.

W2.3 script:
rsna_w2_3_fold_safe_pseudo_labels.py.

For each outer fold:

outer-train gold only
inner OOF calibration
fold-specific gates
held-out gold diagnostic only
unlabeled pseudo labels produced fold-safely

W2.3:
AUROC .7712431043
AP .6832265162
Brier .1590722247.

Per-label:
ACL .926471
MCL .950113
Medial Meniscus .844351
Lateral Meniscus .815528
Medial OA .707752
Lateral OA .659574
PF OA .749678
Effusion .693168
Synovitis .549582
Baker .830616
Contusion .703779
Fracture .824306.

W2.3 was strong for ACL/MCL/menisci/Effusion/Baker/Fracture and partial Contusion. It was weak/no-gate for Medial OA, Lateral OA, PF OA, Synovitis.

W2.4 was an external audit:
rsna_w2_4_high_fidelity_fold_safe_consensus_v2.py
using Pilkwang report_labels_v2.csv.

W2.4 .849857.
Pilkwang .870040.

This was useful ceiling evidence only. Pilkwang must not enter our own production teacher.

W2.5 own Qwen teacher

Model:
Qwen2.5-7B-Instruct.

We went through V1–V6.1:

Kaggle 4-bit failed due bitsandbytes incompatibility.
single-T4 FP16 OOM.
dual-T4 sharded FP16 worked.
schema/evidence failures appeared.
all-12 prompt caused cross-label evidence contamination.
moved to one-label-per-generation.
V6.1 used grounded/nonfatal evidence logic.

State ontology:
P present
A absent
U uncertain
N not addressed.

Additional:
confidence, lean, related evidence, severity, chronicity, literal evidence.

Final W2.5:
W2.3 .771243
Qwen raw .713218
Qwen-only .664891
Qwen+W2 .789624
delta +.01838.
Do not full-extract W2.5.

Key lesson:
literal report interpretation is not the competition label.
Examples were seen where reports explicitly mentioned effusion/fracture but gold was 0.
Also many grounded N states were challenge-positive.

W2.6 FS4

Successful controlled script:
rsna_w2_6_fs4_challenge_mapper_v2_portable.py.

FS4 labels:
Medial OA, Lateral OA, PF OA, Synovitis.

For each held-out gold query:

TF-IDF built from outer-train gold
retrieve 2 positives +2 negatives
query gold unavailable to model
Qwen predicts challenge probability

232 calls total.

Results:
W2.3 .77124
FS4 replace .81342
fixed50 .80929.

Fixed50 delta ~+.03804.
95% CI approximately [.0120,.0673].
P(delta>0) ~.9987.

Per-label fixed50:
Medial OA .8434 vs .7078
Lateral OA .7244 vs .6596
PF OA .8468 vs .7497
Synovitis .7085 vs .5496.

Direct replace had slightly higher AUROC, but fixed50 had better Brier/AP:
W2.3 Brier .15907
replace .15245
fixed50 .15008.

Production rule is locked:

8 established labels = W2.3
4 weak labels = 0.50 W2.3 + 0.50 FS4

Do not do more broad report-prompt research.

W2.6-P production — current blocker

Current script:
rsna_w2_6p_batched_production_mapper.py

Purpose:
generate FS4 probabilities for all 4,349 unlabeled studies:
4,349×4 = 17,396 Qwen cells.

Production uses all 58 gold reports as exemplar pool.
For each query/label:

TF-IDF query-specific retrieval
2 nearest positive examples
2 nearest negative examples
W2 structured report summary
Qwen challenge probability

Final teacher:
8 labels W2.3 cross-fold mean
4 labels 0.5 W2.3 + 0.5 FS4.

Planned output files:
01_w23_production_base_long.csv
02_fs4_exemplar_audit.csv
03_fs4_prompt_metadata.csv
04_fs4_production_long.csv
05_fs4_probabilities_wide.csv
06_final_hybrid_teacher_long.csv
07_final_hybrid_probabilities_wide.csv
08_recommended_teacher_weights_wide.csv
09_recommended_teacher_mask_wide.csv
10_production_summary.json
11_validation_summary.json
w26p_manifest.json.

Cache:
w26p_fs4_production_v1.jsonl.

The current production implementation is too slow.

Kaggle 2×T4:

batch size 8
max context 3584
retry contexts 3072/2560/2048/1536
7B FP16 balanced across both GPUs
reached ~6,768/17,396 after 12h
rate degraded to ~9.4 cells/min
remaining ETA ~18.8h
cell timed out at 43,200s.

The log shows many batches performing several failed OOM attempts before eventually succeeding at a shorter context, wasting significant time.

Desktop 4060Ti 16GB +48GB RAM also projects ~48+ hours even with batch 8/16/32.
Laptop 3060 6GB +64GB eventually runs into VRAM issues.
Apple Silicon 16GB exists but current script has no Apple backend.
Kaggle T4 quota is almost gone.
TPU v5e access exists ~20h; current script deliberately does not implement Qwen TPU inference.

Immediate next task

Do not rerun W2.6-P unchanged.

Inspect:

rsna_w2_6p_batched_production_mapper.py
rsna_w2_6p_console_log.txt
rsna_w2_6_fs4_challenge_mapper_v2_portable.py
rsna_w2_6_v2.zip
any exported w26p_fs4_production_v1.jsonl

Build an optimized production script.

It must:

reuse/merge old cache;
support local CUDA;
support Apple Silicon via MPS and/or MLX;
optionally keep T4 support;
drastically reduce prompt/prefill cost;
avoid repeated OOM ladders;
produce minimal output, preferably only probability;
sort/bucket by token length;
benchmark first 200–500 cells;
print projected full runtime;
auto-stop if runtime is still unacceptable.

Strong candidate optimization directions:

much shorter prompt;
compressed exemplar reports;
compressed W2 summaries;
output {"p":...} only;
fixed lower token budget;
faster inference backend;
Apple MLX quantized Qwen;
fixed exemplar/prefix-cache strategy validated on 232 gold cells;
teacher distillation using partial production Qwen cache.

If the Kaggle cache containing roughly 6,768 cells can be exported, do not recompute those cells. Merge by UID/Label plus prompt SHA.

Before full production, verify any optimized approximation against the existing W2.6 gold 232-cell benchmark so challenge-mapping performance is not silently destroyed.

If full direct Qwen inference still projects above single-digit hours, stop scaling it directly and distill/approximate the teacher.

W6 after production

W6.0:
same frozen Curia CLS architecture as W4, but train with the new W2.6 production teacher.
Purpose: isolate supervision gain.

W6.1:
Curia spatial patch-token model.

W6.2:
partial Curia last-block fine-tuning only if W6.1 warrants it.

Do not reopen Ortho or broad report-model experimentation.

Leakage constraints

Use exact fixed folds for controlled experiments.

Held-out gold never calibrates itself.

W2.6 gold retrieval examples are outer-train only.

Production all-58 labels are final training resources only.

A downstream model trained on all-58-derived pseudo labels cannot claim pristine OOF against those same 58.

Pilkwang stays outside own production.

Hidden test has no reports.

Hardware
RTX 4060 Ti 16GB +48GB RAM — preferred local CUDA machine.
RTX 3060 6GB +64GB RAM — not preferred.
Apple Silicon 16GB unified memory — support explicitly requested.
Kaggle 2×T4 — quota nearly exhausted.
TPU v5e access ~20h — probably preserve for W6 image compute.
Working style

Finish quickly.
Avoid unnecessary branches.
Use controlled apples-to-apples experiments.
Do not reopen already rejected approaches.
The next code should solve the W2.6-P runtime blocker, finish final teacher labels, then move immediately to W6 Curia.

33. FILES TO UPLOAD TO THE NEW CHAT
Essential
rsna_w2_6p_batched_production_mapper.py

Current production implementation. Needed to optimize rather than rebuild from memory.

rsna_w2_6p_console_log.txt

Shows actual throughput, OOM behavior, and 12h timeout.

w26p_fs4_production_v1.jsonl

Upload if you can export it.
Potentially contains ~6,768 completed production cells and may save ~39% of the work.

If you have several partial caches from Kaggle/local runs, upload all of them.

rsna_w2_6_fs4_challenge_mapper_v2_portable.py

Defines the successful controlled W2.6 method.

rsna_w2_6_v2.zip

Contains successful W2.6 gold results and should be used to validate any production optimization.

rsna_w2_3.zip

Contains fold-safe W2.3 probabilities and diagnostics.

04_gold_structured_report_features.csv

Needed for W2/W2.6 features.

08_full_structured_report_labels.csv

Needed for full unlabeled production.

train.csv

Needed for report text, gold/unlabeled split, and labels.

Strongly Recommended
rsna_w2_5_own_report_teacher_gold_gate_v6_1.py

Preserves final grounded report extraction logic and failure lessons.

W2.5 gold output/log archive

Useful for error analysis.

Canonical W4 Curia script + output/cache manifest

Exact canonical filename is UNKNOWN from the compact context. Upload it before W6 work.

rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py

Baseline image implementation.

rsna_w5_0_orthodiffusion_spatial_fold_safe_train_v2.py

Prevents repeating stopped Ortho work.

labeled_dataset_summary.json

Data audit.

Optional
report_labels_v2.csv — external audit only
W2.4 logs
W3.1 logs
W5 cache audit files
public baseline notes
Final project handoff sentence

The project is not waiting for another broad architecture search. It is waiting for a fast, portable, cache-aware replacement for W2.6-P that preserves the successful W2.6 challenge-mapping signal. Once that production teacher is complete, move immediately to W6.0 Curia with improved supervision, then W6.1 spatial Curia, and only then consider partial fine-tuning or final ensemble work.