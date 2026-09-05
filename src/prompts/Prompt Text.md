Project continuation context
--------------------------------------
Attached RSNA_Knee_Project_Continuation_Package.md is the project details that what we already achieved or not and what we need to do finish this project and send submission to Kaggle.
attached rsna_files.zip file contains all the necessary files, log, report, scripts etc - let me know if face any difficulties to read any of this file - I'll upload again.

----------------------------


We are working on the Kaggle **RSNA Knee Abnormality Detection** competition. The goal is a high-performing MRI-only hidden-test model under extreme label scarcity. The training set has 4,407 studies, but only 58 are fully labeled. The remaining 4,349 have no challenge labels. All 4,407 training studies have radiology reports. Hidden test has no reports, so reports are being used only as privileged training-time supervision.

The 12 labels are:  
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

Gold positives among 58:  
| labels           | positives |
| ---------------- | --------- |
| ACL              | 24        |
| MCL              | 9         |
| Medial Meniscus  | 26        |
| Lateral Meniscus | 23        |
| Medial OA        | 15        |
| Lateral OA       | 11        |
| PF OA            | 21        |
| Effusion         | 35        |
| Synovitis        | 27        |
| Baker's          | 12        |
| Contusion        | 19        |
| Fracture         | 18        |

DICOM layout:  
`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`.

`train_series.csv` has:  
`StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`.

There are 24,371 series:  
9,864 sagittal, 8,609 coronal, 5,898 axial.  
The 58 gold studies have 336 series and 10,528 slices, mean 5.793 series/study. All gold studies have all three planes. Structured plane agreed with geometry for 336/336 gold series. `SeriesDescription` is unreliable. Pixel spacing varies and repeated acquisitions cannot be assumed duplicate.

The exact fold assignment is locked:  
5-fold greedy multilabel, StudyUID sorted, seed 42.  
Checksum:  
`1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a`

Do not regenerate folds.

### Image history

Early ResNet experiments:

*   V1.1 frozen ResNet18 + metadata: .54655/.42145/.45195 AUROC/AP/F1
    
*   V2.1 layer4 fine-tune: .4800/.3781/.3978
    
*   V3.1 position: .4973/.3699/.3843
    
*   V4 no metadata: .54577/.43250/.40716
    
*   W3.0 `rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py`: .5459179/.433128/.407160
    
*   W3.1 weak supervision retained .523402, worse than W3.0
    
*   W3.0 public LB ~.608, hybrid ~.610
    

Curia W4 is the current image anchor:  
model `raidium/curia-2`, medical DINOv2-family, ~86.1M params, 512 input, patch16, 12 layers, hidden768, CLS=`last_hidden_state[:,0]`.

We froze Curia, cached one series CLS embedding, and trained a hierarchical study model.

W4:

*   gold .6206400750
    
*   weak .6410767023
    
*   gated .6523168374
    

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
`lt-0123/OrthoDiffusion`, 3 orientation-specific 3D diffusion U-Nets, frozen t100/mid2 features.  
Gold .5775082, weak .5985844750.  
W4+W5 50/50 = .642307, only +.001231.  
Stop Ortho branch.

### Report pipeline

W2 semantic model:  
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.

W2 Stage-A:  
rule exact7state .35  
semantic-only .241667  
hybrid .408333  
binary polarity hybrid .766667.

W2 full Stage-B all58 macro .759289.

W2 exact Stage-B features:  
`EvidencePositiveScore`  
`EvidenceNegativeScore`  
`RelatedScore`  
`UncertaintyFlag`  
`FusedAssertionConfidence`  
`RuleDecidableFlag`  
`SemanticDirectStrength`  
`SemanticDirectMargin`  
`SeverityLow`  
`SeverityModerate`  
`SeverityHigh`  
`SeverityDegenerative`.

Important W2 files:  
`04_gold_structured_report_features.csv`  
`08_full_structured_report_labels.csv`.

W2.3 script:  
`rsna_w2_3_fold_safe_pseudo_labels.py`.

For each outer fold:

*   outer-train gold only
    
*   inner OOF calibration
    
*   fold-specific gates
    
*   held-out gold diagnostic only
    
*   unlabeled pseudo labels produced fold-safely
    

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
`rsna_w2_4_high_fidelity_fold_safe_consensus_v2.py`  
using Pilkwang `report_labels_v2.csv`.

W2.4 .849857.  
Pilkwang .870040.

This was useful ceiling evidence only. Pilkwang must not enter our own production teacher.

### W2.5 own Qwen teacher

Model:  
`Qwen2.5-7B-Instruct`.

We went through V1–V6.1:

*   Kaggle 4-bit failed due bitsandbytes incompatibility.
    
*   single-T4 FP16 OOM.
    
*   dual-T4 sharded FP16 worked.
    
*   schema/evidence failures appeared.
    
*   all-12 prompt caused cross-label evidence contamination.
    
*   moved to one-label-per-generation.
    
*   V6.1 used grounded/nonfatal evidence logic.
    

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
Also many grounded `N` states were challenge-positive.

### W2.6 FS4

Successful controlled script:  
`rsna_w2_6_fs4_challenge_mapper_v2_portable.py`.

FS4 labels:  
Medial OA, Lateral OA, PF OA, Synovitis.

For each held-out gold query:

*   TF-IDF built from outer-train gold
    
*   retrieve 2 positives +2 negatives
    
*   query gold unavailable to model
    
*   Qwen predicts challenge probability
    

232 calls total.

Results:  
W2.3 .77124  
FS4 replace .81342  
fixed50 .80929.

Fixed50 delta ~+.03804.  
95% CI approximately \[.0120,.0673\].  
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

*   8 established labels = W2.3
    
*   4 weak labels = 0.50 W2.3 + 0.50 FS4
    

Do not do more broad report-prompt research.

### W2.6-P production — current blocker

Current script:  
`rsna_w2_6p_batched_production_mapper.py`

Purpose:  
generate FS4 probabilities for all 4,349 unlabeled studies:  
4,349×4 = 17,396 Qwen cells.

Production uses all 58 gold reports as exemplar pool.  
For each query/label:

*   TF-IDF query-specific retrieval
    
*   2 nearest positive examples
    
*   2 nearest negative examples
    
*   W2 structured report summary
    
*   Qwen challenge probability
    

Final teacher:  
8 labels W2.3 cross-fold mean  
4 labels 0.5 W2.3 + 0.5 FS4.

Planned output files:  
`01_w23_production_base_long.csv`  
`02_fs4_exemplar_audit.csv`  
`03_fs4_prompt_metadata.csv`  
`04_fs4_production_long.csv`  
`05_fs4_probabilities_wide.csv`  
`06_final_hybrid_teacher_long.csv`  
`07_final_hybrid_probabilities_wide.csv`  
`08_recommended_teacher_weights_wide.csv`  
`09_recommended_teacher_mask_wide.csv`  
`10_production_summary.json`  
`11_validation_summary.json`  
`w26p_manifest.json`.

Cache:  
`w26p_fs4_production_v1.jsonl`.

The current production implementation is too slow.

Kaggle 2×T4:

*   batch size 8
    
*   max context 3584
    
*   retry contexts 3072/2560/2048/1536
    
*   7B FP16 balanced across both GPUs
    
*   reached ~6,768/17,396 after 12h
    
*   rate degraded to ~9.4 cells/min
    
*   remaining ETA ~18.8h
    
*   cell timed out at 43,200s.
    

The log shows many batches performing several failed OOM attempts before eventually succeeding at a shorter context, wasting significant time.

Desktop 4060Ti 16GB +48GB RAM also projects ~48+ hours even with batch 8/16/32.  
Laptop 3060 6GB +64GB eventually runs into VRAM issues.  
Apple Silicon 16GB exists but current script has no Apple backend.  
Kaggle T4 quota is almost gone.  
TPU v5e access exists ~20h; current script deliberately does not implement Qwen TPU inference.

### Immediate next task

Do not rerun W2.6-P unchanged.

Inspect:

*   `rsna_w2_6p_batched_production_mapper.py`
    
*   `rsna_w2_6p_console_log.txt`
    
*   `rsna_w2_6_fs4_challenge_mapper_v2_portable.py`
    
*   `rsna_w2_6_v2.zip`
    
*   any exported `w26p_fs4_production_v1.jsonl`
    

Build an optimized production script.

It must:

*   reuse/merge old cache;
    
*   support local CUDA;
    
*   support Apple Silicon via MPS and/or MLX;
    
*   optionally keep T4 support;
    
*   drastically reduce prompt/prefill cost;
    
*   avoid repeated OOM ladders;
    
*   produce minimal output, preferably only probability;
    
*   sort/bucket by token length;
    
*   benchmark first 200–500 cells;
    
*   print projected full runtime;
    
*   auto-stop if runtime is still unacceptable.
    

Strong candidate optimization directions:

1.  much shorter prompt;
    
2.  compressed exemplar reports;
    
3.  compressed W2 summaries;
    
4.  output `{"p":...}` only;
    
5.  fixed lower token budget;
    
6.  faster inference backend;
    
7.  Apple MLX quantized Qwen;
    
8.  fixed exemplar/prefix-cache strategy validated on 232 gold cells;
    
9.  teacher distillation using partial production Qwen cache.
    

If the Kaggle cache containing roughly 6,768 cells can be exported, do not recompute those cells. Merge by UID/Label plus prompt SHA.

Before full production, verify any optimized approximation against the existing W2.6 gold 232-cell benchmark so challenge-mapping performance is not silently destroyed.

If full direct Qwen inference still projects above single-digit hours, stop scaling it directly and distill/approximate the teacher.

### W6 after production

W6.0:  
same frozen Curia CLS architecture as W4, but train with the new W2.6 production teacher.  
Purpose: isolate supervision gain.

W6.1:  
Curia spatial patch-token model.

W6.2:  
partial Curia last-block fine-tuning only if W6.1 warrants it.

Do not reopen Ortho or broad report-model experimentation.

### Leakage constraints

Use exact fixed folds for controlled experiments.

Held-out gold never calibrates itself.

W2.6 gold retrieval examples are outer-train only.

Production all-58 labels are final training resources only.

A downstream model trained on all-58-derived pseudo labels cannot claim pristine OOF against those same 58.

Pilkwang stays outside own production.

Hidden test has no reports.

### Hardware

*   RTX 4060 Ti 16GB +48GB RAM — preferred local CUDA machine.
    
*   RTX 3060 6GB +64GB RAM — not preferred.
    
*   Apple Silicon 16GB unified memory — support explicitly requested.
    
*   Kaggle 2×T4 — quota nearly exhausted.
    
*   TPU v5e access ~20h — probably preserve for W6 image compute.
    

### Working style

Finish quickly.  
Avoid unnecessary branches.  
Use controlled apples-to-apples experiments.  
Do not reopen already rejected approaches.  
Since W2.6-P runtime blocker is already fixed in the `rsna_w2_6p_fast_logits_optimized.py`, we need to implement W6 Curia now.
The code structure must be match with `rsna_w2_6p_fast_logits_optimized.py` like the folder structure for kaggle, local etc. Rest is upto you and follow the logic. 
Our last submission was `rsna_w4_0_curia2_final_submission.py` and the leaderboard score was 0.721. We need to pass 0.952+
The submission file `rsna_w4_0_curia2_final_submission.csv` 
The sample submission file `sample_submission.csv` we need to follow this format. 



* * *


# FILES UPLOAD DETAILS

`rsna_w2_6p_batched_production_mapper.py` Current production implementation. Needed to optimize rather than rebuild from memory and it's already optimized as `rsna_w2_6p_fast_logits_optimized.py`
the output are in the folder `rsna_w2_6p_fast`

Potentially contains \~6,768 completed production cells and may save \~39% of the work.
### `rsna_w2_6_fs4_challenge_mapper_v2_portable.py`

Defines the successful controlled W2.6 method.

### `rsna_w2_6_v2.zip`

Contains successful W2.6 gold results and should be used to validate any production optimization.

### `rsna_w2_3.zip`

Contains fold-safe W2.3 probabilities and diagnostics.

### `04_gold_structured_report_features.csv`

Needed for W2/W2.6 features.

### `08_full_structured_report_labels.csv`

Needed for full unlabeled production.

### `train.csv`

Needed for report text, gold/unlabeled split, and labels.


### `rsna_w2_5_own_report_teacher_gold_gate_v6_1.py`

Preserves final grounded report extraction logic and failure lessons.

### W2.5 gold output/log archive 
`rsna_w2_5_gold_gate_console_log.txt` and `rsna_w2_5_gold_gate_gold_console_log.txt`

Useful for error analysis.

### Canonical W4 Curia script + output/cache manifest
`rsna_w4_0_curia2_final_submission.py`
`rsna_w4_0_curia2_final_submission_console_log.txt` and `rsna_w4_0_submission.zip`

Exact canonical filename is **UNKNOWN** from the compact context. Upload it before W6 work.

### `rsna_w3_0_cached_resnet18_reproduction_fixed_v2.py`

Baseline image implementation.

### `rsna_w5_0_orthodiffusion_spatial_fold_safe_train_v2.py`

Prevents repeating stopped Ortho work.

### `labeled_dataset_summary.json`

Data audit.

Optional
--------

*   `report_labels_v2.csv` — external audit only
    
*   W2.4 logs in `rsna_w2_4` folder
    
*   W3.1 logs in `rsna_w3_1` folder
