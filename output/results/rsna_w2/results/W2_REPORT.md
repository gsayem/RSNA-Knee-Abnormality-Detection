# RSNA W2 — Multilingual Structured Report Labeler

- Mode: `full`
- Semantic model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Stage A: W1.2 deterministic concept-local rules + multilingual 3-way NLI
- Stage B: per-label strongly regularized logistic challenge-ontology mapping
- Semantic recovery is target-grounded before NLI may change a W1.2 `not_mentioned` state
- `not_mentioned` and `mentioned_neutral` are kept unavailable for pseudo-supervision in FULL mode
- W1.3 stress-test PPV/NPV values are not used as direct weights
- FULL-mode soft labels are additionally gated by evidence-subset Stage-B validation

## Stage B gold cross-fit

```text
           Label  GoldPositive  GoldNegative  OOF_AUROC   OOF_AP  OOF_Brier  Prior_Brier  BrierImprovementVsPrior  OOF_LogLoss  EvidenceAvailableN  EvidenceCoverage  EvidenceOnly_AUROC  EvidenceOnly_AP  EvidenceOnly_Brier  EvidenceOnly_Prior_Brier  EvidenceOnly_BrierImprovementVsPrior                                                                PseudoLabelGateReason  PseudoLabelGatePass
             ACL            24            34   0.924020 0.938634   0.086224     0.242568                 0.156344     0.318340                  39          0.672414            0.975936         0.981238            0.074017                  0.245891                              0.171874                                                                                 pass                 True
             MCL             9            49   0.959184 0.711672   0.094376     0.131094                 0.036718     0.349057                  30          0.517241            0.904762         0.711672            0.143841                  0.210000                              0.066159                                                                                 pass                 True
 Medial Meniscus            26            32   0.831731 0.842803   0.136149     0.247325                 0.111176     0.443498                  34          0.586207            0.875000         0.909115            0.125388                  0.228374                              0.102986                                                                                 pass                 True
Lateral Meniscus            23            35   0.804969 0.721646   0.163076     0.239298                 0.076223     0.509636                  43          0.741379            0.839912         0.760379            0.147820                  0.246620                              0.098800                                                                                 pass                 True
       Medial OA            15            43   0.765891 0.618527   0.177506     0.191736                 0.014230     0.546001                   7          0.120690            0.600000         0.876190            0.256053                  0.204082                             -0.051971                         insufficient_evidence_n|evidence_brier_not_better_than_prior                False
      Lateral OA            11            47   0.558994 0.389432   0.155358     0.153686                -0.001671     0.492356                   2          0.034483            1.000000         1.000000            0.052032                  0.250000                              0.197968                                                              insufficient_evidence_n                False
           PF OA            21            37   0.720721 0.707427   0.177308     0.230975                 0.053667     0.539214                  10          0.172414            0.333333         0.906041            0.109710                  0.090000                             -0.019710 insufficient_evidence_n|evidence_auc_below_gate|evidence_brier_not_better_than_prior                False
        Effusion            35            23   0.685714 0.752443   0.212316     0.239298                 0.026982     0.613410                  51          0.879310            0.725806         0.771569            0.200140                  0.238370                              0.038230                                                                                 pass                 True
       Synovitis            27            31   0.473716 0.486287   0.251512     0.248811                -0.002701     0.709953                  15          0.258621            0.000000         0.550947            0.286505                  0.195556                             -0.090950                         evidence_auc_below_gate|evidence_brier_not_better_than_prior                False
         Baker's            12            46   0.802536 0.754744   0.111202     0.164090                 0.052889     0.385536                  22          0.379310            0.858333         0.865705            0.160581                  0.247934                              0.087353                                                                                 pass                 True
       Contusion            19            39   0.786775 0.644239   0.181820     0.220273                 0.038454     0.544498                  38          0.655172            0.666667         0.661512            0.232278                  0.249307                              0.017029                                                                                 pass                 True
        Fracture            18            40   0.797222 0.620005   0.161751     0.214031                 0.052280     0.511991                  25          0.431034            0.756410         0.679660            0.183709                  0.249600                              0.065891                                                                                 pass                 True
      MACRO_MEAN           240           456   0.759289 0.682322   0.159050     0.210266                 0.051216     0.496957                 316          0.454023            0.711347         0.806169            0.164339                  0.221311                              0.056972                                                                 macro_not_applicable                False
```

These are cross-fitted results on only 58 gold studies. The W1/W1.1/W1.2 development history means they are not an independent external validation set.

## Candidate soft labels

FULL mode writes candidate report-derived soft labels for the 4,349 unlabeled studies. They are not automatically consumed by the MRI model.

`CandidateSelectionScore` is a ranking/triage score, not a validated loss weight.
