# Security Four / CCF-A Upgrade Experiment Matrix

The current repository has been upgraded with event-level and false-alarm-rate evidence, but it is not yet at CCF-A/security-four standard. The remaining gap is methodological rather than cosmetic: CT&T test04 and strict false-alarm operating points are still weak.

## P0: Required Before a Serious CCF-A/Security-Four Submission

| Priority | Experiment | Hypothesis | Required implementation | Success criterion |
|---|---|---|---|---|
| P0 | Export final anomaly ensemble per-window scores | Current multi-seed anomaly policy may have better event-level behavior than individual normality scores | Add `--save-scores` to `cmf_can.analysis.anomaly_ensemble`; save `sample_id,label,score,policy,seed` | Event recall and FA/hour can be computed for final policy |
| P0 | Context-masked CT&T test02/test04 3-seed | ID-context harms unknown vehicles; context masking is a real repair, not a seed accident | Train/evaluate `wo_context` or shift-aware context gate for seeds 42/2024/2026 | test02 F1 > 0.80 mean; test04 improves over Full with lower variance |
| P0 | Strict false-alarm event metrics | Security reviewers care about false alarms per hour, not only F1 | Report event recall, detection delay, FA events/hour for every main setting | test03/test04 show useful event recall under deployable FA/hour |
| P0 | Final policy ablation | Normality repair must be decomposed | Compare CMF score, PCA, Ledoit-Wolf, robust stats, temporal smoothing, ensemble | Each retained component improves either F1, AUPR, or low-FA event recall |
| P0 | Adaptive attacker stress test | Top security venues require attacker-aware evaluation | Simulate ID replay, payload mimicry, timing jitter, low-rate injection | Report degradation and robust fallback policy |

## P1: Strongly Recommended

| Priority | Experiment | Hypothesis | Required implementation | Success criterion |
|---|---|---|---|---|
| P1 | Self-supervised CAN pretraining | Label efficiency needs representation learning, not only 1% supervised training | Masked ID/payload modeling, next-ID prediction, contrastive window learning | Stable 1% gains across ROAD and CT&T |
| P1 | Segment/top-k pooling | Window labels dilute short attacks | Split 100-frame windows into segments and top-k pool attack evidence | test04 and low attack-frame-ratio recall improve |
| P1 | Shift-aware modality reliability gate | Full fusion fails because context reliability changes under shift | Estimate ID overlap/profile shift and gate context accordingly | Full/adaptive model beats static Full and static -Ctx |
| P1 | Cross-dataset transfer | Top-tier generalization requires transfer beyond same benchmark | ROAD->CT&T, CT&T->ROAD, CrySyS subset transfer | Honest transfer improvement or clear failure analysis |

## P2: Paper Strengthening

| Priority | Experiment | Hypothesis | Required implementation | Success criterion |
|---|---|---|---|---|
| P2 | Calibration/reliability bins for final policy | Low-FPR decisions need calibrated risk | Save bin-level reliability for final score | ECE improves after calibration |
| P2 | Event-level qualitative cases | Security papers need actionable failures | Save top false positives/false negatives with attack/capture context | Failure analysis supports design choices |
| P2 | Online update study | Vehicle IDS is deployed over time | Benign-only normality update with drift guard | Maintains low FA/hour under benign drift |

## Current Hard Evidence Added

Generated files:

- `results/cmf_tables/top_tier_event_level_metrics.csv`
- `results/cmf_tables/top_tier_open_world_policy.csv`
- `results/cmf_figures/top_tier_open_world_event_recall.{png,pdf,svg}`
- `results/cmf_figures/top_tier_low_false_alarm_event_recall.{png,pdf,svg}`
- `results/cmf_diagnostics/ccfa_security4_evidence_upgrade.md`

## Honest Status

The project is now closer to a serious security evaluation because it reports event-level detection and false alarm rate per hour. It still does not meet a CCF-A/security-four bar because CT&T test04 under low false-alarm constraints remains weak and adaptive-attacker experiments are absent.
