# HCRL / Car-Hacking / CrySyS CMF-CAN Execution Status

Date: 2026-06-27

## Scope

Raw data were downloaded from official sources:

- HCRL CAN-Intrusion / OTIDS: official HCRL Dropbox package.
- HCRL Car-Hacking: official HCRL Dropbox package.
- CrySyS: Figshare dataset `10.6084/m9.figshare.23624208.v1`.

## Processed Datasets

| Dataset | Frames | Windows | Split | Labels |
|---|---:|---:|---|---|
| `hcrl_can_intrusion` | 4,613,439 | 46,126 | 60/20/20 by capture chronology | attack files marked anomalous; attack-free marked normal |
| `car_hacking` | 17,558,346 | 175,576 | 60/20/20 by capture chronology | CSV `T` flag anomalous; `R` and normal run benign |
| `crysys_subset` | 16,538,840 | 165,353 | stratified non-overlapping windows | JSON marker start/end intervals anomalous |
| `crysys_family_subset` | 41,052,185 | 410,425 | stratified non-overlapping windows | six attack families, injection-preferred |
| `crysys_family_mod_subset` | 38,503,318 | 384,951 | stratified non-overlapping windows | six attack families, modification-only |

CrySyS subset uses 26 benign captures plus 2 malicious captures per scenario. Current malicious selection is deterministic and mainly covers `ADD-DECR` injection/modification; expand this later for full attack-family coverage.

`crysys_family_subset` and `crysys_family_mod_subset` were added after the first CrySyS run to avoid over-narrow `ADD-DECR` conclusions. Both cover `ADD-DECR`, `ADD-INCR`, `CONST`, `NEG-OFFSET`, `POS-OFFSET`, and `REPLAY` across all 26 scenarios. The injection-preferred subset is much easier; the modification-only subset is the stronger auxiliary stress test.

## Fixes During Execution

- Car-Hacking CSV rows are variable width because DLC can be less than 8. Parser was fixed to treat the final non-empty field as `R/T` and zero-pad payload bytes.
- CrySyS first run used chronological split and produced a test set with no positives. That result was deleted. CrySyS was rebuilt with stratified non-overlapping window splits; test now has 28,516 normal and 4,556 anomalous windows.
- GPU training must run outside the sandbox. Inside the sandbox PyTorch reports `cuda False`; outside it sees `NVIDIA RTX PRO 5000 72GB Blackwell`.

## Result Tables

- `results/cmf_tables/hcrl_main_15ep.csv`
- `results/cmf_tables/car_hacking_main_15ep.csv`
- `results/cmf_tables/crysys_subset_main_15ep.csv`
- `results/cmf_tables/crysys_family_subset_main_15ep.csv`
- `results/cmf_tables/crysys_family_mod_subset_main_15ep.csv`
- `results/cmf_tables/crysys_family_mod_cmf_can_3seed_mean_std.csv`
- `results/cmf_tables/crysys_family_mod_3model_3seed_mean_std.csv`
- `results/cmf_tables/crysys_family_mod_3model_3seed_summary.csv`

## Main Results, Seed 42, 15 Epochs

| Dataset | Model | F1 | AUPR | AUROC | FPR |
|---|---|---:|---:|---:|---:|
| HCRL CAN-Intrusion | transformer | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| HCRL CAN-Intrusion | concat_fusion | 0.9998 | 1.0000 | 1.0000 | 0.0004 |
| HCRL CAN-Intrusion | cmf_can | 0.9967 | 1.0000 | 1.0000 | 0.0063 |
| Car-Hacking | transformer | 0.9998 | 1.0000 | 1.0000 | 0.0000 |
| Car-Hacking | concat_fusion | 0.9983 | 0.9997 | 0.9998 | 0.0001 |
| Car-Hacking | cmf_can | 0.9983 | 1.0000 | 1.0000 | 0.0000 |
| CrySyS subset | transformer | 0.8386 | 0.9383 | 0.9865 | 0.0217 |
| CrySyS subset | concat_fusion | 0.8698 | 0.9576 | 0.9911 | 0.0183 |
| CrySyS subset | cmf_can | 0.8725 | 0.9566 | 0.9906 | 0.0193 |
| CrySyS family inj | cmf_can | 0.9978 | 0.9998 | 0.9999 | 0.0000 |
| CrySyS family mod | transformer | 0.9284 | 0.9117 | 0.9872 | 0.0311 |
| CrySyS family mod | concat_fusion | 0.9307 | 0.9160 | 0.9875 | 0.0306 |
| CrySyS family mod | cmf_can | 0.9311 | 0.9226 | 0.9883 | 0.0289 |

## 3-Seed Result

`crysys_family_mod_subset`, seeds 42/43/44:

| Model | F1 mean | F1 std | AUPR mean | AUPR std | AUROC mean | AUROC std | Recall@FPR<=1e-4 mean | Recall@FPR<=1e-4 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 0.9317 | 0.0030 | 0.9343 | 0.0199 | 0.9900 | 0.0024 | 0.1346 | 0.0617 |
| Concat-Fusion | 0.9307 | 0.0000 | 0.9184 | 0.0053 | 0.9879 | 0.0007 | 0.1391 | 0.0094 |
| CMF-CAN | 0.9319 | 0.0029 | 0.9350 | 0.0156 | 0.9899 | 0.0021 | 0.1623 | 0.0073 |

## Interpretation

HCRL/OTIDS and Car-Hacking are now fully runnable from official raw data. Their results are very high, but HCRL/OTIDS is especially easy because the available raw files do not contain per-frame labels and attack captures are labeled wholesale.

CrySyS is the more meaningful auxiliary stress test. After fixing split leakage/emptiness, `cmf_can` is slightly best by F1 and AUPR on the modification-only 3-seed summary, while Transformer has essentially tied AUROC. The margin is small and should be described as incremental, not decisive.

The expanded CrySyS family experiments clarify the difficulty difference:

- Injection-preferred family coverage is easy for CMF-CAN (`F1=0.9978`), so it should not be over-claimed.
- Modification-only family coverage is much harder and more useful. `cmf_can` is slightly best by F1/AUPR and has the best mean low-FPR recall among the three models, but low-FPR recall remains weak: at FPR about `1e-4`, mean recall is only `0.1623`.

## Exported Paper Artifacts

- `results/cmf_tables/table_hcrl_main_15ep.tex`
- `results/cmf_tables/table_car_hacking_main_15ep.tex`
- `results/cmf_tables/table_crysys_subset_main_15ep.tex`
- `results/cmf_tables/table_crysys_family_mod_3seed.tex`
- `results/cmf_figures/fig_crysys_family_mod_recall_at_fpr_1e4.png`

## Remaining Limitations

1. HCRL/OTIDS and Car-Hacking should be presented as auxiliary sanity/generalization checks, not primary evidence, because the task is near ceiling.
2. CrySyS modification-only low-FPR recall remains weak even after CMF-CAN wins the mean comparison. A low-FPR-specific calibration/training variant would be needed before making a strong deployment claim on this auxiliary dataset.
