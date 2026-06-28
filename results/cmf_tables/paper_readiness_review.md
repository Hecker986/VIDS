# CMF-CAN Paper Readiness Review

## 1. Can the current results support a paper draft?
Yes, but only with a cautious mixed-results narrative. The current package is enough for a first draft around cross-modality CAN IDS and deployment-oriented analysis, not enough for a universal superiority claim.

## 2. Does it support 'CMF-CAN comprehensively outperforms Transformer'?
No. On ROAD, Transformer has higher F1/Macro-F1 (0.8279/0.9102) than CMF-CAN (0.7894/0.8903), while CMF-CAN improves AUROC/AUPR (0.9431/0.8060 vs 0.9335/0.7794).

## 3. Best narrative
Recommended: cross-modality feature fusion for CAN IDS with deployment-oriented low-FPR evidence. Label efficiency is a secondary, setting-dependent result. Generalizable CAN IDS should not be the main claim.

## 4. Strongest results
- CT&T KV-KA: CMF-CAN slightly improves F1/Macro-F1 over Transformer.
- CT&T UV-KA low-FPR: CMF-CAN shows a large recall advantage at measured FPR budgets.
- ROAD ranking metrics: CMF-CAN improves AUROC/AUPR over Transformer despite weaker thresholded F1.
- CrySyS-subset: all three main models are close; CMF-CAN is competitive, not decisively dominant.

## 5. Weakest results
- CT&T KV-UA and UV-UA have low absolute F1, especially test04.
- ROAD F1/Macro-F1 favor Transformer.
- Ablation does not show Full CMF-CAN is always the best thresholded-F1 variant.
- HCRL and Car-Hacking are near-saturated sanity checks and should not be overemphasized.

## 6. Main-paper figures
Figures 1-8 refined are suitable for the main paper if the text states dataset-dependent behavior. Figure 7a is preferable for the main low-FPR story; Figure 7b belongs in the appendix/discussion.

## 7. Appendix figures
Gate weights and ROAD PR/ROC/per-attack figures can go to appendix if the corresponding prediction/gate CSVs are retained. HCRL/Car-Hacking should be appendix sanity checks.

## 8. Conclusions that cannot be written
- Do not claim consistent superiority over Transformer across all datasets and metrics.
- Do not claim unknown vehicle + unknown attack generalization is solved.
- Do not claim every modality is always beneficial.
- Do not claim CT&T few-label superiority is stable at every label ratio.

## 9. Remaining key experiments
- Complete prediction/score dumps for CT&T test02-test04 for all three main models; CT&T test01 is already exported.
- Complete CMF-CAN gate dumps for CT&T test02-test04; CT&T test01 is already exported.
- CT&T unknown-setting ablations for test02/test03/test04.
- Embedding dumps if representation visualization is required.

## 10. CCF B/C or intelligent vehicle security readiness
Basically enough for an initial submission draft if framed as a systematic cross-modality/deployment study and if limitations are explicit.

## 11. CCF A / top-tier security gap
Still short. Needs stronger unknown-shift results, complete per-sample evidence, more rigorous baselines, more seeds for shifted CT&T, deployment validation, and stronger ablation under unknown vehicle/attack shifts.
