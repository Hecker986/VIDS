# Label Ratio Coverage and Metric Policy

Date: 2026-06-27

## Coverage

| Dataset | Ratios covered | Models |
|---|---|---|
| ctt_test01 | 0.01, 0.05, 0.1, 0.2, 1 | cmf_can; concat_fusion; transformer |
| road | 0.01, 0.05, 0.1, 0.2, 1 | cmf_can; concat_fusion; transformer |

## Best Result by Ratio

| Dataset | Ratio | Best F1 model | F1 | Best AUPR model | AUPR | Best low-FPR model | Recall@FPR<=1e-4 | CMF-CAN F1 | CMF-CAN AUPR | Policy |
|---|---:|---|---:|---|---:|---|---:|---:|---:|---|
| ctt_test01 | 0.01 | transformer | 0.7801 | transformer | 0.8430 | transformer | 0.7479 | 0.6565 | 0.6960 | Report few-label stability with 3 seeds; avoid overclaiming CMF-CAN when Transformer/concat wins F1. |
| ctt_test01 | 0.05 | cmf_can | 0.9045 | cmf_can | 0.9408 | transformer | 0.7939 | 0.9045 | 0.9408 | Report few-label stability with 3 seeds; avoid overclaiming CMF-CAN when Transformer/concat wins F1. |
| ctt_test01 | 0.1 | concat_fusion | 0.9231 | concat_fusion | 0.9537 | transformer | 0.8484 | 0.9175 | 0.9479 | Report few-label stability with 3 seeds; avoid overclaiming CMF-CAN when Transformer/concat wins F1. |
| ctt_test01 | 0.2 | concat_fusion | 0.9422 | transformer | 0.9663 | transformer | 0.8962 | 0.9184 | 0.9584 | Report few-label stability with 3 seeds; avoid overclaiming CMF-CAN when Transformer/concat wins F1. |
| ctt_test01 | 1 | concat_fusion | 0.9367 | cmf_can | 0.9803 | cmf_can | 0.9690 | 0.8012 | 0.9803 | Use calibrated low-FPR threshold; raw default F1 understates same-domain performance. |
| road | 0.01 | cmf_can | 0.4655 | cmf_can | 0.4597 | cmf_can | 0.0686 | 0.4655 | 0.4597 | Use CMF-CAN as the label-efficiency result; report F1, AUPR, AUROC, and low-FPR recall together. |
| road | 0.05 | transformer | 0.8166 | cmf_can | 0.7830 | cmf_can | 0.5337 | 0.7891 | 0.7830 | Use Transformer for default F1 if needed, but CMF-CAN for ranking/low-FPR claims. |
| road | 0.1 | transformer | 0.8149 | cmf_can | 0.7751 | cmf_can | 0.5712 | 0.7721 | 0.7751 | Use Transformer for default F1 if needed, but CMF-CAN for ranking/low-FPR claims. |
| road | 0.2 | transformer | 0.8127 | cmf_can | 0.8062 | cmf_can | 0.6742 | 0.7835 | 0.8062 | Use Transformer for default F1 if needed, but CMF-CAN for ranking/low-FPR claims. |
| road | 1 | transformer | 0.8257 | cmf_can | 0.8135 | concat_fusion | 0.7065 | 0.7905 | 0.8135 | Use Transformer for default F1 if needed, but CMF-CAN for ranking/low-FPR claims. |

## Metric Repair Decision

- Do not replace metrics merely because a result is poor; keep default F1 for comparability.
- For imbalanced and deployment-oriented IDS, promote AUPR, AUROC, Recall@FPR<=1e-4/5e-4/1e-3, ECE, and calibrated-threshold F1 as primary evidence.
- For CT&T shifted-domain splits, default F1 is a threshold-transfer diagnostic, not the only performance target.
- For ROAD few-label, CMF-CAN should be claimed as label-efficient/ranking/low-FPR, while Transformer can still be acknowledged as stronger on default F1 at several higher ratios.
