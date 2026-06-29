# D3 ID-context Shift Diagnosis

## Hypothesis
ID-context helps known vehicles but can hurt unknown vehicles because learned ID behavior profiles drift.

## Answers
- Check `d3_id_context_shift.csv` for ID overlap and JS divergence.
- Strong `-Ctx` performance in CT&T test02/test04 is consistent with ID-context shift risk.
- Context masking/downweighting is justified when overlap is low or ID frequency divergence is high.
- Most unstable proxy features are ID frequency, payload profile and transition profile shifts.
