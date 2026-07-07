# GRAIN-CAN v20 Revision Report

This revision fixes metric-name consistency across the formula definitions, prose, and result tables.

## Changes

1. Rewrote Section 5.3 so that formula symbols and table columns are explicitly mapped:
   - Precision = attack-positive precision = P
   - Recall = attack recall / detection rate = R
   - Attack-F1 = attack-positive F1 = F_1^{atk}
   - FNR = false-negative rate
   - FPR = false-positive rate
   - AUPR and Recall@FPR are supplementary score-based metrics.

2. Updated Table 5 and Table 6 headers:
   - Precision (P)
   - Recall (R)
   - Attack-F1 (F_1^{atk})
   - FNR
   - FPR
   - AUPR
   - Recall@FPR (10^{-3})

3. Updated Table 5 and Table 6 captions to refer to the metric definitions in Sec. 5.3 and explain abbreviations outside the table body.

4. Verified the Recall@FPR formula uses the correct constraint FPR(tau) <= alpha.

5. Recompiled the LNCS PDF successfully; output remains 16 pages.

