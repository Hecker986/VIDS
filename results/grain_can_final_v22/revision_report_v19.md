# GRAIN-CAN v19 Revision Report

This revision focuses on metric-name consistency and table clarity.

## Changes

1. Rewrote Section 5.3 `Metrics and Operating Points` to use one consistent metric vocabulary throughout the paper:
   - attack-positive precision
   - attack recall
   - attack-F1
   - false-negative rate (FNR)
   - false-positive rate (FPR)
   - area under the precision-recall curve (AUPR)
   - Recall@FPR
   - FNR@FPR

2. Clarified that attack-F1 is the standard F1 score with attack as the positive class, not a newly proposed metric.

3. Clarified that AUPR and Recall@FPR are supplementary score-based operating metrics and are only compared among locally reproduced detectors with comparable scores.

4. Updated Table 5 and Table 6 captions and headers so that the metric names match Section 5.3 exactly.

5. Corrected table headers from shortened labels such as `False-neg. rate` and `False-pos. rate` to full names:
   - false-negative rate (FNR)
   - false-positive rate (FPR)
   - area under PR curve (AUPR)

6. Confirmed Recall@FPR formula uses the correct constraint: FPR(tau) <= alpha.

7. Prevented the environment table from floating into the middle of the metrics explanation by using a fixed table placement and a float barrier.

## Build

The paper compiles successfully to 16 pages with no LaTeX build failure.
