# Event-Level Metrics Analysis

No official event boundary files were found. Events here are approximated as contiguous positive windows ordered by window_start, so delay is reported in frame-index units and should be treated as coarse evidence only.

Findings:

1. ROAD event recall is low for both Transformer and CAN-Transformer+ same-ID under this coarse construction: 0.1642 with only 2 false positive windows.
2. CT&T test02 Transformer/CMF-CAN show event recall=1.0, but with 58,867 false alarm windows, which is not deployable.
3. CT&T test04 Transformer reaches event recall=1.0 but with 128,693 false alarm windows; CMF-CAN reduces false alarms to 75,567 but event recall falls to 0.8531.
4. Event-level framing is necessary, but current window predictions need thresholding/false-alarm budgets and official event boundary files before this can support a strong paper claim.
