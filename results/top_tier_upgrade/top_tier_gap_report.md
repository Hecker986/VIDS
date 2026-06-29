# Top-Tier Gap Report

Reliable-CMF-CAN is implemented as a research prototype, but the experimental evidence does not yet meet a CCF A/security-four bar.

## Missing or Not Met

- missing prediction: ctt_test01/frame_only
- missing prediction: ctt_test01/stats_only
- missing prediction: ctt_test01/wo_context
- missing prediction: ctt_test01/wo_stats
- missing prediction: ctt_test02/frame_only
- missing prediction: ctt_test02/stats_only
- missing prediction: ctt_test02/wo_context
- missing prediction: ctt_test02/wo_stats
- missing prediction: ctt_test03/frame_only
- missing prediction: ctt_test03/stats_only
- missing prediction: ctt_test03/wo_context
- missing prediction: ctt_test03/wo_stats
- missing prediction: ctt_test04/frame_only
- missing prediction: ctt_test04/stats_only
- missing prediction: ctt_test04/wo_context
- missing prediction: ctt_test04/wo_stats
- E1 missing trained checkpoint/result: road/Reliable-CMF-CAN
- E1 missing trained checkpoint/result: road/Reliable-CMF-CAN w/o shift
- E1 missing trained checkpoint/result: road/Reliable-CMF-CAN w/o normality
- E1 missing trained checkpoint/result: road/Reliable-CMF-CAN w/o segment
- E1 missing trained checkpoint/result: ctt_test01/Reliable-CMF-CAN
- E1 missing trained checkpoint/result: ctt_test01/Reliable-CMF-CAN w/o shift
- E1 missing trained checkpoint/result: ctt_test01/Reliable-CMF-CAN w/o normality
- E1 missing trained checkpoint/result: ctt_test01/Reliable-CMF-CAN w/o segment
- E1 missing trained checkpoint/result: ctt_test02/Reliable-CMF-CAN
- E1 missing trained checkpoint/result: ctt_test02/Reliable-CMF-CAN w/o shift
- E1 missing trained checkpoint/result: ctt_test02/Reliable-CMF-CAN w/o normality
- E1 missing trained checkpoint/result: ctt_test02/Reliable-CMF-CAN w/o segment
- E1 missing trained checkpoint/result: ctt_test04/Reliable-CMF-CAN
- E1 missing trained checkpoint/result: ctt_test04/Reliable-CMF-CAN w/o shift
- E1 missing trained checkpoint/result: ctt_test04/Reliable-CMF-CAN w/o normality
- E1 missing trained checkpoint/result: ctt_test04/Reliable-CMF-CAN w/o segment
- E2 missing trained Reliable-CMF-CAN shift-control checkpoint and context_shift_score dump
- E4 missing trained Reliable-CMF-CAN segment/top-k checkpoint and segment_scores dump
- E5 missing Reliable-CMF-CAN training with val_low_fpr_composite
- E6 missing Reliable-CMF-CAN multi-seed runs
- E7 missing rebuilt window_size=50/200 features and model runs
- E8 missing Reliable-CMF-CAN prediction dump for per-attack analysis

## Assessment

The strongest current direction is Reliable-CMF-CAN as an adaptive low-FPR/open-world system, but the current result set is still closer to CCF B unless P0/P1 Reliable experiments are completed.
