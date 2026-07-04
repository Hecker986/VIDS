# Rewrite Report

## What changed
- Rewrote the paper as a GRAIN-CAN detector / pipeline paper rather than an audit-style evaluation report.
- Removed audit-heavy main-text structure. Window, threshold, metric-availability, and prior-comparability details are no longer main narrative tables.
- Replaced the previous crowded method figure with a clean horizontal detector pipeline.
- Replaced raw CSV-dump figures with five compact publication-style figures: pipeline, mechanism, main CT&T result, raw-window comparison, feature ablation, and supplementary low-FPR operating evidence.
- Kept rule/statistical baselines as diagnostic checks only, not as main competitors.

## New structure
1. Introduction
2. Background and Design Motivation
3. GRAIN-CAN Pipeline
4. Experimental Setup
5. Evaluation
6. Discussion
7. Conclusion

## New main figures
1. GRAIN-CAN pipeline
2. Local evidence dilution mechanism
3. Main CT&T cross-shift results
4. Raw fixed-window versus GRAIN-CAN
5. Feature-group ablation
6. Supplementary low-FPR operating evidence

## New main tables
1. Compared IDS pipelines
2. CT&T shift settings
3. Main CT&T results
4. Raw-window versus GRAIN-CAN
5. Feature-group ablation
6. Pipeline positioning

## Checks
- GRAIN-CAN is written as a supervised feature pipeline, not a neural architecture, rule detector, or evaluation framework.
- ACE-CAN language is not used.
- Defensive disclaimers are moved to Discussion.
- The manuscript keeps attack-positive IDS metrics and score-metric cautions.
