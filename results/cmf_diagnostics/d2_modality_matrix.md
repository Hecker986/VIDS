# D2 Modality Matrix

## Hypothesis
Full CMF-CAN may over-fuse; some modalities may be noisy under shift.

## Evidence
- Existing ablation shows Full is not always best.
- CT&T unknown settings often favor `-Ctx`, `Stats`, or `-Gate` depending on metric.
- Missing pairwise variants are recorded as missing, not imputed.

## Answers
1. ROAD: frame and no-window/no-context variants are competitive; stats-only is weak.
2. CT&T unknown vehicle: stats and removing context can be useful.
3. Full shows over-fusion in several settings.
4. ID-context can hurt unknown vehicle/test04 settings.
5. Stats-only is more stable in shifted settings than in ROAD.
