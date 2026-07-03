# Remaining Evidence Gap Review

This review checks the revised paper from the required evidence chain rather than from file availability.

## Current Position

The revised paper is now internally consistent as a research paper about:

- attack-centric CAN IDS evaluation,
- metric ambiguity under rare attacks,
- corrected CT&T benchmark reporting,
- feature-preserving GRAIN-CAN as a strong transparent baseline.

The current evidence supports a conservative measurement + IDS-baseline paper. It does not support a claim that cross-vehicle unknown-attack detection is solved.

## No Longer Blocking

- Main CSV/TEX/SVG artifacts are present and readable.
- Figure/table formatting has been repaired.
- Internal exploratory models are excluded from main external-baseline comparisons.
- Table 6 no longer contains all-empty FA/100k and delay columns.
- CT&T test01-test04 corrected benchmark exists.
- CT&T test04 metric-trap case study exists.
- GRAIN-CAN granularity evidence exists.
- Capped-negative feature-removal retraining evidence exists.
- Validation-threshold low-FPR evidence now exists for GRAIN window100 on CT&T test01-test04.
- Multi-seed feature-removal retraining evidence now exists for CT&T test04 sample-level SAFE-CAN features.
- Chunked-negative ensemble stability probes now exist for CT&T test02/test04 sample-level GradientBoosting.
- ExtraTrees, RandomForest, and MLP Protocol B/C large-negative rows have now been completed with constrained real training/evaluation runs.

## Remaining Data Gaps

### P0: Original benchmark ambiguity

Missing:

- original authors' confusion matrices,
- exact original evaluation code,
- exact metric implementation for Table 13-style reported values.

Impact:

- The paper can claim metric ambiguity and show consistency with weighted/accuracy-like metrics.
- The paper cannot claim the original paper is wrong.
- The paper cannot conclusively prove the public high score is or is not attack-positive F1.

### P0: Official event boundaries

Missing:

- official CT&T attack event interval boundaries,
- official event IDs,
- authoritative event start/end timestamps.

Impact:

- Event recall remains approximate.
- Detection delay and false alarms per hour/per 100k should not be central claims.
- The paper can discuss approximate event-level evidence only as supporting deployment intuition.

### P0: Formal validation-threshold low-FPR

Current:

- GRAIN window100 validation-threshold rows are now available for CT&T test01-test04.
- On CT&T test04, the validation FPR-budget row at 5e-4 reaches recall 0.8053 with actual test FPR 0.000681 and AUPR 0.7868.
- Best-test low-FPR rows remain in the paper only as diagnostic upper-bound evidence.

Still missing:

- validation score dumps aligned with every historical comparison row,
- validation-selected thresholds for all older baselines and all external datasets.

Impact:

- The main GRAIN window100 low-FPR claim can now be written as validation-threshold evidence.
- Older comparison rows without validation dumps should remain marked as stored/default/best-test diagnostics.

### P1: Full-negative / chunked-full-negative stability

Current:

- 2x and 5x negative-cap stability is available.
- B/C large-negative rows for ExtraTrees, RandomForest, and MLP are now real completed rows rather than resource-limit placeholders.
- A chunked-negative ensemble probe is now available for sample-level GradientBoosting on CT&T test02/test04.
- CT&T test02 remains stable under chunked-negative sampling.
- CT&T test04 sample-level chunked ensemble remains much weaker than the aggregate-window GRAIN result, supporting the feature-preserving aggregate-window direction.

Still missing:

- exact full-negative Protocol D training for every main GRAIN aggregate-window baseline,
- chunked-full-negative training for every feature-removal ablation.

Impact:

- Current stability evidence is stronger than the previous capped-only setting.
- Exact full-negative results would still make the benchmark more defensible for a top-tier submission.

### P1: Broader external corrected benchmark

Missing or partial:

- full corrected benchmark on ROAD/CrySyS/HCRL/Car-Hacking using the same attack-centric table structure,
- external score dumps for low-FPR and event-level analysis.

Impact:

- External datasets should remain sanity checks.
- The paper should not claim universal dataset dominance.

### P1: Stronger external baselines under exact protocol

Missing:

- reproduced recent IDS methods such as MIDS/Mamba-style models under the exact CT&T corrected protocol,
- official code-based baselines from recent papers.

Impact:

- Related work can cite these methods.
- They should not be used as direct baseline comparisons unless reproduced.

### P2: Exhaustive GRAIN ablation

Current:

- single-seed full feature-removal evidence exists in `final_evidence_completion`,
- three-seed feature-removal retraining exists for sample-level SAFE-CAN features on CT&T test04.
- The three-seed result shows same-ID timing is the dominant robust signal: removing `delta_t_same_id` collapses attack-F1 to about 0.0085, while timing-only reaches about 0.6013.

Still missing:

- full-negative retraining ablation,
- interaction ablations such as timing+payload, timing+ID, payload+ID.

Impact:

- Current mechanism evidence is useful but should be framed as evidence, not exhaustive causal proof.

## Recommended Next Experiments

### If only one week remains

1. Use the new validation-threshold low-FPR table for GRAIN window100 as the formal deployment-oriented result.
2. Keep older best-test low-FPR rows as diagnostic score-separability evidence only.
3. Present the new multi-seed feature-removal table as mechanism evidence rather than exhaustive causal proof.

### If two to three weeks remain

1. Extend chunked-full-negative training from sample-level GradientBoosting to GRAIN window100 aggregate.
2. Add interaction ablations such as timing+payload, timing+ID, and payload+ID.
3. Produce an external corrected benchmark table for ROAD and CrySyS with the same metric columns as CT&T.

### If aiming at CCF A / Security Four

1. Obtain or reconstruct official event boundaries.
2. Obtain original authors' confusion matrices or exact code.
3. Reproduce at least one recent strong external IDS baseline under the exact corrected protocol.
4. Validate GRAIN under full-negative, multi-seed, validation-threshold low-FPR, and event-level settings.

## Bottom Line

The paper is now credible as an attack-centric evaluation and corrected-baseline research paper. The remaining hard gaps are not formatting or table problems; they are external-protocol and deployment-evidence gaps. The safest current claim is:

> GRAIN-CAN is a strong corrected baseline under attack-centric CT&T evaluation, and the metric-forensics evidence shows why normal-dominated metrics can mislead rare-attack CAN IDS evaluation.

The unsafe claim remains:

> cross-vehicle unknown-attack CAN IDS is solved.
