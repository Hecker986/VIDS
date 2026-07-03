# Final Submission Readiness

## Verdict

The manuscript is ready for a CIVS-style research-paper submission after compiling with the official template package. It is not a finished Security Four / CCF A artifact in the sense of having official author confusion matrices, exact v1.5 alignment, and official event boundaries. The paper is strongest as a measurement-plus-baseline research paper:

**Attack-centric evaluation framework + metric-forensics case study + corrected benchmark + feature-preserving GRAIN-CAN baseline.**

## What is strong enough

- The core metric-trap argument is well supported.
- The CT&T test04 corrected benchmark is explicit and conservative.
- The manuscript no longer claims unknown attacks are solved.
- Self-designed exploratory CMF/Reliable models are removed from the paper's comparison story.
- GRAIN-CAN is defined as a reproducible feature-preserving baseline, not oversold as a new architecture.
- The final PDF is 16 pages in the local build.
- A strict LLNCS source is available for official-template compilation.

## What remains as limitation

- Original Table 13 metric definitions cannot be definitively resolved without author confusion matrices or code.
- Official event-level deployment evidence cannot be claimed without official event boundaries.
- Exact can-train-and-test-v1.5 alignment remains unresolved.
- Exhaustive full-negative Protocol D is not presented as complete.

## Submission guidance

Submit only with conservative language. The paper should say "metric ambiguity" and "attack-centric correction", not "prior work is wrong". It should say GRAIN-CAN is a strong corrected baseline, not a solved unknown-attack detector.
