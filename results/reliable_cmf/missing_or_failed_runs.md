# Missing or Failed Runs

- CT&T test01/test03/test04 Reliable-CMF-CAN full 50ep runs not completed in this turn.
- Reliable-CMF-CAN ablation variants were registered but not trained: no_reliability, no_shift, no_normality, no_segment.
- Segment evidence CSVs contain real topk_score, but full per-segment score vectors and top-k indices were not exported by the trainer.
- Three-seed CT&T shifted validation was not run because single-seed Reliable-CMF-CAN did not beat existing alternatives.
- Event-level metrics for Reliable-CMF-CAN were not recomputed in this turn.
