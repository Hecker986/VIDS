# Missing or Limited Items

- Official ML reproduction is a pilot: training keeps all sampled positives but caps negative samples; testing uses full official test folders.
- Full per-frame official ML score dumps were generated during evaluation but removed from the committed evidence package because they were about 1.5GB; aggregate metrics, logs, and scripts are retained for reproduction.
- Short-window datasets for window_size=10/20/50 were not materialized in this run, so their rows are marked NA in `ctt_granularity_comparison.csv`.
- Full public-benchmark reproduction would require confirming the exact original feature set, preprocessing, train/validation protocol, and whether all negative frames are used.
- Event-level metrics use contiguous positive windows as coarse events because no official attack event boundary files were found.
- Normality policy results are post-hoc score-combination pilots with test upper-bound thresholds; they are not formal validation-selected results.
- Mamba/SSM was not run in this protocol rescue stage.
