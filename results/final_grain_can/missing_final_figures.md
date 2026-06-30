# Missing Final Figures

No requested final SVG is missing. All paper_fig*.svg files are present and passed the integrity audit.

Remaining evidence gaps, not figure-file gaps:
- Protocol B 2x negative-cap completed for key sample-level models and seeds 42/2024/2026.
- Protocol C 5x negative-cap completed for key sample-level models with seed 42 only.
- Protocol D full-negative/chunked full-negative was not completed and is explicitly marked resource-limited.
- Large-negative ExtraTrees/RandomForest/MLP variants were not completed in this revision; existing capped rows remain available.
- Low-FPR aggregate-window curves are from complete test score dumps but use best-test budget thresholds as upper bounds, not formal validation thresholds.
- Official event boundaries are unavailable; event metrics are approximate.
- LightGBM/XGBoost/Mamba were not run unless prior result files already existed.
