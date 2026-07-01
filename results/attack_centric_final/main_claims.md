# Main Claims

- Weighted/accuracy-like metrics can make all-normal or weak detectors look strong in rare-attack CAN IDS.
- CT&T test04 is an extreme example, but the metric trap appears across rare-attack settings.
- Corrected attack-centric metrics show test04 is not solved.
- Best current corrected test04 row: `GradientBoosting / window_100` with attack-F1=0.6678, AUPR=0.7845.
- GRAIN-CAN is a strong corrected baseline, not a final unknown-attack solution.
