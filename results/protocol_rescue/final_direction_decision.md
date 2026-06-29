# Final Direction Decision

1. CAN-Transformer+ should not be expected to directly exceed Transformer on ROAD under the current window=100 protocol.
2. CT&T official frame-level protocol reveals a much stronger signal on test02: GradientBoosting reaches F1=0.9662, compared with window=100 Transformer F1=0.1600 and -Ctx F1=0.8528.
3. test04 remains unsolved: official sample-level GradientBoosting reaches only F1=0.3332. Do not claim unknown vehicle + unknown attack is solved.
4. window=100 should be abandoned as the default shifted-CT&T protocol until window_size=10/20/50 experiments prove otherwise.
5. The next main direction should be feature-preserving sample/short-window detection with event-level and low-false-alarm evaluation.
6. The most promising CCF A/Security Four framing is not "new fusion model beats Transformer"; it is "protocol-correct, feature-preserving CAN IDS under distribution shift with deployable false-alarm control."
7. If short-window and false-alarm-budgeted detectors cannot preserve the official sample-level test02 strength while improving test04, the topic should be reframed as a protocol-gap/negative-result study or stopped.
