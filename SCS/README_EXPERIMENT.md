# SCS-CAN experiment workspace

Active protocol: v3.0 one-week plan. Main datasets: ROAD, HCRL-CH, HCRL-SA.
CrySyS is excluded from the main run because free disk is below the plan's 15GB gate (~9GB free).
Windowing is dynamic: only frame tables and window indices may be persisted.

## Progress (2026-06-24)

- [x] 完整代码库 `/root/autodl-tmp/scs-can/src/`
- [x] HCRL-CH 预处理：18.5M 帧，185,454 窗口
- [x] ROAD 预处理：28.2M 帧，282,377 窗口
- [x] HCRL-SA 预处理：1.9M 帧，19,077 窗口
- [x] HCRL-CH Transformer baseline（AUROC=1.0）
- [x] Day1-3：Transformer / SCS-CAN w/o SSL / Full SCS-CAN（ROAD + HCRL-CH）
- [ ] Day4-6：少标签 / 跨车 / 消融 / CNN-LSTM（运行中）

Runbook: `README.md`
