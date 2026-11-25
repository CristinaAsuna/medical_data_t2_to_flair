# 实验登记表

每个实验一行，由 `tools/project.py new` 追加。状态：queued → running → done，另有 stopped（达到停止门槛）和 stale（主干已变，结果不再可比）。一句话结论不超过 40 字，写数字。从旧项目搬来的结论也记在这里，"卡片"一栏写旧项目里的路径。

| id | 类型 | 问题 | 影响的决定 | 状态 | 一句话结论 | 卡片 |
|---|---|---|---|---|---|---|
| 2026-11-24-trunk-bridge-fm | trunk | F1 端到端在 main_val 上的 PSNR 和病灶 Dice | 可信则锁 trunk-v1 并开消融；否则查契约或退回 F0 | running | 正式训练已提交，作业 4363186，进度见 STATE.md | [卡片](experiments/2026-11-24-trunk-bridge-fm/CARD.md) |
