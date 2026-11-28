# 实验登记表

每个实验一行，由 `tools/project.py new` 追加。状态：queued → running → done，另有 stopped（达到停止门槛）和 stale（主干已变，结果不再可比）。一句话结论不超过 40 字，写数字。从旧项目搬来的结论也记在这里，"卡片"一栏写旧项目里的路径。

| id | 类型 | 问题 | 影响的决定 | 状态 | 一句话结论 | 卡片 |
|---|---|---|---|---|---|---|
| 2026-11-24-trunk-bridge-fm | trunk | F1 端到端在 main_val 上的 PSNR 和病灶 Dice | 可信则锁 trunk-v1 并开消融；否则查契约或退回 F0 | done | 80k 19.14 dB / Dice 0.8288，锁 trunk-v1 | [卡片](experiments/2026-11-24-trunk-bridge-fm/CARD.md) |
| 2026-11-26-baseline-unet-l1 | ablation | 同网络同 50k 步的 L1 回归相对 F1@50k 的 PSNR/LPIPS/病灶指标 | 不差于 F1 则论文不能声称生成式收益 | done | PSNR +0.458 dB，LPIPS 差 0.0205，病灶 CI 含 0 | [卡片](experiments/2026-11-26-baseline-unet-l1/CARD.md) |
| 2026-11-26-baseline-pix2pix | ablation | 加对抗项后编造病灶数和 LPIPS 怎么变 | 决定论文是否保留"GAN 编造结构"的论述 | done | 编造 1.567，Holm p≥0.38；LPIPS 差 0.0186 | [卡片](experiments/2026-11-26-baseline-pix2pix/CARD.md) |
| 2026-11-27-baseline-selfrdb | ablation | SelfRDB 相对 F1 及两条基线的病灶与图像指标 | 决定如何报告外部桥模型的病灶 Dice | done | Dice 0.7815，比 F1@50k 低 0.0377（p=7e-5） | [卡片](experiments/2026-11-27-baseline-selfrdb/CARD.md) |
| 2026-11-28-f0-noise-start | ablation | 同网络同 50k 步只把起点换成噪声，F1@50k 相对 F0 的假阳性体积（H1） | H1 成立则保留贡献 2；否则删去，F0 作阴性结果 | queued | 待运行 | [卡片](experiments/2026-11-28-f0-noise-start/CARD.md) |
