# 2026-11-26-baseline-pix2pix

类型：ablation（基线）　状态：done（2026-11-26 批准；2026-11-28 核对完成）

## 问题
在 UNet-L1 的基础上加 pix2pix 的对抗项（条件 PatchGAN，L1 权重 100），编造病灶数和 LPIPS 相对 UNet-L1 和 F1@50k 怎么变？

## 影响的决定
- pix2pix 编造病灶数（或假阳性体积）明显高于 UNet-L1 和 F1（配对差的 95% CI 不含 0）：论文保留"对抗训练会编造结构"的论述，并用它作为病灶级评测能区分方法的正例。
- 否则：删掉论文相关工作中"GAN 会编造纹理"的论断，只报数字。

## 预测
- agent：pix2pix 的 LPIPS 比 UNet-L1 好（更清晰），PSNR 比 UNet-L1 低 0.5–2 dB；编造病灶数是三者中最多的。
- 用户：（待填）

## 推翻条件
pix2pix 的编造病灶数不高于 UNet-L1，或者 LPIPS 不比 UNet-L1 好。

## 相对主干的改动
configs/exp/2026-11-26-baseline-pix2pix.yaml：`method.name: pix2pix`（生成器与 UNet-L1 相同；判别器为 70×70 PatchGAN，输入中间 T2 与真/假 FLAIR 两通道，vanilla GAN 损失，`method.l1_weight: 100`）；按 pix2pix 原文用 Adam lr 2e-4、betas (0.5, 0.999)（G 和 D 相同）；`infer.steps: 1`。其余与主干相同，生成器仍用 EMA 和 last.pt（D-006）。

## 评测
同 PIPELINE.md §4（main_val，seed 0，步数 1）。与 UNet-L1、F1@50k 的比较用 `t2flair.compare`；"编造病灶数高于两者"这一条作为本卡唯一的主终点（Holm 校正 2 次比较），其余探索性。

## 计算和存储预算
1×A100 80G，判别器约增加 15%，50k 步约 12 h；存储上限 5 GB。超过 18 h、出现 NaN 或判别器损失塌到 0 就停下来问。

## 要保存的产物
result.json；figures/val_samples.png；figures/train_curve.png；runs/<id>/model/checkpoints/last.pt 和 best.pt（诊断，含判别器状态）；推理缓存。

## 结果
main_val 30 例配对，均为 50k 步。pix2pix 为 17.660 dB / LPIPS 0.1212 / Dice 0.7807。每例编造病灶数 1.567；相对 UNet-L1 差 +0.500（95% CI [−0.200, 1.233]，Holm p=0.381），相对 F1@50k 差 −0.267（[−1.800, 0.933]，Holm p=0.950）。
主终点未成立，且 LPIPS 比 UNet-L1 差 0.0186（95% CI [0.0138, 0.0232]），触发推翻条件。报告观察值，本协议结果不支持“对抗训练会编造结构”的论断。
