# 2026-11-24-trunk-bridge-fm

类型：trunk　状态：running（2026-11-24 用户批准）

## 问题
F1 桥式 FM 端到端（预处理 → 训练 → 50 步推理 → 唯一评测入口）在 main_val 上的 PSNR，以及 S_eval 病灶 Dice 与真实 FLAIR 上界的差，各是多少？

## 影响的决定
- 可信（PSNR 比恒等基线至少高 3 dB，样例目视合理，S_eval 在真实 FLAIR 上 Dice ≥ 0.80）：锁定 trunk-v1；开 ablation 卡片 F0、病灶加权、分割一致性，以及基线 UNet-L1、pix2pix、SelfRDB。
- 不可信：先排查数据契约和评测入口；如果 F1 本身训不出来，就改用 F0 作主干（新记一条决策，取代 D-002）。

## 预测
- agent：val PSNR 比恒等基线高 5–8 dB；合成 FLAIR 的 S_eval Dice 比上界低 0.05–0.15，主要损失在小病灶。
- 用户：（批准时未填写）

## 推翻条件
训练 20k 步时 val PSNR 仍不比恒等基线高 3 dB 以上，或者最终 Dice 比上界低 0.25 以上：说明 F1 这样设置不可用，agent 的预测被推翻。

## 相对主干的改动
本卡片定义主干本身：configs/trunk.yaml = F1（σ=0.5，UNet 基础通道 64，bs 64，lr 1e-4，EMA 0.9999，100k 步）。没有 configs/exp 文件。

## 评测
同 PIPELINE.md §4（main_val，50 步，seed 0）。本卡片同时实现唯一评测入口、S_train 和 S_eval。

## 计算和存储预算
- 集群：预处理 1 个 bme_cpu 作业，约 2 小时；F1 训练 1×A100 80G，约 12 GPU 小时；S_train 和 S_eval 各约 2 GPU 小时。合计约 16 GPU 小时。
- 存储上限 20 GB（runs/ 下全部产物）。超过 1.5 倍预算或出现 NaN 就停下来问。

## 要保存的产物
result.json；figures/val_samples.png；figures/train_curve.png；checkpoints/best.pt（F1、S_train、S_eval 各一个，放在各自的 run 目录）；runs/<id>/samples/val-steps50-seed0/（推理缓存，供后续 analysis 复用）。

## 结果
【运行后填写：数字、与主干的差、结论、这个实验不能说明什么】
