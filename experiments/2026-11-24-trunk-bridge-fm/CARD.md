# 2026-11-24-trunk-bridge-fm

类型：trunk　状态：done（2026-11-24 批准；2026-11-28 核对并锁定 trunk-v1）

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
本卡片定义主干本身：configs/trunk.yaml = F1（σ=0.5，UNet 基础通道 64，bs 64，lr 1e-4，EMA 0.9999，50k 步）。没有 configs/exp 文件。
- 2026-11-25 正式运行前改动（用户批准，预测和推翻条件不变）：smoke 4362831 只有 1.04 it/s，100k 步约 26 h，超过止损线。A100 测速 4362979：注意力头 8→64 维快 1.31 倍（1.40 it/s）；数据加载和 TF32 都不是瓶颈。因此改为 `attn_head_dim: 64`、100k→50k 步，按 1.40 it/s 约 9.9 h。channels_last 实测只有 0.76 倍（测速 4363005，更慢），不开；smoke2 4363121 实测 1.363 it/s，50k 步约 10.2 h。
- 续训规则（运行前定好，只用一次）：如果 45k 和 50k 两次验证中较高的 val_psnr 比 40k 时高出 0.3 dB 以上，就用同一条命令加 `--set train.steps=80000`，从 last.pt 续训到 80k（按 1.40 it/s 算总共约 15.9 h，没有超过 18 h 的止损线）；否则就停在 50k。
- 2026-11-26 续训已触发：作业 4363186 val_psnr 40k 20.39 → 45k 20.62 / 50k 20.73（+0.34 dB）；S_eval 最后一步 0.838（最好 0.860），S_train 0.871。续训前把 50k 的 last.pt 原样复制为 `step50k.pt`，供消融按 50k 预算比较。

## 评测
同 PIPELINE.md §4（main_val，50 步，seed 0）。本卡片同时实现唯一评测入口、S_train 和 S_eval。
- 2026-11-26 评测改动（用户批准，在评测前生效，预测和推翻条件不变；D-006）：生成器和分割器一律用最后一步的 last.pt，不再按 main_val 挑 best（best 只作诊断）；HD95 空预测记 373.13；新增编造病灶数与漏检病灶数，小病灶召回降为探索性。

## 计算和存储预算
- 集群：预处理 1 个 bme_cpu 作业，约 2 小时；F1 训练 1×A100 80G，约 12 GPU 小时；S_train 和 S_eval 各约 2 GPU 小时。合计约 16 GPU 小时。
- 存储上限 20 GB（runs/ 下全部产物）。超过 1.5 倍预算或出现 NaN 就停下来问。

## 要保存的产物
result.json；figures/val_samples.png；figures/train_curve.png；checkpoints/last.pt 和 best.pt（F1、S_train、S_eval 各一套，放在各自的 run 目录）；F1 的 checkpoints/step50k.pt（约 1 GB）；runs/<id>/samples/val-steps50-seed0/（推理缓存，供后续 analysis 复用）。

## 结果
作业 4363758 在 main_val 完成。F1@80k 的 PSNR 19.135 dB，恒等 11.901 dB（+7.234 dB）；病灶 Dice 0.8288，真实 FLAIR 参考 0.8383（差 −0.0095）。F1@50k 为 18.939 dB / 0.8192。三例样图保留主要病灶位置且脑室抑制合理，达标锁定 trunk-v1（D-007）。结果限于 main_val 与 S_eval 协议。
