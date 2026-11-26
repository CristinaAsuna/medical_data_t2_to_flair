# 当前状态

更新：2026-11-26

## 当前主干

- tag：【待填：主干未锁定，锁定后打 trunk-v1】
- commit：【待填：主干未锁定】
- 配置：configs/trunk.yaml（F1 桥式 FM，见 docs/PROJECT.md；内容在主干卡片批准后填写）
- 评测集：main_val（30 例）；main_test（100 例）冻结（D-003）
- 主指标：按受试者统计的 PSNR 和 S_eval 病灶 Dice；基线为恒等（输出 T2）和真实 FLAIR 上界
- checkpoint：无

## 当前唯一问题

F1 端到端在 main_val 上的 PSNR，以及病灶 Dice 与上界的差各是多少？→ 卡片：experiments/2026-11-24-trunk-bridge-fm/CARD.md（running）

## 排队（最多 5 项）

1. ablation：F0（从噪声出发）vs F1
2. ablation：F1 + 病灶加权（α=5）
3. ablation：F1 + 分割一致性
4. 基线：SelfRDB（在 main 划分上重训，需 third_party 适配层）

数据与环境：BraTS 2021 tar 和 LPIPS AlexNet 权重已在 cluster 共享目录登记（见 MACHINES.md）。作业 `4362829` 完成 main 预处理，manifest 验收通过（300/30/100，最大肿瘤裁剪损失 0）。

## 下一步（当前卡片，在 cluster 上执行）

- 主干 50k 作业 `4363186` 已结束登记，`step50k.pt` 与 50k 的 `last.pt` SHA-256 一致（归档作业 `4363757`）。
- 串行作业 `4363758` 于 11-26 10:21 在 `bme_a10080g` 开跑，代码 `4dbeb56`。按 D-006 在 main_val 评测 F1@80k、F1@50k 和两个基线。
- F1 run_id `2026-11-24-trunk-bridge-fm-20261126-102125-cluster`，确认从 50k 恢复，50,100–50,200 步实测 1.381 it/s，预计 11-26 16:34 完成（约 6.2 小时）。
- UNet-L1 run_id `2026-11-26-baseline-unet-l1-20261126-102128-cluster`，训练约 10 小时，预计 11-27 02:34 后完成。
- pix2pix run_id `2026-11-26-baseline-pix2pix-20261126-102130-cluster`，训练约 12 小时，预计 11-27 14:34 后完成。
- 时间均为北京时间，基线预计时刻需加前序推理、评测和出图时间。SLURM 时限 11-28 02:21；作业结束前集群不拉取代码。

## 搁置的想法

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
