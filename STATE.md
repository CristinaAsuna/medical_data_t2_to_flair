# 当前状态

更新：2026-11-25

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
4. 基线：UNet-L1、pix2pix、SelfRDB（在 main 划分上重训）

数据与环境：BraTS 2021 tar 和 LPIPS AlexNet 权重已在 cluster 共享目录登记（见 MACHINES.md）。作业 `4362829` 完成 main 预处理，manifest 验收通过（300/30/100，最大肿瘤裁剪损失 0）。

## 下一步（当前卡片，在 cluster 上执行）

- §5 第 0–2 步已完成，测速与配置决策见卡片。Smoke2 `4363121` 验收通过（1.363 it/s，50k 约 10.2 小时），产物在 `scratch/smoke/2026-11-24-trunk-bridge-fm-smoke2/`。
- §5 第 3 步作业 `4363186` 已提交至 `bme_a10080g`，run_id `2026-11-24-trunk-bridge-fm-20261125-141234-cluster`，训练代码 `8074cf8`。依次运行 F1 50k、S_eval、S_train。
- 提交后检查显示因 `Priority` 排队，正式吞吐待核对。按 Slurm 预排 2026-11-25 17:04 开始、运行 14–15 小时估算，2026-11-26 07:04–08:04 完成（北京时间，随实际开跑时间更新）。
- 作业结束后按卡片比较 40k、45k 和 50k 的 val_psnr，决定是否续训到 80k，再执行 §5 第 4、5 步。

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
