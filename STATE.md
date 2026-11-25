# 当前状态

更新：2026-11-25

上限 40 行。每次会话结束时改写，不追加历史。

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

§5 第 0–2 步已完成：smoke 4362831 跑通，但只有 1.04 it/s。测速 4362979 之后，用户批准主干改为 `attn_head_dim: 64`、50k 步（约 9.9 h，见卡片）。

现在：在 A100 上测 channels_last（`bench_attn.py` 的 `*_cl` 变体）。如果加速明显，就在 trunk.yaml 里加 `model.channels_last: true`。然后跑短 smoke 确认速度，再进入 §5 第 3 步正式训练。

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
