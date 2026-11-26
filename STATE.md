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
4. 基线：SelfRDB（在 main 划分上重训，需 third_party 适配层）

数据与环境：BraTS 2021 tar 和 LPIPS AlexNet 权重已在 cluster 共享目录登记（见 MACHINES.md）。作业 `4362829` 完成 main 预处理，manifest 验收通过（300/30/100，最大肿瘤裁剪损失 0）。

## 下一步（当前卡片，在 cluster 上执行）

- 主干：50k 作业 `4363186` 完成（val_psnr 20.73，S_eval 0.838 / S_train 0.871），续训规则触发（+0.34 dB）→ 保留 `step50k.pt` 后续训到 80k（提示词见卡片），之后用 D-006 规则做 §5 第 4、5 步。
- 基线（用户 2026-11-26 批准，提前于主干锁定开卡）：`2026-11-26-baseline-unet-l1`、`2026-11-26-baseline-pix2pix`，代码已就绪，各 50k 步、约 10–12 h；与 F1 的 `step50k.pt` 比较。
- 作业名额：同时最多 2 个（含排队的 repa 4363187），免费 bme_a10080g 每人 1 张卡；排法见最新会话。

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
