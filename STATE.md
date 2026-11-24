# 当前状态

更新：2026-11-24

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

数据与环境：BraTS 2021 tar 已在 cluster `$LAB_DATA_DIR/brats2021/`（sha256 已核对，见 MACHINES.md）；环境见 PIPELINE.md。

## 下一步（当前卡片，在 cluster 上执行）

代码已完成并在 Mac 上通过 11 个测试（commit 见 git log）；集群上还没跑任何作业。按 docs/PIPELINE.md §5 的 0→5 顺序做：LPIPS 权重 → 预处理 → 真实数据 smoke（停下汇报 it/s 和耗时估算）→ F1 与分割器正式训练 → val 推理与评测 → 填卡片结果，等用户核对。

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
