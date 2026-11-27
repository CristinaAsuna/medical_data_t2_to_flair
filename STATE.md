# 当前状态

更新：2026-11-27

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
4. 基线：SelfRDB 在 main 划分上重训并配对比较（CARD.md 已批准）。A100 smoke `4366755` 通过（1.542 it/s，13.46 GiB，50k≈9.0h）；正式作业 `4366827` 已提交 `bme_a10080g`（run_id `2026-11-27-baseline-selfrdb-20261127-163109-cluster`，含 50k 训练→推理→评测→4 组配对对比→出图）。

数据与环境：BraTS 2021 tar 和 LPIPS AlexNet 权重已在 cluster 共享目录登记（见 MACHINES.md）。作业 `4362829` 完成 main 预处理，manifest 验收通过（300/30/100，最大肿瘤裁剪损失 0）。

## 下一步（当前卡片，在 cluster 上执行）

- §5 第 0–5 步已执行。作业 `4363758` 已完成 F1@80k、F1@50k 和两条 50k 基线的 main_val 评测，三项运行已结束登记，详见各 result.json。
- 配对比较 `4366494` 完成，结果写入基线的 result.json。F1@50k 的 30 例源结果在 cluster 的 `scratch/eval/f1_step50k_result.json`，检查点为已校验的 `step50k.pt`（作业 `4363757`）。
- 三张卡片结果待用户核对。F1@80k 达到卡片的可信门槛，待用户确认后锁定 trunk-v1。main_test 按 D-003 保持冻结。

## 搁置的想法

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
