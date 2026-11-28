# 当前状态

更新：2026-11-28

## 当前主干

- tag：`trunk-v1`（2026-11-28 锁定，见 D-007）
- commit：`4d0e68d`（F1 正式训练与评测提交；锁定时仓库同步提交见 tag）
- 配置：configs/trunk.yaml（F1 布朗桥 FM，σ=0.5，50k 步作消融基准；主干续训至 80k）
- 评测集：main_val（30 例）；main_test（100 例）冻结（D-003）
- 主指标：F1@80k PSNR 19.135 dB / LPIPS 0.0758 / Dice 0.8288（F1@50k 18.939 dB / 0.0822 / 0.8192；恒等 11.901 dB / 0.4575；真实 FLAIR 参考 Dice 0.8383）
- checkpoint：cluster `runs/2026-11-24-trunk-bridge-fm/model/checkpoints/{last.pt(80k),step50k.pt}`；共享分割器 `runs/2026-11-24-trunk-bridge-fm/{s_train,s_eval}/checkpoints/last.pt`

## 当前唯一问题

待开下一张卡片：F0（从噪声出发的条件 FM）相对主干 F1@50k 在假阳性体积（H1 主终点）与 PSNR（−0.5 dB 非劣效）上怎么变？

## 排队（最多 5 项）

1. ablation：F0（从噪声出发）vs F1@50k（检验 H1）
2. ablation：F1 + 病灶加权（α=5，检验 H2）
3. ablation：F1 + 分割一致性（冻结 S_train，t>0.5，检验 H1/H2）

数据与环境：BraTS 2021 tar 与 LPIPS AlexNet 权重已在 cluster 共享目录登记（MACHINES.md）；`main` 预处理完成（300/30/100，最大肿瘤裁剪损失 0）；SelfRDB 独立环境在 `cluster:/public_bme2/bme-cuizhm/maxquan/envs/selfrdb`。

## 下一步

- 四张初始卡片（F1、UNet-L1、pix2pix、SelfRDB）均已核对结案为 `done`，`trunk-v1` 已锁定。
- 已生成四项实验的 `tools/project.py audit` 清单，待用户确认后按需清理冗余产物。
- 待用户批准后用 `tools/project.py new` 开 F0 消融卡片并进入下一轮实验；`main_test` 继续冻结。

## 搁置的想法

- Full（两种病灶损失都加）；采样步数曲线 {1,2,4,10,50}；文献协议 40 例
- σ 扫描 {0.25, 0.5, 1.0}；小波高频损失；α=10；加大训练例数；5 次采样的不确定性
