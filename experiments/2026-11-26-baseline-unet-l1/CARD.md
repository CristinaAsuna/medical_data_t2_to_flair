# 2026-11-26-baseline-unet-l1

类型：ablation（基线）　状态：queued（2026-11-26 用户批准开卡；提前于主干锁定，见 STATE.md）

## 问题
同一个 UNet、同样 50k 步，只把训练目标换成从 T2 直接 L1 回归 FLAIR（一步出图），在 main_val 上相对 F1@50k 的 PSNR、LPIPS、漏检率和编造病灶数各差多少？

## 影响的决定
- UNet-L1 在 PSNR、假阳性体积和漏检率上都不差于 F1@50k（配对差的 95% CI 不偏向 F1）：桥式生成在本协议上没有优势，论文不能声称生成式带来收益；病灶感知改动要同时加在 UNet-L1 上做对照。
- 否则（F1 在 LPIPS 或至少一项病灶指标上更好，且 PSNR 不差 0.5 dB 以上）：UNet-L1 只作参考行，按原计划做 F0 和病灶感知消融。

## 预测
- agent：UNet-L1 的 PSNR 与 F1@50k 相差 1 dB 以内（可能略高，回归偏向平滑）；LPIPS 比 F1 差 0.02 以上；编造病灶更少，但漏检率更高（小病灶被抹平）。
- 用户：（待填）

## 推翻条件
UNet-L1 的 LPIPS 不比 F1 差，或者漏检率不比 F1 高：说明"回归会抹平病灶"的预测不成立。

## 相对主干的改动
configs/exp/2026-11-26-baseline-unet-l1.yaml：`method.name: regression`（网络输入 x_t 固定为 T2、t=0，一次前向输出 FLAIR，L1 损失）；`infer.steps: 1`。网络、数据、优化器、EMA、50k 步和检查点规则（last.pt，D-006）都与主干相同。对比对象是主干的 `step50k.pt`（`--set infer.ckpt=step50k`），训练量相同。

## 评测
同 PIPELINE.md §4（main_val，seed 0，步数 1）。与 F1@50k 的比较用 `t2flair.compare`，全部是探索性的（参考行，不做 Holm）。

## 计算和存储预算
1×A100 80G，按 F1 每步约 0.74 s 估计 50k 步约 10.3 h；存储上限 5 GB（last/best 共约 1 GB 硬链接，推理缓存约 0.3 GB）。超过 15 h 或出现 NaN 就停下来问。

## 要保存的产物
result.json；figures/val_samples.png；figures/train_curve.png；runs/<id>/model/checkpoints/last.pt 和 best.pt（诊断）；推理缓存。

## 结果
【运行后填写：数字、与主干的差、结论、这个实验不能说明什么】
