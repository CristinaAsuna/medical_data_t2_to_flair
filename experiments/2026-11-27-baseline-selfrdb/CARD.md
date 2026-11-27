# 2026-11-27-baseline-selfrdb

类型：ablation（外部方法基线）　状态：queued（2026-11-27 用户批准）

## 问题
同一 BraTS main 划分上从头训练 SelfRDB 后，其病灶 Dice 相对 F1@50k 如何？与 UNet-L1、pix2pix 和续训后的 F1@80k 在图像、病灶指标及计算成本上各差多少？

## 影响的决定
SelfRDB 相对 F1@50k 的 Dice 配对均值差若为正且双侧 95% CI 下界高于 0，在论文中将其列为本验证集上病灶 Dice 更高的外部参照。若上界低于 0，报告 F1 的优势。区间含 0 则保留两者，报告差值与不确定性。

## 预测
agent 预测 SelfRDB 的平均 Dice 高于 F1@50k（参考值 0.8192）。这是运行前的方向性判断。

## 推翻条件
SelfRDB−F1@50k 的 Dice 点估计 ≤0 即推翻方向预测。点估计和区间分别报告。

## 相对主干的改动
以 icon-lab/SelfRDB 官方 T2→FLAIR 实现替换生成方法，使用 main_train 的同一 300 例，从随机初始化训练 50k 次参数更新。依 D-004 固定上游 commit 并保存原样副本，项目侧适配读写和评测。
上游 6262ede。单通道中心 T2→同层 FLAIR；项目裁剪 192，以背景 −1 居中补至 256，输出裁回 192。FP32、batch 4、Adam G/D 学习率 0.00016/0.0001、betas (0.5,0.9)，保留官方对抗与 L1、梯度惩罚及调度。50k 个训练 batch 各更新 G、D 一次，共 20 万张切片。推理 10 步、每步最多递归 2 次、阈值 0.01；采用项目评测 seed 0。

## 评测
同 PIPELINE.md §4。固定最后一步、seed 0、main_val 30 例及同一 S_eval；采用官方 T2→FLAIR 推理设置，记录采样步数及实际网络调用。核验预测的病例、切片顺序、空间对齐和强度范围，再调用唯一评测入口。main_test 冻结。
唯一主比较为 SelfRDB−F1@50k 的病例配对 Dice。用 t2flair.compare 报配对差、95% bootstrap CI 和 Wilcoxon p。与 UNet-L1@50k、pix2pix@50k、F1@80k 的比较及 PSNR、SSIM、LPIPS、HD95、假阳性和漏检病灶指标均为探索性。
按病例 ID 逐一匹配已有 result.json；F1@50k 使用 scratch/eval/f1_step50k_result.json。报告各方法的训练步数、有效 batch、推理开销与可取得的实测耗时。

## 计算和存储预算
拟用 1×A100 80 GB，smoke 限 30 分钟。正式训练和验证暂按 30 GPU 小时预算，存储上限 20 GB；两者均为拟议预算，正式提交前用 smoke 稳态吞吐、推理耗时、显存峰值和产物大小复算。预计超预算先询问。
报错、NaN、病例对齐失败或实际耗时超过核准预算 1.5 倍时停下汇报。

## 要保存的产物
result.json（含各对照的逐病例配对统计）、figures/train_curve.png、figures/val_samples.png；runs/<id>/ 下的 last.pt、best.pt（诊断）、训练日志和 main_val 推理缓存。

## 结果
待运行。
