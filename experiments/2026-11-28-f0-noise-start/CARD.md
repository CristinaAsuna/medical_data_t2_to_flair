# 2026-11-28-f0-noise-start

类型：ablation　状态：running（2026-11-28 用户批准开卡，指定终点预测）

## 问题
同一网络、同样 50k 步，只把起点从 T2 换成高斯噪声（F0），在 main_val 上 F1@50k 相对 F0 的假阳性体积（H1 主终点）差多少，PSNR 是否非劣（−0.5 dB）？

## 影响的决定
- F1−F0 的假阳性体积配对均值差 <0 且 95% CI 上界 <0，同时 PSNR 非劣（单侧 95% 下界 > −0.5 dB）：H1 成立。论文保留贡献 2"从源图出发比从噪声出发少编造病灶（受控对比）"；F0 不再进入病灶感知消融。
- 否则：H1 不成立。论文删去贡献 2 的主张，F0 作为阴性结果报告（起点不改变病灶编造）；贡献收窄为评测协议和 H2。
- 主终点 p 值最后与 H2 一起做 Holm 校正（D-006）；Holm 后不显著也按"不成立"处理。

## 预测
- agent：F0 的假阳性体积比 F1@50k 高 10% 以上（F1@50k 为 1230 体素），编造病灶数更多；PSNR 与 F1@50k 相差 0.5 dB 以内。30 例下 CI 可能含 0。
- 用户：（待填）

## 推翻条件
F1−F0 的假阳性体积点估计 ≥0（F0 不比 F1 编造得多）：agent 的预测被推翻。

## 相对主干的改动
configs/exp/2026-11-28-f0-noise-start.yaml：`method.name: noise_fm`（`sigma: 0.0` 只区分缓存键）。路径改为 x_t=(1−t)ε+t·x，t=0 为纯噪声，T2 只经三层条件进入网络；仍用终点预测、对 FLAIR 的 MSE。采样从 ε 出发做 50 步确定性 Euler：x_s=x_t+(s−t)/(1−t)·(x̂−x_t)，最后一步输出 x̂（bridge.noise_sample）。网络、2.5D 条件、bs 64、lr、EMA、50k 步、推理 50 步、last.pt 规则（D-006）都与主干相同。对比对象是主干 `step50k.pt` 的评测结果。

## 评测
同 PIPELINE.md §4（main_val，50 步，seed 0，last.pt）。主比较（A=F1@50k，B=F0）：
`python -m t2flair.compare pair --a scratch/eval/f1_step50k_result.json --b experiments/2026-11-28-f0-noise-start/result.json --name f1_50k_vs_f0 --primary fp_volume:lower --noninf psnr:higher:0.5 --metrics ssim mae lpips dice hd95 miss_rate fp_lesions missed_lesions lesion_detection --into experiments/2026-11-28-f0-noise-start/result.json`
F0 对 UNet-L1、pix2pix 的比较及其余指标均为探索性。

## 计算和存储预算
1×A100 80G（bme_a10080g）。网络与 F1 相同，按 1.36 it/s 计 50k 步约 10.2 h，推理与评测约 1 h，合计约 12 GPU 小时。存储上限 5 GB。超过 18 h、出现 NaN 或 val_psnr 在 20k 步仍不比恒等基线高 3 dB 就停下来问。

## 要保存的产物
result.json；figures/val_samples.png；figures/train_curve.png；runs/<id>/model/{checkpoints/{last,best}.pt,swanlog/}；main_val 推理缓存（50 步，seed 0）。

## 结果
【运行后填写：数字、与主干的差、结论、这个实验不能说明什么】
