# 项目：目标、背景和已有结论

永久文档，上限 150 行。只在目标或结论变化时改写。

## 目标

输入一张脑 MRI 的 T2 轴向切片（及相邻两张），生成同位置的 FLAIR。做成的标准：在 PSNR/SSIM 不低于条件 FM 基线的前提下，病灶级幻觉更少，即用独立分割器 S_eval 测得的假阳性体积和漏检率更低、病灶 Dice 更接近真实 FLAIR 上界（定义见 PIPELINE.md §4）。

课程：上科大《医学大数据与人工智能》，4 人，约 2 周，产出 IEEE TMI 格式论文、项目报告、带录屏的 PPT 和写作 Skill。

## 背景

- FLAIR 和 T2 的主要差别：FLAIR 抑制脑脊液信号（偏全局、低频的对比度变化），水肿和病灶更突出。合成时的主要风险是把病灶抹掉，或者编造出不存在的病灶。
- 方法演进：GAN（pix2pix、ResViT）→ 从噪声出发的条件扩散（DDPM、SynDiff）→ 以源图像为端点的扩散桥（BBDM、I2SB、SelfRDB，MedIA 2025）→ 桥加小波分解（FADM，J. Imaging 2025；WRB，MICCAI 2026）。
- 已发表工作的共同不足：只报整图 PSNR/SSIM；数据子集很小（WRB 只用 40 例，测试 10 例）；按切片统计；没有病灶级评估。不同论文的 T2→FLAIR PSNR 在 26–31 dB 之间，口径不一致，不可直接比较。
- 能复现的强基线只有 SelfRDB（MIT 协议，有代码和权重）。WRB 和 FADM 目前都没有放出代码。
- 写论文前逐条核对原文。⚠ 已知疑点：WRB 表中 T2→FLAIR 的 BBDM/DDPM 数字与 SelfRDB 表中 I2SB/DDPM 的数字逐位相同，但两者的数据子集不同。

## 关键文献（链接）

- WRB（MICCAI 2026）：https://papers.miccai.org/miccai-2026/0884-Paper3942.html ；T2→FLAIR 27.51 dB / 91.33%，40 例，代码未放出
- FADM（J. Imaging 2025）：https://pmc.ncbi.nlm.nih.gov/articles/PMC12112230/ ；T2→FLAIR 30.85 dB / 0.958，100 例，无代码
- SelfRDB（MedIA 2025）：https://arxiv.org/abs/2405.06789 ，代码 https://github.com/icon-lab/SelfRDB （仓库报告 T2→FLAIR 27.98 dB）
- D3Bridge（BraTS 切分脚本）：https://github.com/yufengxiecs/D3Bridge
- GAN/扩散/FM 的 T1w→T2w 基准：https://arxiv.org/abs/2507.14575 ；MPFlow（用分割 Dice 衡量肿瘤幻觉）：https://arxiv.org/abs/2603.03710
- MICCAI 2025 可控 FM（加肿瘤约束）：https://papers.miccai.org/miccai-2025/paper/1113_paper.pdf

## 方法概要

- 共用骨架：2D UNet（diffusers UNet2DModel，基础通道 64，倍率 1/2/4/8，在 48 和 24 分辨率处加注意力），条件为 2.5D 的三张 T2 切片。
- F0（对照）：条件 FM，从高斯噪声出发积分到 FLAIR；与 F1 同网络、同条件、同终点预测损失，只差起点和路径。
- **F1（主干）**：桥式 FM，从中间切片的 T2 出发：x_t = (1−t)·y + t·x + σ·√(t(1−t))·ε。网络直接预测 FLAIR；采样用布朗桥 SDE：x ← x + (x̂ − x)/(1−t)·Δt + σ·√Δt·z。
- 病灶感知改动（作为消融，叠加在 F1 上）：病灶加权损失 W = 1 + (α−1)·dilate(M)；分割一致性损失（冻结的 S_train，只在 t>0.5 时启用）。两者都加即为 Full。
- 对比方法：UNet-L1 回归、pix2pix、SelfRDB（在本项目划分上重训）、F0。

## 已有结论

每条一行，写数字，并标明证据强度：`[已验证]` 表示在当前评测入口上复现过，`[历史]` 表示来自旧项目且未在本仓库复现，`[观察]` 表示定性印象。

- （暂无。主干尚未锁定。）

## 已排除的方向

每条一行：试过什么、结果如何、为什么停。细节指向 EXPERIMENTS.md 里的那一行。

- IXI T1 ×4 层间超分 + 观测一致性投影（未运行，选题阶段排除）：健康脑没有病灶，"抑制幻觉"无法直接量化；方法贡献偏后处理分析。见 D-001。
