# 流程：从数据到评测

永久文档，上限 120 行。只在命令或接口变化时改写。每一步写一条可以直接运行的命令。

> 状态：接口已定，代码尚未实现（等主干卡片批准）。下面的命令是约定的接口，实现时按此对齐；改动接口要同步改这里。

## 环境

- mac：`/Users/max/codebase/.ml/.venv/bin/python`，只做编码、L0 测试和不超过 5 分钟的 CPU smoke；不放数据。
- cluster：`<work_root>/envs/t2flair`（uv 建的 venv，torch 的 CUDA 版本按集群驱动选择），用 `pip install -r requirements.txt` 安装。确切版本装好后用 `pip freeze` 锁进 requirements.txt。
- 每次运行前先 `source tools/env.sh`；集群作业脚本的第一条命令也是它。

## 1. 准备数据

```bash
python -m t2flair.splits --data $LAB_DATA_DIR/brats2021 --out configs/splits      # 生成 main_* 和 lit_* 两套 ID 列表（只需一次）
python -m t2flair.prep --data $LAB_DATA_DIR/brats2021 --split configs/splits/main --out $LAB_DATA_DIR/brats2021/processed/main
```

## 2. 训练

```bash
python -m t2flair.train --config configs/trunk.yaml                                   # 主干 F1
python -m t2flair.train --config configs/trunk.yaml --exp configs/exp/<id>.yaml       # 实验：只覆盖差异
python -m t2flair.train_seg --config configs/trunk.yaml --role {s_train,s_eval}       # 分割器（各训一次，所有实验共用）
```

## 3. 推理

```bash
python -m t2flair.infer --run runs/<id> --split val --steps 50 --seed 0
```

- 输出写到 `runs/<id>/samples/<split>-steps<N>-seed<S>/`，每个受试者一个 `.npz`。同一组 checkpoint、划分、步数和种子只生成一次，已存在就跳过。

## 4. 评测（唯一入口）

```bash
python -m t2flair.evaluate --run runs/<id> --split val --steps 50 --seed 0 --out experiments/<id>/result.json
```

- 评测集：开发阶段用 `main_val`（30 例）；`main_test`（100 例）冻结，只在最终报告时评测一次（D-003）。
- 主指标（全部先按受试者求值，再在受试者之间取均值±标准差）：
  - 画质：PSNR（dB，↑）、SSIM（↑）、MAE（↓）、LPIPS（↓），都只在脑区 mask 内计算，取值范围按 [-1,1] → [0,1] 换算。
  - 病灶：用 S_eval 分割合成的 FLAIR，得到预测掩膜 P；同一 S_eval 分割真实 FLAIR 得到 P_real（上界）；GT 掩膜为 G。报告 Dice(P,G)、HD95、假阳性体积 |P \ dilate(G,5px)|、漏检率 1−|P∩G|/|G|、小病灶召回，以及无肿瘤切片误报率。
- 必须对比的基线：恒等基线（直接输出中间切片的 T2）在同一入口上的数值；真实 FLAIR 上 S_eval 的 Dice（上界）。
- 统计：实验之间用按受试者配对的 Wilcoxon 检验，并报配对差的 95% bootstrap 置信区间。
- 输出：`experiments/<id>/result.json`

所有实验都用这一条命令评测，结果才能互相比较。不另写评测脚本。
