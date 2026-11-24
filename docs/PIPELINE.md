# 流程：从数据到评测

永久文档，上限 120 行。只在命令或接口变化时改写。每一步写一条可以直接运行的命令。

> 所有命令在项目根目录运行，先 `source tools/env.sh`（它把 `src/` 加进 PYTHONPATH）。集群作业脚本在 `tools/jobs/`，日志写到 `scratch/slurm/`（提交前 `mkdir -p scratch/slurm`）。L0 测试：`python -m pytest tests -q`（含合成数据的端到端 smoke）。

## 环境

- mac：`/Users/max/codebase/.ml/.venv/bin/python`，只做编码、L0 测试和不超过 5 分钟的 CPU smoke；不放数据。
- cluster：`/public_bme2/bme-cuizhm/maxquan/envs/t2flair`（uv venv，Python 3.12.13，torch 2.6.0+cu124）；项目在 `/public_bme2/bme-cuizhm/maxquan/Projects/medical_data_t2_to_flair`。集群是 CentOS 7 / glibc 2.17，只能装 manylinux_2_17 的预编译包，在登录节点运行：
  `uv pip install --python <env>/bin/python --python-platform x86_64-manylinux_2_17 --only-binary :all: -r requirements.txt`
- 每次运行前先 `source tools/env.sh`；集群作业脚本的第一条命令也是它。

## 1. 准备数据

```bash
python -m t2flair.splits --subjects <每行一个 ID 的文件> --out configs/splits   # 已生成并提交（1251 例），不要重跑
sbatch tools/jobs/prep.sbatch main    # bme_cpu：解压 tar 中的 t2/flair/seg 到 raw/（只做一次），再运行
# python -m t2flair.prep --raw $LAB_DATA_DIR/brats2021/raw --splits configs/splits/main --protocol main \
#     --out $LAB_DATA_DIR/brats2021/processed/main --workers 16
```

## 2. 训练

```bash
python -m t2flair.train --config configs/trunk.yaml                                   # 主干 F1
python -m t2flair.train --config configs/trunk.yaml --exp configs/exp/<id>.yaml       # 实验：只覆盖差异
python -m t2flair.train_seg --config configs/trunk.yaml --role {s_train,s_eval}       # 分割器（各训一次，所有实验共用）
# 集群：sbatch -p bme_a10080g --gres=gpu:NVIDIAA10080GBPCIe:1 -J <名称> tools/jobs/gpu.sbatch "train --config configs/trunk.yaml"
```

- 产物：`runs/<exp>/model/`（生成器），`runs/<seg.runs_exp>/{s_train,s_eval}/`（分割器），各含 `log.jsonl` 和 `checkpoints/{last,best}.pt`；中断后同一命令从 last.pt 续训。

## 3. 推理

```bash
python -m t2flair.infer --config configs/trunk.yaml [--exp ...] --split val --steps 50 --seed 0
```

- 输出写到 `runs/_cache/samples/<key 哈希>/`（`artifacts.cache_dir`，key = exp、best.pt 的指标、划分、步数、种子、σ），每个受试者一个 `.npz`。同一组 key 只生成一次，已完成就跳过。

## 4. 评测（唯一入口）

```bash
python -m t2flair.evaluate --config configs/trunk.yaml [--exp ...] --split val --steps 50 --seed 0 \
    --out experiments/<id>/result.json --run-id <lab run id>
python -m t2flair.figures --config configs/trunk.yaml --out experiments/<id>/figures   # train_curve.png、val_samples.png
```

- LPIPS 用 AlexNet 权重，需事先在登录节点下载到 `$LAB_MODELS_DIR/torch`（计算节点不能联网）。

- 评测集：开发阶段用 `main_val`（30 例）；`main_test`（100 例）冻结，只在最终报告时评测一次（D-003）。
- 主指标（全部先按受试者求值，再在受试者之间取均值±标准差）：
  - 画质：PSNR（dB，↑）、SSIM（↑）、MAE（↓）、LPIPS（↓），都只在脑区 mask 内计算，取值范围按 [-1,1] → [0,1] 换算。
  - 病灶：用 S_eval 分割合成的 FLAIR，得到预测掩膜 P；同一 S_eval 分割真实 FLAIR 得到 P_real（上界）；GT 掩膜为 G。报告 Dice(P,G)、HD95、假阳性体积 |P \ dilate(G,5px)|、漏检率 1−|P∩G|/|G|、小病灶召回，以及无肿瘤切片误报率。
- 必须对比的基线：恒等基线（直接输出中间切片的 T2）在同一入口上的数值；真实 FLAIR 上 S_eval 的 Dice（上界）。
- 统计：实验之间用按受试者配对的 Wilcoxon 检验，并报配对差的 95% bootstrap 置信区间。
- 输出：`experiments/<id>/result.json`

所有实验都用这一条命令评测，结果才能互相比较。不另写评测脚本。
