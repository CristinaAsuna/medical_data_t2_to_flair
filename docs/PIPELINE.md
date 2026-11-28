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
# 基线（同一 UNet，只换目标，见 baselines.py）：--exp configs/exp/2026-11-26-baseline-{unet-l1,pix2pix}.yaml；推理/评测用 --steps 1
# F0（同一 UNet，只换起点为噪声，见 bridge.py）：--exp configs/exp/2026-11-28-f0-noise-start.yaml；推理/评测用默认 50 步
# 与 F1 同训练量对比：主干推理加 --set infer.ckpt=step50k
# 集群：sbatch -p bme_a10080g --gres=gpu:NVIDIAA10080GBPCIe:1 -J <名称> tools/jobs/gpu.sbatch "train --config configs/trunk.yaml"
```

- 产物：`runs/<exp>/model/`（生成器），`runs/<seg.runs_exp>/{s_train,s_eval}/`（分割器），各含 `log.jsonl` 和 `checkpoints/{last,best}.pt`；中断后同一命令从 last.pt 续训。

## 3. 推理

```bash
python -m t2flair.infer --config configs/trunk.yaml [--exp ...] --split val --steps 50 --seed 0
```

- 用 `infer.ckpt` 指定的检查点（主干为 `last`：最后一步的 EMA 权重，不按 main_val 挑选；D-006）。
- 输出写到 `runs/_cache/samples/<key 哈希>/`（`artifacts.cache_dir`，key = exp、检查点名和它的步数（best 则为指标）、划分、步数、种子、σ），每个受试者一个 `.npz`。同一组 key 只生成一次，已完成就跳过。

## 4. 评测（唯一入口）

```bash
python -m t2flair.evaluate --config configs/trunk.yaml [--exp ...] --split val --steps 50 --seed 0 \
    --out experiments/<id>/result.json --run-id <lab run id>
python -m t2flair.figures --config configs/trunk.yaml --out experiments/<id>/figures   # train_curve.png、val_samples.png
```

- LPIPS 用 AlexNet 权重，需事先在登录节点下载到 `$LAB_MODELS_DIR/torch`（计算节点不能联网）。

- 评测集：开发阶段用 `main_val`（30 例）；`main_test`（100 例）冻结，只在最终报告时评测一次（D-003）。
- 主指标（全部先按受试者求值，再在受试者之间取均值±标准差）：
  - 画质：PSNR（dB，↑）、SSIM（↑）、MAE（↓）只在脑区 mask 内计算，取值范围按 [-1,1] → [0,1] 换算；LPIPS（↓）按整张切片计算（AlexNet，输入 [-1,1]）。
  - 病灶：用 S_eval（加载 `seg.ckpt`，主干为 `last`）分割合成的 FLAIR 得到 P；同一 S_eval 分割真实 FLAIR 得到参考 P_real（不是上界）；GT 为 G。报告 Dice(P,G)、HD95（恰有一方为空时记 373.13，并报空预测数）、假阳性体积 |P \ dilate(G,5)|、漏检率 1−|P∩G|/|G|、编造病灶数（P 中 ≥10 体素且不接触 dilate(G,5) 的连通域）、漏检病灶数和病灶检出率（G 中 ≥50 体素且 P 未接触的连通域）。小病灶召回和无肿瘤切片误报率为汇总比例，只作探索性指标。连通域都按 26 连通；阈值在 `eval` 配置里。
- 必须对比的参考：恒等（直接输出中间切片的 T2）和真实 FLAIR 在同一入口上的数值。
- 统计（D-006）：`python -m t2flair.compare pair --a <A>/result.json --b <B>/result.json --name <名称> --primary <指标>:lower|higher [--noninf <指标>:<方向>:<界限或百分比>] --into <A>/result.json`，按受试者配对：Wilcoxon p、配对均值差的 95% bootstrap CI；非劣效看单侧 95% 界；汇总比例用受试者 bootstrap。结果存进 A 的 result.json 的 `comparisons`。多个假设：`python -m t2flair.compare holm <A>/result.json:<名称> ...`（Holm 校正）。每个假设的主终点和界限在卡片里运行前写好，其余指标都是探索性的。
- 输出：`experiments/<id>/result.json`

所有实验都用这一条命令评测，结果才能互相比较。不另写评测脚本。

## 5. 在集群上执行主干卡片（2026-11-24-trunk-bridge-fm）

项目根：`/public_bme2/bme-cuizhm/maxquan/Projects/medical_data_t2_to_flair`；环境：`/public_bme2/bme-cuizhm/maxquan/envs/t2flair`。登录节点只做 git、下载和 sbatch，单个进程不超过 1 小时；计算全走 Slurm。同时最多 2 个作业、3 张 GPU（bme-cluster）。每一步执行前按 AGENTS.md §8 告知并等确认。

0. 登录节点：`git pull && mkdir -p scratch/slurm && source tools/env.sh && source <环境>/bin/activate`，然后 `python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)"` 下载 AlexNet 到 `$LAB_MODELS_DIR/torch`，登记总台 MACHINES.md。
1. 预处理：`sbatch tools/jobs/prep.sbatch main`。验收：`processed/main/manifest.json` 中 counts 为 300/30/100，`max_tumor_lost_by_crop` < 0.01（否则停下汇报）。
2. 真实数据 smoke（`bme_gpu`，约 20 分钟，产物只在 scratch/）：
   `sbatch -p bme_gpu --gres=gpu:NVIDIAA10080GBPCIe:1 -J t2f-smoke tools/jobs/gpu.sbatch "train --smoke --set train.steps=300 train.val_every=150 train.log_every=50" "train_seg --role s_eval --smoke --set seg.steps=300 seg.val_every=300" "infer --smoke --split val --steps 5" "evaluate --smoke --split val --steps 5 --out scratch/smoke/result_smoke.json" "figures --smoke --steps 5 --out scratch/smoke/figures"`
   验收：无报错、loss 下降、val_psnr 有限；用 `scratch/smoke/<exp>/model/log.jsonl` 的 time 字段算 it/s，估算 100k 步耗时。超过卡片预算 1.5 倍就停下问用户。
3. 正式运行（两个作业并行；每个提交后立即 `lab run start 2026-11-24-trunk-bridge-fm --job <作业号> -- "<同一命令>"`）：
   - `sbatch -p bme_a10080g --gres=gpu:NVIDIAA10080GBPCIe:1 -J t2f-f1 tools/jobs/gpu.sbatch "train --config configs/trunk.yaml"`
   - `sbatch -p bme_gpu --gres=gpu:NVIDIAA10080GBPCIe:1 -J t2f-seg tools/jobs/gpu.sbatch "train_seg --role s_eval" "train_seg --role s_train"`
   作业中断就用同一命令重交（从 last.pt 续训）。结束后 `lab run end <run_id> --status done|failed --outputs cluster:<runs 路径>`。
4. 续训（如触发）结束后，先在集群上 `git pull` 拿到评测代码，再：`sbatch -p <有空的 A100 分区> ... -J t2f-eval tools/jobs/gpu.sbatch "infer --split val --steps 50 --seed 0" "evaluate --split val --steps 50 --seed 0 --out experiments/2026-11-24-trunk-bridge-fm/result.json --run-id <run_id>" "figures --out experiments/2026-11-24-trunk-bridge-fm/figures"`，同样登记 lab run。
5. 收尾：把 result.json 的数字填进卡片"结果"一节（与恒等基线和上界比较、写这个结果不能说明什么），更新 EXPERIMENTS.md 一行和 STATE.md，`tools/project.py check`，提交并 `lab sync`。状态保持 running，等用户核对后再改 done、按 AGENTS.md §7 锁 trunk-v1；不要自己评测 main_test。
