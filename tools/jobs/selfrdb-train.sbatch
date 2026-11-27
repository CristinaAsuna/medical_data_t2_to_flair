#!/bin/bash
# Formal SelfRDB 50k training + main_val inference (envs/selfrdb) followed by unified
# evaluation, paired comparisons, and figures (envs/t2flair).
# Usage:
#   sbatch -p bme_a10080g --gres=gpu:NVIDIAA10080GBPCIe:1 -J selfrdb-50k \
#       tools/jobs/selfrdb-train.sbatch <lab_run_id>
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o scratch/slurm/%x-%j.out
#SBATCH -e scratch/slurm/%x-%j.err
set -euo pipefail

RUN_ID="${1:?usage: sbatch ... tools/jobs/selfrdb-train.sbatch <lab_run_id>}"
EXP_CFG="configs/exp/2026-11-27-baseline-selfrdb.yaml"
OUT_JSON="experiments/2026-11-27-baseline-selfrdb/result.json"
OUT_FIGS="experiments/2026-11-27-baseline-selfrdb/figures"

source tools/env.sh
echo "node ${SLURM_JOB_NODELIST:-local} start $(date) run_id=${RUN_ID}"
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv

# Phase 1: train + infer in isolated subshell with envs/selfrdb + GCC 7.3.1 + CUDA 11.8
(
  module load compiler/gcc/7.3.1 cuda/7/11.8
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export CUDAHOSTCXX="$CXX"
  unset SLURM_NTASKS
  export PATH="/public_bme2/bme-cuizhm/maxquan/envs/selfrdb/bin:$PATH"
  export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-/tmp}/selfrdb-ext-${SLURM_JOB_ID}"
  export MAX_JOBS=2
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export MPLBACKEND=Agg
  export PYTHONNOUSERSITE=1

  echo "== Phase 1a: SelfRDB 50k train ($(date))"
  python -u experiments/2026-11-27-baseline-selfrdb/train.py \
    --config configs/trunk.yaml --exp "$EXP_CFG"

  echo "== Phase 1b: SelfRDB main_val infer ($(date))"
  python -u experiments/2026-11-27-baseline-selfrdb/infer.py \
    --config configs/trunk.yaml --exp "$EXP_CFG" \
    --split val --steps 10 --seed 0 --batch 32
)

# Phase 2: single evaluation entry point, paired comparisons, and figures in envs/t2flair
echo "== Phase 2: t2flair evaluate + compare + figures ($(date))"
source /public_bme2/bme-cuizhm/maxquan/envs/t2flair/bin/activate
export MPLBACKEND=Agg

python -m t2flair.evaluate \
  --config configs/trunk.yaml --exp "$EXP_CFG" \
  --split val --steps 10 --seed 0 \
  --out "$OUT_JSON" --run-id "$RUN_ID"

METRICS="psnr ssim mae lpips hd95 fp_volume miss_rate fp_lesions missed_lesions lesion_detection empty_slice_false_alarm small_lesion_recall"

python -m t2flair.compare pair \
  --a "$OUT_JSON" --b scratch/eval/f1_step50k_result.json \
  --name selfrdb_vs_f1_50k --primary dice:higher --metrics $METRICS --into "$OUT_JSON"

python -m t2flair.compare pair \
  --a "$OUT_JSON" --b experiments/2026-11-26-baseline-unet-l1/result.json \
  --name selfrdb_vs_unet_l1 --primary dice:higher --metrics $METRICS --into "$OUT_JSON"

python -m t2flair.compare pair \
  --a "$OUT_JSON" --b experiments/2026-11-26-baseline-pix2pix/result.json \
  --name selfrdb_vs_pix2pix --primary dice:higher --metrics $METRICS --into "$OUT_JSON"

python -m t2flair.compare pair \
  --a "$OUT_JSON" --b experiments/2026-11-24-trunk-bridge-fm/result.json \
  --name selfrdb_vs_f1_80k --primary dice:higher --metrics $METRICS --into "$OUT_JSON"

python -m t2flair.figures \
  --config configs/trunk.yaml --exp "$EXP_CFG" \
  --steps 10 --seed 0 --out "$OUT_FIGS"

echo "end $(date)"
