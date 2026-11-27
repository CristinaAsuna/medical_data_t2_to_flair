#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o scratch/slurm/%x-%j.out
#SBATCH -e scratch/slurm/%x-%j.err
set -euo pipefail
source tools/env.sh
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
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv
python -u experiments/2026-11-27-baseline-selfrdb/smoke.py
