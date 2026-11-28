#!/bin/bash
# GPU job running one or more t2flair modules in sequence. Partition and GPU are given at submit time:
#   sbatch -p bme_a10080g --gres=gpu:NVIDIAA10080GBPCIe:1 -J <name> tools/jobs/gpu.sbatch "train --config configs/trunk.yaml" ["infer ..."]
#   sbatch -p bme_gpu     --gres=gpu:NVIDIAA10080GBPCIe:1 -J <name> tools/jobs/gpu.sbatch "..."
# An argument whose first word ends in .py runs that script instead of a t2flair module.
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 48:00:00
#SBATCH -o scratch/slurm/%x-%j.out
#SBATCH -e scratch/slurm/%x-%j.err
source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
source tools/env.sh
source /public_bme2/bme-cuizhm/maxquan/envs/t2flair/bin/activate
echo "node ${SLURM_JOB_NODELIST:-local} start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"
for step in "$@"; do
  if [[ "${step%% *}" == *.py ]]; then  # an experiments/<id>/ script, e.g. "experiments/<id>/bench_attn.py --loader"
    echo "== python $step  ($(date))"
    python $step
  else
    echo "== python -m t2flair.$step  ($(date))"
    python -m t2flair.$step
  fi
done
echo "end $(date)"
