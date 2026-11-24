#!/bin/bash
# CPU job: extract T2/FLAIR/seg from the BraTS 2021 tar (once) and preprocess one protocol.
#   sbatch tools/jobs/prep.sbatch [main|lit]      (submit from the project root)
#SBATCH -J t2flair-prep
#SBATCH -N 1
#SBATCH -p bme_cpu
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o scratch/slurm/%x-%j.out
#SBATCH -e scratch/slurm/%x-%j.err
set -euo pipefail
source ~/.bashrc >/dev/null 2>&1 || true
source tools/env.sh
source /public_bme2/bme-cuizhm/maxquan/envs/t2flair/bin/activate
PROTOCOL="${1:-main}"
DATA="$LAB_DATA_DIR/brats2021"
echo "node ${SLURM_JOB_NODELIST:-local} start $(date)"
if [ ! -f "$DATA/raw/.extracted" ]; then
  mkdir -p "$DATA/raw"
  tar -xf "$DATA/BraTS2021_Training_Data.tar" -C "$DATA/raw" --wildcards '*_t2.nii.gz' '*_flair.nii.gz' '*_seg.nii.gz'
  n=$(ls -d "$DATA"/raw/BraTS2021_* | wc -l)
  echo "extracted $n subjects"
  [ "$n" -eq 1251 ] && touch "$DATA/raw/.extracted"
fi
python -m t2flair.prep --raw "$DATA/raw" --splits "configs/splits/$PROTOCOL" --protocol "$PROTOCOL" \
  --out "$DATA/processed/$PROTOCOL" --workers 16
du -sh "$DATA/raw" "$DATA/processed/$PROTOCOL"
echo "end $(date)"
