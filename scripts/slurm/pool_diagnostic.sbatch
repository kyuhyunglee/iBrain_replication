#!/bin/bash
# Pooling diagnostic without training, and the collapse metrics of the baseline R0 (2026-10-08). Submit from the repo root:
#   sbatch scripts/slurm/pool_diagnostic.sbatch [checkpoint ...]
#SBATCH --job-name=ibrain-pooldiag
#SBATCH --partition=debug
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /scratch/connectome/zres0710/iBrain_replication
mkdir -p logs
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
CKPTS=("$@")
[ ${#CKPTS[@]} -eq 0 ] && CKPTS=(runs/smoke_s0/final.pt)
python scripts/pool_diagnostic.py --ckpt "${CKPTS[@]}"
for c in "${CKPTS[@]}"; do python scripts/check_collapse.py --ckpt "$c"; done
