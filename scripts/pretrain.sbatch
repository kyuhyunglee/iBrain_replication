#!/bin/bash
# Lab server (node1~4) pretraining. Submit: sbatch scripts/pretrain.sbatch <config> <seed> <out> [--spike-only]
# GPU only via Slurm. Do not run python scripts/pretrain.py directly from a login shell.
#SBATCH --job-name=ibrain-pretrain
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12      # per-GPU maximum; the Neural Pile is held in memory (U30)
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -e
mkdir -p logs
CONFIG=${1:-configs/paper.yaml}
SEED=${2:-0}
OUT=${3:-runs/paper_s${SEED}}
EXTRA=${4:-}                      # e.g. --spike-only for M3
source activate ibrain
# May be preempted (requeue), so resume if ckpt.pt exists (SPEC resource rules)
RESUME=""
[ -f "$OUT/ckpt.pt" ] && RESUME="--resume $OUT/ckpt.pt"
python scripts/pretrain.py --config "$CONFIG" --seed "$SEED" --out "$OUT" $RESUME $EXTRA
