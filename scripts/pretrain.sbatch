#!/bin/bash
# Lab server (node1~4) pretraining. Submit: sbatch scripts/pretrain.sbatch <config> <seed> <out>
# GPU only via Slurm. Do not run python scripts/pretrain.py directly from a login shell.
#SBATCH --job-name=ibrain-pretrain
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -e
mkdir -p logs
CONFIG=${1:-configs/paper.yaml}
SEED=${2:-0}
OUT=${3:-runs/paper_s${SEED}}
source activate ibrain
# May be preempted (requeue), so resume if ckpt.pt exists (SPEC resource rules)
RESUME=""
[ -f "$OUT/ckpt.pt" ] && RESUME="--resume $OUT/ckpt.pt"
python scripts/pretrain.py --config "$CONFIG" --seed "$SEED" --out "$OUT" $RESUME
