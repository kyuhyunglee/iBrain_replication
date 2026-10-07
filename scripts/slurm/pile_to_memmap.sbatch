#!/bin/bash
# One-time Neural Pile conversion for data.spike.format pile_mmap (U30). CPU only (node2/node4). Submit from the repo root:
#   sbatch scripts/slurm/pile_to_memmap.sbatch [split]
# Memory: 8 workers each hold one decompressed parquet file; MaxRSS was 50 GB on 2026-10-08 (4.6 min), so 12 x 5G.
#SBATCH --job-name=pile-memmap
#SBATCH --partition=debug
#SBATCH --exclude=node1,node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /scratch/connectome/zres0710/iBrain_replication
mkdir -p logs
# .venv only runs on node1 (uv Python under the node-local /home); this env lives on /scratch
PY=/scratch/connectome/zres0710/envs/ibrain-cpu/bin/python
$PY scripts/pile_to_memmap.py --root /storage/connectome/zres0710/iBrain_pretrain/spikes/neural-pile-primate \
    --out /scratch/connectome/zres0710/iBrain_cache/neural-pile-primate-mmap --split "${1:-train}" --workers 8
