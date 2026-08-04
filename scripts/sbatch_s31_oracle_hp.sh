#!/bin/bash
# Stage 31 round 6B oracle draft for submission from the grammar login node.
# grammar's default CPU partition is intentionally used: no GPU partition/GRES.
# PREPARE ONLY: this file does not submit itself.
#
# N=64 one-worker smoke arithmetic time: 0.985 s.
# Conservative O(N^2), no-parallel-speedup extrapolation:
#   N=2048: 0.985*(2048/64)^2 = 1009 s = 16.8 min
#   N=4096: 0.985*(4096/64)^2 = 4035 s = 67.3 min
#   serial sum = 84.1 min; 03:00 gives >2.1x wall-time margin.
#SBATCH --job-name=s31_ka2_hp80
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/s31_oracle_hp_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/s31_oracle_hp_%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this draft with sbatch from the repository root}
cd "$REPO_ROOT"

test -f scripts/s31_ka2_oracle_hp.py
test -d docs/s31_results
python3 -c 'import mpmath; assert mpmath.mp.dps >= 15'

export PYTHONUNBUFFERED=1
python3 scripts/s31_ka2_oracle_hp.py \
    --nref 2048 4096 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out docs/s31_results/ka2_oracle_hp.json \
    --checkpoint-dir docs/s31_results/ka2_oracle_hp.checkpoints \
    --log docs/s31_results/ka2_oracle_hp.progress.log

# Lightweight judgment is deliberately a separate, explicit driver action:
# python3 scripts/s31_ka2_judge.py \
#   --oracle docs/s31_results/ka2_oracle_hp.json \
#   --out docs/s31_results/ka2_oracle_hp_judgment.json
