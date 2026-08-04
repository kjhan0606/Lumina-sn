#!/bin/bash
#SBATCH --job-name=as_ca3_target
#SBATCH --partition=a10,a40,a100,a100_pcie
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# AUTOSTRUCTURE Ca III target-only structure run (Ar-like, Z=20).
# Generates LEVELS, TERMS files used to build adasin for DR step.
# MXCONF=6: ground 3p^6 + 5 single excitations (3p->3d,4s,4p,4d,4f).

cd /home/kjhan/local/autostructure/runs/ca3_target_full

echo "=== AS Ca III target (Ar-like Z=20) ==="
echo "Host: $(hostname)  Time: $(date)"

/home/kjhan/local/autostructure/aslm.x < das > stdout.log 2> stderr.log
rc=$?
echo "AS exit=$rc"

if [ -f LEVELS ]; then
    echo "--- LEVELS ---"
    head -20 LEVELS
fi

echo "Done: $(date)"
