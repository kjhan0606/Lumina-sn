#!/bin/bash
#SBATCH --job-name=as_dr_planC
#SBATCH --partition=a10,a40,a100,h100
#SBATCH --cpus-per-task=1
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Generic AUTOSTRUCTURE plan-C DR + adasdr post-processor.
# Usage: sbatch --job-name=as_dr_<ion> --export=ALL,RUN_DIR=/path/to/dr_planC slurm_as_dr_planC.sh
# Expects: $RUN_DIR/das (DR deck), $RUN_DIR/adasin (parent list).
# Writes: $RUN_DIR/{olg,oic,ols,o1,adf09,adasdr.log,adasout}.

set -e

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR not set"; exit 1
fi
echo "Host: $(hostname)  Time: $(date)  RUN_DIR: $RUN_DIR"

cd "$RUN_DIR"
echo "=== AS structure + DR run ==="
/home/kjhan/local/autostructure/aslm.x < das > stdout.log 2>&1
rc=$?
echo "  AS exit=$rc, files:"
ls -lh oic olg ols 2>/dev/null

if [ ! -s oic ]; then
    echo "ERROR: oic missing or empty"; exit 1
fi

ln -sf oic o1
echo "=== adasdr post-processor ==="
/home/kjhan/local/autostructure/post-procs/adasdr.x < adasin > adasdr.log 2>&1
rc2=$?
echo "  adasdr exit=$rc2"
ls -lh adf09 2>/dev/null

echo "Done: $(date)"
