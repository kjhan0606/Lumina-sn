#!/bin/bash
#SBATCH --job-name=as_sc3_full
#SBATCH --partition=long,short,medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# AUTOSTRUCTURE Sc III full-physics DR deck (MXCONF=5, MXCCF=15, COREX='3-4').
# Pilot was MXCONF=2 (3d+4s only), undersized. Full deck adds 4p/4d/4f channels
# expected to raise α_DR by 2-5×. Output: oic → o1 (symlink) → adasdr → adf09.

cd /home/kjhan/local/autostructure/runs/sc3_dr_full

echo "=== AS Sc III FULL-PHYSICS DR ==="
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "deck:"; cat das | head -30; echo "..."

/home/kjhan/local/autostructure/aslm.x < das > as_stdout.log 2> as_stderr.log
rc_as=$?
echo "AS exit=$rc_as  $(date)"
echo "Files: $(ls -la oic olg ols 2>&1 | head -5)"

if [ -f oic ] && [ $rc_as -eq 0 ]; then
    ln -sf oic o1
    echo "--- adasdr post-process ---"
    /home/kjhan/local/autostructure/post-procs/adasdr.x < adasin > adasdr.log 2>&1
    rc_pp=$?
    echo "adasdr exit=$rc_pp"
    if [ -f adf09 ]; then
        echo "adf09 size: $(ls -l adf09 | awk '{print $5}') bytes"
        echo "--- adf09 header ---"
        head -10 adf09
    fi
fi

echo "Done: $(date)"
