#!/bin/bash
#SBATCH --job-name=adasdr_ca3
#SBATCH --partition=a10,a40,a100,a100_pcie
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Run adasdr.x post-processor on Ca III FULL DR oic
# Inputs:
#   - adasin (copied from ca3_target_full/adasin_template)
#   - o1 (symlink to ca3_dr_full/oic)
# Output: adf09 (Burgess-fittable resonance table)

cd /home/kjhan/local/autostructure/runs/ca3_dr_full

# Copy adasin from target run
cp -v /home/kjhan/local/autostructure/runs/ca3_target_full/adasin_template ./adasin

# Ensure o1 symlink exists
if [ ! -L o1 ]; then
    ln -sf oic o1
fi

echo "=== adasdr Ca III ==="
echo "Host: $(hostname)  Time: $(date)"
echo "adasin: NTAR1/NTAR2: $(head -2 adasin | tail -1)"
echo "o1 -> $(readlink o1)"

/home/kjhan/local/autostructure/post-procs/adasdr.x < adasin > adasdr.log 2>&1
rc=$?
echo "adasdr exit=$rc  $(date)"

if [ -f adf09 ]; then
    sz=$(ls -l adf09 | awk '{print $5}')
    nl=$(wc -l < adf09)
    echo "adf09: size=$sz bytes, $nl lines"
    echo "--- adf09 header ---"
    head -10 adf09
    echo "--- adf09 final ---"
    tail -5 adf09
fi

echo "Done: $(date)"
