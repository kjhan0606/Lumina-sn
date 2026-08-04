#!/bin/bash
#SBATCH --job-name=adasdr_only
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/adasdr_%j.log
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/adasdr_%j.err
#SBATCH --time=UNLIMITED
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -u
cd "${RUN_DIR:?RUN_DIR required}"
echo "[$(date)] adasdr rerun in $RUN_DIR"
ls -lh oic adasin
[ -e o1 ] || ln -sf oic o1
/home/kjhan/local/autostructure/post-procs/adasdr.x < adasin > adasdr.log 2>&1
rc=$?
echo "[$(date)] adasdr exit=$rc"
ls -lh adf09 adasdr.log adasout 2>&1 || true
tail -40 adasdr.log
exit $rc
