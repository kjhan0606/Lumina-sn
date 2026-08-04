#!/bin/bash
#SBATCH --job-name=parity_runner2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
# Second parity runner. Executes in the GPFS clone /gpfs/kjhan/lumina_runner2 so
# repo-root artifacts NEVER collide with runner 1 (see runner_spool/README.INTERLOCK
# — the collision constraint is the shared root, so runner2 gets its own root).
# h200 ONLY: judgment runs are compared bytewise/numerically against H200 baselines;
# other GPU models are an unverified FP-determinism variable (register: cross-GPU
# probe required before opening h100/a100).
# Spool bookkeeping stays in the MAIN repo (runner2_spool/) so verdict tooling and
# backups see everything; after every job the clone's logs/ are rsynced back.
# Launchers queued here MUST cd "$LUMINA_RUN_ROOT" (exported below), not the repo.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C=/gpfs/kjhan/lumina_runner2
export LUMINA_RUN_ROOT=$C
SPOOL=$R/runner2_spool
mkdir -p "$SPOOL/queue" "$SPOOL/running" "$SPOOL/done" "$C/logs"
cd "$C"
echo "runner2 up: host=$(hostname) job=$SLURM_JOB_ID root=$C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} GPU=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
while true; do
  if [ -e "$SPOOL/STOP" ]; then echo "runner2: STOP file — exiting"; rm -f "$SPOOL/STOP"; break; fi
  next=$(ls -1 "$SPOOL/queue" 2>/dev/null | sort | head -1)
  if [ -n "$next" ]; then
    mv "$SPOOL/queue/$next" "$SPOOL/running/$next"
    echo "runner2: START $next  $(date '+%F %T')"
    bash "$SPOOL/running/$next"; rc=$?
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.rc$rc"
    rsync -a "$C/logs/" "$R/logs/" && echo "runner2: logs synced -> main repo"
    echo "runner2: END $next rc=$rc  $(date '+%F %T')"
  else
    sleep 30
  fi
done
