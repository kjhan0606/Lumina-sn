#!/bin/bash
#SBATCH --job-name=parity_runner
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
# parity-ladder runner: holds one GPU via slurm and executes queued run scripts
# sequentially INSIDE the allocation (proper accounting, no squatting, no per-run
# queue wait, and structurally serialized — repo-root artifacts never collide).
# Protocol: drop an executable .sh into runner_spool/queue/ ; it is moved to
# running/ during execution and to done/<name>.<rc> after. Touch runner_spool/STOP
# to make the runner exit after the current script.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$R"
SPOOL=$R/runner_spool
mkdir -p "$SPOOL/queue" "$SPOOL/running" "$SPOOL/done"
echo "runner up: host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} GPU=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
while true; do
  if [ -e "$SPOOL/STOP" ]; then echo "runner: STOP file — exiting"; rm -f "$SPOOL/STOP"; break; fi
  next=$(ls -1 "$SPOOL/queue" 2>/dev/null | sort | head -1)
  if [ -n "$next" ]; then
    mv "$SPOOL/queue/$next" "$SPOOL/running/$next"
    echo "runner: START $next  $(date '+%F %T')"
    bash "$SPOOL/running/$next"; rc=$?
    mv "$SPOOL/running/$next" "$SPOOL/done/$next.rc$rc"
    echo "runner: END $next rc=$rc  $(date '+%F %T')"
  else
    sleep 30
  fi
done
