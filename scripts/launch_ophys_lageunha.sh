#!/bin/bash
# O-PHYS solve 수동 런처 (lageunha). CMFGEN OMP=16 절대 규약.
# STAGE-1 (2026-08-06, user 승인 선택지 1): T 고정으로 populations/NETRATE/TOTRATE/
# CHI/ETA 를 먼저 확보한다. free-T 는 stage 2. FIX_T 는 반드시 명시 선언한다.
RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
cd "$RUN" || exit 2
export SLURM_CPUS_PER_TASK=16
export SLURM_JOB_ID="stage1-tfixed-$(date -u +%Y%m%dT%H%M%SZ)"
export OPHYS_MODE=solve
export OPHYS_FIX_T=T          # STAGE-1 선언. stage 2 에서는 F 로 되돌린다.
setsid nohup bash "$RUN/submit_cmfgen_ophys.slurm" \
  > "$RUN/seq_logs/manual_solve.log" 2>&1 < /dev/null &
echo "launched pid=$! job_tag=$SLURM_JOB_ID FIX_T=$OPHYS_FIX_T"
