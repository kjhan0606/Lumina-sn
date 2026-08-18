#!/bin/bash
# O-PHYS capture (NUM_ITS=1 x4) — user 승인 선택지 2:
# **미수렴 스냅샷**을 기계 게이트(L-1bb·L-4·L-3·L-5) 전용 ORACLE_INPUT 으로 추출.
# 공시 의무: run.fix_t=true · temperature_solved=false · heat_residual 미달 ·
# 정지 시점 MAXCH. 판정 이름은 PASS 가 아니라 PASS_UNCONVERGED_ORACLE 을 쓴다.
RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
cd "$RUN" || exit 2
export SLURM_CPUS_PER_TASK=16
export SLURM_JOB_ID="capture-unconverged-$(date -u +%Y%m%dT%H%M%SZ)"
export OPHYS_MODE=capture
export OPHYS_FIX_T=T
setsid nohup bash "$RUN/submit_cmfgen_ophys.slurm" \
  > "$RUN/seq_logs/manual_capture.log" 2>&1 < /dev/null &
echo "launched pid=$! tag=$SLURM_JOB_ID"
