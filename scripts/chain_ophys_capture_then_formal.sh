#!/bin/bash
# capture(4회) 정상 종료를 확인한 뒤에만 formal 을 이어 실행한다.
# capture 가 실패하면 formal 을 돌리지 않는다(잘못된 진리 생산 방지).
RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
LOG=$RUN/seq_logs/chain_capture_formal.log
: > "$LOG"
for i in $(seq 1 400); do
  pgrep -x cmfgen_dev.exe >/dev/null 2>&1 || break
  sleep 120
done
NC=$(ls "$RUN/seq_logs/captures" 2>/dev/null | wc -l)
NB=$(stat -c%s "$RUN/NETRATE" 2>/dev/null || echo 0)
TB=$(stat -c%s "$RUN/TOTRATE" 2>/dev/null || echo 0)
echo "CAPTURE_DONE captures=$NC netrate=$NB totrate=$TB" >> "$LOG"
if [ "$NC" -lt 4 ] || [ "$NB" -lt 1000 ] || [ "$TB" -lt 1000 ]; then
  echo "FORMAL_SKIPPED_CAPTURE_INCOMPLETE" >> "$LOG"; exit 3
fi
cd "$RUN" || exit 2
export SLURM_CPUS_PER_TASK=16
export SLURM_JOB_ID="formal-unconverged-$(date -u +%Y%m%dT%H%M%SZ)"
export OPHYS_MODE=formal
export OPHYS_FIX_T=T
echo "FORMAL_LAUNCH tag=$SLURM_JOB_ID" >> "$LOG"
bash "$RUN/submit_cmfgen_ophys.slurm" >> "$RUN/seq_logs/manual_formal.log" 2>&1
echo "FORMAL_RC=$?" >> "$LOG"
for n in CHI_DATA CHI_DATA_INFO ETA_DATA ETA_DATA_INFO; do
  echo "  $n=$(stat -c%s $RUN/$n 2>/dev/null || echo MISSING)" >> "$LOG"
done
