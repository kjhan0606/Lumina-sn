#!/bin/bash
# Watch LUMINA pure-CMFGEN slurm jobs; block until all watched jobs leave the
# queue, then print a completion summary and EXIT (so the agent is re-invoked).
# Usage: slurm_watch.sh [jobid ...]   (default: all my ddc15_pc* jobs)
U=${USER:-kjhan}
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
watch=("$@")
if [ ${#watch[@]} -eq 0 ]; then
  mapfile -t watch < <(squeue -u "$U" -h -o "%i %j" | awk '$2 ~ /ddc15_pc/ {print $1}')
fi
if [ ${#watch[@]} -eq 0 ]; then echo "[slurm_watch] no LUMINA jobs to watch"; exit 0; fi
echo "[slurm_watch] watching: ${watch[*]}  (poll 90s)"
while :; do
  active=0
  for j in "${watch[@]}"; do
    [ -n "$(squeue -j "$j" -h -o '%t' 2>/dev/null)" ] && active=$((active+1))
  done
  [ "$active" -eq 0 ] && break
  sleep 90
done
echo "=================================================================="
echo "[slurm_watch] DONE — jobs finished: ${watch[*]}"
echo "=================================================================="
for j in "${watch[@]}"; do
  d=$(ls -d "$ROOT"/logs/ddc15_pc_phase3_*_"$j" 2>/dev/null | head -1)
  echo "--- job $j  dir=$d"
  [ -n "$d" ] && tail -2 "$d/stdout.log" 2>/dev/null
  [ -n "$d" ] && { echo "  dumps:"; ls -la "$d"/lumina_sl_vs_B.csv "$d"/nlte_budget_*.csv \
      "$d"/lumina_spectrum*.csv "$d"/lumina_plasma_state.csv 2>/dev/null | awk '{print "   ",$5,$NF}'; }
  [ -n "$d" ] && { fb=$(grep -c NLTE-FALLBACK "$d/stderr.log" 2>/dev/null); echo "  NLTE-FALLBACK lines: $fb"; }
done
