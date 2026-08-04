#!/bin/bash
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$ROOT"
wait_terminal() {
  local j=$1
  until sacct -j "$j" --format=State -n 2>/dev/null | head -1 | grep -qiE 'COMPLETED|FAILED|CANCEL|TIMEOUT|OUT_OF'; do
    sleep 60
  done
  sacct -j "$j" --format=State,Elapsed -n 2>/dev/null | head -1
}
echo "### waiting superSE radeq-OFF 161616 ..."; echo "OFF: $(wait_terminal 161616)"
echo "### waiting superSE radeq-ON  161617 ...";  echo "ON:  $(wait_terminal 161617)"
for pair in "161616 superSE_n12" "161617 superSE_radeq_n12"; do
  set -- $pair
  L=$(ls -d logs/*${2}_${1} 2>/dev/null | head -1)/stdout.log
  echo ""; echo "=========== plasma trailer $1 ($2) ==========="
  grep -E 'Super-levels: (ACTIVE|off)|Mean \|W error|Mean \|T_rad error|T_inner final' "$L" 2>/dev/null | tail -4
  echo "=========== SCORE $1 ($2) ==========="
  python3 scripts/score_blondin_fscl_sn2002bo.py "$1" "$2" 2>&1 | grep -vE '^\s*$' | sed -n '1,40p'
done
echo ""; echo "### DONE $(date)"
