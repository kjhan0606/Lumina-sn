#!/bin/bash
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$ROOT"
OFF_JID=161586; OFF_TAG=superlev_n12
ON_JID=161597;  ON_TAG=superlev_radeq_n12
wait_terminal() {
  local j=$1
  until sacct -j "$j" --format=State -n 2>/dev/null | head -1 | grep -qiE 'COMPLETED|FAILED|CANCEL|TIMEOUT|OUT_OF'; do
    sleep 60
  done
  sacct -j "$j" --format=State,Elapsed -n 2>/dev/null | head -1
}
echo "### waiting super-level radeq-OFF $OFF_JID ..."; echo "OFF: $(wait_terminal $OFF_JID)"
echo "### waiting super-level radeq-ON  $ON_JID ...";  echo "ON:  $(wait_terminal $ON_JID)"
for pair in "$OFF_JID $OFF_TAG" "$ON_JID $ON_TAG"; do
  set -- $pair
  L=$(ls -d logs/*${2}_${1} 2>/dev/null | head -1)/stdout.log
  echo ""; echo "=========== plasma trailer $1 ($2) ==========="
  grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' "$L" 2>/dev/null | tail -3
  echo "=========== SCORE $1 ($2) ==========="
  python3 scripts/score_blondin_fscl_sn2002bo.py "$1" "$2" 2>&1 | grep -vE '^\s*$' | sed -n '1,40p'
done
echo ""; echo "### DONE $(date)"
