#!/bin/bash
# Wait for both radeq A/B legs, score each, print comparison + stability trace.
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$ROOT"
ON_JID=${1:-161421}; ON_TAG=${2:-radeq_milne_n12}
OFF_JID=${3:-161427}; OFF_TAG=${4:-radeq_off_n12}

wait_terminal() {
  local j=$1
  until sacct -j "$j" --format=State -n 2>/dev/null | head -1 | grep -qiE 'COMPLETED|FAILED|CANCEL|TIMEOUT|OUT_OF'; do
    sleep 60
  done
  sacct -j "$j" --format=State,Elapsed -n 2>/dev/null | head -1
}

echo "### waiting for radeq-ON $ON_JID ..."; ST_ON=$(wait_terminal "$ON_JID"); echo "ON state: $ST_ON"
echo "### waiting for radeq-OFF $OFF_JID ..."; ST_OFF=$(wait_terminal "$OFF_JID"); echo "OFF state: $ST_OFF"

ON_LOG=$(ls -d logs/*${ON_TAG}_${ON_JID} 2>/dev/null | head -1)/stdout.log
echo ""
echo "=========== radeq-ON shell0/shell29 T_e/T_rad per iter (stability) ==========="
grep -E "\[RADEQ\] T_e/T_rad" "$ON_LOG" 2>/dev/null
echo "--- radeq-ON convergence trailer ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|NaN/Inf=' "$ON_LOG" 2>/dev/null | tail -6

echo ""
echo "=========== SCORE radeq-OFF baseline ($OFF_JID) ==========="
python3 scripts/score_blondin_fscl_sn2002bo.py "$OFF_JID" "$OFF_TAG" 2>&1 | grep -vE '^\s*$' | sed -n '1,60p'
echo ""
echo "=========== SCORE radeq-ON ($ON_JID) ==========="
python3 scripts/score_blondin_fscl_sn2002bo.py "$ON_JID" "$ON_TAG" 2>&1 | grep -vE '^\s*$' | sed -n '1,60p'
echo ""
echo "### DONE $(date)"
