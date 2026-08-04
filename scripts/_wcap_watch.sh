#!/bin/bash
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; cd "$ROOT"
JID=161630; TAG=superSE_wcap_n12
# Early check: first W table -> confirm shell0 W<=1
for i in $(seq 1 80); do
  L=$(ls -d logs/*${TAG}_${JID} 2>/dev/null | head -1)/stdout.log
  if [ -f "$L" ] && grep -q 'W_LUMINA.*W_TARDIS' "$L" 2>/dev/null; then
    echo "=== EARLY: first W table (cap verify, shell0 should be <=1.0) ==="
    awk '/Shell.*W_LUMINA/{p=1} p{print} /^    9 /{exit}' "$L" | head -12
    break
  fi
  sleep 30
done
# Wait terminal
until sacct -j "$JID" --format=State -n 2>/dev/null | head -1 | grep -qiE 'COMPLETED|FAILED|CANCEL|TIMEOUT|OUT_OF'; do sleep 60; done
echo ""; echo "### $JID terminal: $(sacct -j $JID --format=State,Elapsed -n 2>/dev/null|head -1)"
L=$(ls -d logs/*${TAG}_${JID} 2>/dev/null | head -1)/stdout.log
echo "=== FINAL per-shell W/T_rad (capped) shells 0-5 + last ==="
awk '/Shell.*W_LUMINA/{p=1} p{print}' "$L" | sed -n '1,9p;33,36p'
echo "=== plasma trailer ==="
grep -E 'Super-levels: (ACTIVE|off)|Mean \|W error|Mean \|T_rad error|T_inner final' "$L" 2>/dev/null | tail -4
echo "=== SCORE (W-cap on) ==="
python3 scripts/score_blondin_fscl_sn2002bo.py "$JID" "$TAG" 2>&1 | grep -vE '^\s*$' | sed -n '1,40p'
echo ""; echo "### DONE $(date)"
