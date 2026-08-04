#!/bin/bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
# wait for S1 (s2fph) to finish
until grep -q "Done (pure-CMFGEN)" logs/stage1_toy06_s2fph/stdout.log 2>/dev/null || ! pgrep -x lumina_cuda >/dev/null; do sleep 20; done
sleep 8
echo "=== S1 (window 1000-4000A) DONE; final T_e[49]=$(grep '\[CMFGEN\] iter' logs/stage1_toy06_s2fph/stdout.log | tail -1) ==="
# ensure GPU free then launch S2
sleep 5
while pgrep -x lumina_cuda >/dev/null; do sleep 5; done
echo "=== launching S2 (LAMLO=228, OMP=64) ==="
bash scripts/run_s2_fph2.sh
echo "=== S2 DONE ==="
