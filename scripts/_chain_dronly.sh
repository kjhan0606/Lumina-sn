#!/bin/bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
# 1. wait for baldiag to FULLY finish (no two-on-one-GPU)
while pgrep -x lumina_cuda >/dev/null; do sleep 30; done
sleep 5
echo "===== baldiag DONE — converged clean baseline =====" > logs/chain_dronly.log
echo "-- final T_e[0,25,49] --" >> logs/chain_dronly.log
grep "\[CMFGEN\] iter" logs/stage1_toy06_baldiag/stdout.log | tail -2 >> logs/chain_dronly.log
echo "-- converged RADEQ-BAL (s=25,37,43,49) --" >> logs/chain_dronly.log
grep "RADEQ-BAL" logs/stage1_toy06_baldiag/stdout.log | tail -4 >> logs/chain_dronly.log
grep "RADEQ-DIAG s=49" logs/stage1_toy06_baldiag/stdout.log | tail -1 >> logs/chain_dronly.log
# 2. launch DR-only
echo "===== launching DR-only (FROZENIN_DR=1, no boost/NT) =====" >> logs/chain_dronly.log
nohup bash scripts/run_dronly.sh > logs/dronly.driver.log 2>&1 &
sleep 8
pgrep -x lumina_cuda >/dev/null && echo "DR-only RUNNING" >> logs/chain_dronly.log || echo "DR-only FAILED to start" >> logs/chain_dronly.log
