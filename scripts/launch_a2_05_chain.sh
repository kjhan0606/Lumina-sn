#!/bin/bash
# A2-05 CHAIN lane launcher (lageunha). Absolute paths throughout.
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$REPO" || exit 2
make l1bf_fixture > /tmp/a205_fixture_build.log 2>&1 || { echo "make FAILED"; exit 2; }
setsid nohup python3 scripts/a2_05_chain_lane.py --workers 32 \
  > "$REPO/validation/a2_05/chain_lane_run.log" 2>&1 < /dev/null &
echo "launched pid=$!"
