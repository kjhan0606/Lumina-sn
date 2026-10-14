#!/usr/bin/env bash
# Invoke on syntax; all calculations are supervised on the authorized remote host.
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 lageunha \
 'systemd-run --user --unit=lumina-r2-ada-20261014 --wait --pipe \
 -p RuntimeMaxSec=86280 -p TimeoutStopSec=120 -p MemoryMax=192G \
 -p CPUQuota=1600% -p TasksMax=512 -p KillMode=control-group -p Restart=no \
 /bin/bash /gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014/tools/run_r2_ada.sh'
