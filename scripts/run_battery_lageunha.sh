#!/bin/bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 2
timeout 600 python3 scripts/run_gate_battery.py 2>&1 | tail -1
