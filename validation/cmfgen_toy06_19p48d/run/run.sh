#!/bin/bash
cd /gpfs/kjhan/cmfgen_runs/toy06_19.48d
ulimit -s unlimited 2>/dev/null
export OMP_NUM_THREADS=16
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores
rm -f OUTGEN batch.log
nice -n 19 /gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe > batch.log 2>&1
echo "CMFGEN_EXIT=$?" >> batch.log
