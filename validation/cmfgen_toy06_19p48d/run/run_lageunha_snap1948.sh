#!/bin/bash
# run_lageunha_snap1948.sh -- direct 19.48d CMFGEN toy06 snapshot, MANUAL run on LagEunha.
#
# WHY MANUAL / WHY HERE (2026-07-16 23:40, user decision):
#   grammar (CPU slurm) is full -- 100/112 nodes alloc, the largest partial-node
#   gap is 12 cores, so a 16-core request cannot backfill and slurm estimated
#   StartTime=07:00 (~7.5 h away).  LagEunha is a SEPARATE server (10.0.200.25)
#   that shares /home and /gpfs, so the run dir, atomic data and exe are the same
#   files -- nothing is copied.  This does NOT violate the 2026-07-16 10:00
#   directive, which banned manual CMFGEN on the *syntax login node* only;
#   LagEunha is where manual runs are normal (feedback_lageunha_omp32).
#   grammar job 364521 was CANCELLED before this started: it would have woken at
#   07:00 in THIS SAME run dir and rm'd this run's scratch out from under it.
#
# ***** OMP_NUM_THREADS=16 IS MANDATORY -- DO NOT USE THE LAGEUNHA OMP=60 RULE *****
#   The standing lageunha manual-run rule (OMP=60) MUST NOT be applied to CMFGEN.
#   comp_opac.f:88 sums continuum opacity with an array REDUCTION under
#   SCHEDULE(DYNAMIC), so the FP summation order is thread-count dependent; at
#   32/64 threads it tips net chi negative at the cold outer plateau and
#   METHOD=LOGMON then takes LOG(chi<=0)=NaN (README R7).  16 is the only
#   thread count this model has ever run clean at.  60 is squarely in the
#   poisoned range.
#
# CORE BINDING: ramses_final3d holds 32 procs pinned to the EVEN cores 0,2,...,62
#   -- i.e. it spans BOTH sockets, it is NOT confined to socket 0.  The free
#   physical cores are the ODD ones.  We take the 16 odd cores that lie in NUMA
#   node1 (32-63): 33,35,...,63.  Zero core collision with ramses, and all 16
#   threads on one NUMA node.  Memory: ramses holds ~570 GB, ~423 GB free;
#   CMFGEN peaks at ~92 GB RSS.
set -u
DIR=/gpfs/kjhan/cmfgen_runs/toy06_19.48d
EXE=/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe
CORES=33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63

cd "$DIR" || exit 1
ulimit -s unlimited 2>/dev/null
export OMP_NUM_THREADS=16 OMP_STACKSIZE=512M OMP_PROC_BIND=close OMP_PLACES=cores

# fresh clean start (no restart-read of any stale scratch; the dead 364439
# EDDFACTOR is still on disk from 20:52 and MUST go)
rm -f OUTGEN batch.log EDDFACTOR* SCRTEMP POINT1 POINT2 \
      STEQ_VALS CORRECTION_SUM BAMAT* CSCRATCH* BA_ASCI* \
      JH_AT_CURRENT_TIME* NEG_OPAC MEANOPAC R_GRID_SELECTION fort.*

{ echo "HOST=$(hostname)"; echo "MODE=manual (no slurm)"; echo "OMP=16"; echo "CORES=$CORES"
  echo "PID=$$"
  grep -E '\[SN_AGE\]|\[T_MIN\]|\[DC_METH\]|\[MAX_LAM\]|\[MAX_LIN\]|\[LSTAR\]|\[DO_DDT\]' VADAT
  grep NUM_ITS IN_ITS; echo "START=$(date '+%F %T')"; } > run_lageunha.info

echo "[snap1948-lageunha] host=$(hostname) OMP=16 cores=$CORES start=$(date '+%F %T')"
taskset -c "$CORES" nice -n 5 "$EXE" > batch.log 2>&1
echo "CMFGEN_EXIT=$?" >> batch.log
echo "[snap1948-lageunha] end $(date '+%F %T')  $(grep CMFGEN_EXIT batch.log | tail -1)"
