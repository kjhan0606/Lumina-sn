#!/bin/bash
# run_lageunha_armF.sh -- STINT-2 next rung, ARM F (fresh damped-LAMBDA insurance), LagEunha.
#
# ARM F DESIGN (insurance/hedge, orthogonal to Arm N): a FRESH LTE@trueT cold start with the
#   SAME (T,ne)-seeded SN_HYDRO_DATA, FIX_T=T, LAM_VAL kept 400 (stock LAMBDA-dominated
#   cadence), MAX_LIN=MAX_LAM 10->3 (damped), NUM_ITS=60. Tests whether damping the LAMBDA
#   step to 3x + more iterations lets the 23-dex demand decay PASS the depth-47 plateau
#   WITHOUT the it35-style re-excitation that wrecked stint-1's ring (the ring may be a
#   10x-cap overshoot artifact). Distinct from Arm N (which forces Newton coupling).
#
# CONFIG PREP ALREADY APPLIED to the clone dir (this wrapper only WIPES SCRATCH for a fresh
#   start; it makes NO config edit): VADAT MAX_LIN=MAX_LAM=3, FIX_T=T, LAM_VAL=400, DC_METH=LTE
#   (pure LTE, ZERO *_IN). IN_ITS NUM_ITS=60, DO_LAM_IT=F.
#
# OMP_NUM_THREADS=16 MANDATORY. CORES: NUMA node0 ODD 1..31; ramses owns EVEN; disjoint from
#   ramses AND from Arm N (node1 odd 33-63).
set -u
DIR=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_armF
EXE=/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe
CORES=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31

cd "$DIR" || { echo "[armF] cd $DIR failed" >&2; exit 1; }
ulimit -s unlimited 2>/dev/null
export OMP_NUM_THREADS=16 OMP_STACKSIZE=512M OMP_PROC_BIND=close OMP_PLACES=cores

# drop cloned-in archive subdirs + stale run metadata (clutter from cp -a); keep config+atomic links
rm -rf crashbak_* unconv_* OUTGEN_stint1_fixT batch_stint1.log run_*.info CMFGEN_PID 2>/dev/null
# FULL scratch clean => NEWMOD fresh LTE start (no *_IN in this dir; nothing seed-like removed)
rm -f OUTGEN batch.log EDDFACTOR* SCRTEMP POINT1 POINT2 \
      STEQ_VALS CORRECTION_SUM CORRECTION_LINK BAMAT* BAMATPNT CSCRATCH* BA_ASCI* \
      JH_AT_CURRENT_TIME* NEG_OPAC MEANOPAC RVTJ R_GRID_SELECTION ADIABAT_CHK \
      COLLISION_SUMMARY CFDAT_OUT CONT_FREQ fort.*

{ echo "HOST=$(hostname)"; echo "ARM=F (fresh LTE@trueT, FIX_T=T, damped MAX_LIN=MAX_LAM=3, LAM_VAL=400, 60 iters)"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"; echo "CORES=$CORES"; echo "WRAPPER_PID=$$"
  grep -E '\[FIX_T\]|\[LAM_VAL\]|\[MAX_LIN\]|\[MAX_LAM\]|\[DC_METH\]|\[SN_AGE\]|\[T_MIN\]|\[DO_DDT\]' VADAT
  echo "XzV_IN count (must be 0): $(ls *_IN 2>/dev/null | wc -l)"
  grep NUM_ITS IN_ITS; echo "START=$(date '+%F %T')"; } > run_armF.info

echo "[armF] host=$(hostname) OMP=$OMP_NUM_THREADS cores=$CORES start=$(date '+%F %T')"
taskset -c "$CORES" nice -n 5 "$EXE" > batch.log 2>&1 &
CPID=$!
echo "$CPID" > CMFGEN_PID
echo "[armF] cmfgen pid=$CPID cwd=$DIR"
wait "$CPID"
echo "CMFGEN_EXIT=$?" >> batch.log
echo "[armF] end $(date '+%F %T') $(grep CMFGEN_EXIT batch.log | tail -1)"
