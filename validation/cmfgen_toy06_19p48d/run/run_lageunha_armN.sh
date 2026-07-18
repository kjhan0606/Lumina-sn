#!/bin/bash
# run_lageunha_armN.sh -- STINT-2 next rung, ARM N (forced-Newton, T held), MANUAL LagEunha.
#
# FORENSIC BASIS (stint-1 corpse toy06_19.48d, all source-cited):
#   Q1: FIX_T=T does NOT force LAMBDA. LAMBDA_ITERATION is init = RD_LAMBDA(=DO_LAM_IT=F)
#       [cmfgen_sub.f:608], NOT from FIX_T. Two FULL-LINEARIZATION iters DID run under
#       FIX_T=T (it1 & it35, both "BA computed" in OUTGEN). The all-LAMBDA lock came from
#       solve_for_pops.f:298 `IF(MAXCH .GT. 1.0E+05) LAMBDA_ITERATION=.TRUE.; FIXED_T=.TRUE.`
#       -- the ~1e6% demand never fell below the HARDCODED 1e5% override, so every iter was
#       forced LAMBDA. The lone Newton step (it35, fired when it34 demand 3.46e4 < 1e5 &
#       CNT_LAM>=NUM_LAM=2) OVERSHOT 3.46e4 -> 2.66e5 and moved the correction front d38->d47.
#   => "T release = Newton entry" is FALSE. The config Newton-entry knob is LAM_VAL:
#      solve_for_pops.f:286 branch A (MAXCH < VAL_DO_LAM) sets LAMBDA_ITERATION=.FALSE. and
#      FIXED_T=RD_FIX_T, BYPASSING the 1e5 override in branch B. Raising LAM_VAL 400 -> 1e8
#      makes the ~1e6% demand select FULL LINEARIZATION (Newton) while keeping T held.
#   Q2: the stuck 1e6-1e7% demand at depth 47 (v~14060 km/s, tau_Ross~0.07, optically-thin
#       outer IME/Fe shell) is the TRACE HIGH-ION tail -- FeSEV(FeVII)/FeSIX(FeVI)/NkSIX(NiVI)/
#       SkSIX(SVI), abundance-floor ions, ~1e5% corrections (CORRECTION_LINK). Dominant Fe II
#       is saturated (~0.98). Same disease as the SCRTEMP LTE-init killer; FIX_T contained it
#       (0 NaN) but LAMBDA's diagonal-only step cannot resolve the coupled ionization tail.
#
# ARM N DESIGN: continuation from it40 SCRTEMP, FIX_T=T KEPT, LAM_VAL 400->1e8 (force full
#   linearization = off-diagonal Jacobian coupling the trace tail needs), MAX_LIN=MAX_LAM
#   10->3 (damp the it35-style overshoot). Tests: does damped Newton-with-T-held drive the
#   trace-tail demand below 1e5% (break the LAMBDA lock) and toward convergence?
#
# CONFIG PREP ALREADY APPLIED (this wrapper makes NO edit, NO rm -- continuation):
#   VADAT: LAM_VAL 400->1.0D+08 ; MAX_LIN 10->3 ; MAX_LAM 10->3 ; FIX_T=T kept.
#   IN_ITS: NUM_ITS=40 (continuation adds iters 41..80; POINT1 NITSF=40 verified).
#   SCRATCH PRESERVED: SCRTEMP/POINT1/POINT2/EDDFACTOR* from it40 are the restart state.
#   Pristine corpse backed up -> unconv_stint1_fixT_bak/.
#
# OMP_NUM_THREADS=16 MANDATORY (comp_opac.f:88 REDUCTION NaN at 32/64). CORES: NUMA node1
#   ODD 33..63; ramses_final3d owns EVEN 0-62 (verified taskset); intersection EMPTY.
set -u
DIR=/gpfs/kjhan/cmfgen_runs/toy06_19.48d
EXE=/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe
CORES=33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63

cd "$DIR" || { echo "[armN] cd $DIR failed" >&2; exit 1; }
ulimit -s unlimited 2>/dev/null
export OMP_NUM_THREADS=16 OMP_STACKSIZE=512M OMP_PROC_BIND=close OMP_PLACES=cores

# archive stint-1 OUTGEN/batch (OUTGEN opens APPEND; keep a clean stint-1 copy)
cp -f OUTGEN OUTGEN_stint1_fixT 2>/dev/null || true
cp -f batch.log batch_stint1.log 2>/dev/null || true

{ echo "HOST=$(hostname)"; echo "ARM=N (forced-Newton LAM_VAL=1e8, FIX_T=T, damped 3x, it40 continuation)"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"; echo "CORES=$CORES"; echo "WRAPPER_PID=$$"
  grep -E '\[FIX_T\]|\[LAM_VAL\]|\[MAX_LIN\]|\[MAX_LAM\]|\[DC_METH\]|\[SN_AGE\]|\[T_MIN\]|\[DO_DDT\]' VADAT
  grep NUM_ITS IN_ITS; echo "POINT1: $(sed -n 2p POINT1)"; echo "START=$(date '+%F %T')"; } > run_armN.info

echo "[armN] host=$(hostname) OMP=$OMP_NUM_THREADS cores=$CORES start=$(date '+%F %T')"
taskset -c "$CORES" nice -n 5 "$EXE" > batch.log 2>&1 &
CPID=$!
echo "$CPID" > CMFGEN_PID
echo "[armN] cmfgen pid=$CPID cwd=$DIR"
wait "$CPID"
echo "CMFGEN_EXIT=$?" >> batch.log
echo "[armN] end $(date '+%F %T') $(grep CMFGEN_EXIT batch.log | tail -1)"
