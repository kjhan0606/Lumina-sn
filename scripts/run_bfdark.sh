#!/bin/bash
# hot-band s36-40 neighbour-illumination FALSIFIER (diagnostic, not physics).
# Base = repro22 champion config (commit 64de233, bit-identical verified).
# Arms: A = LUMINA_DIAG_BF_DARK=36:40:30 (band's own >=30eV bf lamp off)
#       B = LUMINA_DIAG_BF_DARK=36:49:30 (band + all outer lamps off)
# Verdict: A hot => external illumination sustains the hot root;
#          B cold (~13-20kK) => bf-lamp family is the whole sustainer;
#          B hot => bf-lamp framing refuted.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true

export OMP_NUM_THREADS=60 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_CMF_SOLVE_GPU=1
MODEL=data/tardis_reference_toy06_19p48d

export CUDA_VISIBLE_DEVICES=0
export LUMINA_NLTE=1
export LUMINA_NLTE_GREY_ITERS=2 SUPER_LEVELS=1 LUMINA_NLTE_GREY_TAU=2
export LUMINA_MACROATOM_NEUTRAL_E=1 LUMINA_SUPER_CUTOFF=100
export LUMINA_GAMMA_DEP=1 LUMINA_NLTE_ASSEMBLE_GPU=1
export LUMINA_DEPOSITION_FILE=$MODEL/deposition_cmfgen.csv
export LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=12 LUMINA_CMFGEN_ALI_ITER=8
export LUMINA_BF_OPACITY=1 LUMINA_CMFGEN_SIGMA_BF=$MODEL/cmfgen_sigma_bf.bin
export LUMINA_DYNAMIC_TRANSPROB=1 LUMINA_NLTE_SKIP_Z=14 LUMINA_NLTE_START_ITER=2
export LUMINA_NLTE_FLOOR_REG=0 LUMINA_NLTE_INV_CEIL=1e4
export LUMINA_RADEQ_TE=1 LUMINA_RADEQ_COOL_ESCAPE=0
export LUMINA_RADEQ_COOL_NONNEG=0 LUMINA_RADEQ_COOL_NLTE_ONLY=0
export LUMINA_RADEQ_LINE_RESPOND=1 LUMINA_RADEQ_DAMP=0.5
export LUMINA_COUPLED_NEWTON=0 LUMINA_DIP_TRACE=1 LUMINA_CN_DAMP=0.5 LUMINA_COUPLED_NEWTON_SMIN=20 LUMINA_COUPLED_JNU_PHOTOION=1 LUMINA_FROZENIN=0
export LUMINA_NLTE_PER_ION_RESCALE=1 LUMINA_COUPLED_JNU_LSTAR=0
export LUMINA_COUPLED_LAMBDA_STAR=1 LUMINA_COUPLED_TDEP=1 LUMINA_RADEQ_LINE_RE=0
export LUMINA_TE_TRAD_RATIO=1.0 LUMINA_LINE_INTERACTION=macroatom
export LUMINA_TAU_BY_ION=1 LUMINA_DIFFUSE_INNER_BC=1 LUMINA_ENERGY_BUDGET=1
export LUMINA_NLTE_LTE_FLOOR=1 LUMINA_NLTE_COLL_FIX=1 LUMINA_NLTE_ION_LOCK=1
export LUMINA_NLTE_LOCK_START_ITER=0 LUMINA_NLTE_FALLBACK_TE=1
export LUMINA_CMFGEN_FROZEN_CONT=1 LUMINA_CMFGEN_FROZEN_ALI=60
export LUMINA_MAX_INTERACTIONS=1000 LUMINA_MACROATOM_EWEIGHT=1
export LUMINA_CMFGEN_THEN_MC=0
export LUMINA_CMFGEN_LINE_EPS_PHYS=1

export LUMINA_STAGE0_DIAG=1
export LUMINA_RADEQ_DIAG=1
export LUMINA_CN_RTRUTH=1
export LUMINA_COUPLED_NT=1
export LUMINA_FROZENIN_DR=1
export LUMINA_TDEP_EQIC=1
export LUMINA_RADEQ_FB_RATE=1
export LUMINA_HRESP_CLAMP=1.0
export LUMINA_TE_STEP_CLAMP=1
export LUMINA_BF_RATE_POPS=1
export LUMINA_ETLA_ALLOW_HEAT=1
export LUMINA_RADEQ_SIMUL=1
export LUMINA_TRAD_COLOR_FIX=1
export LUMINA_J_DAMP=0.5
export LUMINA_RADEQ_VR_STD=1
export LUMINA_ION_POP_DUMP=1
export LUMINA_INNER_BB_SCALE=1.0
export LUMINA_CMF_DEP_SOURCE=1

# falsifier extras
export LUMINA_CMFGEN_JDUMP=1
export LUMINA_SIMUL_TRACE=38

run() {  # $1 = tag, $2 = LUMINA_DIAG_BF_DARK value ("" = control)
  local OUT=logs/stage1_toy06_$1
  mkdir -p "$OUT"
  if [ -n "$2" ]; then export LUMINA_DIAG_BF_DARK="$2"; else unset LUMINA_DIAG_BF_DARK; fi
  export LUMINA_STAGE0_CSV=$OUT/stage0.csv
  rm -f "$LUMINA_STAGE0_CSV"
  echo "[run] $1  LUMINA_DIAG_BF_DARK=${2:-<off>} -> $OUT"
  ./lumina_cuda "$MODEL" 100000 8 spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "[done] $1 rc=$?"
  for csv in lumina_spectrum_formal.csv lumina_spectrum.csv lumina_plasma_state.csv lumina_cmfgen_jnu.csv; do
    [ -f "$csv" ] && cp -f "$csv" "$OUT/$csv"
  done
}

case "${1:-all}" in
  A)   run bfdarkA "36:40:30" ;;
  B)   run bfdarkB "36:49:30" ;;
  D)   export LUMINA_DIAG_LINE_DARK=1; run bfdarkD "36:49:30" ;;
  E)   export LUMINA_DIAG_LINE_DARK=1 LUMINA_DIAG_DARK_ITERS=8; run bfdarkE "36:49:30" ;;
  M)   export LUMINA_CMF_BF_MILNE=1; run milne2 "" ;;
  M2)  export LUMINA_CMF_BF_MILNE=2; run milne3 "" ;;
  P)   export LUMINA_CMF_EPAY=1; run epay1 "" ;;
  P2)  export LUMINA_CMF_EPAY=1; run epay2 "" ;;
  P3)  export LUMINA_CMF_EPAY=1 LUMINA_CMF_EPAY_SMIN=5; run epay3 "" ;;
  P4)  export LUMINA_CMF_EPAY=2 LUMINA_CMF_EPAY_SMIN=5 LUMINA_CMF_BF_MILNE=2; run epay4 "" ;;
  P5)  export LUMINA_CMF_EPAY=2 LUMINA_CMF_EPAY_SMIN=5 LUMINA_CMF_BF_MILNE=2; run epay5 "" ;;
  all) run bfdarkA "36:40:30"; run bfdarkB "36:49:30" ;;
esac
echo "ALL DONE"
