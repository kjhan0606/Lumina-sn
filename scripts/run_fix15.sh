#!/bin/bash
# Stage 1 (transport-coupled T_e) A/B: deposition injected into the formal-solve
# emissivity (LUMINA_CMF_DEP_SOURCE) OFF vs ON. Gate = emergent luminosity must
# rise by ~Sum_s H_gamma[s]*V_shell (energy conservation). Sequential (one GPU).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true

export OMP_NUM_THREADS=60 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_CMF_SOLVE_GPU=1
MODEL=data/tardis_reference_toy06_19p48d

export OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=0
export LUMINA_CMF_SOLVE_GPU=1
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
export LUMINA_J_DAMP=0.5
export LUMINA_RADEQ_VR_STD=1
export LUMINA_ION_POP_DUMP=1
# inner blackbody amplitude: 1.0 = default (central luminosity dominates, double-
# counts deposition); 0 = ARTIS-style (deposition is the SOLE power source).
export LUMINA_INNER_BB_SCALE=${INNER_BB:-1.0}
SUF=${SUF:-}

run() {  # $1 = tag, $2 = dep_source (0|1)
  local OUT=logs/stage1_toy06_$1$SUF
  mkdir -p "$OUT"
  export LUMINA_CMF_DEP_SOURCE=$2
  export LUMINA_STAGE0_CSV=$OUT/stage0.csv
  rm -f "$LUMINA_STAGE0_CSV"
  echo "[run] $1  LUMINA_CMF_DEP_SOURCE=$2 -> $OUT"
  ./lumina_cuda "$MODEL" 100000 8 spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "[done] $1 rc=$?"
  # outputs land in cwd and are overwritten by the next run — snapshot them now
  for csv in lumina_spectrum_formal.csv lumina_spectrum.csv lumina_plasma_state.csv; do
    [ -f "$csv" ] && cp -f "$csv" "$OUT/$csv"
  done
}

run fix15 1
echo "ALL DONE"
