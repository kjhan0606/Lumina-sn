#!/bin/bash
# Stage 0 (transport-coupled T_e) mechanism probe on toy06 + radioactive deposition.
# Read-only diagnostic: dumps 4pi*Int chi(B-J)dnu (line+cont) vs H_gamma at each
# deposition shell + per-bin J/B(Te) CSV. NO solver changes. See
# docs/TRANSPORT_COUPLED_TE_DESIGN.md Stage 0.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true

MODEL=data/tardis_reference_toy06_19p48d
OUT=logs/stage0_toy06_diag
mkdir -p "$OUT"

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
# --- toy06 config replicated from the last toy06 run banner ---
export LUMINA_NLTE_GREY_ITERS=2 SUPER_LEVELS=1 LUMINA_NLTE_GREY_TAU=2
export LUMINA_MACROATOM_NEUTRAL_E=1 LUMINA_SUPER_CUTOFF=100
export LUMINA_GAMMA_DEP=1 LUMINA_NLTE_ASSEMBLE_GPU=1
export LUMINA_DEPOSITION_FILE=$MODEL/deposition_cmfgen.csv
export LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=8 LUMINA_CMFGEN_ALI_ITER=8
export LUMINA_BF_OPACITY=1 LUMINA_CMFGEN_SIGMA_BF=$MODEL/cmfgen_sigma_bf.bin
export LUMINA_DYNAMIC_TRANSPROB=1 LUMINA_NLTE_SKIP_Z=14 LUMINA_NLTE_START_ITER=2
export LUMINA_NLTE_FLOOR_REG=0 LUMINA_NLTE_INV_CEIL=1e4
export LUMINA_RADEQ_TE=1 LUMINA_RADEQ_DIAG=1 LUMINA_RADEQ_COOL_ESCAPE=1
export LUMINA_RADEQ_COOL_NONNEG=0 LUMINA_RADEQ_COOL_NLTE_ONLY=0
export LUMINA_RADEQ_LINE_RESPOND=0 LUMINA_RADEQ_DAMP=0.3
export LUMINA_COUPLED_NEWTON=0 LUMINA_COUPLED_JNU_PHOTOION=1 LUMINA_FROZENIN=1
export LUMINA_NLTE_PER_ION_RESCALE=1 LUMINA_COUPLED_JNU_LSTAR=0
export LUMINA_COUPLED_LAMBDA_STAR=1 LUMINA_COUPLED_TDEP=1 LUMINA_RADEQ_LINE_RE=0
export LUMINA_TE_TRAD_RATIO=1.0 LUMINA_LINE_INTERACTION=macroatom
export LUMINA_TAU_BY_ION=1 LUMINA_DIFFUSE_INNER_BC=1 LUMINA_ENERGY_BUDGET=1
export LUMINA_NLTE_LTE_FLOOR=1 LUMINA_NLTE_COLL_FIX=1 LUMINA_NLTE_ION_LOCK=1
export LUMINA_NLTE_LOCK_START_ITER=0 LUMINA_NLTE_FALLBACK_TE=1
export LUMINA_CMFGEN_FROZEN_CONT=1 LUMINA_CMFGEN_FROZEN_ALI=60
export LUMINA_MAX_INTERACTIONS=1000 LUMINA_MACROATOM_EWEIGHT=1
# Stage 0 itself: skip the MC spectrum (plasma-only), enable the probe
export LUMINA_CMFGEN_THEN_MC=0
export LUMINA_STAGE0_DIAG=1
export LUMINA_STAGE0_CSV=$OUT/lumina_stage0_jminusb.csv

# enable_nlte requires the "nlte" positional arg (argv[5]) OR LUMINA_NLTE=1;
# without it compute_radiative_equilibrium_te early-returns (NLTE state NULL) and
# the RADEQ/STAGE0 loop never runs. argv = <model> <n_packets> <iters> spectrum nlte
export LUMINA_NLTE=1

rm -f "$LUMINA_STAGE0_CSV"
./lumina_cuda "$MODEL" 100000 8 spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log"
echo "DONE rc=$?  log=$OUT/stdout.log"
