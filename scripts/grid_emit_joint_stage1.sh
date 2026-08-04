#!/bin/bash
# Path 2 — Stage 1 joint W1×W2×W3 grid (3×3×3 = 27 points) for UV blanketing optimization.
# Runs on top of σ_bf-on config (path 1 fix). 200K pkts, 10 iters, NLTE start=5.
# Each point ~37 min on A40; cluster runs in parallel where slots are available.
#
# Stage 1 grid (around 151409 champion W1=1.5/W2=0.3/W3=1.0):
#   W1 ∈ {1.2, 1.5, 1.8}    LUMINA_UVOPT_EMIT_BOOST  (UVtg lam window 3200-3800)
#   W2 ∈ {0.2, 0.3, 0.4}    LUMINA_UVOPT_EMIT_BOOST2 (red    lam window 5800-7000)
#   W3 ∈ {0.9, 1.0, 1.1}    LUMINA_UVOPT_EMIT_BOOST3 (CaK    lam window 2600-3200)
#                           — uses third emit-window (must be wired in binary)
#
# Usage: bash scripts/grid_emit_joint_stage1.sh [dry-run]
set -e
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
BINARY=$ROOT/lumina_cuda_a40_dualemit
REFDIR=$ROOT/data/tardis_reference_strat6_higherL_aulboost
SIGBF=$ROOT/data/atomic/cmfgen_sigma_bf.bin
DRY="${1:-}"

W1_LIST=(1.2 1.5 1.8)
W2_LIST=(0.2 0.3 0.4)
W3_LIST=(0.9 1.0 1.1)

# Note: W3 currently NOT in emit window 1/2; if dualemit binary lacks W3,
# this grid degenerates to a 9-point W1×W2 sweep. Run dry-run first to check.
N=0
for W1 in "${W1_LIST[@]}"; do
  for W2 in "${W2_LIST[@]}"; do
    for W3 in "${W3_LIST[@]}"; do
      N=$((N+1))
      JOB="grid_w1${W1//./p}_w2${W2//./p}_w3${W3//./p}"
      EXPORT="ALL,LUMINA_BIN=$BINARY,REF_DIR=$REFDIR,N_PKT=200000,N_ITER=10,MODE=spectrum,NLTE_FLAG=nlte"
      EXPORT="$EXPORT,LUMINA_DYNAMIC_TRANSPROB=1,LUMINA_NLTE_START_ITER=5"
      EXPORT="$EXPORT,LUMINA_CMFGEN_SIGMA_BF=$SIGBF"
      EXPORT="$EXPORT,LUMINA_UVOPT_EMIT_BOOST=$W1,LUMINA_UVOPT_EMIT_LAM_MIN=3200,LUMINA_UVOPT_EMIT_LAM_MAX=3800"
      EXPORT="$EXPORT,LUMINA_UVOPT_EMIT_BOOST2=$W2,LUMINA_UVOPT_EMIT_LAM_MIN2=5800,LUMINA_UVOPT_EMIT_LAM_MAX2=7000"
      EXPORT="$EXPORT,LUMINA_UVOPT_EMIT_BOOST3=$W3,LUMINA_UVOPT_EMIT_LAM_MIN3=2600,LUMINA_UVOPT_EMIT_LAM_MAX3=3200"
      CMD="sbatch --job-name=$JOB --time=UNLIMITED --export=$EXPORT $ROOT/scripts/slurm_lumina_a40.sh"
      if [ "$DRY" = "dry-run" ]; then
        echo "[$N/27] $CMD"
      else
        echo "[$N/27] submitting $JOB"
        eval "$CMD"
        sleep 2
      fi
    done
  done
done
echo "Total: $N jobs"
