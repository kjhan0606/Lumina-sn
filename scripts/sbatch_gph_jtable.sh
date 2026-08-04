#!/bin/bash
#SBATCH --job-name=a10_kx_jtable
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# #33 GRADIENT-TRANSPLANT DIAGNOSTIC  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (scripts/sbatch_gph_alllevel.sh MODE=all, the
# a10_kx_gphall config) PLUS the external CMFGEN J_nu(v) field injection:
#     export LUMINA_GPH_JTABLE=data/cmfgen_jtable_toy06_19p48d.bin
# Binary: lumina_cuda.withJtable (default-OFF gate; byte-identical when env unset).
#
# Causality test (#32 GRADIENT_BUDGET_VERDICT): the missing ~4.4-dex Fe recomb
# gradient is carried almost entirely by the FIELD folded into Gph. Force CMFGEN's
# measured field SHAPE into Gph only -> does the ionization gradient appear?
# This is a SURGICAL probe: the table overrides J in the Gph rate integral ONLY;
# thermal balance / line transfer / estimators are untouched.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#   PASS: photosphere Fe f(IV)=IV/(III+IV) at s6/s8/s10 drops from 0.75/0.46/0.49
#         to <=0.1 (CMFGEN 0.07/0.02/0.02) AND deep s0-s4 stays >=0.7.
#   PARTIAL: drops but stalls 0.1-0.3 => T_e/population share confirmed.
#   FAIL-null: no movement => FIRST check the [JTABLE] counters (wiring) before
#              any physics conclusion:
#         [JTABLE] loaded ... nonzero_bins=... J[s0,FUVband]=... J[s8,FUVband]=...
#         [JTABLE] gph_evals_using_table=<N>   (printed once per iteration; N==0
#                  with the gate ON = wiring bug, NOT a physics null).
#   Secondary watch (not gating): photospheric Co f(IV) should stay ~0.1 (already
#         correct; a large overshoot flags over-injection); S III response logged.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withJtable binary as ./lumina_cuda for run_coevolve_s01.sh ----
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
JT_BIN=lumina_cuda.withJtable
[ -x "$JT_BIN" ] || { echo "FATAL: $JT_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[jtable] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$JT_BIN" lumina_cuda
echo "[jtable] installed $JT_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1

# --- #33 gradient-transplant injection -----------------------------------------
export LUMINA_GPH_JTABLE=data/cmfgen_jtable_toy06_19p48d.bin
[ -f "$LUMINA_GPH_JTABLE" ] || { echo "FATAL: $LUMINA_GPH_JTABLE missing (run: python3 scripts/build_cmfgen_jtable.py)"; exit 2; }

TAG="a10_kx_jtable"
mkdir -p logs/coevolve_consume_${TAG}
rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="$TAG"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_${TAG}/$f"
done
echo "${TAG} DONE -> logs/coevolve_consume_${TAG}/"
echo "[jtable] verify wiring: grep '\[JTABLE\]' logs/coevolve_consume_${TAG}/stdout.log"
