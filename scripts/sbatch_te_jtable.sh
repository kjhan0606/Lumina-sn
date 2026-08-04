#!/bin/bash
#SBATCH --job-name=a10_kx_tetab_jtab
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# F3-T TEMPERATURE-TABLE + JTABLE END-MEMBER  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (scripts/sbatch_gph_alllevel.sh MODE=all, the
# a10_kx_gphall config) PLUS BOTH offline CMFGEN pins:
#     export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv   (T_e(v) pin)
#     export LUMINA_GPH_JTABLE=data/cmfgen_jtable_toy06_19p48d.bin   (#33 Gph J_nu)
# Binary: lumina_cuda.withTetab (carries BOTH gates; byte-identical when both unset).
#
# This is the "Lumina with CMFGEN's T AND CMFGEN's Gph field" END-MEMBER: the closest
# offline-achievable CMFGEN-twin. Its ionfrac vs CMFGEN measures everything that STILL
# differs once both the temperature structure and the ionizing field are transplanted.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATE (verbatim -- do not move the goalposts after the run):
#   Twin test: Fe f(IV)=IV/(III+IV) profile vs CMFGEN at ALL shells.
#   Pre-register: the s6 transition-lag (0.304 in #33) should CLOSE toward 0.07
#                 if the lag was T_e's fault.
#   Secondary watch (not gating): deep T_e-sensitive observables -- n_e(s0) should
#         rise toward CMFGEN 5.09e9; watch whether the pin BREAKS n_e agreement
#         (would reveal compensating errors). Co lands at 0.005-0.026 photosphere
#         (CMFGEN 0.10) / 0.50 deep at s2 (CMFGEN 0.98) -- the Co-specific rate
#         deficit remains measurable in isolation with both pins active.
#   Wiring (check FIRST on any null):
#         [TETAB] loaded ... T[s0]=... T[s8]=...          (once, on load)
#         [TETAB] shells_pinned=50                         (once per iteration)
#         [JTABLE] loaded ... nonzero_bins=...             (once, on load)
#         [JTABLE] gph_evals_using_table=<N>  (N==0 with the gate ON = wiring bug)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withTetab binary as ./lumina_cuda for run_coevolve_s01.sh -------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
TT_BIN=lumina_cuda.withTetab
[ -x "$TT_BIN" ] || { echo "FATAL: $TT_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[tetab] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$TT_BIN" lumina_cuda
echo "[tetab] installed $TT_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value so
# this clone matches the B-run env verbatim.
unset LUMINA_NLTE_BK_CEIL

# --- F3-T temperature-table pin + #33 gradient-transplant Gph field --------------
export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv
[ -f "$LUMINA_TE_TABLE" ] || { echo "FATAL: $LUMINA_TE_TABLE missing (run: python3 scripts/build_cmfgen_te_table.py)"; exit 2; }
export LUMINA_GPH_JTABLE=data/cmfgen_jtable_toy06_19p48d.bin
[ -f "$LUMINA_GPH_JTABLE" ] || { echo "FATAL: $LUMINA_GPH_JTABLE missing (run: python3 scripts/build_cmfgen_jtable.py)"; exit 2; }

TAG="a10_kx_tetab_jtab"
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
echo "[tetab] verify wiring: grep -E '\[TETAB\]|\[JTABLE\]' logs/coevolve_consume_${TAG}/stdout.log"
