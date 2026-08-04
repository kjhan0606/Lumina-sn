#!/bin/bash
#SBATCH --job-name=a10_kx_tetab
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# F3-T TEMPERATURE-TABLE PROBE  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (scripts/sbatch_gph_alllevel.sh MODE=all, the
# a10_kx_gphall config) PLUS the CMFGEN electron-temperature pin:
#     export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv
# Binary: lumina_cuda.withTetab (default-OFF gate; byte-identical when env unset;
#         also carries the #33 jtable gate, env-OFF here).
# NO jtable: the field is FREE to respond -- that free response IS the measurement.
#
# The probe (docs/FUV_GRADIENT_ATTACK_DESIGN.md, 2026-07-19 F0b verdict):
# F0b established Lumina's deep FUV field is fully thermalized to its OWN cold T_e
# (13120 K at s0 vs CMFGEN 18900 K); the missing gradient is ~1.6 dex temperature-
# inherited (Wien color + dilution) with transport residual <=0.5 dex. F3-T asks the
# causal question: give Lumina CMFGEN's T(v) -- does the transported field acquire
# the gradient? This is a WHOLE-STATE T_e pin (radeq-solved T_e REPLACED by the table
# per shell, BEFORE the ion ladder / n_e commit) so every T_e consumer follows it:
# Saha/NLTE populations, GPU emissivities, k-packet redistribution, coevolve birth-
# Planck SED, cooling.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#   PASS: deep mc_J(918-1290A, s0) rises >=1 dex vs B-run (1.4e-6 -> >=1.4e-5) AND
#         the s0->s8 mc_J slope goes from -0.11 dex to >= +1.0 dex
#         => T_e confirmed causally upstream of the field; fix address = radeq
#            missing heating (backwarming/EPAY).
#   PARTIAL: rises but <1 dex => shared with transport residual.
#   NULL: no movement => FIRST check the [TETAB] counters (wiring), THEN the
#         transport-residual re-audit.
#           [TETAB] loaded ... T[s0]=... T[s8]=...   (once, on load)
#           [TETAB] shells_pinned=50                 (once per iteration; a value
#                   < 50 or absent with the gate ON = silent no-op, NOT a physics
#                   null.)
#   Secondary watch (not gating): deep T_e-sensitive observables -- n_e(s0) should
#         rise toward CMFGEN 5.09e9. The B-run matched n_e(s0) at the WRONG T; watch
#         whether the pin BREAKS n_e agreement (would reveal compensating errors).
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

# --- F3-T temperature-table pin ------------------------------------------------
export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv
[ -f "$LUMINA_TE_TABLE" ] || { echo "FATAL: $LUMINA_TE_TABLE missing (run: python3 scripts/build_cmfgen_te_table.py)"; exit 2; }

TAG="a10_kx_tetab"
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
echo "[tetab] verify wiring: grep '\[TETAB\]' logs/coevolve_consume_${TAG}/stdout.log"
