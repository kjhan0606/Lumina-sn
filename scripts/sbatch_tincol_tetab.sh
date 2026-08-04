#!/bin/bash
#SBATCH --job-name=a10_kx_tincol_tetab
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# F3-B COLOR + F3-T TEMPERATURE END-MEMBER  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (a10_kx_gphall config) PLUS BOTH:
#     export LUMINA_TINNER_COLOR=18760                              (F3-B boundary color)
#     export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv  (F3-T whole-state T_e)
# Binary: lumina_cuda.withTincol (jtable + tetab + tincol gates; all env-OFF by default).
# NO jtable: the field is FREE to respond -- that free response IS the measurement.
#
# This is the FULL TEMPERATURE END-MEMBER: CMFGEN's boundary COLOR (recolors the diffuse
# inner-BC re-emission from Planck(~10020 K) to Planck(18760 K); energy/L/controller
# unchanged) AND CMFGEN's whole-state T_e(v) pin (every T_e consumer follows the table:
# Saha/NLTE pops, GPU emissivities, k-packet redistribution, coevolve birth-Planck SED,
# cooling). It asks: with local T_e AND the boundary color both CMFGEN-consistent, does
# the transported deep FUV field self-consistently build the CMFGEN gradient WITHOUT the
# #33 jtable Gph field injection?
#
# WHY (F3-T + twin 179595/179596, docs/FUV_GRADIENT_ATTACK_DESIGN.md):
#   F3-T (T_e pin alone) lifted deep FUV only +0.30 dex. Twin 179596 (T_e + Gph both
#   pinned) reproduces Fe f(IV) essentially exactly yet the transported mc_J stays flat
#   (+0.31 dex vs CMFGEN +2.42). The boundary COLOR is the only surviving suspect. This
#   card adds the color to the T_e pin: local temperature (source-term) + boundary color
#   (transported-field seed) together = the temperature-only reconstruction of CMFGEN,
#   short of directly injecting CMFGEN's field.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#   PASS:  deep mc_J(918-1290A, s0) approaches CMFGEN within a factor ~3
#          (>= 6e-5, CMFGEN 2.02e-4) AND the s0->s8 mc_J slope >= +2.0 dex
#          => temperature (local T_e + boundary color) alone reconstructs the
#             transported deep FUV field.
#   PARTIAL: deep mc_J(s0) rises materially but stays < 6e-5 or slope < +2.0
#          => temperature is dominant but a residual field/rate term remains
#             (candidate = the #33 jtable Gph contribution / Co rate deficit).
#   NULL:  no material movement over the F3-B color-only card => temperature end-member
#          is insufficient; FIRST check [TINCOL]/[TETAB] counters (wiring), THEN re-audit.
#   SECONDARY (not gating): Fe f(IV)=IV/(III+IV) profile vs CMFGEN. Does the s6
#          transition-lag (0.304 in #33; 0.094 in the both-pinned twin 179596) CLOSE
#          toward CMFGEN 0.069 WITHOUT the jtable field injection? That would demonstrate
#          the self-consistent chain temperature -> field -> ionization (the field the
#          #33 probe injected by hand would instead ARISE from correct temperature).
#          Watch n_e(s0) toward CMFGEN 5.09e9; Co remains ~10x rate-deficient in isolation.
#   Wiring (check FIRST on any null):
#          [TINCOL] active color=18760.0K (T_inner_energy=...K)   (once, at init)
#          [TINCOL] it NN: recolored_packets=<N>                  (per iter; N==0 = no-op)
#          [TETAB] loaded ... T[s0]=... T[s8]=...                 (once, on load)
#          [TETAB] shells_pinned=50                               (per iter; <50 = no-op)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withTincol binary as ./lumina_cuda for run_coevolve_s01.sh ------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
TT_BIN=lumina_cuda.withTincol
[ -x "$TT_BIN" ] || { echo "FATAL: $TT_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[tincol] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$TT_BIN" lumina_cuda
echo "[tincol] installed $TT_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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

# --- F3-B inner-boundary COLOR pin + F3-T whole-state T_e(v) pin -----------------
export LUMINA_TINNER_COLOR=18760
export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv
[ -f "$LUMINA_TE_TABLE" ] || { echo "FATAL: $LUMINA_TE_TABLE missing (run: python3 scripts/build_cmfgen_te_table.py)"; exit 2; }

TAG="a10_kx_tincol_tetab"
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
echo "[tincol] verify wiring: grep -E '\[TINCOL\]|\[TETAB\]' logs/coevolve_consume_${TAG}/stdout.log"
