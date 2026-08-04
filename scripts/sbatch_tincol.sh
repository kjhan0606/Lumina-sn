#!/bin/bash
#SBATCH --job-name=a10_kx_tincol
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# F3-B INNER-BOUNDARY COLOR PROBE  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (scripts/sbatch_gph_alllevel.sh MODE=all, the
# a10_kx_gphall config -- same env as sbatch_te_table.sh's B-run block) PLUS the
# SINGLE new variable:
#     export LUMINA_TINNER_COLOR=18760
# Binary: lumina_cuda.withTincol (carries jtable + tetab + tincol gates; ALL three
#         env-OFF by default; byte-identical to the B-run when TINNER_COLOR unset).
# NO tetab, NO jtable: T_e and the field are FREE to respond -- the free response IS
# the measurement. Only the inner-boundary COLOR changes.
#
# WHAT CHANGES (color only, energy conserved):
#   In the coevolve-consume config the deep region is diffusion-dominated, so packets
#   recycle many times through LUMINA_DIFFUSE_INNER_BC=1, which re-emits each returned
#   packet as a Planck at T_inner (=config.T_inner ~10020 K in this run: a COLD color
#   thermostat replacing CMFGEN's 19-23 kK v<3900 interior). This probe recolors ONLY
#   that re-emission (and any boundary-SED Planck injection) to Planck(18760 K)
#   [= CMFGEN T at v=4264; the true interior is 19-23 kK -- 18760 is the conservative
#   default]. Packet ENERGIES/weights, L_inner and the T_inner controller are
#   bit-unchanged. Causal question: does the transported deep FUV field acquire the
#   missing amplitude/gradient from boundary color ALONE?
#
# WHY (F3-T + twin 179595/179596 outcome, docs/FUV_GRADIENT_ATTACK_DESIGN.md):
#   F3-T pinned T_e(v) whole-state to CMFGEN and REFUTED the "deep FUV is at its local
#   thermal ceiling" model: deep FUV mc_J rose only +0.30 dex while EUV rose +5.8 dex
#   (the pin propagated -- deep FUV simply did not follow). The twin 179596 (T_e AND Gph
#   both pinned) reproduces CMFGEN's Fe f(IV) essentially exactly (s6 lag 0.304->0.094
#   vs 0.069) yet the TRANSPORTED mc_J STAYS FLAT (+0.31 dex vs CMFGEN +2.42). With
#   ionization and local T_e both correct, the boundary COLOR is the only surviving
#   suspect for the transported field. Wien penalty at 1100 A for 10020 K vs ~19000 K
#   ~= -2.7 dex, matching the measured -2.3 dex deep FUV deficit.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#   PASS:  deep mc_J(918-1290A, s0) rises >= +1.5 dex vs B-run
#          (1.4e-6 -> >= 4.4e-5, toward CMFGEN 2.02e-4) AND the s0->s8 mc_J slope
#          goes from -0.19 dex to >= +1.5 dex
#          => the inner-boundary COLOR thermostat is CONFIRMED as the dominant cause
#             of the flat transported deep FUV field.
#   PARTIAL: deep mc_J(s0) rises +0.5 .. +1.5 dex => color is a major but not sole
#            contributor (shared with transport residual / dilution).
#   NULL:  < +0.5 dex => FIRST check the [TINCOL] counters (wiring), THEN the
#          residual-transport re-audit.
#          Wiring (check FIRST on any null):
#            [TINCOL] active color=18760.0K (T_inner_energy=...K)     (once, at init)
#            [TINCOL] it NN: recolored_packets=<N>                    (once per iter;
#                    N==0 with the gate ON prints a *** WARNING *** and is a wiring
#                    no-op, NOT a physics null. In INJECT=2 mode only the diffuse
#                    inner-BC re-emission site fires -- expect N ~ returned-packet
#                    count, i.e. thousands+ per iter.)
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

# --- F3-B inner-boundary COLOR pin (the ONLY new variable) ----------------------
export LUMINA_TINNER_COLOR=18760

TAG="a10_kx_tincol"
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
echo "[tincol] verify wiring: grep '\[TINCOL\]' logs/coevolve_consume_${TAG}/stdout.log"
