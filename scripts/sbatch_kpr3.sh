#!/bin/bash
#SBATCH --job-name=a10_kx_kpr3
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR3 = KPR2 + the B3 EXIT REFINEMENT: chi_line-weighted thermal re-emission
#                                                              (PREPARE ONLY)
#   the -4 k-packet B(T_e) exit re-emits INSIDE the deep line forest, not into
#   the low-opacity continuum window                (LUMINA_KPEMISS_BSRC_SRC=2)
# ============================================================================
# Exact clone of scripts/sbatch_kpr2.sh (STAGE4-R2 A1/A2/A3 + KPR B1/B2/B3 +
# DB_FB + COOLGUARD + Fork B), with ONE new gate:
#
#   THE DEFECT KPR2 LEFT (established): the B3 -4 exit fixed the Co IV fluorescence
#   FUNNEL (mc/cs 39->~1.9) but LEAKS deep energy: u(s0)=230 vs CMFGEN 695. The
#   pure-Planck(T_e) draw scatters the re-emitted photon into low-opacity CONTINUUM
#   windows between the forest lines, where it escapes deep instead of re-trapping.
#   The formation map (validation/.../formation_map/FORMATION_MAP_VERDICT.md) shows
#   CMFGEN's deep emissivity is ~99.7% LINES at thermal strength (S~=B): emission
#   happens INSIDE the forest, so photons re-trap where they were born.
#
#   THE REFINEMENT (env-gated, default SRC=1 => byte-identical to withKpr2):
#     LUMINA_KPEMISS_BSRC_SRC=2   cuda.cu: sample the -4 exit COMOVING FREQUENCY
#                             from a per-shell frequency CDF of
#                             chi_line(nu)*B_nu(T_e[shell])*dnu  (the FULL Sobolev
#                             expansion line opacity cs.chi_line, rebuilt each iter
#                             in cmfgen_assemble) instead of pure Planck(T_e). The
#                             photon lands in the line forest and re-traps; deep
#                             continuum windows no longer bleed the field. ENERGY is
#                             untouched (frequency-only); direction mu drawn as
#                             before. Within-bin: uniform-in-nu (inverse-CDF residual
#                             of the single bin-select uniform). A qualifying shell
#                             with an all-zero chi CDF falls back to pure Planck,
#                             counted as chi_fallback.
#
# Binary: lumina_cuda.withKpr3 (= withKpr2 source + the SRC=2 knob; SRC unset/=1 =>
#         draw-identical to withKpr2, so master-off => byte-identical baseline).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move)
#   Yardstick = CMFGEN toy06 @19.48d at Lumina velocities. The refinement targets
#   the DEEP-u leak the pure-Planck B3 exit could not close:
# ---------------------------------------------------------------------------
#  PRIMARY (the leak this knob attacks):
#    * u_bol(s0)  230 -> >= 400   (toward CMFGEN 695; forest re-trap refills deep u)
#  RETAINED KPR2 GAINS (must hold — the refinement must not undo them):
#    * FUV(918-1290, s0) >= 1e-5      (deep FUV not re-collapsed)
#    * funnel dead: mc/cs @1450-1650 <= 3x   (Co IV pile stays killed)
#    * T_e(s0)  <= 21000              (no self-heat runaway; the DB_FB thermostat
#                                      + COOLGUARD hold. NOTE: the Phase-1a fix
#                                      (SE-pops / source substrate) may lower T_e(s0)
#                                      FURTHER when it lands -- THIS CARD MAY BE
#                                      RE-CUT after 1a; the <=21000 ceiling is the
#                                      no-runaway guard, not a target.)
#    * all other kpr2 retained gains held (EUV non-thermal, f(FeIV) slope, pins)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * unchanged from kpr2: T_rad pin 10470 (C9); Co f(IV) rate deficit (C7/C10);
#      MC blue-tilt / far-outer hot-band (H9/H10). This knob is FREQUENCY-ONLY at
#      the B3 exit; it moves WHERE deep photons re-thermalize, not the rate physics.
#  WIRING (check FIRST on any null):
#    [KPR] ... src=2 (chi_line*B_nu forest)               (once, init; confirms SRC=2 armed)
#    [KPR] it NN: src=2 bteq_exits=<n> cdf_exits=<m> chi_fallback=<k>   (per iter;
#            bteq>0 => B3 firing; chi_fallback small => CDFs non-degenerate)
#    [DBFB] selfcheck s0: net(J=B)/H = <x>                 (once; MUST be < 1e-6)
#    [SIMUL] done: pins hi=<n> lo=<m> of 50                (per iter)
#  HARD KILL: if u_bol(s0) does NOT climb above ~400 while the funnel stays dead
#    (mc/cs <=3x) AND chi_fallback is small (CDFs valid), the "re-trap in forest"
#    thesis for the deep-u leak is wrong -> escalate to a transport-side deep
#    continuum-thermalization audit (is tau_cont actually >=1 deep?).
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr3 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr3
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr3] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr3] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_EVENT_LOG_ESCATTER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value.
unset LUMINA_NLTE_BK_CEIL
# Keep the #33-transplant / thermostat / other-repair gates OFF: T_e and the field
# are FREE to respond (the whole point). unset even if inherited from the shell.
unset LUMINA_GPH_JTABLE LUMINA_TE_TABLE LUMINA_TINNER_COLOR \
      LUMINA_MACROATOM_BF LUMINA_LINE_THERM

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
export LUMINA_STAGE4_BK_CAP=1000     # A1 per-level b_k cap inside the gate

# --- KPEMISS_REPAIR (Part B) master gate + knobs --------------------------------
export LUMINA_KPEMISS_REPAIR=1       # master gate (off => byte-identical)
export LUMINA_KPEMISS_SE_POPS=1      # B1 SE/NLTE pops into kp_emiss (plasma.c:2117)
export LUMINA_KPKT_FB_MULTI=1        # B2 real per-edge fb recombination continuum floor
export LUMINA_KPEMISS_BSRC_TAU=0.13  # B3 B(T_e) k-packet exit where W>this (deep only)
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: -4 exit nu ~ chi_line(nu)*B_nu(Te)  << NEW

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

TAG="a10_kx_kpr3"
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
echo "[kpr3] verify SRC=2 wiring: grep -E '\[KPR\] .*src=2' logs/coevolve_consume_${TAG}/stdout.log   # init + per-iter"
echo "[kpr3] verify gates:        grep -E '\[KPR\]|\[STAGE4\]|\[FB-MULTI\]|\[BSRC\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr3] chi_fallback should be small (CDFs non-degenerate); bteq_exits>0 => B3 firing"
