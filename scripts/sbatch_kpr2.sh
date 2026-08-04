#!/bin/bash
#SBATCH --job-name=a10_kx_kpr2
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR2 = KPR + the PRINCIPLED THERMAL-LEDGER FIX               (PREPARE ONLY)
#   make radeq's bf cooling the DETAILED-BALANCE partner of its photo-heating
# ============================================================================
# Exact clone of scripts/sbatch_kpr.sh (STAGE4-R2 A1/A2/A3 + KPR B1/B2/B3 + Fork B),
# with TWO new gates that fix the KPR T_e-runaway diagnosed in
#   validation/cmfgen_toy06_19p48d/analysis/kpr_runaway_trace/TRACE_LEDGER.txt
#
#   THE DEFECT (established): simul_r1 (plasma.c) balances FIELD-integrated bf
#   photo-heating H_photo = Sum_p nion[ip]*Hex[p] (reads the repaired, bright MC
#   field) against an ANALYTIC, field-decoupled recombination cooling
#   C_fb = n_e*n+*frozenin_alpha_rr(T)*(chi+kTe). The two halves are NOT a
#   detailed-balance pair. When B1/B2/B3 brighten the ionizing field, H_photo
#   scales up but C_fb does not track it -> net heating unbounded -> no cold root
#   -> pin_hi HOLD -> T_e ratchets (s0 23kK, s4 65kK w/ f(FeV)=0.998 coolant
#   burnout). Root cause: TRACE_LEDGER.txt:64-79.
#
#   THE FIX (env-gated, default-OFF, byte-identical when unset):
#     LUMINA_RADEQ_DB_FB=1    plasma.c simul_r1: REPLACE the analytic C_fb with
#                             the emit_nu/Wien DETAILED-BALANCE partner of H_photo,
#                             built bin-consistently in the Gph loop (SAME sigma_bf
#                             grid, f_above weight, lagged pops). bf net =
#                             n*Sum_bb 4pi sigma f_above dnu (J - B_nu^Wien(T)) ->
#                             cancels bin-by-bin at J=B(T_e) BY CONSTRUCTION, so a
#                             brightening field can no longer produce unbounded net
#                             heating. Startup self-check prints
#                             [DBFB] selfcheck s0: net(J=B)/H = <x>  (must be <1e-6).
#     LUMINA_KPEMISS_COOLGUARD=1  cuda.cu: secondary guard. In shells where the Fe
#                             LINE coolant has burned out (f(Fe stage>=V) > 0.5)
#                             SKIP the B3 B(T_e) k-packet exit AND the FB-MULTI
#                             thermal fb exit (revert to legacy single-edge fb) so
#                             the transport stops piling hard-UV into a coolant-dead
#                             shell -> defuses the strip attractor.
#
# Binary: lumina_cuda.withKpr2 (= withKpr source + the two gates above; both
#         default-OFF/COOLGUARD-tied-to-master, so master-off => byte-identical).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (from the runaway trace's OWN predictions -- do NOT move)
#   TRACE_LEDGER.txt committed T_e was s0=22924, s3=31378, s4=65273 (runaway);
#   CMFGEN truth s0~18760, s3~14000, s4~13500. The DB_FB thermostat should land:
# ---------------------------------------------------------------------------
#  RUNAWAY DEFUSED (primary):
#    * T_e(s0)  -> 18-19 kK      (NOT 23k pin_hi, NOT 13k pre-repair floor)
#    * T_e(s4)  -> 13-15 kK      (NOT 65k; the mid-shell self-heat is bounded)
#    * pins_hi  <  5 of 50       (was 31/50 and ratcheting)
#    * f(FeV) s4  < 0.1          (was 0.998 -- coolant no longer burns out)
#    * f(FeIV) s8 <= 0.25        (downstream flood cleared)
#  DEEP GAINS RETAINED (the KPR repair must survive the thermostat):
#    * FUV(918-1290, s0) >= 1e-5     (deep FUV not re-collapsed)
#    * EUV(300-450,  s0) >= 1e-9     (non-thermal EUV preserved; selective iii)
#    * f(FeIV) slope s0->s8 >= +1.0  (field un-flattened)
#    * funnel dead: mc/cs @1450-1650 <= 3x
#    * u_bol(s0) >= 400             (deep u retained toward 695)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * T_rad pin 10470 uniq (C9)                 -- separate estimator fix
#    * Co f(IV) twin-style rate deficit (C7/C10) -- needs SE-weighted Gph, outside DB_FB
#    * MC blue-tilt / far-outer hot-band (H9/H10)-- separate physics, untouched
#  WIRING (check FIRST on any null):
#    [DBFB] selfcheck s0: net(J=B)/H = <x>       (once, init; MUST be < 1e-6, else FATAL abort)
#    [DBFB] LUMINA_RADEQ_DB_FB=1: simul_r1 bf cooling = emit_nu/Wien ...   (once, init)
#    [KPR] ... COOLGUARD=1 (f(FeV)>0.5 -> skip B3+FB-MULTI)                (once, init)
#    [SIMUL] done: pins hi=<n> lo=<m> of 50      (per iter; n should FALL toward <5)
#    [KPR] it NN: bteq_exits=<n> cdf_exits=<m>   (B3; COOLGUARD zeroes bteq in burned shells)
#  HARD KILL: if pins_hi does NOT fall below ~5 and T_e(s4) stays > 30 kK, the bf
#    term is still not the binding partner -> escalate (the analytic C_ff/C_ad or
#    the line coolant, not C_fb, dominates the mid-shell balance).
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr2 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr2
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr2] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr2] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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

# --- KPR2: the principled thermal-ledger fix (this run's new physics) -----------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

TAG="a10_kx_kpr2"
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
echo "[kpr2] verify DB_FB wiring: grep -E '\[DBFB\]' logs/coevolve_consume_${TAG}/stdout.log   # selfcheck < 1e-6"
echo "[kpr2] verify gates:        grep -E '\[KPR\]|\[STAGE4\]|\[FB-MULTI\]|\[BSRC\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr2] verify A3 (Ti):      grep -c 'info=199' logs/coevolve_consume_${TAG}/stderr.log   # expect 0"
