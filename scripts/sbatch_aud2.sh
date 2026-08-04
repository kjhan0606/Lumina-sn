#!/bin/bash
#SBATCH --job-name=a10_kx_aud2
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# AUD2 = GATE-AUDIT arm B: SUSPECT-B (b_k band-aids) OFF; SUSPECT-A kept.
#        Binary UNCHANGED = lumina_cuda.withKpr9. Only the ENV changes: this
#        measures whether the b_k floor/cap band-aids (FLOOR_MODE + FLOOR_BKMAX
#        + STAGE4_BK_CAP) are now DEAD WEIGHT because METACOLL drains the Fe III
#        metastable pileup (level 17) at the SOURCE, so nothing pins against the
#        b_k ceiling anymore.
# ============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% SEQUENTIAL SUBMISSION REQUIRED -- DO NOT run aud1/aud2/aud3 concurrently. %%
# %% All three install lumina_cuda.withKpr9 as ./lumina_cuda in the SHARED    %%
# %% repo and restore the original on EXIT. If two run at once, one job's     %%
# %% EXIT-restore can swap ./lumina_cuda MID-RUN of another. Submit ONE AT A  %%
# %% TIME (wait for DONE), or run aud3 first. (No per-job copy is used -- the  %%
# %% runner run_coevolve_s01.sh hardcodes ./lumina_cuda.)                     %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ---------------------------------------------------------------------------
# SOURCE-COUPLING (verified src/lumina_plasma.c + lumina_cuda.cu) -- WHY turning
# SUSPECT-B OFF does NOT disable STAGE4 (a KEEP gate) and preserves the deep win:
#   * STAGE4_BK_CAP is a per-level departure CLAMP applied INSIDE the stage4
#     use_nlte loop [plasma.c:5834-5843]; disabling it removes only the ceiling,
#     the STAGE4 depth-gated NLTE-weight comb (LUMINA_NLTE_STAGE4, KEPT) still runs.
#     The clamp is disabled ONLY by cap<=0: `if (g_stage4_bk_cap > 0.0 && ...)`
#     [plasma.c:5839]. Default is 1000 [plasma.c:5777], so UNSETTING leaves it
#     capping at 1000 -- it must be set =0 to actually retire it.
#   * Source-cited deep safety: the cap "does NOT touch the deep physical drain
#     (s0 comb << cap)" [plasma.c:5837] -- at s0 the departure comb is far below
#     1000, so removing the cap cannot change the deep-s0 populations => the deep
#     win is preserved by construction; the cap only ever bit the hot super-thermal
#     photospheric combs (Ni III s8 b_k~2e9) that this audit is probing.
#   * FLOOR_MODE=1 is the global NLTE writeback floor/cap policy; default 0 =
#     legacy flat 1e-30 negative-clamp, byte-identical [cuda.cu:833, plasma.c:11409].
#     Unsetting FLOOR_MODE (and FLOOR_BKMAX) reverts to that legacy floor -- it does
#     NOT disable the NLTE solve, only the LTE-relative floor + b_k<=1000 writeback cap.
# DEEP-CORE ENV IS BYTE-PRESERVED: every KEEP gate below (METACOLL, STAGE4+WTHR,
#   KPEMISS_REPAIR/SE_POPS/FB_MULTI/BSRC_TAU/BSRC_SRC, RADEQ_DB_FB, LINE_BSRC(+MODE),
#   RADEQ_PUMP_FIELD/FALLBACK) is identical to sbatch_kpr9.sh. SUSPECT-A (the
#   photospheric OTS family + COOLGUARD) is ALSO kept identical to kpr9 this arm.
# ---------------------------------------------------------------------------
# PRE-REGISTERED RETIREMENT CRITERIA (verbatim -- identical in aud1/aud2/aud3):
#   A gate family is RETIRABLE if, with it OFF: (1) DEEP WIN HELD -- FUV(s0)>=1.5e-4,
#   slope>=+2.0, funnel<=3x, deep f(FeIV) s0>=0.95, u(s0)>=450; AND (2) PHOTOSPHERE
#   NO-WORSE than kpr9 -- f(FeIV) s6/s8 not higher than kpr9's 0.781/0.775, Fe III
#   lvl17 b_k s4 stays ~thermal(<3). If OFF holds both => the family was dead weight,
#   RETIRE. If OFF regresses deep => the gate was load-bearing (unexpected,
#   investigate). If OFF regresses ONLY the photosphere => the patch was partially
#   masking; note which metric.
#
# THIS ARM (aud2) UNSET/OFF LIST (SUSPECT-B):
#   unset LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX   (=> legacy 1e-30 floor)
#   export LUMINA_STAGE4_BK_CAP=0   (=0 disables the clamp; unset would leave 1000)
#   SUSPECT-A KEPT: BSRC_PHOT(+SRC+XION), FB_OTS(+NUMIN), OTS_MODE(+TAU), COOLGUARD=1.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr9 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr9
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[aud2] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[aud2] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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

# ============================================================================
# KPR9 ROOT FIX: the MISSING collisional de-excitation channel for
#                metastable levels lacking radiative decay (COLD-CASE-P).
# ============================================================================
export LUMINA_NLTE_METASTABLE_COLL=1   # add Axelrod (Omega=1) meta->ground drain
                                       #   for every drainless-metastable level

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
# [SUSPECT-B OFF] retire the per-level b_k clamp. =0 disables it (plasma.c:5839);
# unsetting would fall back to the default 1000 and keep capping. STAGE4 depth-comb
# (LUMINA_NLTE_STAGE4, above) is UNAFFECTED -- only the ceiling is removed.
export LUMINA_STAGE4_BK_CAP=0

# --- KPEMISS_REPAIR (Part B) master gate + knobs --------------------------------
export LUMINA_KPEMISS_REPAIR=1       # master gate (off => byte-identical)
export LUMINA_KPEMISS_SE_POPS=1      # B1 SE/NLTE pops into kp_emiss (plasma.c:2117)
export LUMINA_KPKT_FB_MULTI=1        # B2 real per-edge fb recombination continuum floor
export LUMINA_KPEMISS_BSRC_TAU=0.13  # B3 B(T_e) k-packet exit where W>this (deep only)
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: deep -4 exit nu ~ chi_line(nu)*B_nu(Te)

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # [SUSPECT-A KEPT] skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend (inherited)
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J

# --- PHASE-1 FINAL LEVER: floor policy + zero-count pump fallback (inherited) -----
# [SUSPECT-B OFF] retire the LTE-relative floor + b_k<=1000 writeback cap. Unset both
# => default 0 = legacy flat 1e-30 negative-clamp (cuda.cu:833, plasma.c:11409).
unset LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# --- PHOTOSPHERIC EUV REPAIR: [SUSPECT-A KEPT this arm] Prong A + Prong B ----------
export LUMINA_KPEMISS_BSRC_PHOT=1          # Prong A: extend -4 B(Te) exit to tau_bf- phot shells
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te); deep keeps SRC=2
export LUMINA_KPEMISS_FB_OTS=1             # Prong B: case-B/OTS redirect of EUV ground-edge (-3) fb
export LUMINA_KPEMISS_FB_OTS_NUMIN=7.40e15 # 405A: redirect ONLY Fe III & bluer IGE EUV ground edges
export LUMINA_KPEMISS_BSRC_PHOT_XION=1     # phot -4 exit thermalizes CROSS-ION re-excites only
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

TAG="a10_kx_aud2"
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
echo "[aud2] verify SUSPECT-B is OFF: grep -E '\[FLOORM\]' logs/coevolve_consume_${TAG}/stdout.log  # should be ABSENT (mode=1 banner gone)"
echo "[aud2] verify METACOLL drain holds WITHOUT the cap: grep -E '\[METACOLL-PROBE\]' logs/coevolve_consume_${TAG}/stdout.log  # FeIII lvl17 b_k/gnd sLAST must stay O(1-10), NOT run away"
echo "[aud2] verify STAGE4 comb still runs: grep -E '\[STAGE4\]' logs/coevolve_consume_${TAG}/stdout.log"
echo "[aud2] RETIRE-B if: DEEP HELD (FUV s0>=1.5e-4, slope>=+2.0, funnel<=3x, f(IV,s0)>=0.95, u(s0)>=450) AND PHOT NO-WORSE (f(FeIV) s6/s8 <= 0.781/0.775, Fe III lvl17 b_k s4 <3). If lvl17 b_k runs away => the cap was still load-bearing (METACOLL drain insufficient alone)."
