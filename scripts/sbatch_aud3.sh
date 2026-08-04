#!/bin/bash
#SBATCH --job-name=a10_kx_aud3
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# AUD3 = GATE-AUDIT PARSIMONY ENDPOINT: SUSPECT-A AND SUSPECT-B BOTH OFF.
#        Minimal core = deep-repair (KPEMISS B1/B2/B3 deep) + STAGE4 depth-comb
#        + METACOLL root-fix ONLY. Binary UNCHANGED = lumina_cuda.withKpr9; only
#        the ENV strips both suspect families. If the DEEP WIN holds and the
#        PHOTOSPHERE is no-worse with EVERYTHING suspect removed, the entire
#        symptom-patch quilt (7 photospheric OTS gates + 3 b_k band-aids) is dead
#        weight and retirable in one shot -- the anti-patch-quilt endpoint.
# ============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% SEQUENTIAL SUBMISSION REQUIRED -- DO NOT run aud1/aud2/aud3 concurrently. %%
# %% All three install lumina_cuda.withKpr9 as ./lumina_cuda in the SHARED    %%
# %% repo and restore the original on EXIT. If two run at once, one job's     %%
# %% EXIT-restore can swap ./lumina_cuda MID-RUN of another. Submit ONE AT A  %%
# %% TIME (wait for DONE). RECOMMENDED: run THIS (aud3) FIRST -- it is the     %%
# %% parsimony endpoint; if it PASSES both criteria, aud1/aud2 become          %%
# %% confirmatory. (No per-job copy is used -- run_coevolve_s01.sh hardcodes    %%
# %% ./lumina_cuda.)                                                           %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ---------------------------------------------------------------------------
# SOURCE-COUPLING (verified src/lumina_cuda.cu + lumina_plasma.c) -- the deep win
# survives with BOTH families off:
#   * Deep B3 -4 B(T_e) exit master kpr_bsrc_on=(kpr_master && kpr_bsrc_tau>0.0)
#     [cuda.cu:5397] depends ONLY on REPAIR + BSRC_TAU(>0); BSRC_TAU=0.13 & SRC=2
#     KEPT below => deep chi_line*B_nu forest exit stays ARMED regardless of the
#     photospheric prongs (bsrc_phot_on=kpr_bsrc_on && bsrc_phot [cuda.cu:5441]).
#   * OTS_MODE armed to 0 when both phot prongs off [cuda.cu:5461] -> inert.
#   * COOLGUARD off (=0) only re-enables thermal exits in f(FeV)>0.5 shells
#     [cuda.cu:5462-5471]; INACTIVE at the deep s0 (f(FeIV,s0)>=0.95).
#   * STAGE4_BK_CAP=0 removes only the ceiling; "does NOT touch the deep physical
#     drain (s0 comb << cap)" [plasma.c:5837-5839]; STAGE4 depth-comb still runs.
#   * FLOOR_MODE unset => legacy 1e-30 floor [cuda.cu:833]; NLTE solve intact.
# DEEP-CORE ENV IS BYTE-PRESERVED: every KEEP gate below (METACOLL, STAGE4+WTHR,
#   KPEMISS_REPAIR/SE_POPS/FB_MULTI/BSRC_TAU/BSRC_SRC, RADEQ_DB_FB, LINE_BSRC(+MODE),
#   RADEQ_PUMP_FIELD/FALLBACK) is identical to sbatch_kpr9.sh.
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
# THIS ARM (aud3) UNSET/OFF LIST (SUSPECT-A + SUSPECT-B):
#   unset LUMINA_KPEMISS_BSRC_PHOT LUMINA_KPEMISS_BSRC_PHOT_SRC
#         LUMINA_KPEMISS_BSRC_PHOT_XION LUMINA_KPEMISS_FB_OTS
#         LUMINA_KPEMISS_FB_OTS_NUMIN LUMINA_KPEMISS_OTS_MODE LUMINA_KPEMISS_OTS_TAU
#         LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX
#   export LUMINA_KPEMISS_COOLGUARD=0    (defaults ON => must be =0)
#   export LUMINA_STAGE4_BK_CAP=0        (=0 disables clamp; unset would leave 1000)
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
                && echo "[aud3] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[aud3] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
# [SUSPECT-B OFF] retire the per-level b_k clamp. =0 disables it (plasma.c:5839).
export LUMINA_STAGE4_BK_CAP=0

# --- KPEMISS_REPAIR (Part B) master gate + knobs --------------------------------
export LUMINA_KPEMISS_REPAIR=1       # master gate (off => byte-identical)
export LUMINA_KPEMISS_SE_POPS=1      # B1 SE/NLTE pops into kp_emiss (plasma.c:2117)
export LUMINA_KPKT_FB_MULTI=1        # B2 real per-edge fb recombination continuum floor
export LUMINA_KPEMISS_BSRC_TAU=0.13  # B3 B(T_e) k-packet exit where W>this (deep only)
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: deep -4 exit nu ~ chi_line(nu)*B_nu(Te)

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
# [SUSPECT-A OFF] COOLGUARD defaults ON under the master gate => set =0 to disable.
export LUMINA_KPEMISS_COOLGUARD=0

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend (inherited)
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J

# --- PHASE-1 FINAL LEVER: floor policy + zero-count pump fallback (inherited) -----
# [SUSPECT-B OFF] retire the LTE-relative floor + b_k<=1000 writeback cap.
unset LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# ============================================================================
# [SUSPECT-A OFF] PHOTOSPHERIC EUV / OTS FAMILY RETIRED FOR THIS ARM.
#   Source-verified above: unsetting them disables ONLY the photospheric exits;
#   the deep B3 (BSRC_TAU=0.13, BSRC_SRC=2 KEPT above) stays ARMED (cuda.cu:5397).
# ============================================================================
unset LUMINA_KPEMISS_BSRC_PHOT \
      LUMINA_KPEMISS_BSRC_PHOT_SRC \
      LUMINA_KPEMISS_BSRC_PHOT_XION \
      LUMINA_KPEMISS_FB_OTS \
      LUMINA_KPEMISS_FB_OTS_NUMIN \
      LUMINA_KPEMISS_OTS_MODE \
      LUMINA_KPEMISS_OTS_TAU \
      LUMINA_KPEMISS_BSRC_PHOT_WFLOOR LUMINA_KPEMISS_BSRC_WFLOOR

TAG="a10_kx_aud3"
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
echo "[aud3] verify BOTH suspect families OFF (expect NO phot & NO FLOORM banners): grep -E '\[BSRC_PHOT\]|\[BSRC_PHOT_XION\]|\[FB_OTS\]|\[OTS-TAUBF\]|\[FLOORM\]' logs/coevolve_consume_${TAG}/stdout.log  # should be ABSENT"
echo "[aud3] verify DEEP CORE INTACT: grep -E '\[KPR\].*BSRC_TAU=0.130 -> B\(T_e\) exit ARMED src=2|\[METACOLL\]|\[METACOLL-PROBE\]|\[STAGE4\]' logs/coevolve_consume_${TAG}/stdout.log"
echo "[aud3] PARSIMONY PASS if: DEEP HELD (FUV s0>=1.5e-4, slope>=+2.0, funnel<=3x, f(IV,s0)>=0.95, u(s0)>=450) AND PHOT NO-WORSE (f(FeIV) s6/s8 <= 0.781/0.775, Fe III lvl17 b_k s4 <3). Then the whole 10-gate symptom-patch quilt is dead weight -> RETIRE both families."
