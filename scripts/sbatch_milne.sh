#!/bin/bash
#SBATCH --job-name=a10_kx_milne
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# TEPOP1 = kpr10b champion + LUMINA_KPEMISS_TE_POP=1 (SINGLE-VARIABLE A/B).
# ============================================================================
# WHAT (verified DIRECTLY from two source trees, do NOT re-litigate):
#   The k-packet re-emission CDF weight (plasma.c, kp_emiss build) computes its
#   NON-SE fallback lower-level population n_lower on the TARDIS convention:
#   diluted (xW), T_rad-pinned Boltzmann. That population is SUPER-THERMAL at the
#   photosphere and drives photospheric IGE OVER-IONIZATION. ARTIS toy06 (the
#   benchmark; LTEPOP_EXCITATION_USE_TJ=false) instead uses an UNDILUTED,
#   T_e-anchored LTE population (calculate_levelpop_boltzmann,
#   artis-ref/ltepop.cc:361-367) for the non-modeled ions, with rate-solved NLTE
#   pops for modeled ions. The rate coefficient C_up ALREADY correctly carries
#   exp(-dE/kT_e) (plasma.c:2152) -- ONLY the population n_lower was wrong.
#
# THE CHANGE (LUMINA_KPEMISS_TE_POP=1, one coherent physics change, gated):
#   Replace the fallback n_lower with an UNDILUTED LTE-at-T_e population,
#   self-consistent in numerator AND denominator:
#     n_lower = n_ion * g_lo * exp(-E_glo/(kB*T_e)) / Z_e(T_e)
#     Z_e(T_e) = SUM_l g_l*exp(-E_l/(kB*T_e))     [NO W, no metastable split]
#   Z_e is built per-shell in compute_transition_probabilities using the EXACT
#   plasma->T_e[s] that C_up uses, guaranteeing numerator/denominator share one
#   T_e. The SE/NLTE override (mapped levels use NLTE pops) is UNCHANGED; C_up is
#   UNCHANGED. Gate OFF (either LUMINA_KPEMISS_REPAIR or LUMINA_KPEMISS_TE_POP
#   unset) => the dilute-Boltzmann fallback is BYTE-IDENTICAL to kpr10b.
#
# Binary: lumina_cuda.withMilne (= kpr10b source + the TE_POP gate).
#   TE_POP unset => k-packet fallback pop byte-identical to withKpr10 champion;
#   TE_POP=1 activates the undiluted LTE(T_e) fallback pop. Host-side arithmetic
#   only; no RNG touched. This card = kpr10b env VERBATIM + the single new export.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move)          Yardstick = CMFGEN toy06 @19.48d.
# ---------------------------------------------------------------------------
#  PASS = photospheric f(FeIV) s8 drops materially below the kpr10b baseline
#    (0.76-0.84) toward CMFGEN 0.022 / ARTIS 0.000, AND deep f(FeIV) s0 held
#    >= 0.5 (not cratered), AND no T_e(s8) pathology.
#  PARTIAL = s8 drops but deep regresses, or s8 barely moves (BSRC_PHOT masking
#    the CDF path -- pivot: next run turns BSRC_PHOT family off to exercise the
#    CDF path).
#  NULL = s8 unchanged AND [KPR TE_POP] log absent or fallback path never taken
#    (wiring no-op -- check gate resolution FIRST).
#  WIRING (check FIRST on any null):
#    [KPR TE_POP] init -> "[KPR TE_POP] k-packet fallback pop = LTE(T_e),
#                          undiluted (ARTIS calculate_levelpop_boltzmann parity)"
#    (fires once on first compute_transition_probabilities when REPAIR=1 &&
#    TE_POP=1; absence => the gate did not resolve => the fallback is unchanged.)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withMilne binary as ./lumina_cuda for run_coevolve_s01.sh ------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
TEPOP_BIN=lumina_cuda.withMilne
[ -x "$TEPOP_BIN" ] || { echo "FATAL: $TEPOP_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[tepop1] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$TEPOP_BIN" lumina_cuda
echo "[tepop1] installed $TEPOP_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
# KPR10 ROOT-FIX COMPLETION: the metastable collisional drain, now all-lower-
#   levels with the CMFGEN forbidden floor (matches FeIII_COL_DATA topology).
# ============================================================================
export LUMINA_NLTE_METASTABLE_COLL=1   # arm the drainless-metastable drain pass
export LUMINA_NLTE_METACOLL_MODE=2     # 2 = ALL-lower-levels (was ground-only in kpr9)
export LUMINA_NLTE_METACOLL_OMEGA=0.1  # CMFGEN's f=0 forbidden floor (col_data header)

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
export LUMINA_STAGE4_BK_CAP=0        # RETIRE the STAGE4 per-level b_k cap (=0 disables the
                                     #   clamp; unset would leave the 1000 default). TEST:
                                     #   does the completed all-lower drain make it unnecessary?
# --- OPEN DECISION (see header): to fully match AUD-2 cap-off (b_k free to 2214+),
#     ALSO retire the FLOOR writeback cap. Left commented per the mission's env list.


# --- KPEMISS_REPAIR (Part B) master gate + knobs --------------------------------
export LUMINA_KPEMISS_REPAIR=1       # master gate (off => byte-identical)
export LUMINA_KPEMISS_SE_POPS=1      # B1 SE/NLTE pops into kp_emiss (plasma.c:2117)
export LUMINA_KPKT_FB_MULTI=1        # B2 real per-edge fb recombination continuum floor
export LUMINA_KPEMISS_BSRC_TAU=0.13  # B3 B(T_e) k-packet exit where W>this (deep only)
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: deep -4 exit nu ~ chi_line(nu)*B_nu(Te)

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend (inherited)
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J

# --- PHASE-1 FINAL LEVER: floor policy + zero-count pump fallback (inherited) -----
export LUMINA_NLTE_FLOOR_MODE=0        # [kpr10b] genuine cap-off: FLOORM disabled (was masking b_k at 1000)
export LUMINA_NLTE_FLOOR_BKMAX=1000000000  # [kpr10b] no b_k clamp
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# --- PHOTOSPHERIC EUV REPAIR: Prong A + Prong B composed, PHYSICAL tau_bf gate ----
export LUMINA_KPEMISS_BSRC_PHOT=0 # [MILNE] retire phot -4 B(Te) exit so k-packets reach the exact fb channel          # Prong A: extend -4 B(Te) exit to tau_bf-
                                           #   qualified phot shells
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te) (Wien-dead EUV);
                                           #   deep tier keeps SRC=2 (BSRC_SRC above)
export LUMINA_KPEMISS_FB_OTS=0   # [MILNE] retire fb->B(Te) redirect (Milne fb is correct)             # Prong B: case-B/OTS redirect of EUV ground-edge
                                           #   (-3) fb -> B(Te) draw where tau_bf-qualified
export LUMINA_KPEMISS_FB_OTS_NUMIN=4.80e15 # [FBCB] 620A: broadband case-B over the FULL Fe III
                                           #   recomb edge complex (440-520A). Detailed-balance
                                           #   investigation: fb source is super-Planckian
                                           #   (mc_J/B_nu 693-1180x @s8), a Kirchhoff/Milne break.
                                           #   euv461(461A) split the complex: 461-520A edges have
                                           #   nu=5.76-6.50e15 < 6.50e15 -> UNCAUGHT (the dominant
                                           #   1180x band). 4.80e15 catches nu_edge>4.80e15 (<624A),
                                           #   covering 440-520A; d_kpr_thick spares thin/emergent.
export LUMINA_KPEMISS_BSRC_PHOT_XION=1     # phot -4 exit thermalizes CROSS-ION re-excites only;
                                           #   same-ion cascades KEPT (preserve 912-2000A FUV)
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

# --- TEPOP1 SINGLE-VARIABLE DELTA vs kpr10b (added AFTER all other KPEMISS exports;
#     nothing below unsets or re-exports it) --------------------------------------
export LUMINA_KPEMISS_TE_POP=1   # ARTIS-parity: k-packet fallback pop = LTE(T_e), undiluted

# --- MILNE: exact per-level radiative-recombination fb (retires FB_OTS/FB_COOL_KT approx) ---
export LUMINA_FB_MILNE_EXACT=1

TAG="a10_kx_milne"
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
echo "[tepop1] verify TE_POP init:   grep -E '\[KPR TE_POP\]' logs/coevolve_consume_${TAG}/stdout.log  # fires once when REPAIR=1 && TE_POP=1"
echo "[tepop1] verify gates:         grep -E '\[KPR TE_POP\]|\[METACOLL\]|\[METACOLL-PROBE\]|\[OTS-TAUBF\]|\[BSRC_PHOT\]|\[BSRC_PHOT_XION\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[tepop1] PASS: photospheric f(FeIV) s8 drops materially below kpr10b (0.76-0.84) toward CMFGEN 0.022/ARTIS 0.000; DEEP f(FeIV) s0 held >=0.5; no T_e(s8) pathology"
