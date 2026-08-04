#!/bin/bash
#SBATCH --job-name=a10_kx_kpr6
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR6 = KPR5 + the two-pronged PHOTOSPHERIC EUV REPAIR (case-B / OTS)  (PREPARE ONLY)
# ============================================================================
# Exact clone of scripts/sbatch_kpr5.sh (STAGE4-R2 + KPEMISS_REPAIR B1/B2/B3
# SRC=2 + DB_FB + COOLGUARD + Fork B + PUMP_FIELD alpha=1 + FLOORM + PUMPF), with
# the two new composed gates from docs/PHOTOSPHERIC_EUV_REPAIR_DESIGN.md. The
# design shows only A+B TOGETHER crosses the recombination threshold, so this card
# runs the composed config; each prong is an independent env gate for A/B attribution.
#
#   PRONG A (LUMINA_KPEMISS_BSRC_PHOT) -- extend the B3 `-4` B(Te) k-packet exit to
#     the photosphere. Host-only qualify-mask change: the -4 exit (a Planck/thermal
#     r-packet, no line-forest re-excitation) is ALSO taken in the thick photospheric
#     tier (WFLOOR<W<=TAU), not only the deep tier (W>TAU). Photospheric shells use
#     PURE PLANCK (SRC=1) -- Wien-dead in the EUV -- instead of the deep SRC=2 chi-
#     forest, so the -4 exit removes the S III cross-ion line-EUV attractor (~49% of
#     local EUV creation, the EXCITED Gph channel ~76% of Gph) without re-seeding it.
#
#   PRONG B (LUMINA_KPEMISS_FB_OTS) -- the transport detailed-balance partner. At the
#     kpkt-fb `-3` emission site, an EUV GROUND-edge recombination photon (comoving
#     nu0 > 912A) in the continuum-thick tier is re-absorbed ON THE SPOT (case B):
#     the emission is redirected to the `-4` B(Te) draw instead of free-streaming the
#     404A Fe III ground edge (whose bf re-absorption opacity the runaway collapsed).
#     Energy conserved (frequency redistribution only). This is the transport analog
#     of the radeq [DBFB] fix and mirrors the host-side LUMINA_CMF_OTS for eta_bf.
#     Removes the 40.5% fb-continuum EUV and the GROUND Gph channel (6.56/s @404A).
#     FUV/optical ground edges (>912A) and thin (W<WFLOOR) shells are UNCHANGED
#     (protects the deep FUV gains and legitimate escaping nebular EUV).
#
# Binary: lumina_cuda.withKpr6 (= withKpr5 source + the two gates; both unset/=0
#         => byte-identical to withKpr5, so master-off => byte-identical baseline).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move -- from PHOTOSPHERIC_EUV_REPAIR_DESIGN.md 5.1/5.2)
#   Yardstick = CMFGEN toy06 @19.48d at Lumina velocities.
# ---------------------------------------------------------------------------
#  PRIMARY (the composed A+B target; each prong ALONE stays above threshold):
#    * f(FeIV, s8) <= 0.25      (expect toward 0.022; kpr5 now 0.982; kpr5jt forced 0.008)
#    * Gph(FeIII, s8) within 10x of CMFGEN 3.5e-5  (<= ~3.5e-4/s; kpr5 now 27.37/s)
#    * EUV(300-450, s8) outward decline STEEP, CMFGEN-like (>> 1e3x; toward 1.42e6;
#                                kpr5 now only ~53x -- under-absorbed)
#    * T_e(s8) toward 10.4 kK   (<= 11.4 kK; kpr5 now 12208; CMFGEN 10383)
#  RETAINED KPR5 GAINS (must HOLD -- both prongs act on thick-tier phot/mid shells,
#                       and B only on EUV edges; the deep -4 exit and FUV fb untouched):
#    * FUV(918-1290, s0) >= 1.5e-4     (deep FUV not re-collapsed)
#    * FUV gradient slope >= +2.0      (outward steepening held)
#    * u_bol(s0) >= 450                (bath energy held)
#    * funnel dead: mc/cs @1450-1650 <= 3x  (Co IV pile stays killed)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * Co twin f(IV) rate deficit (~10x) unchanged -- a RATE defect, orthogonal to transport.
#    * MC blue-tilt / far-outer hot-band untouched; T_rad pin (C9).
#  WIRING (check FIRST on any null):
#    [BSRC_PHOT] ... WFLOOR=0.020 PHOT_SRC=1                 (once, init; Prong A armed)
#    [FB_OTS] ... NUMIN=3.290e+15 Hz (912 A)                 (once, init; Prong B armed)
#    [KPR] itNN: ... phot_bteq=<n> fb_ots_redirects=<m>      (per iter)
#            phot_bteq must become LARGE at/after iter 1 (currently the disease is
#            bteq_exits=0 cdf_exits=59M at the photosphere); fb_ots_redirects>0 after
#            iter 0 => Prong B firing on EUV ground edges.
#    [FLOORM]/[PUMPF]/[FB-MULTI]/[DBFB]/[SIMUL] done          (inherited kpr5 wiring)
#  FALSIFIERS (from the design 6.1):
#    (i)  if A+B fire (counters confirm) but f(FeIV,s8) stays > 0.5, the field is not
#         the whole lock -> re-open ionization balance (alpha/DR), not transport.
#    (ii) if EUV(300-450) fails to steepen while [FB_OTS] demonstrably fires, the
#         B(Te) redirect under-absorbs (Te +1.9kK high) -> escalate to the sigma_bf
#         immediate-rescatter fallback (design 3.2).
#    (iii) if FUV(s0) drops, the NUMIN cutoff is wrong (B touched FUV edges).
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr6 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr6
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr6] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr6] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
export LUMINA_NLTE_FLOOR_MODE=1      # FIX-1: LTE-relative floor + b_k cap (was flat 1e-30)
export LUMINA_NLTE_FLOOR_BKMAX=1000  # FIX-1: departure cap b_k<=1000
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# --- PHOTOSPHERIC EUV REPAIR: Prong A + Prong B composed         <<<<< NEW --------
export LUMINA_KPEMISS_BSRC_PHOT=1          # Prong A: extend the -4 B(Te) exit to the
                                           #   photospheric thick tier (WFLOOR<W<=TAU)
export LUMINA_KPEMISS_BSRC_PHOT_WFLOOR=0.02 # phot-tier lower W bound (excludes W<0.02 thin nebula)
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te) (Wien-dead EUV);
                                           #   deep tier keeps SRC=2 (BSRC_SRC above)
export LUMINA_KPEMISS_FB_OTS=1             # Prong B: case-B/OTS redirect of EUV ground-edge
                                           #   (-3) fb -> B(Te) draw in the thick tier
export LUMINA_KPEMISS_FB_OTS_NUMIN=3.29e15 # EUV cutoff = 912A; FUV/optical edges keep raw fb

TAG="a10_kx_kpr6"
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
echo "[kpr6] verify PRONG-A wiring: grep -E '\[BSRC_PHOT\]|phot_bteq' logs/coevolve_consume_${TAG}/stdout.log  # init + per-iter phot_bteq>0"
echo "[kpr6] verify PRONG-B wiring: grep -E '\[FB_OTS\]|fb_ots_redirects' logs/coevolve_consume_${TAG}/stdout.log  # init + per-iter redirects>0"
echo "[kpr6] verify gates:         grep -E '\[BSRC_PHOT\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr6] PRIMARY: f(FeIV,s8)<=0.25, Gph(FeIII,s8)<=~3.5e-4, EUV(300-450,s8) steep decline, T_e(s8)<=11.4kK; RETAINED: FUV(s0)>=1.5e-4, slope>=+2.0, u(s0)>=450, funnel<=3x"
