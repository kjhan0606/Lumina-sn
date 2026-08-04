#!/bin/bash
# ============================================================================
# FEIII_COLDATA A/B: milnec2ne64 champion env + real Zhang 1996 Fe III
#   collisional strengths (LUMINA_FEIII_COLDATA=1). SINGLE-VARIABLE delta.
# ============================================================================
# ROOT CAUSE (verified): the over-populated photospheric Fe III levels
#   (25 b_k=72, 17 b_k=109, 18/28/31/32 b_k=5-14) drain only through FORBIDDEN
#   radiative lines (f_lu ~ 1e-8). Lumina's collision proxy = van Regemorter
#   (C ~ f_lu) -> ~0 collisional drain; the METACOLL Omega=0.1 floor is ~13x too
#   weak. CMFGEN drains them with real close-coupling Omega ~ 1-9 to EVERY lower
#   level (25->17 Omega=8.76, 17->ground 1.36). THE FIX imports the exact Zhang
#   FeIII_COL_DATA (22139 trans x 20 T) and makes it the SOLE Fe III collision
#   source (per-line proxy + METACOLL suppressed for Fe III -> CMFGEN parity).
#
# Binary: lumina_cuda.withColdata (= withMilneC2_ne64 source + FEIII_COLDATA gate
#   + loader). FEIII_COLDATA unset => byte-identical to withMilneC2_ne64.
#
# Yardstick = CMFGEN toy06 @19.48d. PASS = FeIII s8 b_k(25/17/18/28/31/32) drop
#   toward ~1; s8 mc_J/B_nu(461-520A) drops from ~20x toward 0.06x; f(FeIV) s8
#   toward 0.022; deep f(FeIV) s0 held ~0.98.
# ============================================================================
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# --- [PARALLEL] own-binary via LUMINA_BIN (no ./lumina_cuda cp contention) ---
export LUMINA_BIN=lumina_cuda.withColdata
[ -x "$LUMINA_BIN" ] || { echo "FATAL: $LUMINA_BIN missing/not built"; exit 2; }
echo "[parallel] using $LUMINA_BIN directly (md5=$(md5sum $LUMINA_BIN | cut -d' ' -f1))"

# --- OMP: single run -> all 32 cores (user requested OMP=32) ---
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_EVENT_LOG_ESCATTER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
unset LUMINA_NLTE_BK_CEIL
unset LUMINA_GPH_JTABLE LUMINA_TE_TABLE LUMINA_TINNER_COLOR \
      LUMINA_MACROATOM_BF LUMINA_LINE_THERM

# --- metastable collisional drain (kept ON for Co/Ni; Fe III now handled by Zhang) ---
export LUMINA_NLTE_METASTABLE_COLL=1
export LUMINA_NLTE_METACOLL_MODE=2
export LUMINA_NLTE_METACOLL_OMEGA=0.1

# --- STAGE4-ROUND2 ---
export LUMINA_NLTE_STAGE4=1
export LUMINA_STAGE4_GPH_WTHR=0.13
export LUMINA_STAGE4_BK_CAP=0

# --- KPEMISS_REPAIR ---
export LUMINA_KPEMISS_REPAIR=1
export LUMINA_KPEMISS_SE_POPS=1
export LUMINA_KPKT_FB_MULTI=1
export LUMINA_KPEMISS_BSRC_TAU=0.13
export LUMINA_KPEMISS_BSRC_SRC=2
export LUMINA_RADEQ_DB_FB=1
export LUMINA_KPEMISS_COOLGUARD=1
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1
export LUMINA_RADEQ_PUMP_FIELD=1
export LUMINA_NLTE_FLOOR_MODE=0
export LUMINA_NLTE_FLOOR_BKMAX=1000000000
export LUMINA_RADEQ_PUMP_FALLBACK=1
export LUMINA_KPEMISS_BSRC_PHOT=0
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1
export LUMINA_KPEMISS_FB_OTS=0
export LUMINA_KPEMISS_FB_OTS_NUMIN=4.80e15
export LUMINA_KPEMISS_BSRC_PHOT_XION=1
export LUMINA_KPEMISS_OTS_MODE=2
export LUMINA_KPEMISS_OTS_TAU=1.0
export LUMINA_KPEMISS_TE_POP=1

# --- MILNE: exact per-level radiative-recombination fb (compose with the fix) ---
export LUMINA_FB_MILNE_EXACT=1

# ============================================================================
# THE SINGLE NEW DELTA vs withMilneC2_ne64: real Zhang Fe III collisions.
# ============================================================================
export LUMINA_FEIII_COLDATA=1

TAG="a10_kx_coldata"
mkdir -p logs/coevolve_consume_${TAG}
rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="$TAG"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin lumina_levelpop.csv lumina_ion_pops.csv lumina_plasma_state.csv; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_${TAG}/$f"
done
echo "${TAG} DONE -> logs/coevolve_consume_${TAG}/"
