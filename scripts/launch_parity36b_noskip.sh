#!/bin/bash
# parity36b-noskip: TREATMENT arm of the SKIP_Z A/B (2026-07-27 night).
#
# Single effective variable vs parity36a: LUMINA_NLTE_SKIP_Z is set to the EMPTY
# string, so silicon leaves the skip list and is treated like every other NLTE
# element. Verified offline that this is config-only, no code path added:
#   run_coevolve_s01.sh:28  LUMINA_NLTE_SKIP_Z=${LUMINA_NLTE_SKIP_Z-14}
#     -> `${VAR-14}` substitutes 14 only when VAR is UNSET; set-but-empty stays
#        empty (checked in bash directly, both branches).
#   plasma.c nlte_skip_z_load:  if (!e || !*e) return;   -> no element skipped
#   cuda.cu:1692                if (e && *e) {...}       -> no element skipped,
#                                                          and NO banner printed
#
# WHY this is the orthodox configuration, not another knob:
# tau and S_l must come from the SAME populations. The code's Sobolev tau carries
# stim_corr = 1 - (g_l n_u)/(g_u n_l) = d/(1+d) while the line source is
# S_l = (2hv^3/c^2)/d, so the product is
#     S_l * tau = (2hv^3/c^2) * C * f_lu * lambda * t_exp * n_upper * (g_l/g_u)
# — d cancels EXACTLY (verified numerically on 4 probe cells, ratio 1.000000).
# A consistent NLTE pair therefore cannot manufacture energy no matter how close
# the populations sit to inversion: the emergent contribution is proportional to
# n_upper. SKIP_Z breaks that pairing (nebular thick tau x NLTE S_l), which is
# what produced parity35's FORMAL-CONS 3.484 -> 5973 (1714x, 83% of the excess in
# 1000-1300A). Removing the element from the skip list restores the pairing at
# the source instead of gating the symptom.
#
# Expected cost side (measured from parity35's dump for Si III): tau_NLTE is
# 50-2000x LARGER than nebular in the photosphere, so this arm ADDS silicon line
# blanketing. That opacity rides on b_k values the campaign has not yet validated
# against CMFGEN (s8 b4=2.83, b9=18.7), so the spectrum is expected to move and
# the run is a MEASUREMENT, not a candidate for promotion.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityS
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity36b
export LUMINA_NLTE_SKIP_Z=
export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1 LUMINA_RADEQ_OMEGA_FLOOR=1
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
export LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 LUMINA_GPH_ALLLEVEL_NLTE=1
export LUMINA_EVENT_LOG_CAP=400 LUMINA_JNU_FINE_DUMP=1
export LUMINA_NLTE_FINAL_RESOLVE=1 LUMINA_JBAR_DUMP=1
export LUMINA_JBAR_DUMP_IONS=14:1,14:2
export LUMINA_C1_SUPERBIN_TEPIN=1 LUMINA_C1_BIN_DUMP=1
export LUMINA_RADEQ_DB_FB=1
export LUMINA_CMF_ADV_SPLIT=1 LUMINA_CMF_FINE_ALI=20000
export LUMINA_LINE_THERM=1 LUMINA_LINE_THERM_SMAX=49
export LUMINA_CMF_FINE_LINEDUMP=1 LUMINA_CMF_FINE_LINEDUMP_SHELL=8

OUT=logs/coevolve_consume_$P0TAG
bash scripts/run_coevolve_s01.sh consume

# --- self-preservation (see parity36a for why the launcher does this) --------
for f in cmf_fine_linedump_s8.csv lumina_c1_bins.csv lumina_jbar_dump.csv \
         lumina_levelpop_resolve_raw.csv lumina_levelpop_resolve_ema.csv; do
  if [ -f "$f" ] && [ "$f" -nt "$OUT/.run_start" ]; then
    cp -f "$f" "$OUT/$f"; echo "[preserve] $f"
  else
    echo "[preserve] SKIP $f (missing or older than run start)"
  fi
done

# --- post-run envcheck -------------------------------------------------------
echo "=== envcheck (from the binary's own RESOLVED CONFIG block) ==="
for g in "LUMINA_NLTE_SKIP_Z=" "LUMINA_JBAR_DUMP_IONS=14:1,14:2" \
         "LUMINA_JBAR_DUMP=1" "LUMINA_BIN=lumina_cuda.withParityS"; do
  if grep -qxF "  $g" "$OUT/stdout.log"; then echo "  OK   [$g]"
  else echo "  FAIL [$g]  <-- gate did not reach the process; run is void"; fi
done
# The GPU skip banner must be ABSENT — it prints only when the list is non-empty.
if grep -q "NLTE_SKIP_Z active" "$OUT/stdout.log"; then
  echo "  FAIL skip banner present -> silicon still skipped; run is void"
else
  echo "  OK   no SKIP_Z banner (silicon is NLTE like every other element)"
fi
echo "DONE parity36b"
