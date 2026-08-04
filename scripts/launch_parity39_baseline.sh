#!/bin/bash
# parity39-baseline: REGRESSION PIN for the gates promoted 2026-07-28.
#
# Physics is exactly parity38. The only difference is HOW the promoted gates are
# set: parity38 exported them inline, parity39 sources scripts/parity_baseline.env.
# So this run does two jobs at once —
#   (1) verifies the baseline env file wires correctly end to end, and
#   (2) pins the promoted configuration as the reference every later parity run
#       is diffed against.
#
# REGISTERED CONTROL (hard): identical gates + identical binary + a deterministic
# pipeline must reproduce parity38 exactly.
#     lumina_c1_bins.csv / lumina_plasma_state.csv / lumina_ion_pops.csv /
#     lumina_spectrum_formal.csv  ->  0 differing rows vs parity38
#     FORMAL-CONS = 4.37
# Precedent that this standard is achievable: parity36a reproduced parity33 with 0
# differing rows across a change of host AND thread count; parity37 reproduced
# parity36b with 0 differing rows across a binary rebuild carrying a new dump.
# A nonzero diff here means the env file does not reproduce the inline exports —
# find the missing or extra gate before using the baseline for anything.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity39

# --- the promoted baseline (SKIP_Z empty, LTE_FLOOR=0, standard observers) ----
. scripts/parity_baseline.env

export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1 LUMINA_RADEQ_OMEGA_FLOOR=1
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
export LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 LUMINA_GPH_ALLLEVEL_NLTE=1
export LUMINA_EVENT_LOG_CAP=400 LUMINA_JNU_FINE_DUMP=1
export LUMINA_NLTE_FINAL_RESOLVE=1
export LUMINA_C1_SUPERBIN_TEPIN=1 LUMINA_C1_BIN_DUMP=1
export LUMINA_RADEQ_DB_FB=1
export LUMINA_CMF_ADV_SPLIT=1 LUMINA_CMF_FINE_ALI=20000
export LUMINA_LINE_THERM=1 LUMINA_LINE_THERM_SMAX=49

OUT=logs/coevolve_consume_$P0TAG
bash scripts/run_coevolve_s01.sh consume

for f in cmf_fine_linedump_s8.csv cmf_fine_linedump_s45.csv cmf_fine_linedump_s49.csv \
         lumina_c1_bins.csv lumina_jbar_dump.csv \
         lumina_levelpop_resolve_raw.csv lumina_levelpop_resolve_ema.csv; do
  if [ -f "$f" ] && [ "$f" -nt "$OUT/.run_start" ]; then
    cp -f "$f" "$OUT/$f"; echo "[preserve] $f"
  else
    echo "[preserve] SKIP $f (missing or older than run start)"
  fi
done

echo "=== envcheck (binary's own RESOLVED CONFIG) ==="
for g in "LUMINA_NLTE_LTE_FLOOR=0" "LUMINA_NLTE_SKIP_Z=" \
         "LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49" "LUMINA_JBAR_DUMP_IONS=14:1,14:2" \
         "LUMINA_BIN=lumina_cuda.withParityU"; do
  if grep -qxF "  $g" "$OUT/stdout.log"; then echo "  OK   [$g]"
  else echo "  FAIL [$g]  <-- baseline env did not reach the process"; fi
done
grep -q "NLTE_SKIP_Z active" "$OUT/stdout.log" && echo "  FAIL skip banner present" || echo "  OK   no SKIP_Z banner"
echo "=== REGISTERED CONTROL: must reproduce parity38 ==="
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
for f in lumina_c1_bins.csv lumina_plasma_state.csv lumina_ion_pops.csv lumina_spectrum_formal.csv; do
  b=logs/coevolve_consume_parity38/$f
  if [ -f "$b" ] && [ -f "$OUT/$f" ]; then
    n=$(diff "$b" "$OUT/$f" | grep -c '^<' || true)
    echo "  $f: $n differing lines vs parity38"
  fi
done
echo "DONE parity39"
