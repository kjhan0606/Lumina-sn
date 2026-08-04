#!/bin/bash
# parity40-tepin: does the T_e deficit explain front D (wavelength redistribution)?
#
# Single effective variable vs parity39: LUMINA_TE_TABLE pins T_e to the CMFGEN
# 19.48 d structure. Everything else is the promoted baseline.
#
# WHY. parity38/39 vs the published CMFGEN spectrum (StaNdaRT toy06, 19.48 d slice,
# overlap window 2500-19995 A) is an ENERGY-REDISTRIBUTION defect, not a budget
# error: deficit area 22.4% of L_truth, excess area 23.8%, net +1.4%. The
# cumulative (ours-truth) runs -5.8% at 4000 A, -11.0% at 5500 A (minimum), -1.1%
# at 10000 A, +1.4% at the end — blue/optical energy comes out in the IR instead.
# The excess is CONTINUUM (continuum ratio ~ total ratio: 1.848 vs 1.899 over
# 10000-14000 A), so this is a different front from the line-fluorescence work.
#
# The matching structural defect, measured against the same CMFGEN file
# (phys_toy06_cmfgen.txt, #TIME 19.480):
#     n_e  ours/CMFGEN : median 0.975  (0.797-1.160)  -> ionisation is validated
#     T_e  ours/CMFGEN : median 0.859, and it degrades outward —
#                        s8 1.088, s35 0.800, s45 0.606, s49 0.449
#                        (11057 K where CMFGEN has 24600 K)
# A far-outer layer that is up to 2.2x too cold cannot radiate its share in the
# blue/UV, which is the shape of the redistribution seen in the spectrum.
#
# WHAT THE GATE DOES (read at lumina_plasma.c:8042 before use): whole-state pin.
# The table value replaces the radeq-solved T_e for every shell BEFORE
# simul_ladder, so the ion ladder, n_e, collisional rates, the GPU k-packet ff/fb
# and the co-evolve birth Planck SED all inherit it. Table
# data/cmfgen_te_table_toy06_19p48d.csv was built from the same phys file
# (header records the provenance) and must fill all 50 shells or the gate reports
# INACTIVE — the loader is fail-closed, so check the [TETAB] banner.
#
# THIS IS A DIAGNOSTIC, NOT A CANDIDATE CONFIG. Pinning T_e removes the energy
# balance that the temperature is supposed to satisfy, so a good spectrum here
# proves attribution, not a repair.
#
# REGISTERED
#  W1 wiring (hard): "[TETAB] ... (WHOLE-STATE T_e pin ACTIVE)" present, shells=50,
#     and the run's lumina_plasma_state.csv T_e matches the table within 1%.
#  F1 (the test, directional): if the T_e deficit is the cause of front D, the
#     redistribution must shrink — deficit AND excess areas both below 12% of
#     L_truth (from 22.4% / 23.8%), and the cumulative minimum at ~5500 A above
#     -5% (from -11.0%). If both stay near their parity39 values, T_e is NOT the
#     driver and front D needs a different mechanism.
#  M1 characterisation, no threshold: band ratios vs truth (parity39 reference
#     2500-3500 0.906 / 3500-4500 0.609 / 5500-7000 1.366 / 10000-14000 1.899),
#     total L/L_truth (1.014), shape score mean|log10| (0.1321).
#  M2 watch: n_e after the pin. It was already right (median 0.975); if pinning a
#     hotter T_e breaks it, the two are coupled and the attribution is muddier.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity40

. scripts/parity_baseline.env
export LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv

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

echo "=== W1 wiring ==="
grep -m1 "\[TETAB\]" "$OUT/stdout.log" || echo "  FAIL no [TETAB] banner -- gate inactive, run is void"
for g in "LUMINA_TE_TABLE=data/cmfgen_te_table_toy06_19p48d.csv" \
         "LUMINA_NLTE_LTE_FLOOR=0" "LUMINA_NLTE_SKIP_Z="; do
  grep -qxF "  $g" "$OUT/stdout.log" && echo "  OK   [$g]" || echo "  FAIL [$g]"
done
echo "=== scalars ==="
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
echo "DONE parity40"
