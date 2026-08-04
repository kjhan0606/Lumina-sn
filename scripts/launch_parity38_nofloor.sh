#!/bin/bash
# parity38-nofloor: is the LTE floor the source of the 1000-1300 A pathology?
#
# Single effective variable vs parity37: LUMINA_NLTE_LTE_FLOOR=0.
# Everything else — physics config, binary (withParityU), dump shells — identical.
#
# WHAT parity37's tau observer established (all measured, not inferred):
#  * The (tau, S_l) pair IS matched once silicon is out of SKIP_Z: inverting the
#    dump (d = (2hv^3/c^2)/S_l -> stim = d/(1+d) -> n_l implied by tau) reproduces
#    the levelpop n_l to a median 0.76 / 1.26 / 0.95 at s8 / s45 / s49. So the
#    cancellation identity holds and no energy is manufactured by mispairing —
#    that defect was parity35's alone (FORMAL 5973 vs 34.89 here, 171x).
#  * The residual 10x excess is a SECOND, independent defect. In 1000-1300 A at
#    s45/s49, silicon's 195 lines carry 99.8-99.9% of all line emission (98,556
#    other lines carry the rest), and attributing that emission by level
#    provenance gives:
#        mixed pair (one level floored)  99.8% / 99.9%
#        both levels floored              0.0%
#        both levels SOLVED               0.0%
#    Mixed means: lower level solved at b = 1e-4..1e-3, upper level replaced by
#    the LTE floor at EXACTLY b = 1.0000 (cuda.cu:1490-1514 puts every level with
#    x[i] <= xmax*1e-12 at its LTE@Te value relative to the ion ground). Raising
#    the upper level to LTE while the lower stays 3-4 decades below it drives
#    (g_u n_l)/(g_l n_u) toward 1, i.e. d -> 0, i.e. S_l/B = 1e3..1.8e5.
#
# THE TEST: with the floor off, the writeback keeps the solve's own (tiny)
# excited populations — legacy branch `if (x[i] < 0.0) x[i] = 1e-30`. If the floor
# is the source, those upper levels go back to ~0, d becomes large, and S_l
# collapses toward B.
#
# REGISTERED, before the run:
#  F1 (the test, directional): sum of Sl_times_esc over Si lines in 1000-1300 A
#      must fall by >= 10x at BOTH s45 and s49 (parity37: 5.922e-04 / 2.679e-04).
#  F2 (scalar): FORMAL-CONS must fall below 15 (parity37/36b: 34.89; the
#      silicon-LTE baseline parity33/36a: 3.484). A value still near 34.89 means
#      the floor is NOT the source and the mixed-pair attribution above is wrong.
#  R1 (RISK — the floor exists for a reason): the FLOORM comment (cuda.cu:952)
#      says the flat 1e-30 clamp, after the per-ion rescale, becomes an absolute
#      population floor that pins a trace ion's near-threshold comb at b_k ~ 1e8.
#      Count levels with b_k > 1e4 in lumina_levelpop_resolve_raw.csv and compare
#      with parity37. If that count explodes, "remove the floor" is the wrong
#      repair and the right one is to fix the floor's REFERENCE (floor relative to
#      the ion's solved departure instead of at b=1).
#  C1 (control): iterations 0-1 run before NLTE starts (NLTE_START_ITER=2), so
#      c1_bins rows for iter 0-1 must match parity37 exactly.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity38
export LUMINA_NLTE_SKIP_Z=
export LUMINA_NLTE_LTE_FLOOR=0
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
export LUMINA_CMF_FINE_LINEDUMP=1 LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49

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

echo "=== envcheck (from the binary's own RESOLVED CONFIG block) ==="
for g in "LUMINA_NLTE_LTE_FLOOR=0" "LUMINA_NLTE_SKIP_Z=" \
         "LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49" "LUMINA_BIN=lumina_cuda.withParityU"; do
  if grep -qxF "  $g" "$OUT/stdout.log"; then echo "  OK   [$g]"
  else echo "  FAIL [$g]  <-- gate did not reach the process; run is void"; fi
done
echo "=== F2 scalar (parity37 = 34.89; silicon-LTE baseline = 3.484) ==="
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
echo "=== R1 risk counter: levels with b_k > 1e4 (parity37 comparison) ==="
for r in parity37 parity38; do
  f=logs/coevolve_consume_$r/lumina_levelpop_resolve_raw.csv
  [ -f "$f" ] && echo "  $r: $(awk -F, 'NR>1 && $9>1e4' "$f" | wc -l)"
done
echo "DONE parity38"
