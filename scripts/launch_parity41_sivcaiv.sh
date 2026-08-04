#!/bin/bash
# parity41-sivcaiv: import the four ions the campaign has been missing since
# 2026-07-19 — S IV, S V, Ca IV, Ca V.
#
# Single effective variable vs parity39 (the promoted baseline): the atomic data
# reference directory. Same binary (withParityU), same gates, same everything.
#
# WHY. Without those four stages, S IV/V and Ca IV/V are level-less destination
# rungs, so LUMINA_SIMUL_CAP_TOPION truncates the ladder with r:=0 and their
# populations are identically zero. Measured against the published CMFGEN toy06
# @19.48d (StaNdaRT ionfrac files):
#     Ca III  ours 1.0000 in EVERY shell   vs truth 0.84-0.97 (Ca IV 2.5-16%)
#     S II  @s8  ours 0.0021              vs truth 0.1522   (72x too low)
#     Fe IV @s8  ours 0.359               vs truth 0.0219   (16x too high)
# and dig_F18 recorded the deeper cost: the cap breaks the energy ledger --
# photo-heating is charged to the element while the fb cooling term is
# identically zero because nion[upper] == 0. CMFGEN carries all four ions with
# osc_data + phot_data_A + col_data (19apr23), so this is faithfulness, not an
# extension. The importer, the procedure and the priority (Ca IV >= S IV) were
# all written down on 2026-07-19 and never executed.
#
# DATA BUILD (data/tardis_reference_toy06_19p48d_sivcaiv), verified before this run:
#   new ions      S IV 194 / S V 200 / Ca IV 378 / Ca V 200 levels
#   sigma_bf      100% / 100% / 99.5% / 98.5% coverage (existing S III is 67.6%)
#   ionisation    S 3->5 stages, Ca 3->5 stages
#   SINGLE VAR    zero existing ions changed level count. Ti/Cr/Mn II were about
#                 to ride along at 600->1000 levels (a super-level entry added to
#                 the importer on 2026-06-04 and never rebuilt); held back so the
#                 A/B stays one variable.
#   re-indexing   inserting Z=16/20 stages shifts every global level index above
#                 them, so ma_radrecomb_target.bin (keyed by target_gidx) and the
#                 three .npy caches (sized by n_lines 2,565,342 -> 2,584,132) were
#                 REBUILT. ige_col_*.bin (per-ion indices, Z/ion0 in the header)
#                 and level_multiplicity.csv ((Z,ion,level_num) keys) are index-
#                 independent and were reused.
#   ladder room   sum(npop-1) 49 -> 53 against SIM_MAXP 96. plasma.c:8315 drops a
#                 whole element from the ladder with NO warning past that limit,
#                 so this was checked before importing, not after.
#
# REGISTERED
#  W1 wiring (hard): RESOLVED CONFIG shows LUMINA_MODEL_DIR=...sivcaiv and the
#     argv carries that directory; the run's ion_pops must contain S stage 4 and
#     Ca stage 4 rows (they cannot exist in the baseline).
#  F1 (the test, directional): Ca III must leave the 1.0000 rail. Truth has Ca IV
#     at 2.5-16% depending on shell; anything above ~1% of Ca IV counts as the
#     rail breaking. If Ca III stays at exactly 1.0000 the import did not reach
#     the ladder and the run is void, not negative.
#  F2 (directional): S II at s8 must rise from 0.0021 toward truth 0.1522.
#     Registered as "moves by more than 2x"; closing the full 72x is NOT expected
#     from this change alone.
#  M1 characterisation: full ion-fraction table vs CMFGEN for Si/S/Ca/Fe/Co/Ni.
#  M2 characterisation: the front-D numbers — line blocking per 1000 A at
#     s8/s20/s35 (baseline 244.8 near-UV -> 0.06 IR), emergent L/L_truth over
#     2500-19995 A (baseline 1.014), redistribution deficit/excess areas
#     (baseline 22.4% / 23.8%), band ratios (3500-4500 0.609, 10000-14000 1.899).
#  M3 watch, no threshold: T_e and n_e. The cap previously zeroed the fb cooling
#     of these ions, so restoring them changes the energy ledger; movement here is
#     expected and is the point, but a runaway is a stop signal.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity41

# --- the single variable ------------------------------------------------------
export LUMINA_MODEL_DIR=data/tardis_reference_toy06_19p48d_sivcaiv

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

echo "=== W1 wiring ==="
for g in "LUMINA_MODEL_DIR=data/tardis_reference_toy06_19p48d_sivcaiv" \
         "LUMINA_NLTE_LTE_FLOOR=0" "LUMINA_NLTE_SKIP_Z=" \
         "LUMINA_BIN=lumina_cuda.withParityU"; do
  grep -qxF "  $g" "$OUT/stdout.log" && echo "  OK   [$g]" || echo "  FAIL [$g]"
done
grep -m1 "argv:" "$OUT/stdout.log"
echo "  S stage-4 rows in ion_pops : $(awk -F, 'NR>1 && $2==16 && $3==4' "$OUT/lumina_ion_pops.csv" 2>/dev/null | wc -l)  (0 = import did not reach the run)"
echo "  Ca stage-4 rows in ion_pops: $(awk -F, 'NR>1 && $2==20 && $3==4' "$OUT/lumina_ion_pops.csv" 2>/dev/null | wc -l)"
echo "=== F1 quick look: is Ca III still on the 1.0000 rail? ==="
awk -F, 'NR>1 && $2==20 {t[$1]+=$4; if($3==2) c3[$1]=$4; if($3==3) c4[$1]=$4}
         END {for (s=8; s<=30; s+=11) if (t[s]>0) printf "  s%-3d Ca III %.4f  Ca IV %.4f\n", s, c3[s]/t[s], c4[s]/t[s]}' \
    "$OUT/lumina_ion_pops.csv" 2>/dev/null
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
echo "DONE parity41"
