#!/bin/bash
# parity42-gphoff: roll back the three GPH gates that broke Fe and S ionisation.
#
# Single effective variable vs parity41: LUMINA_GPH_ALLLEVEL,
# LUMINA_GPH_ALLLEVEL_NLTE and LUMINA_GPH_SIGMA_CMFGEN go from 1 to unset.
# Everything else — binary, atomic data (the S IV/V + Ca IV/V build), all other
# gates — is held.
#
# WHY. The s8 ion-fraction timeline across the whole parity series pins the
# regression to one run:
#     parity10-22   Fe IV 0.0198-0.0235   (truth 0.0219)   S II 0.21-0.86
#     parity23-41   Fe IV 0.249 -0.607    (17-28x over)    S II 0.0017-0.0038
# and the binary's own RESOLVED CONFIG diff between parity22 and parity23 is
# EXACTLY those three lines, nothing else. So one gate flip broke two elements at
# once, and it has been on in every run since — including the baseline promoted
# earlier today (scripts/parity_baseline.env). Everything built on top of it
# (the SKIP_Z work, the LTE-floor work, front D) was measured on a broken
# ionisation state.
#
# The mechanism is a compensating pair, and today's Saha self-test sees both
# halves independently:
#   LUMINA_ALPHA_SPINGATE  cuts Fe III alpha by 10.5x at 5 kK — this is the gate
#                          that CLOSED the Fe gap at parity12 (f(FeIV) 0.0049 ->
#                          0.0213 vs truth 0.022).
#   LUMINA_GPH_ALLLEVEL    raises Gamma (G_all/G_gnd = 40.7).
# Each was adopted against a different baseline. Together they over-ionise Fe by
# ~17x, and the J=B closure test scores the champion set at Fe III 3.95.
#
# The ledger already contained the clue and did not act on it: dig_F18 recorded
# "parity17/22 = the campaign's best outer-shell result (= before GPH all-level)"
# without connecting it to the ionisation regression.
#
# REGISTERED
#  W1 wiring (hard): RESOLVED CONFIG must contain NO LUMINA_GPH_ALLLEVEL,
#     LUMINA_GPH_ALLLEVEL_NLTE or LUMINA_GPH_SIGMA_CMFGEN line. If any is
#     present the rollback did not take and the run is void, not negative.
#  F1 (the test, directional): Fe IV at s8 must return to the 0.02 band.
#     Registered as < 0.06 (parity41 is 0.6072, truth 0.0219, the pre-regression
#     runs sat at 0.0198-0.0235).
#  F2 (directional): S II at s8 must rise above 0.05 (parity41 0.0011,
#     truth 0.1522, pre-regression 0.21-0.86 — those overshot, so landing
#     anywhere in 0.05-0.9 counts as the regression undone).
#  M1 characterisation: full ion-fraction table vs CMFGEN for Si/S/Ca/Fe/Co/Ni,
#     and whether the new Ca IV / S IV stages survive the rollback.
#  M2 characterisation: front-D metrics — emergent L/L_truth (parity41 0.960),
#     redistribution deficit/excess (26.6% / 22.7%), band ratios (3500-4500
#     0.540, 10000-14000 1.917), line blocking per 1000 A at s8/s20/s35.
#     If over-ionisation was driving front D, the blocking collapse should ease.
#  M3 watch: T_e and n_e vs CMFGEN (parity41 medians 0.858 / 1.002).
#
# NOT a promotion candidate by itself: rolling back restores Fe/S but also
# removes the excited-level photoionisation that Bug1 is about. The point is to
# establish which baseline the last 19 runs should have been measured against.
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity42

export LUMINA_MODEL_DIR=data/tardis_reference_toy06_19p48d_sivcaiv
. scripts/parity_baseline.env

export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1 LUMINA_RADEQ_OMEGA_FLOOR=1
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
# --- the single variable: the three GPH gates are NOT exported -----------------
# (parity41 had: LUMINA_GPH_SIGMA_CMFGEN=1 LUMINA_GPH_ALLLEVEL=1 LUMINA_GPH_ALLLEVEL_NLTE=1)
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

echo "=== W1 wiring: the three GPH gates must be ABSENT ==="
for g in LUMINA_GPH_ALLLEVEL LUMINA_GPH_ALLLEVEL_NLTE LUMINA_GPH_SIGMA_CMFGEN; do
  if grep -qE "^  $g=" "$OUT/stdout.log"; then echo "  FAIL $g still set -> run is VOID"
  else echo "  OK   $g absent"; fi
done
grep -qxF "  LUMINA_MODEL_DIR=data/tardis_reference_toy06_19p48d_sivcaiv" "$OUT/stdout.log" \
  && echo "  OK   atomic data held (sivcaiv)" || echo "  FAIL atomic data not held"
echo "=== F1/F2 quick look at s8 (truth: Fe IV 0.0219, S II 0.1522) ==="
awk -F, 'NR>1 && $1==8 {t[$2]+=$4; v[$2","$3]=$4}
         END {printf "  Fe IV %.4f   Fe III %.4f   S II %.4f   S III %.4f   Ca IV %.4f\n",
              v["26,3"]/t[26], v["26,2"]/t[26], v["16,1"]/t[16], v["16,2"]/t[16], v["20,3"]/t[20]}' \
    "$OUT/lumina_ion_pops.csv" 2>/dev/null
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
echo "DONE parity42"
