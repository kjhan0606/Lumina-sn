#!/bin/bash
# parity37-taudump: parity36b replayed on the tau-observer binary (2026-07-28).
#
# PHYSICS CONFIG IS BYTE-FOR-BYTE parity36b (SKIP_Z empty = silicon NLTE like
# every other element). The ONLY differences are diagnostic:
#   binary  lumina_cuda.withParityU  (= withParityS sources + the linedump edit)
#   env     LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49   (was 8)
#
# WHY: after parity36b there is still no observer for tau_sobolev anywhere in the
# pipeline, and the instrument that looked like a stand-in is blind. beta in
# lumina_jbar_dump.csv is recorded during rate assembly, but compute_plasma_state
# rewrites tau_sobolev from the nebular ion pops at the top of every iteration and
# the NLTE tau is written at the END of nlte_solve_all_gpu — so that beta is the
# NEBULAR tau in every arm. Measured proof: 89% of Si III betas are byte-identical
# between the SKIP_Z=14 and SKIP_Z-empty runs, and the rest differ in the 6th
# digit. Every tau number produced tonight is therefore unverified.
#
# The edit adds tau_sob and Sl_times_esc = S_l*(1-e^-tau) to the fine linedump,
# read at the point the formal solver consumes line_source_S — so the (tau, S_l)
# pair can finally be inspected AS CONSUMED. The shell list targets the band that
# carries 84.5% of parity36b's excess (1000-1300 A, dominated by s40-49): s8 keeps
# continuity with every earlier dump, s45 and s49 are the pathology.
#
# REGISTERED CONTROL (write it down before the run): the edit touches only a
# diagnostic block, so this run MUST reproduce parity36b. Required:
#     lumina_c1_bins.csv / lumina_plasma_state.csv / lumina_ion_pops.csv /
#     lumina_spectrum_formal.csv   ->  0 differing rows vs parity36b
#     FORMAL-CONS = 34.89
# If any of those move, the diagnostic edit perturbed physics and the tau numbers
# describe a DIFFERENT run — they must not then be compared with parity36b.
# (Precedent: parity36a reproduced parity33 with 0 differing rows across a host
# and thread-count change, so this pipeline does support that standard.)
#
# TIMING RISK, stated up front: parity36b took 94.8 min and slurm 183402 expires
# 2026-07-28 03:28. If this is killed mid-run, the script sits in
# runner_spool/running/ — move it back to runner_spool/queue/ to retry on the
# next allocation (backup job 186309 is pending).
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityU
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity37
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
for g in "LUMINA_NLTE_SKIP_Z=" "LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49" \
         "LUMINA_BIN=lumina_cuda.withParityU"; do
  if grep -qxF "  $g" "$OUT/stdout.log"; then echo "  OK   [$g]"
  else echo "  FAIL [$g]  <-- gate did not reach the process; run is void"; fi
done
if grep -q "NLTE_SKIP_Z active" "$OUT/stdout.log"; then
  echo "  FAIL skip banner present -> silicon still skipped; run is void"
else
  echo "  OK   no SKIP_Z banner"
fi
grep -c "LINEDUMP wrote" "$OUT/stderr.log" | sed 's/^/  linedump files written (must be 3): /'
echo "=== registered control: must reproduce parity36b ==="
grep -E "FORMAL-CONS" "$OUT/stdout.log" | tail -1
for f in lumina_c1_bins.csv lumina_plasma_state.csv lumina_ion_pops.csv lumina_spectrum_formal.csv; do
  b=logs/coevolve_consume_parity36b/$f
  if [ -f "$b" ] && [ -f "$OUT/$f" ]; then
    n=$(diff <(cat "$b") <(cat "$OUT/$f") | grep -c '^<' || true)
    echo "  $f: $n differing lines vs parity36b"
  fi
done
echo "DONE parity37"
