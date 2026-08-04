#!/bin/bash
# parity36a-base: CONTROL arm of the SKIP_Z A/B (2026-07-27 night).
#
# Identical to parity33 in physics; the ONLY change is that the per-line jbar
# observer now also dumps Si II (LUMINA_JBAR_DUMP_IONS=14:1,14:2 instead of the
# 14:2 default). The dump is write-only observation inside an omp critical
# section, so it must not move any number — that is registered as the control.
#
# Why this run exists: the jbar dump's `beta` for a SKIP_Z element is
# radeq_beta_esc(tau_sobolev) evaluated on the NEBULAR tau that SKIP_Z preserves
# (plasma.c:13087). Inverting beta therefore recovers tau_nebular per
# (line,shell) — the only offline route to it, since partition functions are not
# dumped. parity36b re-runs this exact config with SKIP_Z removed, where the same
# column yields tau_NLTE. The pair measures, per line and shell, whether the
# stated SKIP_Z rationale still holds:
#   "NLTE rate matrices can collapse populations of dominant ions (e.g. Si II)
#    in inner shells ... producing tau values many orders of magnitude below the
#    Saha-Boltzmann nebular estimate"  (lumina_plasma.c, above nlte_skip_z_load)
# Measured already for Si III from parity35's dump: NO collapse — tau_NLTE is
# 50-2000x LARGER in the photosphere and nebular-thick lines stay thick. Si II is
# the ion the comment names and the one with no data, hence this run.
#
# Binary: withParityS = the parity33 binary (LUMINA_SL_WRITE_SKIPZ is dead and
# stays unset in BOTH arms; that gate was judged structurally wrong 2026-07-27).
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn || exit 1
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8 | tr '\n' '; ')"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_BIN=lumina_cuda.withParityS
export LUMINA_KPACKET=1
export LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity36a
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

# --- self-preservation -------------------------------------------------------
# The slurm runner (sbatch_parity_runner.sh) has no preserve() step, unlike the
# manual runner, and run_coevolve_s01.sh copies only 5 CSVs. Everything the
# judgment battery reads must be copied here or the next queued run overwrites
# it in repo root. Same freshness rule as run_coevolve_s01.sh: only products
# newer than $OUT/.run_start are this run's.
for f in cmf_fine_linedump_s8.csv lumina_c1_bins.csv lumina_jbar_dump.csv \
         lumina_levelpop_resolve_raw.csv lumina_levelpop_resolve_ema.csv; do
  if [ -f "$f" ] && [ "$f" -nt "$OUT/.run_start" ]; then
    cp -f "$f" "$OUT/$f"; echo "[preserve] $f"
  else
    echo "[preserve] SKIP $f (missing or older than run start)"
  fi
done

# --- post-run envcheck -------------------------------------------------------
# Authority is the binary's own environ walk (cuda.cu:5899 "RESOLVED CONFIG"),
# not this launcher's exports: an intermediate script can unset a gate silently
# (run_coevolve_s01.sh:115 idiom; that accident voided parity34's 83 GPU-min).
echo "=== envcheck (from the binary's own RESOLVED CONFIG block) ==="
for g in "LUMINA_NLTE_SKIP_Z=14" "LUMINA_JBAR_DUMP_IONS=14:1,14:2" \
         "LUMINA_JBAR_DUMP=1" "LUMINA_BIN=lumina_cuda.withParityS"; do
  if grep -qF "  $g" "$OUT/stdout.log"; then echo "  OK   $g"
  else echo "  FAIL $g  <-- gate did not reach the process; run is void"; fi
done
grep -c "^  LUMINA_SL_WRITE_SKIPZ" "$OUT/stdout.log" | sed 's/^/  SL_WRITE_SKIPZ occurrences (must be 0): /'
echo "DONE parity36a"
