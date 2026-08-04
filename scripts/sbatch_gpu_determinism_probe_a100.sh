#!/bin/bash
#SBATCH --job-name=gpuprobe_a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
# CROSS-GPU DETERMINISM PROBE (a100 vs H200 baseline parity44). NOT a physics run.
# PREREGISTERED: exact replica of 51_parity44_droff env + binary withParityW on a
# a100 GPU. A) all physical CSVs byte-identical to logs/coevolve_consume_parity44
#   -> a100 OPENS for judgment runs. B) any diff -> quantify (max rel dev), a100
#   STAYS CLOSED, deviation registered quietly (B9 instrument debt). No tuning.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export LUMINA_RUN_ROOT=/gpfs/kjhan/lumina_probe_a100
cd "$LUMINA_RUN_ROOT"
echo "probe up: host=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1 LUMINA_BIN=lumina_cuda.withParityW LUMINA_KPACKET=1 LUMINA_EVENT_LOG=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PKTS=100000 NITER=12 P0TAG=parity44a100probe
export LUMINA_MODEL_DIR=data/tardis_reference_toy06_19p48d_sivcaiv
. scripts/parity_baseline.env
export LUMINA_MA_REAL_UPSILON=1 LUMINA_MA_LINE_DESTRUCT=1 LUMINA_ALPHA_SPINGATE=1
export LUMINA_SIMUL_CAP_TOPION=1 LUMINA_FB_COOL_KT=1
export LUMINA_OMEGA_CMFGEN=1
export LUMINA_FROZENIN_DR=0
export LUMINA_DR_BOOST_BADNELL=0 LUMINA_DR_BOOST_NORAD=0
export LUMINA_DR_BOOST_MAZZOTTA=0 LUMINA_DR_BOOST_AUTOSTRUCT=0
export LUMINA_MA_RADRECOMB=1 LUMINA_C1_DEGEN_FALLBACK=1 LUMINA_SUPER_LEVELS=1
export LUMINA_EVENT_LOG_CAP=400 LUMINA_JNU_FINE_DUMP=1
export LUMINA_NLTE_FINAL_RESOLVE=1 LUMINA_JBAR_DUMP=1 LUMINA_JBAR_DUMP_IONS=14:1,14:2
export LUMINA_C1_SUPERBIN_TEPIN=1 LUMINA_C1_BIN_DUMP=1
export LUMINA_RADEQ_DB_FB=1
export LUMINA_CMF_ADV_SPLIT=1 LUMINA_CMF_FINE_ALI=20000
export LUMINA_LINE_THERM=1 LUMINA_LINE_THERM_SMAX=49
export LUMINA_CMF_FINE_LINEDUMP=1 LUMINA_CMF_FINE_LINEDUMP_SHELL=8,45,49
OUT=logs/coevolve_consume_$P0TAG
bash scripts/run_coevolve_s01.sh consume
echo "=== DETERMINISM VERDICT vs parity44 (H200) ==="
BASE=$R/logs/coevolve_consume_parity44
for f in lumina_ion_pops.csv lumina_plasma_state.csv lumina_spectrum_formal.csv lumina_levelpop.csv; do
  if cmp -s "$OUT/$f" "$BASE/$f"; then echo "  IDENTICAL $f"; else
    echo "  **DIFF** $f"; python3 - "$OUT/$f" "$BASE/$f" <<'PY'
import sys,csv
a=list(csv.reader(open(sys.argv[1]))); b=list(csv.reader(open(sys.argv[2])))
mx=0.0
for ra,rb in zip(a[1:],b[1:]):
    for xa,xb in zip(ra,rb):
        try:
            fa,fb=float(xa),float(xb)
            if fb!=0: mx=max(mx,abs(fa-fb)/abs(fb))
        except ValueError: pass
print(f"    max rel dev = {mx:.3e}  rows {len(a)-1}/{len(b)-1}")
PY
  fi
done
echo "=== JITTER BASE-RATE (diagnostic dumps; measurement only, NOT verdict) ==="
for f in lumina_c1_bins.csv lumina_jbar_dump.csv lumina_levelpop_resolve_raw.csv \
         lumina_levelpop_resolve_ema.csv lumina_c2_bfr_dump.csv \
         cmf_fine_linedump_s8.csv cmf_fine_linedump_s45.csv cmf_fine_linedump_s49.csv; do
  [ -f "$f" ] && cp -f "$f" "$OUT/$f"
  if [ -f "$OUT/$f" ] && [ -f "$BASE/$f" ]; then
    n=$(cmp -l "$OUT/$f" "$BASE/$f" 2>/dev/null | wc -l)
    echo "  $f diff_bytes=$n"
  else
    echo "  $f MISSING"
  fi
done
mkdir -p "$R/logs/coevolve_consume_parity44a100probe" && rsync -a "$OUT/" "$R/logs/coevolve_consume_parity44a100probe/"
echo "DONE probe a100"
