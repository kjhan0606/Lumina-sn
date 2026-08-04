#!/bin/bash
#SBATCH --job-name=parity_smoke
#SBATCH --partition=h100,h200,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ARTIS-parity Group A structural smoke: NITER=1, tiny PKTS. GOAL = confirm
#   (1) the LUMINA_ARTIS_PARITY banner + the 3 ige_col (Fe II / Co III / Ni III)
#       loads fire and the Fe III Zhang table arms, and
#   (2) the parity NLTE assembler (A1 metastable full-connect, A2 vR+Bethe+Gaunt,
#       A3 generic col_ion pass, A4 coll-ioniz + 3-body recomb) runs one solve
#       without crash / NaN.
#   NOT a physics verdict. STAGE4 on so Fe/Co/Ni III sit as lower ions of a pair
#   (their generic col_data pass + A4 ionization channel then actually execute).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

BIN=./lumina_cuda.withParity
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; exit 2; }
echo "[smoke] $BIN md5=$(md5sum $BIN | cut -d' ' -f1)"

export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export LUMINA_ARTIS_PARITY=1
export LUMINA_NLTE_STAGE4=1

MODEL=data/tardis_reference_toy06_19p48d
OUT=logs/parity_smoke_${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
echo "[smoke] running: $BIN $MODEL 5000 1 spectrum nlte"
"$BIN" "$MODEL" 5000 1 spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log" || echo "  rc=$?"

echo "==================== PARITY LOAD / BANNER ===================="
grep -nE "ARTIS_PARITY = ON|Group A collisional|ion col_data Z=|Omega ARMED|generic ion tables loaded" "$OUT/stdout.log" || echo "  (banner/loads NOT found)"
echo "==================== SANITY (NaN / crash) ===================="
grep -niE "nan|inf|segfault|assert|error|fatal" "$OUT/stdout.log" "$OUT/stderr.log" | head -20 || echo "  (clean)"
echo "==================== TAIL ===================="
tail -15 "$OUT/stdout.log"
echo "[smoke] DONE -> $OUT"
