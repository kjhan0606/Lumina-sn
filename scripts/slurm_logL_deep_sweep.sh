#!/bin/bash
#SBATCH --job-name=logL_deep
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-4
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# (1) Deep log_L sweep — extend below 0.50× to find knee.
# Initial sweep 155368: monotone 309% (0.50×) → 783% (1.41×).
# Hypothesis: minimum is below 0.50×. Probe 0.25, 0.32, 0.40, 0.50, 0.63 ×.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

SCALES=(0.25 0.32 0.40 0.50 0.63)
LABELS=(d060 d050 d040 d030 d020)

idx=$SLURM_ARRAY_TASK_ID
scale=${SCALES[$idx]}
label=${LABELS[$idx]}

work_root="$ROOT/logs/logL_deep_${SLURM_ARRAY_JOB_ID}_${label}"
mkdir -p "$work_root"
cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$SRC_REF"/*; do
    bn=$(basename "$f")
    ln -sf "$f" "$REF_DIR/$bn"
done
rm -f "$REF_DIR/config.json"
python3 -c "
import json
with open('$SRC_REF/config.json') as f: c = json.load(f)
c['luminosity_inner_erg_s'] = c['luminosity_inner_erg_s'] * $scale
with open('$REF_DIR/config.json','w') as f: json.dump(c, f, indent=2)
"

echo "=== log_L deep sweep [${label}] scale=${scale} (log=$(python3 -c "import math; print(f'{math.log10(${scale}):+.3f}')")) ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "REF_DIR:   $REF_DIR (luminosity scaled by ${scale})"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -1
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==27 && $3==0' | tail -1
fi
echo "Done: $(date)"
