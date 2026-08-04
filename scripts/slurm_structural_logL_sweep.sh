#!/bin/bash
#SBATCH --job-name=logL_sweep
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-4
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# (1) Structural sweep — log_L axis only.
# Baseline TARDIS reference has luminosity_inner_erg_s = 2.0e43.
# Scales: 0.50× (Δlog=-0.30), 0.71× (-0.15), 1.00× (0), 1.41× (+0.15), 2.00× (+0.30).
# Hypothesis: less L → less photoionization → less over-ionization → smaller W err.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

# Scale factors indexed by SLURM_ARRAY_TASK_ID
SCALES=(0.50 0.71 1.00 1.41 2.00)
LABELS=(m030 m015 base p015 p030)

idx=$SLURM_ARRAY_TASK_ID
scale=${SCALES[$idx]}
label=${LABELS[$idx]}

work_root="$ROOT/logs/logL_sweep_${SLURM_ARRAY_JOB_ID}_${label}"
mkdir -p "$work_root"
cd "$work_root"

# Build modified reference dir (symlink everything, override config.json)
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

echo "=== log_L sweep [${label}] scale=${scale} ==="
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
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -2
fi
echo "Done: $(date)"
