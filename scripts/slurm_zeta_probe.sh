#!/bin/bash
#SBATCH --job-name=zeta_probe
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# P4: ζ table override sensitivity probe.
# Carsus placeholder rows confirmed:
#   Z=22 Ti III ≡ Z=27 Co III ≡ Z=28 Ni III   (zeta_ion=2 row identical)
#   Z=21 Sc II  ≡ Z=27 Co II                  (zeta_ion=1 row identical)
# Targets: increase k=1 (II→III) ζ for Co/Ni → less III over-ionization
# (when ζ↑, phi_neb leans on ζ·(Te/Trad)·phi_e which is smaller than W·(1-ζ)·phi_r).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_zlock"

N_PKT=50000
N_ITER=8

ZMASK=${ZP_ZMASK:-27,28}
IONMASK=${ZP_IONMASK:-1}
VAL=${ZP_VAL:-0.8}

cell_dir="$ROOT/logs/zeta_probe_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== ζ-override sensitivity probe ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"
echo "ζ-override: ZMASK=$ZMASK  IONMASK=$IONMASK  VAL=$VAL"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_ZETA_OVERRIDE_ZMASK=$ZMASK \
LUMINA_ZETA_OVERRIDE_IONMASK=$IONMASK \
LUMINA_ZETA_OVERRIDE_VAL=$VAL \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- ζ-override banner ---"
grep -E '\[ζ-override\]|\[zeta' stdout.log | head -1
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -2
    echo "--- Co II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==27 && $3==0' | tail -2
    echo "--- Ni II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==28 && $3==0' | tail -2
fi
echo "Done: $(date)"
