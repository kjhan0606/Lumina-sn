#!/bin/bash
#SBATCH --job-name=phys_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Smoke test of physical 2011fe ref (t_exp=17.2d, L=1.3e43, v_in=10400 km/s).
# Same params as struct3d_155420 cells: 50K×8, NLTE_START=5, lumina_cuda_h100_cefix.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_cefix"

work_root="$ROOT/logs/phys_smoke_${SLURM_JOB_ID}"
mkdir -p "$work_root"
cd "$work_root"

# Per-run REF_DIR with fresh state files (stale plasma_state/n_e symlinks → wrong physics).
REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    bn=$(basename "$f")
    ln -sf "$f" "$REF_DIR/$bn"
done
rm -f "$REF_DIR/plasma_state.csv" "$REF_DIR/electron_densities.csv"

python3 <<PYEOF
import json, numpy as np, pandas as pd
ref="$REF_DIR"
with open(f"{ref}/config.json") as f: cfg=json.load(f)
T_inner=cfg["T_inner_K"]
geo=pd.read_csv(f"{ref}/geometry.csv")
dens=pd.read_csv(f"{ref}/density.csv")
r_in0=geo["r_inner"].iloc[0]
r_in=geo["r_inner"].values
W=0.5*(r_in0/r_in)**2
T_rad=np.full(len(geo), T_inner, dtype=float)
pd.DataFrame({"shell_id":np.arange(len(geo)),"W":W,"T_rad":T_rad}).to_csv(f"{ref}/plasma_state.csv", index=False)
m_H=1.6726e-24
ne=(dens["rho"].values/m_H)/14.0
pd.DataFrame({"shell_id":np.arange(len(geo)),"n_e":ne}).to_csv(f"{ref}/electron_densities.csv", index=False)
print(f"  fresh state: W[0]={W[0]:.3f} T_rad[0]={T_rad[0]:.0f}K n_e[0]={ne[0]:.3e}")
PYEOF

echo "=== physical ref smoke ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
cat "$REF/config.json"
echo ""

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
"$BIN" "$REF_DIR" 50000 8 spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -6
echo "Done: $(date)"
