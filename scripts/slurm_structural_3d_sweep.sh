#!/bin/bash
#SBATCH --job-name=struct3d
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-26
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# Structural 3D sweep at fixed log_L = 0.40× (RMS min from log_L sweep 155368/155379).
# Axes: log_rho_0 offset {-0.3, 0, +0.3}, density_exp {7,8,9}, v_inner_kms {9500,11500,13500}
# 3^3 = 27 cells. Goal: find structural minimum of RMS = mean(|I_mod - I_HST|/I_HST).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_cefix"

L_SCALE=0.40
N_PKT=50000
N_ITER=8

RHO_OFFSETS=(-0.30 0.00 0.30)
DEXPS=(7.0 8.0 9.0)
VINS=(9500 11500 13500)

idx=$SLURM_ARRAY_TASK_ID
i=$((idx / 9))
j=$(( (idx % 9) / 3 ))
k=$((idx % 3))

ro=${RHO_OFFSETS[$i]}
de=${DEXPS[$j]}
vi=${VINS[$k]}

# label like r-030_n70_v09500  (rho-offset _ density-exp _ v_inner)
label=$(printf "r%+05.2f_n%03.1f_v%05d" "$ro" "$de" "$vi" | tr '+' 'p' | tr '.' 'p' | tr '-' 'm')

work_root="$ROOT/logs/struct3d_${SLURM_ARRAY_JOB_ID}_${label}"
mkdir -p "$work_root"
cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$SRC_REF"/*; do
    bn=$(basename "$f")
    ln -sf "$f" "$REF_DIR/$bn"
done
rm -f "$REF_DIR/config.json" "$REF_DIR/density.csv" "$REF_DIR/geometry.csv"

python3 <<PYEOF
import json, numpy as np, pandas as pd
src = "$SRC_REF"
dst = "$REF_DIR"
ro, de, vi_kms, l_scale = $ro, $de, $vi, $L_SCALE

with open(f"{src}/config.json") as f: cfg = json.load(f)
t_exp = cfg["time_explosion_s"]
n_shells = cfg["n_shells"]

geom_src = pd.read_csv(f"{src}/geometry.csv")
dens_src = pd.read_csv(f"{src}/density.csv")
# Use actual geometry.csv outer (28750 km/s); config.json v_outer_max_cm_s is stale.
v_outer_max = geom_src["v_outer"].iloc[-1] / 1e5

v_grid_kms = np.linspace(vi_kms, v_outer_max, n_shells + 1)
v_inner = v_grid_kms[:-1] * 1e5
v_outer = v_grid_kms[1:] * 1e5
r_inner = v_inner * t_exp
r_outer = v_outer * t_exp

# Calibrate rho_0 (at v_ref=11000 km/s) from reference power-law fit
v_ref_kms = 11000
v_mid_src = (geom_src["v_inner"].values + geom_src["v_outer"].values) / 2 / 1e5
slope, intercept = np.polyfit(np.log(v_mid_src), np.log(dens_src["rho"].values), 1)
rho0_base = np.exp(intercept + slope * np.log(v_ref_kms))
rho0 = rho0_base * 10**ro

v_mid_new = (v_inner + v_outer) / 2 / 1e5
rho_new = rho0 * (v_mid_new / v_ref_kms)**(-de)

pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "r_inner": r_inner,
    "r_outer": r_outer,
    "v_inner": v_inner,
    "v_outer": v_outer,
}).to_csv(f"{dst}/geometry.csv", index=False)

pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "rho": rho_new,
}).to_csv(f"{dst}/density.csv", index=False)

cfg["luminosity_inner_erg_s"] = cfg["luminosity_inner_erg_s"] * l_scale
cfg["v_inner_min_cm_s"] = vi_kms * 1e5
cfg["v_outer_max_cm_s"] = v_outer_max * 1e5
with open(f"{dst}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  built: v_in={vi_kms} v_out={v_outer_max:.0f}  rho0={rho0:.3e}  exp={de}  L_scale={l_scale}")
PYEOF

echo "=== struct3d [${label}]  rho_off=${ro}  dexp=${de}  v_in=${vi} km/s  L=${L_SCALE}x ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

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
echo "Done: $(date)"
