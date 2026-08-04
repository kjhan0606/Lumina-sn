#!/bin/bash
#SBATCH --job-name=struct3d_phys
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-26
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# 3D structural sweep at PHYSICAL 2011fe ref (t_exp=17.2d, L=1.3e43, v_in=10400 km/s).
# Axes: rho_log_offset {-0.3, 0, +0.3}, density_exp {7,8,9}, v_inner_kms {8500, 10400, 12500}.
# Cells re-build geometry + density on the physical reference baseline.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

RHO_OFFSETS=(-0.30 0.00 0.30)
DEXPS=(7.0 8.0 9.0)
VINS=(8500 10400 12500)

idx=$SLURM_ARRAY_TASK_ID
i=$((idx / 9))
j=$(( (idx % 9) / 3 ))
k=$((idx % 3))

ro=${RHO_OFFSETS[$i]}
de=${DEXPS[$j]}
vi=${VINS[$k]}

label=$(printf "r%+05.2f_n%03.1f_v%05d" "$ro" "$de" "$vi" | tr '+' 'p' | tr '.' 'p' | tr '-' 'm')

work_root="$ROOT/logs/struct3d_phys_${SLURM_ARRAY_JOB_ID}_${label}"
mkdir -p "$work_root"
cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$SRC_REF"/*; do
    bn=$(basename "$f")
    ln -sf "$f" "$REF_DIR/$bn"
done
rm -f "$REF_DIR/config.json" "$REF_DIR/density.csv" "$REF_DIR/geometry.csv"
rm -f "$REF_DIR/plasma_state.csv" "$REF_DIR/electron_densities.csv"

python3 <<PYEOF
import json, numpy as np, pandas as pd
src = "$SRC_REF"
dst = "$REF_DIR"
ro, de, vi_kms = $ro, $de, $vi

with open(f"{src}/config.json") as f: cfg = json.load(f)
t_exp = cfg["time_explosion_s"]
n_shells = cfg["n_shells"]
L_inner = cfg["luminosity_inner_erg_s"]

geom_src = pd.read_csv(f"{src}/geometry.csv")
dens_src = pd.read_csv(f"{src}/density.csv")
v_outer_max = geom_src["v_outer"].iloc[-1] / 1e5

v_grid_kms = np.linspace(vi_kms, v_outer_max, n_shells + 1)
v_inner = v_grid_kms[:-1] * 1e5
v_outer = v_grid_kms[1:] * 1e5
r_inner = v_inner * t_exp
r_outer = v_outer * t_exp

# Calibrate rho_0 (at v_ref=11000 km/s) from physical-ref power-law fit
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

# Update v_inner in config; KEEP L_inner at physical 1.3e43 (no scale factor).
sigma_SB = 5.670374e-5
T_inner = (L_inner / (4*np.pi*r_inner[0]**2 * sigma_SB))**0.25
cfg["v_inner_min_cm_s"] = vi_kms * 1e5
cfg["v_outer_max_cm_s"] = v_outer_max * 1e5
cfg["T_inner_K"] = round(T_inner, -1)
with open(f"{dst}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# Fresh plasma_state.csv + electron_densities.csv (no stale TARDIS state).
W_init = 0.5 * (r_inner[0] / r_inner)**2
T_rad_init = np.full(n_shells, T_inner, dtype=float)
pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "W": W_init,
    "T_rad": T_rad_init,
}).to_csv(f"{dst}/plasma_state.csv", index=False)
m_H = 1.6726e-24
ne_init = (rho_new / m_H) / 14.0  # rough singly-ionized estimate
pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "n_e": ne_init,
}).to_csv(f"{dst}/electron_densities.csv", index=False)
print(f"  built: v_in={vi_kms} v_out={v_outer_max:.0f}  rho0={rho0:.3e}  exp={de}  L={L_inner:.2e}  T_in_init={T_inner:.0f}")
print(f"         init W[0]={W_init[0]:.3f} T_rad[0]={T_rad_init[0]:.0f} n_e[0]={ne_init[0]:.3e}")
PYEOF

echo "=== struct3d_phys [${label}]  rho_off=${ro}  dexp=${de}  v_in=${vi} km/s ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -6
echo "Done: $(date)"
