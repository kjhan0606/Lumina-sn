#!/bin/bash
#SBATCH --job-name=outerFe_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Path B smoke: Champion cell (rho_off=0, n=9.0, v_in=12500) at outerFe abundance ref.
# Outer Fe (shells 8-11: 1e-3, shells 12-15: 5e-4) replacing 1e-6 floor.
# Same physical IC base: t_exp=17.2d, L=1.3e43.
# 50K x 8 iter, NLTE_START=5, lumina_cuda_h100_cefix.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical_outerFe"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

# Champion cell
ro=0.00
de=9.0
vi=12500
label=$(printf "outerFe_r%+05.2f_n%03.1f_v%05d" "$ro" "$de" "$vi" | tr '+' 'p' | tr '.' 'p' | tr '-' 'm')

work_root="$ROOT/logs/outerFe_smoke_${SLURM_JOB_ID}_${label}"
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

sigma_SB = 5.670374e-5
T_inner = (L_inner / (4*np.pi*r_inner[0]**2 * sigma_SB))**0.25
cfg["v_inner_min_cm_s"] = vi_kms * 1e5
cfg["v_outer_max_cm_s"] = v_outer_max * 1e5
cfg["T_inner_K"] = round(T_inner, -1)
with open(f"{dst}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)

W_init = 0.5 * (r_inner[0] / r_inner)**2
T_rad_init = np.full(n_shells, T_inner, dtype=float)
pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "W": W_init,
    "T_rad": T_rad_init,
}).to_csv(f"{dst}/plasma_state.csv", index=False)
m_H = 1.6726e-24
ne_init = (rho_new / m_H) / 14.0
pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int),
    "n_e": ne_init,
}).to_csv(f"{dst}/electron_densities.csv", index=False)
print(f"  built: v_in={vi_kms} v_out={v_outer_max:.0f}  rho0={rho0:.3e}  exp={de}  L={L_inner:.2e}  T_in_init={T_inner:.0f}")

# Spot-check outer Fe in abundances
ab = pd.read_csv(f"{src}/abundances.csv")
fe = ab[ab["atomic_number"]==26].iloc[0]
ni = ab[ab["atomic_number"]==28].iloc[0]
print(f"  abundances (outerFe): Fe[8]={fe['8']:.2e} Fe[12]={fe['12']:.2e} Fe[20]={fe['20']:.2e}")
print(f"                        Ni[8]={ni['8']:.2e} Ni[12]={ni['12']:.2e} Ni[20]={ni['20']:.2e}")
PYEOF

echo "=== outerFe smoke [${label}]  rho_off=${ro}  dexp=${de}  v_in=${vi} km/s ==="
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
