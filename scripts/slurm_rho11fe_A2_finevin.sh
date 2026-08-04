#!/bin/bash
#SBATCH --job-name=rho11fe_A2
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-3
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# A2: A1 후속. v_in fine bracket {9550,9650,9700,9750} + X_Fe inner plateau cap.
# A1 결과: v_in=9500 → Si II trough 9665 km/s ✓ (HST diff -269) BUT RMS_bn=0.476 폭주.
# 원인: ρ-11fe X_Fe(v<11000)=0.10 → v_in이 9500까지 내려가면 inner shells가 Fe-plateau 영역
#       을 차지 → 3000-3300Å UV blanketing 폭주.
# Fix: X_Fe inner plateau 0.10 → 0.05 (Fe trace level), 다른 ρ-11fe profile 유지.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

VINS=(9550 9650 9700 9750)
vi=${VINS[$SLURM_ARRAY_TASK_ID]}
ro=0.00
de=9.0
label=$(printf "rho11feFe05_v%05d" "$vi")

work_root="$ROOT/logs/rho11feA2_${SLURM_ARRAY_JOB_ID}_${label}"
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
rm -f "$REF_DIR/abundances.csv"

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
    "r_inner": r_inner, "r_outer": r_outer,
    "v_inner": v_inner, "v_outer": v_outer,
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
    "W": W_init, "T_rad": T_rad_init,
}).to_csv(f"{dst}/plasma_state.csv", index=False)
m_H = 1.6726e-24
ne_init = (rho_new / m_H) / 14.0
pd.DataFrame({
    "shell_id": np.arange(n_shells, dtype=int), "n_e": ne_init,
}).to_csv(f"{dst}/electron_densities.csv", index=False)

# ρ-11fe composition WITH X_Fe inner plateau cap (0.10 -> 0.05)
def rho11fe_profile(v):
    if   v < 12500: X_Si = 0.55
    elif v < 16000: X_Si = 0.55 - 0.50*(v-12500)/3500
    else:           X_Si = 0.02
    if   v < 11000: X_O = 0.18
    elif v < 16000: X_O = 0.18 + 0.60*(v-11000)/5000
    else:           X_O = 0.78
    if   v < 12000: X_C = 0.02
    elif v < 17000: X_C = 0.02 + 0.11*(v-12000)/5000
    else:           X_C = 0.13
    X_S  = X_Si * 0.13
    X_Ca = X_Si * 0.045
    X_Mg = 0.04 if v < 16000 else 0.02
    # X_Fe inner plateau: 0.10 -> 0.05 (Fe-trace cap)
    if   v < 11000: X_Fe = 0.05
    elif v < 15000: X_Fe = 0.05 * np.exp(-(v-11000)/1500.0)
    else:           X_Fe = 1e-3
    if   v < 11000: X_Ni = 5e-3
    elif v < 14000: X_Ni = 5e-3 * np.exp(-(v-11000)/1000.0)
    else:           X_Ni = 1e-5
    if   v < 11000: X_Co = 3e-3
    elif v < 14000: X_Co = 3e-3 * np.exp(-(v-11000)/1000.0)
    else:           X_Co = 1e-6
    X_Al = 4e-3; X_Sc = 1e-5; X_Ti = 1e-4; X_V = 5e-5
    X_Cr = X_Fe * 0.05; X_Mn = X_Fe * 0.03
    return {6:X_C, 8:X_O, 12:X_Mg, 13:X_Al, 14:X_Si, 16:X_S,
            20:X_Ca, 21:X_Sc, 22:X_Ti, 23:X_V, 24:X_Cr, 25:X_Mn,
            26:X_Fe, 27:X_Co, 28:X_Ni}

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s in range(n_shells):
    prof = rho11fe_profile(v_mid_new[s])
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
for s in range(n_shells):
    X[:, s] /= X[:, s].sum()

cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)

print(f"  built: v_in={vi_kms} v_out={v_outer_max:.0f}  rho0={rho0:.3e}  T_in_init={T_inner:.0f}")
print(f"  ρ-11fe+Fe05 shell 0 (v={v_mid_new[0]:.0f}): X_Si={X[Z_LIST.index(14),0]:.3f} X_O={X[Z_LIST.index(8),0]:.3f} X_Fe={X[Z_LIST.index(26),0]:.3e}")
PYEOF

echo "=== ρ-11fe A2 [${label}]  v_in=${vi} km/s  +SkipSi  X_Fe_inner=0.05 ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_SKIP_Z=14 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -6
echo "Done: $(date)"
