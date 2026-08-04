#!/bin/bash
#SBATCH --job-name=rho11fe_A3
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-3
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# A3: A1/A2 후속. X_Si plateau를 narrow IME shell로 좁힘 — Mazzali ρ-11fe 진짜 geometry.
# A1/A2 진단: X_Si=0.55 plateau v<12500 광범위 → Si II opacity peak v=11000-12500 잠금
#             → v_in 변경에도 trough 위치 saturation lock.
# Fix: X_Si=0.55 in [v_in, v_in+2500] (narrow 2500 km/s IME shell), outside decay.
#      Si II opacity peak가 v_in 근처에 모임 → trough position이 v_in을 따라감.
# Inner Fe cap (0.05) carry over from A2.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

VINS=(9800 10100 10400 10800)
vi=${VINS[$SLURM_ARRAY_TASK_ID]}
ro=0.00
de=9.0
label=$(printf "rho11feNarrowSi_v%05d" "$vi")

work_root="$ROOT/logs/rho11feA3_${SLURM_ARRAY_JOB_ID}_${label}"
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

# Narrow IME shell + Fe05 cap
def rho11fe_narrow(v, v_in):
    # X_Si: narrow IME shell [v_in, v_in+2500]
    v_si_hi = v_in + 2500
    v_si_decay_hi = v_si_hi + 3500
    if   v < v_si_hi:        X_Si = 0.55
    elif v < v_si_decay_hi:  X_Si = 0.55 - 0.50*(v-v_si_hi)/3500
    else:                    X_Si = 0.02
    # X_O ramp tied to v_in (Mazzali: O picks up just above IME shell)
    v_o_lo = v_si_hi      # O rises just above Si shell
    v_o_hi = v_o_lo + 5000
    if   v < v_o_lo: X_O = 0.18
    elif v < v_o_hi: X_O = 0.18 + 0.60*(v-v_o_lo)/5000
    else:            X_O = 0.78
    # X_C trace
    v_c_lo = v_si_hi
    v_c_hi = v_c_lo + 5000
    if   v < v_c_lo: X_C = 0.02
    elif v < v_c_hi: X_C = 0.02 + 0.11*(v-v_c_lo)/5000
    else:            X_C = 0.13
    X_S  = X_Si * 0.13
    X_Ca = X_Si * 0.045
    X_Mg = 0.04 if v < v_si_decay_hi else 0.02
    # Fe-group: inner cap 0.05, decay above v_in+1000 (Fe is decay product)
    v_fe_decay = v_in + 1000
    v_fe_hi = v_fe_decay + 4000
    if   v < v_fe_decay: X_Fe = 0.05
    elif v < v_fe_hi:    X_Fe = 0.05 * np.exp(-(v-v_fe_decay)/1500.0)
    else:                X_Fe = 1e-3
    if   v < v_fe_decay: X_Ni = 5e-3
    elif v < v_fe_hi:    X_Ni = 5e-3 * np.exp(-(v-v_fe_decay)/1000.0)
    else:                X_Ni = 1e-5
    if   v < v_fe_decay: X_Co = 3e-3
    elif v < v_fe_hi:    X_Co = 3e-3 * np.exp(-(v-v_fe_decay)/1000.0)
    else:                X_Co = 1e-6
    X_Al = 4e-3; X_Sc = 1e-5; X_Ti = 1e-4; X_V = 5e-5
    X_Cr = X_Fe * 0.05; X_Mn = X_Fe * 0.03
    return {6:X_C, 8:X_O, 12:X_Mg, 13:X_Al, 14:X_Si, 16:X_S,
            20:X_Ca, 21:X_Sc, 22:X_Ti, 23:X_V, 24:X_Cr, 25:X_Mn,
            26:X_Fe, 27:X_Co, 28:X_Ni}

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s in range(n_shells):
    prof = rho11fe_narrow(v_mid_new[s], vi_kms)
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
for s in range(n_shells):
    X[:, s] /= X[:, s].sum()

cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)

print(f"  built: v_in={vi_kms} v_out={v_outer_max:.0f}  rho0={rho0:.3e}  T_in_init={T_inner:.0f}")
print(f"  Si IME shell: [{vi_kms}, {vi_kms+2500}] km/s")
print(f"  sh   v_mid    X_Si    X_O    X_Fe")
for s in [0, 2, 4, 6, 10, 15, 25]:
    print(f"  {s:>2}  {v_mid_new[s]:>6.0f}  {X[Z_LIST.index(14),s]:.3f}  {X[Z_LIST.index(8),s]:.3f}  {X[Z_LIST.index(26),s]:.3e}")
PYEOF

echo "=== ρ-11fe A3 [${label}]  v_in=${vi} km/s  +SkipSi  narrow IME ==="
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
