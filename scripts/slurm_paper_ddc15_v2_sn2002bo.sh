#!/bin/bash
#SBATCH --job-name=paper_ddc15_v2_2002bo
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# #284 Paper-faithful DDC15 vs SN 2002bo at Bmax.
# v1 (157708/157786) used synthetic ddc15_profile (X_Si=0.40 at v=9019, vs paper 0.214 = 87% over).
# v2 retunes composition to match Blondin Table 2 photospheric X exactly at v_τ=9019:
#   X(Si)=0.214, X(Ca)=0.0560, X(Fe)=0.0927, X(O)≈2e-6
# AND includes 56Ni → 56Co → 56Fe decay chain at t=17.76d (split 13.3% / 74.0% / 12.7%).
# X(56Co)≈0.46 at photosphere → dramatically more Co II/III bb opacity → should crush UV.
# Density profile (DE=10.5) STILL from 2011fe tuning; that's stage 2 if v2 isn't enough.

VI_KMS=9019
L_SCALE_ARG=1.0
DE_ARG=10.5

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qi H100; then
    BIN="$ROOT/lumina_cuda_oTri_h100"
elif echo "$GPU_NAME" | grep -qi H200; then
    BIN="$ROOT/lumina_cuda_oTri_h100"
else
    BIN="$ROOT/lumina_cuda_oTri_a100"
fi

# Smoke vs production toggle.
if [ "${1:-prod}" = "smoke" ]; then
    N_PKT=200000
    N_ITER=3
    RUN_TAG=smoke
else
    N_PKT=800000
    N_ITER=12
    RUN_TAG=prod
fi

SRC_REF="$ROOT/data/tardis_reference_v3_femerge_capraise"
L_scale=$L_SCALE_ARG

t_exp_s=1534464              # 17.76 d (Blondin Table 2 t_rise)
L_base=1.22e+43              # L_bol from Blondin Table 2

L_inner=$(python3 -c "print(f'{$L_base * $L_scale:.6e}')")

vi=$VI_KMS
ro=0.00
de=$DE_ARG

vi_tag=$(echo "$vi" | tr '.' 'p')
l_tag=$(echo "$L_scale" | tr '.' 'p')
de_tag=$(echo "$de" | tr '.' 'p')
label="paperDDC15v2_2002bo_vi${vi_tag}_L${l_tag}_de${de_tag}_${RUN_TAG}"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"
cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$SRC_REF"/*; do
    bn=$(basename "$f")
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$bn"
done
rm -f "$REF_DIR/config.json" "$REF_DIR/density.csv" "$REF_DIR/geometry.csv"
rm -f "$REF_DIR/plasma_state.csv" "$REF_DIR/electron_densities.csv"
rm -f "$REF_DIR/abundances.csv"

python3 <<PYEOF
import json, numpy as np, pandas as pd
src = "$SRC_REF"; dst = "$REF_DIR"
ro, de, vi_kms = $ro, $de, $vi
t_exp = $t_exp_s; L_inner = $L_inner

with open(f"{src}/config.json") as f: cfg = json.load(f)
n_shells = cfg["n_shells"]
cfg["time_explosion_s"] = float(t_exp)
cfg["luminosity_inner_erg_s"] = float(L_inner)

geom_src = pd.read_csv(f"{src}/geometry.csv")
dens_src = pd.read_csv(f"{src}/density.csv")
v_outer_max = geom_src["v_outer"].iloc[-1] / 1e5

v_grid_kms = np.linspace(vi_kms, v_outer_max, n_shells + 1)
v_inner = v_grid_kms[:-1] * 1e5; v_outer = v_grid_kms[1:] * 1e5
r_inner = v_inner * t_exp; r_outer = v_outer * t_exp

v_ref_kms = 11000
v_mid_src = (geom_src["v_inner"].values + geom_src["v_outer"].values) / 2 / 1e5
slope, intercept = np.polyfit(np.log(v_mid_src), np.log(dens_src["rho"].values), 1)
rho0_base = np.exp(intercept + slope * np.log(v_ref_kms))
with open(f"{src}/config.json") as f: cfg_src = json.load(f)
t_exp_base = cfg_src["time_explosion_s"]
rho0 = rho0_base * 10**ro * (t_exp_base / t_exp)**3
v_mid_new = (v_inner + v_outer) / 2 / 1e5
rho_new = rho0 * (v_mid_new / v_ref_kms)**(-de)

pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "r_inner": r_inner, "r_outer": r_outer,
              "v_inner": v_inner, "v_outer": v_outer}).to_csv(f"{dst}/geometry.csv", index=False)
pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "rho": rho_new}).to_csv(f"{dst}/density.csv", index=False)

sigma_SB = 5.670374e-5
T_inner = (L_inner / (4*np.pi*r_inner[0]**2 * sigma_SB))**0.25
cfg["v_inner_min_cm_s"] = vi_kms * 1e5
cfg["v_outer_max_cm_s"] = v_outer_max * 1e5
cfg["T_inner_K"] = round(T_inner, -1)
with open(f"{dst}/config.json", "w") as f: json.dump(cfg, f, indent=2)

W_init = 0.5 * (r_inner[0] / r_inner)**2
T_rad_init = np.full(n_shells, T_inner, dtype=float)
pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "W": W_init, "T_rad": T_rad_init}).to_csv(f"{dst}/plasma_state.csv", index=False)
m_H = 1.6726e-24
ne_init = (rho_new / m_H) / 14.0
pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "n_e": ne_init}).to_csv(f"{dst}/electron_densities.csv", index=False)

# === Paper-faithful DDC15 composition (Blondin+ 2013 Tables 1,2) ===
# At v_τ=9019: X(Si)=0.214, X(Ca)=0.0560, X(Fe)=0.0927, X(O)=2.05e-6 (paper)
# 56Ni→56Co→56Fe decay at t_exp=17.76d: split 13.3% / 74.0% / 12.7%
# Sum-to-1 enforced via X(56Co) absorbing residual (large Co II/III bb opacity boost).
def ddc15_profile_v2(v):
    V_CORE_TOP = 7000.
    V_NI_TOP   = 11200.   # paper Table 1: v(56Ni) shell upper bound
    V_OUT_BOT  = 13500.

    if v < V_CORE_TOP:
        # Inner core: deflagration-processed, high init 56Ni
        X_init_Ni = 0.55
        X_stableFe = 0.10
        X_Si = 0.03; X_S = 0.008; X_Ca = 0.005
        X_Mg = 0.005; X_Al = 0.001
        X_O  = 1e-6; X_C  = 1e-6
        X_Ti = 1e-4; X_V = 5e-5; X_Sc = 1e-5
    elif v < V_NI_TOP:
        # IME + 56Ni shell — photospheric layer paper-pinned at v=9019
        X_init_Ni  = 0.625    # tuned so sum=1 at v=9019 after Co/Ni placement
        X_stableFe = 0.014    # so total X(Fe)= 0.014 + 0.127*0.625 = 0.0937 ≈ paper 0.0927
        X_Si = 0.214          # paper
        X_S  = 0.060          # IME-typical S/Si ≈ 0.28
        X_Ca = 0.0560         # paper
        X_Mg = 0.020
        X_Al = 0.004
        X_O  = 2e-6           # paper (essentially zero)
        X_C  = 1e-6
        X_Ti = 1e-4; X_V = 5e-5; X_Sc = 1e-5
    elif v < V_OUT_BOT:
        # Transition IME → unburned outer
        t = (v - V_NI_TOP) / (V_OUT_BOT - V_NI_TOP)
        X_init_Ni  = 0.625 * (1 - t)**2     # 56Ni shell falls off above 11200
        X_stableFe = 0.014 * (1 - t) + 0.001 * t
        X_Si = 0.214 * (1 - t) + 0.020 * t
        X_S  = 0.060 * (1 - t) + 1e-3 * t
        X_Ca = 0.0560 * (1 - t) + 5e-4 * t
        X_Mg = 0.020 * (1 - t) + 0.005 * t
        X_Al = 0.004 * (1 - t) + 5e-4 * t
        X_O  = 2e-6 + (0.30 - 2e-6) * t
        X_C  = 1e-6 + (0.10 - 1e-6) * t
        X_Ti = 1e-4 * (1 - t); X_V = 5e-5 * (1 - t); X_Sc = 1e-5
    else:
        # Outer unburned C/O
        X_init_Ni  = 0.0
        X_stableFe = 1e-3
        X_Si = 0.020; X_S = 1e-3; X_Ca = 1e-4
        X_Mg = 0.005; X_Al = 5e-4
        X_O  = 0.65
        X_C  = 0.20
        X_Ti = 1e-5; X_V = 1e-5; X_Sc = 1e-6

    # Apply decay chain at t=17.76d
    F_Ni_REM = 0.1334    # 2^(-17.76/6.10)
    F_CO_NOW = (1 - F_Ni_REM) * 0.8534  # 2^(-17.76/77.27)
    F_FE_NOW = (1 - F_Ni_REM) * (1 - 0.8534)

    X_56Ni = F_Ni_REM * X_init_Ni
    X_56Co = F_CO_NOW * X_init_Ni
    X_56Fe = F_FE_NOW * X_init_Ni

    X_Fe = X_stableFe + X_56Fe
    X_Co = X_56Co + 1e-6
    X_Ni = X_56Ni + 1e-6
    X_Cr = X_Fe * 0.05
    X_Mn = X_Fe * 0.03

    return {6:X_C, 8:X_O, 12:X_Mg, 13:X_Al, 14:X_Si, 16:X_S,
            20:X_Ca, 21:X_Sc, 22:X_Ti, 23:X_V, 24:X_Cr, 25:X_Mn,
            26:X_Fe, 27:X_Co, 28:X_Ni}

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s in range(n_shells):
    prof = ddc15_profile_v2(v_mid_new[s])
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
# Sanity print BEFORE renorm — should show paper X at the shell straddling v=9019
ix_phot = int(np.argmin(np.abs(v_mid_new - 9019)))
print(f"  shell {ix_phot} v={v_mid_new[ix_phot]:.0f} km/s pre-renorm sum={X[:,ix_phot].sum():.4f}")
print(f"    X_Si={X[Z_LIST.index(14),ix_phot]:.4f}  X_Ca={X[Z_LIST.index(20),ix_phot]:.4f}")
print(f"    X_Fe={X[Z_LIST.index(26),ix_phot]:.4f}  X_Co={X[Z_LIST.index(27),ix_phot]:.4f}  X_Ni={X[Z_LIST.index(28),ix_phot]:.4f}")
print(f"    X_O={X[Z_LIST.index(8),ix_phot]:.2e}")
for s in range(n_shells):
    tot = X[:, s].sum()
    if tot > 0: X[:, s] /= tot
print(f"  post-renorm at shell {ix_phot}: X_Si={X[Z_LIST.index(14),ix_phot]:.4f}  X_Fe={X[Z_LIST.index(26),ix_phot]:.4f}  X_Co={X[Z_LIST.index(27),ix_phot]:.4f}")
cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)
print(f"  t_exp={t_exp:.0f}s ({t_exp/86400:.2f}d)  L={L_inner:.3e}  T_inner={T_inner:.1f}K  v_i={vi_kms}km/s  rho0_v11k={rho0:.3e}")
PYEOF

echo "=== PAPER-DDC15-v2 vs SN 2002bo  VI=$VI_KMS L=$L_scale DE=$DE_ARG  RUN=$RUN_TAG [${label}] ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Time: $(date)"
echo "Paper photospheric X (Table 2) ENFORCED at v=9019; 56Ni decay chain at t=17.76d."

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$SRC_REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"

echo ""
echo "--- composition sanity (from stdout) ---"
grep -E "shell .* v=|X_Si=|X_Co=|t_exp=" stdout.log | head -10

echo ""
echo "--- NLTE init ---"
grep -E "Total NLTE levels|Z=8 ion=|^  \[NLTE\]" stdout.log | head -10

echo ""
echo "--- spectrum/convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted|Snifs|RMS_bn' stdout.log | tail -10

echo ""
echo "--- error trailer (last 20 lines stderr) ---"
tail -20 stderr.log

ls -l lumina_spectrum_formal.csv 2>&1 | head -3
echo "Done: $(date)"
