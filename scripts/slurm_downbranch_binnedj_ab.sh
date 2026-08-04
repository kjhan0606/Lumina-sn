#!/bin/bash
#SBATCH --job-name=dbj
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Task #27 downbranch (fluorescence) + binned-J estimator A/B. THE decisive test.
# Phase-2c showed paper-faithful fluorescence (downbranch) catastrophically over-
# redshifts under the MOMENT estimator: T_rad 5786->2297K, W 0.48->23, blue
# [3500-4500] x0.04, NIR x3 -- a Lucy (W,T_rad) estimator COLLAPSE, not a
# transport bug. The binned-J A/B (162151/162152) then proved the dilute-Planck
# SHAPE FIT to the J_nu histogram keeps W physical (shell5 W 1.10->0.77) and
# T_rad up (no red-tail drag) in scatter mode. This run combines the two: turn
# binned-J ON for BOTH arms and flip line interaction, to test whether a robust
# estimator makes FLUORESCENCE usable (the collapse was the blocker).
#   A (ARM=A): scatter    + binned-J   (scatter baseline w/ new estimator)
#   B (ARM=B): downbranch + binned-J   (paper-faithful fluorescence, fixed est.)
# BOTH: binned-J ON, diffusive BC ON, self-pin ON, DDC15-strat, 200k, N_ITER=10,
# DYNAMIC_TRANSPROB=0 (clean static transprobs, dodge #257). Single variable =
# LINE_INTERACTION. Pre-registered: if the estimator was the blocker, B's T_rad
# does NOT collapse to ~2300K and blue[3500-4500] does NOT vanish (vs Phase-2c
# downbranch x0.04); fluorescence then redistributes blue->red GENTLY (blue
# deficit rises toward 1, green/red excess eases). If B still collapses, the
# estimator alone is insufficient -> direct-J_nu in tau_sobolev/phi_neb next.
#   sbatch --export=ALL,ARM=A scripts/slurm_downbranch_binnedj_ab.sh
#   sbatch --export=ALL,ARM=B scripts/slurm_downbranch_binnedj_ab.sh

ARM=${ARM:-A}
DIFFUSE_BC=1                      # both arms build on the Phase-2a fix
BINNEDJ=1                         # both arms use the binned-J fit estimator
if [ "$ARM" = "B" ]; then
    LINE_INT=downbranch          # paper-faithful fluorescence
else
    LINE_INT=scatter             # baseline
fi

VI_KMS=9019
L_SCALE_ARG=1.0
N_PKT=200000
N_ITER=10
RUN_TAG="dbj_${ARM}"

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qiE "H100|H200"; then
    BIN="$ROOT/lumina_cuda_binnedj_h100"
else
    BIN="$ROOT/lumina_cuda_binnedj_a40"   # sm_80+sm_86+PTX: a100(native) / a40 / a10
fi

SRC_REF="$ROOT/data/${REF_NAME:-tardis_reference_cmfgen_superlev_ionfix_ddc15strat}"
L_scale=$L_SCALE_ARG

t_exp_s=1534464              # 17.76 d
L_base=1.22e+43
L_inner=$(python3 -c "print(f'{$L_base * $L_scale:.6e}')")

vi=$VI_KMS
vi_tag=$(echo "$vi" | tr '.' 'p')
l_tag=$(echo "$L_scale" | tr '.' 'p')
label="paperDDC15einsteinFix_2002bo_vi${vi_tag}_L${l_tag}_${RUN_TAG}"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

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
vi_kms = $vi
t_exp  = $t_exp_s
L_inner = $L_inner
M_SUN = 1.989e33

with open(f"{src}/config.json") as f: cfg = json.load(f)
n_shells = cfg["n_shells"]
cfg["time_explosion_s"] = float(t_exp)
cfg["luminosity_inner_erg_s"] = float(L_inner)

geom_src = pd.read_csv(f"{src}/geometry.csv")
v_outer_max = geom_src["v_outer"].iloc[-1] / 1e5

v_grid_kms = np.linspace(vi_kms, v_outer_max, n_shells + 1)
v_inner = v_grid_kms[:-1] * 1e5
v_outer = v_grid_kms[1:] * 1e5
r_inner = v_inner * t_exp
r_outer = v_outer * t_exp
v_mid_new = (v_inner + v_outer) / 2 / 1e5

V_BREAK_KMS = 11000.0
DE_IN  = 4.0
DE_OUT = 13.0
M_TARGET = 0.614 * M_SUN

rho_rel = np.where(v_mid_new < V_BREAK_KMS,
                   (v_mid_new / V_BREAK_KMS)**(-DE_IN),
                   (v_mid_new / V_BREAK_KMS)**(-DE_OUT))
dV = (4.0/3.0) * np.pi * (r_outer**3 - r_inner**3)
M_unnorm = float((rho_rel * dV).sum())
rho_break = M_TARGET / M_unnorm
rho_new = rho_rel * rho_break

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

def ddc15_profile_v3(v):
    V_CORE_TOP = 7000.
    V_NI_TOP   = 11200.
    V_OUT_BOT  = 13500.
    if v < V_CORE_TOP:
        X_init_Ni  = 0.55; X_stableFe = 0.10
        X_Si = 0.03; X_S = 0.008; X_Ca = 0.005
        X_Mg = 0.005; X_Al = 0.001; X_O  = 1e-6; X_C  = 1e-6
        X_Ti = 1e-4; X_V = 5e-5; X_Sc = 1e-5
    elif v < V_NI_TOP:
        X_init_Ni  = 0.625; X_stableFe = 0.014
        X_Si = 0.214; X_S = 0.060; X_Ca = 0.0560
        X_Mg = 0.020; X_Al = 0.004; X_O  = 2e-6; X_C  = 1e-6
        X_Ti = 1e-4; X_V = 5e-5; X_Sc = 1e-5
    elif v < V_OUT_BOT:
        t = (v - V_NI_TOP) / (V_OUT_BOT - V_NI_TOP)
        X_init_Ni  = 0.625 * (1 - t)**2
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
        X_init_Ni  = 0.0; X_stableFe = 1e-3
        X_Si = 0.020; X_S = 1e-3; X_Ca = 1e-4
        X_Mg = 0.005; X_Al = 5e-4; X_O  = 0.65; X_C  = 0.20
        X_Ti = 1e-5; X_V = 1e-5; X_Sc = 1e-6
    F_Ni_REM = 0.1334
    F_CO_NOW = (1 - F_Ni_REM) * 0.8534
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
            26:X_Fe, 27:X_Co, 28:X_Ni}, X_init_Ni

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s in range(n_shells):
    prof, xini = ddc15_profile_v3(v_mid_new[s])
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
for s in range(n_shells):
    tot = X[:, s].sum()
    if tot > 0: X[:, s] /= tot
cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)
print(f"  t_exp={t_exp:.0f}s ({t_exp/86400:.2f}d)  L={L_inner:.3e}  T_inner_init={T_inner:.1f}K")
PYEOF

echo "=== Task #27 downbranch+binned-J A/B  ARM=$ARM  BINNED_J=$BINNEDJ  LINE_INT=$LINE_INT  DIFFUSE_BC=$DIFFUSE_BC  VI=$VI_KMS L=$L_SCALE_ARG  N_PKT=$N_PKT N_ITER=$N_ITER ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Time: $(date)"
echo "Both arms: binned-J ON, diffusive BC ON, self-pin ON. A=scatter, B=downbranch fluorescence. Decisive test: does the robust estimator stop the Phase-2c downbranch T_rad collapse? Static clean transprobs (DYNAMIC_TRANSPROB=0)."

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$SRC_REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=0 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    LUMINA_BINNED_J_ESTIMATOR=$BINNEDJ \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_DIFFUSE_INNER_BC=$DIFFUSE_BC \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- binned-J estimator activation ---"
grep -E "\[binned-J\]" stdout.log
echo ""
echo "--- energy budget per iteration (escaped / reabsorbed / truncated of L_req) ---"
grep -E "E-BUDGET" stdout.log
echo ""
echo "--- T_inner trajectory (self-pin) ---"
grep -E 'T_inner:|T_inner final' stdout.log | tail -14
echo ""
echo "--- convergence trailer (W err / T_rad err should NOT blow up) ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -6
echo "Done: $(date)"
