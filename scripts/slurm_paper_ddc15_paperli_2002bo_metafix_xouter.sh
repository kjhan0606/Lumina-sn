#!/bin/bash
#SBATCH --job-name=paperli_2002bo_xouter
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Paper comparison run for SN 2002bo Bmax against Blondin+2013 Fig 6.
# User directive 2026-05-28: do NOT use macro-atom in paper comparison;
# Blondin's CMFGEN is per-line resonance/cascade, so use LUMINA scatter
# mode (LINE_SCATTER = pure resonance per-line, no internal redistribution).
#
# LINE_INTERACTION knob added 2026-05-28 (lumina_main.c:93, lumina_cuda.cu:2188).
# Binary: lumina_cuda_paperli_{a100,h100} — built 2026-05-28 with Bug #7
# (Doppler μ order in BF-cascade re-emission) also patched.
#
# Configuration matches paperDDC15v3asEfloor_bbfix_2002bo:
#   t_rise = 17.76 d, L_bol = 1.22e43 erg/s, v_in = 9019 km/s
# Atomic data: tardis_reference_v3_femerge_capraise_bbfix_metafix_asE
# (Bug #2 metastable fix active — 650 metastable levels vs OLD 44, 15× more).
# Tests metastable-flag lever vs 158852 (scatter, OLD metastable).

LINE_INT=${LINE_INT:-scatter}   # scatter | downbranch | macroatom
VI_KMS=9019
L_SCALE_ARG=1.0
N_PKT=${N_PKT:-100000}
N_ITER=${N_ITER:-3}
SKIP_Z=${SKIP_Z-14}             # default Si II skip (#219e workaround). SKIP_Z="" enables Si NLTE.
XO_Z=${LUMINA_X_OUTER_SCALE_Z:-26}    # Z to scale (default Fe=26)
XO_FAC=${LUMINA_X_OUTER_SCALE_FAC:-1.0}  # multiplier (1.0 = no-op)
RUN_TAG=paperli_metafix_xouter_${LINE_INT}_Z${XO_Z}_f$(echo "$XO_FAC" | tr '.' 'p')
[ "$SKIP_Z" = "14" ] || RUN_TAG="${RUN_TAG}_skipZ${SKIP_Z:-none}_iter${N_ITER}"

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qi H100; then
    BIN="$ROOT/lumina_cuda_paperli_h100"
elif echo "$GPU_NAME" | grep -qi H200; then
    BIN="$ROOT/lumina_cuda_paperli_h100"
else
    BIN="$ROOT/lumina_cuda_paperli_a100"
fi

SRC_REF="$ROOT/data/tardis_reference_v3_femerge_capraise_bbfix_metafix_asE"
L_scale=$L_SCALE_ARG

t_exp_s=1534464              # 17.76 d
L_base=1.22e+43
L_inner=$(python3 -c "print(f'{$L_base * $L_scale:.6e}')")

vi=$VI_KMS
vi_tag=$(echo "$vi" | tr '.' 'p')
l_tag=$(echo "$L_scale" | tr '.' 'p')
label="paperDDC15_${RUN_TAG}_2002bo_vi${vi_tag}_L${l_tag}_prod"
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
vi_kms = $vi
t_exp  = $t_exp_s
L_inner = $L_inner
XO_Z  = int("$XO_Z")
XO_FAC = float("$XO_FAC")
V_OUTER_SCALE_KMS = 11200.0   # apply scale for v_mid >= this
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
        X_init_Ni  = 0.55
        X_stableFe = 0.10
        X_Si = 0.03; X_S = 0.008; X_Ca = 0.005
        X_Mg = 0.005; X_Al = 0.001
        X_O  = 1e-6; X_C  = 1e-6
        X_Ti = 1e-4; X_V = 5e-5; X_Sc = 1e-5
    elif v < V_NI_TOP:
        X_init_Ni  = 0.625
        X_stableFe = 0.014
        X_Si = 0.214; X_S = 0.060; X_Ca = 0.0560
        X_Mg = 0.020; X_Al = 0.004
        X_O  = 2e-6; X_C  = 1e-6
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
        X_init_Ni  = 0.0
        X_stableFe = 1e-3
        X_Si = 0.020; X_S = 1e-3; X_Ca = 1e-4
        X_Mg = 0.005; X_Al = 5e-4
        X_O  = 0.65; X_C  = 0.20
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
X_init_Ni_arr = np.zeros(n_shells)
for s in range(n_shells):
    prof, xini = ddc15_profile_v3(v_mid_new[s])
    X_init_Ni_arr[s] = xini
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]

# Outer-X ablation: scale Z=XO_Z by XO_FAC in shells with v_mid >= V_OUTER_SCALE_KMS
if XO_FAC != 1.0 and XO_Z in Z_LIST:
    zi = Z_LIST.index(XO_Z)
    mask = v_mid_new >= V_OUTER_SCALE_KMS
    n_scaled = int(mask.sum())
    X[zi, mask] *= XO_FAC
    print(f"  [XOUTER] Z={XO_Z} scaled by {XO_FAC} in {n_scaled} outer shells (v>={V_OUTER_SCALE_KMS} km/s)")

for s in range(n_shells):
    tot = X[:, s].sum()
    if tot > 0: X[:, s] /= tot

cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)
print(f"  t_exp={t_exp:.0f}s ({t_exp/86400:.2f}d)  L={L_inner:.3e}  T_inner_init={T_inner:.1f}K")
PYEOF

echo "=== Paper comparison (NO macro-atom)  LINE_INTERACTION=$LINE_INT  VI=$VI_KMS L=$L_scale  N_PKT=$N_PKT N_ITER=$N_ITER ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Time: $(date)"

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$SRC_REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    ${SKIP_Z:+LUMINA_NLTE_SKIP_Z=$SKIP_Z} \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_RATE_DUMP=1 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"

echo ""
echo "--- line interaction mode ---"
grep -E "Line interaction:" stdout.log | head -3

echo ""
echo "--- convergence trailer ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -10

echo ""
echo "--- spectrum file ---"
ls -la lumina_spectrum_formal.csv 2>&1

echo "Done: $(date)"
