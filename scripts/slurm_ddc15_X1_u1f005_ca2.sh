#!/bin/bash
#SBATCH --job-name=ddc15_X1_u1f005ca2
#SBATCH --partition=h200,h100,a100,a100_pcie
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-3
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# X1: stack U1_f0p05 (Ni II opt champion) + W1 Ca II boost (joint champ).
#
# W1 (job 156520) results on U1_f0p1 base:
#   ctrl 0.2002/J=0.265  →  b1p5 0.1919/0.254  →  b2p0 0.1874/0.246  →  b3p0 0.1874/0.254
# U1 champion: f0p05 RMS_bn=0.1833 (best), J=0.252.
# Hypothesis: Ni II opt (Z=28, [4000,7500]) and Ca II (Z=20, [3700,9000]) are
# orthogonal in Z and partially overlapping in λ → can stack.
#
# Array cells: SCALE7 Ni II opt f=0.05 (U1 champion), SCALE8 Ca II f∈{1.0,1.5,2.0,3.0}
#   0 — ctrl  : Ca II OFF (= U1_f0p05 reproducibility @ 0.1833)
#   1 — b1p5  : Ca II f=1.5
#   2 — b2p0  : f=2.0  (W1 knee — primary champion candidate)
#   3 — b3p0  : f=3.0  (W1 plateau — does deeper Ni II shift the Ca II knee?)

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"

if [[ "$SLURM_JOB_PARTITION" == h100* || "$SLURM_JOB_PARTITION" == h200* ]]; then
    BIN="$ROOT/lumina_cuda_h100_aulscale8"
else
    BIN="$ROOT/lumina_cuda_a100_aulscale8"
fi

N_PKT=800000
N_ITER=12

LABELS=(ctrl  b1p5 b2p0 b3p0)
CA2_BOOSTS=(1.0 1.5  2.0  3.0)
tag=${LABELS[$SLURM_ARRAY_TASK_ID]}
ca2=${CA2_BOOSTS[$SLURM_ARRAY_TASK_ID]}

FE_FAC=0.3
CO_FAC=0.3
RED_FAC=0.3
SI_OPT_FAC=0.10
NI2_OPT_FAC=0.05   # U1 champion
EPS_UV=0.7

vi=10400
ro=0.00
de=9.0
xfeo=0.05

label="ddc15X1_${tag}"
work_root="$ROOT/logs/ddc15X1_${SLURM_ARRAY_JOB_ID}_${label}"
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
ro, de, vi_kms, X_FE_OUTER = $ro, $de, $vi, $xfeo

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
with open(f"{dst}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)

W_init = 0.5 * (r_inner[0] / r_inner)**2
T_rad_init = np.full(n_shells, T_inner, dtype=float)
pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "W": W_init, "T_rad": T_rad_init}).to_csv(f"{dst}/plasma_state.csv", index=False)
m_H = 1.6726e-24
ne_init = (rho_new / m_H) / 14.0
pd.DataFrame({"shell_id": np.arange(n_shells, dtype=int),
              "n_e": ne_init}).to_csv(f"{dst}/electron_densities.csv", index=False)

def ddc15_profile(v, x_fe_outer):
    if   v < 9000:           X_Si = 0.05
    elif v < 11500:          X_Si = 0.40
    elif v < 14000:          X_Si = 0.40 - 0.35*(v-11500)/2500
    else:                    X_Si = 0.05
    if   v < 10000:          X_O = 0.18
    elif v < 12500:          X_O = 0.18 + 0.20*(v-10000)/2500
    elif v < 15000:          X_O = 0.38 + 0.42*(v-12500)/2500
    else:                    X_O = 0.80
    if   v < 11000:          X_C = 0.01
    elif v < 14000:          X_C = 0.01 + 0.10*(v-11000)/3000
    elif v < 17000:          X_C = 0.11 + 0.15*(v-14000)/3000
    else:                    X_C = 0.26
    X_S  = X_Si * 0.13
    X_Ca = X_Si * 0.045
    X_Mg = 0.03 if v < 14000 else 0.01
    if   v < 11000:          X_Fe = 0.15
    elif v < 13500:          X_Fe = x_fe_outer * np.exp(-(v-11000)/1500.0)
    else:                    X_Fe = max(5e-4, x_fe_outer * np.exp(-2500/1500.0))
    if   v < 11000:          X_Ni = 8e-3
    elif v < 13500:          X_Ni = 8e-3 * np.exp(-(v-11000)/1200.0)
    else:                    X_Ni = 5e-6
    if   v < 11000:          X_Co = 5e-3
    elif v < 13500:          X_Co = 5e-3 * np.exp(-(v-11000)/1200.0)
    else:                    X_Co = 5e-7
    X_Al = 4e-3; X_Sc = 1e-5; X_Ti = 1e-4; X_V = 5e-5
    X_Cr = X_Fe * 0.05; X_Mn = X_Fe * 0.03
    return {6:X_C, 8:X_O, 12:X_Mg, 13:X_Al, 14:X_Si, 16:X_S,
            20:X_Ca, 21:X_Sc, 22:X_Ti, 23:X_V, 24:X_Cr, 25:X_Mn,
            26:X_Fe, 27:X_Co, 28:X_Ni}

Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
X = np.zeros((len(Z_LIST), n_shells))
for s in range(n_shells):
    prof = ddc15_profile(v_mid_new[s], X_FE_OUTER)
    for i, Z in enumerate(Z_LIST):
        X[i, s] = prof[Z]
for s in range(n_shells):
    tot = X[:, s].sum()
    if tot > 0: X[:, s] /= tot

cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
rows = [[Z] + list(X[i]) for i, Z in enumerate(Z_LIST)]
pd.DataFrame(rows, columns=cols).to_csv(f"{dst}/abundances.csv", index=False)
PYEOF

echo "=== X1 [${label}]  U1_f0p05 base + Ca2_boost(SCALE8) f=${ca2}  N_PKT=${N_PKT}  N_ITER=${N_ITER} ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary: $BIN"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_EPS_UV="$EPS_UV" \
LUMINA_EPS_UV_RED_ONLY=1 \
LUMINA_AUL_SCALE_FACTOR=0.3 \
LUMINA_AUL_SCALE_LAMBDA_MAX=4000 \
LUMINA_AUL_SCALE_ZMASK=28 \
LUMINA_AUL_SCALE_IONMASK=1 \
LUMINA_AUL_SCALE2_FACTOR=0.05 \
LUMINA_AUL_SCALE2_LAMBDA_MAX=4000 \
LUMINA_AUL_SCALE2_ZMASK=14 \
LUMINA_AUL_SCALE2_IONMASK=1 \
LUMINA_AUL_SCALE3_FACTOR="$FE_FAC" \
LUMINA_AUL_SCALE3_LAMBDA_MAX=4000 \
LUMINA_AUL_SCALE3_ZMASK=26 \
LUMINA_AUL_SCALE3_IONMASK=1 \
LUMINA_AUL_SCALE4_FACTOR="$CO_FAC" \
LUMINA_AUL_SCALE4_LAMBDA_MAX=4000 \
LUMINA_AUL_SCALE4_ZMASK=27 \
LUMINA_AUL_SCALE4_IONMASK=1 \
LUMINA_AUL_SCALE5_FACTOR="$RED_FAC" \
LUMINA_AUL_SCALE5_LAMBDA_MIN=4500 \
LUMINA_AUL_SCALE5_LAMBDA_MAX=7500 \
LUMINA_AUL_SCALE5_ZMASK="24,26,27" \
LUMINA_AUL_SCALE5_IONMASK=2 \
LUMINA_AUL_SCALE6_FACTOR="$SI_OPT_FAC" \
LUMINA_AUL_SCALE6_LAMBDA_MIN=4000 \
LUMINA_AUL_SCALE6_LAMBDA_MAX=7000 \
LUMINA_AUL_SCALE6_ZMASK=14 \
LUMINA_AUL_SCALE6_IONMASK=1 \
LUMINA_AUL_SCALE7_FACTOR="$NI2_OPT_FAC" \
LUMINA_AUL_SCALE7_LAMBDA_MIN=4000 \
LUMINA_AUL_SCALE7_LAMBDA_MAX=7500 \
LUMINA_AUL_SCALE7_ZMASK=28 \
LUMINA_AUL_SCALE7_IONMASK=1 \
LUMINA_AUL_SCALE8_FACTOR="$ca2" \
LUMINA_AUL_SCALE8_LAMBDA_MIN=3700 \
LUMINA_AUL_SCALE8_LAMBDA_MAX=9000 \
LUMINA_AUL_SCALE8_ZMASK=20 \
LUMINA_AUL_SCALE8_IONMASK=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- AUL_SCALE summary ---"
grep -E '\[AUL_SCALE' stdout.log | head -10
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted|EPS_UV' stdout.log | tail -16
echo "Done: $(date)"
