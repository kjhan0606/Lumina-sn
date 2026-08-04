#!/bin/bash
#SBATCH --job-name=nlte3_diag
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# #219e Phase 1: NLTE iron-peak III diagnostic dump.
# Single cell at V3.1 best corner (vi=10600 L=1.55 cleanz, #264 cell 9).
# 800K x 12 iter with LUMINA_NLTE_RATE_DUMP=1 + LUMINA_NLTE_LEVEL_DUMP=1.
# Outputs: nlte_rate_balance.csv + nlte_levels_iter*.csv per iteration.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
    *H100*|*H200*)   BIN="$ROOT/lumina_cuda_h100_aulscale9" ;;
    *A100*)          BIN="$ROOT/lumina_cuda_a100_aulscale9" ;;
    *)               BIN="$ROOT/lumina_cuda_h100_aulscale9" ;;
esac

N_PKT=800000
N_ITER=12

SRC_REF="$ROOT/data/tardis_reference_zeta_clean"
L_scale=1.55
suffix=nlte3diag

t_exp_s=1788480
L_base=1.1e+43
snifs=sn2011fe_p4d2d.csv
hst_tag=phase+03.7
tag=p+3.7d

L_inner=$(python3 -c "print(f'{$L_base * $L_scale:.6e}')")

FE_FAC=0.3
CO_FAC=0.3
SI_OPT_FAC=0.10
NI2_OPT_FAC=0.05
CA2_FAC=3.0
EPS_UV=0.7
S9_FAC=0.05

vi=10600
ro=0.00
de=9.5
xfeo=0.05

label="nlte3diag_${tag}"
work_root="$ROOT/logs/nlte3diag_${SLURM_JOB_ID}_${label}"
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
ro, de, vi_kms, X_FE_OUTER = $ro, $de, $vi, $xfeo
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
    X_S  = X_Si * 0.13; X_Ca = X_Si * 0.045
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
print(f"  t_exp={t_exp:.0f}s  L={L_inner:.3e}  T_inner={T_inner:.1f}K  rho0_v11k={rho0:.3e}")
print(f"  SRC_REF = {src}")
PYEOF

echo "=== NLTE3-DIAG [${label}] L=${L_inner} N_PKT=${N_PKT} ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary: $BIN  Snifs: $snifs  HST tag: $hst_tag"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_NLTE_START_ITER=2 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_NLTE_LEVEL_DUMP=1 \
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
LUMINA_AUL_SCALE8_FACTOR="$CA2_FAC" \
LUMINA_AUL_SCALE8_LAMBDA_MIN=3700 \
LUMINA_AUL_SCALE8_LAMBDA_MAX=9000 \
LUMINA_AUL_SCALE8_ZMASK=20 \
LUMINA_AUL_SCALE8_IONMASK=1 \
LUMINA_AUL_SCALE9_FACTOR="$S9_FAC" \
LUMINA_AUL_SCALE9_LAMBDA_MIN=7500 \
LUMINA_AUL_SCALE9_LAMBDA_MAX=10000 \
LUMINA_AUL_SCALE9_ZMASK=28 \
LUMINA_AUL_SCALE9_IONMASK=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"

echo ""
echo "--- diagnostic file sizes ---"
ls -la nlte_rate_balance.csv nlte_levels_iter*.csv 2>/dev/null | head -20

echo ""
echo "--- iron-peak III rate balance (final iteration) ---"
# Pick last unique iteration per (Z, ion_lo, shell=0)
python3 - <<PYEOF
import pandas as pd
try:
    df = pd.read_csv("nlte_rate_balance.csv")
    # Take last (final iter) for each (Z, ion_lo, shell=0)
    df0 = df[(df["shell"] == 0)].drop_duplicates(["Z","ion_lo"], keep="last")
    ip3 = df0[df0["Z"].isin([24, 26, 27, 28]) & (df0["ion_lo"] == 2)]
    print(ip3[["Z","ion_lo","shell","T_e","T_rad","W","n_ion_lo_neb","n_ion_hi_neb",
               "sum_R_bf","sum_R_rec","ratio_RrecRbf"]].to_string(index=False))
    print()
    print("--- For ref: ion=1 (II) lower-pair entries (II->III) ---")
    ip2 = df0[df0["Z"].isin([24, 26, 27, 28]) & (df0["ion_lo"] == 1)]
    print(ip2[["Z","ion_lo","shell","T_e","T_rad","W","n_ion_lo_neb","n_ion_hi_neb",
               "sum_R_bf","sum_R_rec","ratio_RrecRbf"]].to_string(index=False))
except Exception as e:
    print(f"Rate dump analysis failed: {e}")
PYEOF

echo ""
echo "--- iron-peak III level count + n_pop tally (final iter, shell=0) ---"
python3 - <<PYEOF
import glob, pandas as pd
files = sorted(glob.glob("nlte_levels_iter*.csv"))
if not files:
    print("No level dumps found")
else:
    last = files[-1]
    print(f"Reading {last}")
    df = pd.read_csv(last)
    # shell 0 for Cr/Fe/Co/Ni III
    for Z, name in [(24,"Cr"),(26,"Fe"),(27,"Co"),(28,"Ni")]:
        for ion, ion_name in [(1,"II"),(2,"III")]:
            sub = df[(df["Z"]==Z) & (df["ion"]==ion) & (df["shell"]==0)]
            if len(sub):
                n_total = sub["n_pop"].sum()
                n_ion_ref = sub["n_ion_total"].iloc[0]
                ratio = n_total / n_ion_ref if n_ion_ref > 0 else 0.0
                print(f"  {name} {ion_name} shell 0: n_levels={len(sub):3d}  n_total_NLTE={n_total:.3e}  n_ion_neb={n_ion_ref:.3e}  ratio={ratio:.3e}")
PYEOF

echo ""
echo "--- shell-0 NLTE n_total vs nebular n_ion (from stdout built-in dump) ---"
grep -E "shell 0: NLTE n_total" stdout.log | tail -28

echo ""
echo "--- convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -10

echo "Done: $(date)"
