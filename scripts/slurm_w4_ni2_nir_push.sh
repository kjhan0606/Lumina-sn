#!/bin/bash
#SBATCH --job-name=w4_ni2_nir_push
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-7
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# W4: Push W3 winner n9F03Ni1 (Ni II NIR damp [7500,10000] f=0.30).
# W3 champ: HST[3000,8000]=0.1707, Red 2.09→1.91. Push factor + band + stack.
#   0 — repro            L=1.30, Ni II f=0.30 [7500,10000]   (W3 champ repro)
#   1 — niF020           L=1.30, Ni II f=0.20 [7500,10000]   (push factor)
#   2 — niF010           L=1.30, Ni II f=0.10 [7500,10000]   (aggressive push)
#   3 — niF030b7000      L=1.30, Ni II f=0.30 [7000,10000]   (wider band)
#   4 — niF030b6500      L=1.30, Ni II f=0.30 [6500,10000]   (widest band)
#   5 — niCoF030         L=1.30, Ni+Co II f=0.30 [7500,10000] (stack Co II)
#   6 — L145niF030       L=1.45, Ni II f=0.30 [7500,10000]   (hot + damp)
#   7 — L145niF010       L=1.45, Ni II f=0.10 [7500,10000]   (hot + aggressive)

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_strat6_2011fe_physical"
BIN="$ROOT/lumina_cuda_h100_aulscale9"

N_PKT=800000
N_ITER=12

t_exp_s=1788480
L_BASE=1.10e+43
snifs="sn2011fe_p4d2d.csv"
hst_tag="phase+03.7"

FE_FAC=0.3
CO_FAC=0.3
SI_OPT_FAC=0.10
NI2_OPT_FAC=0.05
CA2_FAC=3.0
EPS_UV=0.7
vi=10400
ro=0.00
de=9.0
xfeo=0.05

# 0:repro 1:niF020 2:niF010 3:niF030b7000 4:niF030b6500 5:niCoF030 6:L145niF030 7:L145niF010
L_SCALE_ARR=(1.30 1.30 1.30 1.30 1.30 1.30 1.45 1.45)
S9_FAC_ARR=(0.30 0.20 0.10 0.30 0.30 0.30 0.30 0.10)
S9_LMIN_ARR=(7500 7500 7500 7000 6500 7500 7500 7500)
S9_Z_ARR=("28" "28" "28" "28" "28" "27,28" "28" "28")
TAG_ARR=(repro niF020 niF010 niF030b7000 niF030b6500 niCoF030 L145niF030 L145niF010)

L_SCALE=${L_SCALE_ARR[$SLURM_ARRAY_TASK_ID]}
S9_FAC=${S9_FAC_ARR[$SLURM_ARRAY_TASK_ID]}
S9_LMIN=${S9_LMIN_ARR[$SLURM_ARRAY_TASK_ID]}
S9_Z=${S9_Z_ARR[$SLURM_ARRAY_TASK_ID]}
tag=${TAG_ARR[$SLURM_ARRAY_TASK_ID]}

L_inner=$(python3 -c "print(${L_BASE} * ${L_SCALE})")

label="w4_${tag}"
work_root="$ROOT/logs/w4_${SLURM_ARRAY_JOB_ID}_${label}"
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
PYEOF

echo "=== W4 [${label}] L=${L_SCALE} SCALE9: Z=${S9_Z} ion=1 f=${S9_FAC} band=[${S9_LMIN},10000] ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_NLTE_START_ITER=2 \
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
LUMINA_AUL_SCALE7_LAMBDA_MAX=7500 \
LUMINA_AUL_SCALE7_LAMBDA_MIN=4000 \
LUMINA_AUL_SCALE7_ZMASK=28 \
LUMINA_AUL_SCALE7_IONMASK=1 \
LUMINA_AUL_SCALE8_FACTOR="$CA2_FAC" \
LUMINA_AUL_SCALE8_LAMBDA_MIN=3700 \
LUMINA_AUL_SCALE8_LAMBDA_MAX=9000 \
LUMINA_AUL_SCALE8_ZMASK=20 \
LUMINA_AUL_SCALE8_IONMASK=1 \
LUMINA_AUL_SCALE9_FACTOR="$S9_FAC" \
LUMINA_AUL_SCALE9_LAMBDA_MIN="$S9_LMIN" \
LUMINA_AUL_SCALE9_LAMBDA_MAX=10000 \
LUMINA_AUL_SCALE9_ZMASK="$S9_Z" \
LUMINA_AUL_SCALE9_IONMASK=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"

echo ""
echo "--- convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -10

echo ""
echo "=== score [${label}] L=${L_SCALE} Z=${S9_Z} f=${S9_FAC} band=[${S9_LMIN},10000] ==="
python3 <<PYEOF
import os, glob, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT="$ROOT"; C_KMS=2.998e5
def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    if sel.sum() < 5: return float("nan")
    return float(np.trapezoid(flu[sel], lam[sel]))
def smooth(lam, flu, fwhm=20000.0):
    dl = np.median(np.diff(lam)); mid = 0.5*(lam[0]+lam[-1])
    sig = (fwhm/C_KMS)*mid/2.355/dl
    return gaussian_filter1d(flu, sig, mode='nearest')
def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, wl_lo, wl_hi):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0 or gO <= 0: return float("nan")
    mod_flu_n = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi); ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return float("nan")
    sO = smooth(ol, of)
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100); ml, mf = mod_lam[selM], mod_flu_n[selM]
    if len(ml) < 10: return float("nan")
    sM = smooth(ml, mf)
    common = (ol>=ml[0]) & (ol<=ml[-1])
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

m = pd.read_csv("$work_root/lumina_spectrum.csv")
mlam, mflu = m["wavelength_angstrom"].values, m["flux"].values

o = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/$snifs", comment='#')
olam, oflu = o["wavelength_angstrom"].values, o["flux_erg_s_cm2_angstrom"].values
gO = band_int(olam, oflu, 4500, 5800)
r_snifs = rms_bn(mlam, mflu, olam, oflu, gO, 3300, 8000)
print(f"  W4 ${tag}  Snifs RMS_bn [3300,8000]  = {r_snifs:.4f}")

hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]:
    pat = f"{hst_dir}/CCD_{grat}_*${hst_tag}*sx1.csv"
    files = sorted(glob.glob(pat))
    if not files: continue
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0].copy()
    if len(dfs) > 1:
        stack = np.column_stack([np.interp(base["wavelength_angstrom"].values,
                                            d["wavelength_angstrom"].values,
                                            d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
        base["flux_erg_s_cm2_angstrom"] = np.nanmean(stack, axis=1)
    sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
    parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
if parts:
    hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
    hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
    hl, hf = hst["wavelength_angstrom"].values, hst["flux_erg_s_cm2_angstrom"].values
    gH = band_int(hl, hf, 4500, 5800)
    if gH > 0:
        r_hst_can = rms_bn(mlam, mflu, hl, hf, gH, 3000, 8000)
        r_hst_uv  = rms_bn(mlam, mflu, hl, hf, gH, 1700, 3000)
        print(f"  W4 ${tag}  HST RMS_bn [3000,8000]   = {r_hst_can:.4f}")
        print(f"  W4 ${tag}  HST RMS_bn [1700,3000] = {r_hst_uv:.4f}")
        gM_ref = band_int(mlam, mflu, 4500, 5800)
        mflu_n = mflu * (gH/gM_ref)
        DIAG_BANDS = [
            ("UV",    1700, 2500),
            ("Blue",  3400, 4500),
            ("Ref",   4500, 5800),
            ("Red",   6500, 10000),
            ("NIR",   8200, 10000),
        ]
        print(f"  W4 ${tag}  --- SED band ratios (LUMINA/HST, normed to Ref) ---")
        for name, lo, hi in DIAG_BANDS:
            mb = band_int(mlam, mflu_n, lo, hi)
            ob = band_int(hl, hf, lo, hi)
            r = mb/ob if (ob and ob > 0) else float("nan")
            print(f"  W4 ${tag}    band {name:6s} [{lo:5d},{hi:5d}]  L/HST = {r:.3f}")
PYEOF

echo "Done: $(date)"
