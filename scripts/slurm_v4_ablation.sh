#!/bin/bash
#SBATCH --job-name=v4_ablation
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-9
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# V4 ablation: which SCALE band-aids are still needed on CLEAN CMFGEN atomic data?
#
# 10 cells × p+3.7d × L×1.45 (V3 champion: Snifs 0.143 / HST opt 0.170 / UV 1.016):
#   0: baseline (all SCALE on)        = same as V4 smoke cell 1
#   1: drop SCALE1 (Ni II UV)
#   2: drop SCALE2 (Si II UV)
#   3: drop SCALE3 (Fe II UV)
#   4: drop SCALE4 (Co II UV)
#   5: drop SCALE6 (Si II opt)
#   6: drop SCALE7 (Ni II opt)
#   7: drop SCALE8 (Ca II opt boost)
#   8: drop SCALE9 (Ni II NIR)
#   9: drop ALL SCALE (pure CMFGEN clean baseline)
#
# Best (lowest Snifs+HST joint) → next-stack candidate.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_cmfgen"
BIN="$ROOT/lumina_cuda_h100_aulscale9"

N_PKT=800000
N_ITER=12

tag=p+3.7d
t_exp_s=1788480
L_base=1.1e+43
L_scale=1.45
snifs=sn2011fe_p4d2d.csv
hst_tag=phase+03.7

L_inner=$(python3 -c "print(f'{$L_base * $L_scale:.6e}')")

idx=$SLURM_ARRAY_TASK_ID
LABELS=(baseline drop_S1_Ni2UV drop_S2_Si2UV drop_S3_Fe2UV drop_S4_Co2UV drop_S6_Si2opt drop_S7_Ni2opt drop_S8_Ca2 drop_S9_Ni2NIR drop_ALL)
SUFFIX=${LABELS[$idx]}

# Baseline factors (W5 niF005, identical to V4 smoke cell 1)
FE_FAC=0.3
CO_FAC=0.3
SI_OPT_FAC=0.10
NI2_OPT_FAC=0.05
CA2_FAC=3.0
EPS_UV=0.7
S9_FAC=0.05
NI_UV_FAC=0.3     # SCALE1
SI_UV_FAC=0.05    # SCALE2

# Override per-cell — set factor to 1.0 to "drop" (i.e., A_ul stays at clean CMFGEN value)
case "$idx" in
  0) ;;                                                # baseline, all on
  1) NI_UV_FAC=1.0 ;;                                  # drop SCALE1
  2) SI_UV_FAC=1.0 ;;                                  # drop SCALE2
  3) FE_FAC=1.0 ;;                                     # drop SCALE3
  4) CO_FAC=1.0 ;;                                     # drop SCALE4
  5) SI_OPT_FAC=1.0 ;;                                 # drop SCALE6
  6) NI2_OPT_FAC=1.0 ;;                                # drop SCALE7
  7) CA2_FAC=1.0 ;;                                    # drop SCALE8
  8) S9_FAC=1.0 ;;                                     # drop SCALE9
  9) NI_UV_FAC=1.0; SI_UV_FAC=1.0; FE_FAC=1.0; CO_FAC=1.0
     SI_OPT_FAC=1.0; NI2_OPT_FAC=1.0; CA2_FAC=1.0; S9_FAC=1.0 ;;  # drop ALL
esac

vi=10400
ro=0.00
de=9.0
xfeo=0.05

label="v4_abl_${SUFFIX}"
work_root="$ROOT/logs/v4_abl_${SLURM_ARRAY_JOB_ID}_${label}"
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
print(f"  t_exp={t_exp:.0f}s  L={L_inner:.3e}  T_inner={T_inner:.1f}K  rho0_v11k={rho0:.3e}")
PYEOF

echo "=== V4 ABLATION [${label}] ${SUFFIX} L=${L_inner} N_PKT=${N_PKT} ==="
echo "Factors: NI_UV=$NI_UV_FAC SI_UV=$SI_UV_FAC FE=$FE_FAC CO=$CO_FAC SI_OPT=$SI_OPT_FAC NI2_OPT=$NI2_OPT_FAC CA2=$CA2_FAC S9=$S9_FAC"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "REF: $SRC_REF (CMFGEN clean atomic data)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_NLTE_START_ITER=2 \
LUMINA_EPS_UV="$EPS_UV" \
LUMINA_EPS_UV_RED_ONLY=1 \
LUMINA_AUL_SCALE_FACTOR="$NI_UV_FAC" \
LUMINA_AUL_SCALE_LAMBDA_MAX=4000 \
LUMINA_AUL_SCALE_ZMASK=28 \
LUMINA_AUL_SCALE_IONMASK=1 \
LUMINA_AUL_SCALE2_FACTOR="$SI_UV_FAC" \
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
echo "--- convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -10

echo ""
echo "=== score [${label}] vs Snifs $snifs + HST $hst_tag ==="
python3 <<PYEOF
import os, glob, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT="$ROOT"; C_KMS=2.998e5
def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return float(np.trapezoid(flu[sel], lam[sel]))
def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, fwhm=20000.0, wl_lo=3300., wl_hi=8000.):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0: return float("nan")
    mod_flu = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi); ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return float("nan")
    dl = np.median(np.diff(ol)); mid = 0.5*(ol[0]+ol[-1])
    sig = (fwhm/C_KMS)*mid/2.355/dl
    sO = gaussian_filter1d(of, sig, mode='nearest')
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100); ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10: return float("nan")
    dlM = np.median(np.diff(ml)); midM = 0.5*(ml[0]+ml[-1])
    sigM = (fwhm/C_KMS)*midM/2.355/dlM
    sM = gaussian_filter1d(mf, sigM, mode='nearest')
    common = (ol>=ml[0]) & (ol<=ml[-1])
    if common.sum() < 10: return float("nan")
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

m = pd.read_csv("$work_root/lumina_spectrum.csv")
mlam, mflu = m["wavelength_angstrom"].values, m["flux"].values

o = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/$snifs", comment='#')
olam, oflu = o["wavelength_angstrom"].values, o["flux_erg_s_cm2_angstrom"].values
gO = band_int(olam, oflu, 4500, 5800)
r_snifs = rms_bn(mlam, mflu, olam, oflu, gO, wl_lo=3300., wl_hi=8000.)
print(f"  V4_abl ${SUFFIX}  Snifs RMS_bn [3300,8000]  = {r_snifs:.4f}")

hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]:
    pat = f"{hst_dir}/CCD_{grat}_*${hst_tag}*sx1.csv"
    files = sorted(glob.glob(pat))
    if not files: continue
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0].copy()
    if len(dfs) > 1:
        flux_stack = np.column_stack([np.interp(base["wavelength_angstrom"].values,
                                                d["wavelength_angstrom"].values,
                                                d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
        base["flux_erg_s_cm2_angstrom"] = np.nanmean(flux_stack, axis=1)
    sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
    parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
if parts:
    hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
    hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
    hl, hf = hst["wavelength_angstrom"].values, hst["flux_erg_s_cm2_angstrom"].values
    gH = band_int(hl, hf, 4500, 5800)
    if gH > 0:
        r_hst_can = rms_bn(mlam, mflu, hl, hf, gH, wl_lo=3000., wl_hi=8000.)
        r_hst_uv  = rms_bn(mlam, mflu, hl, hf, gH, wl_lo=1700., wl_hi=3000.)
        print(f"  V4_abl ${SUFFIX}  HST RMS_bn [3000,8000]   = {r_hst_can:.4f}")
        print(f"  V4_abl ${SUFFIX}  HST RMS_bn [1700,3000]   = {r_hst_uv:.4f}")
PYEOF

echo "Done: $(date)"
