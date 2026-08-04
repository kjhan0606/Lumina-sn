#!/bin/bash
#SBATCH --job-name=v4_inversion
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-3
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# V4 inversion probe: V3 W5 stack on V4 (smoke 157176) gave Snifs 0.30-0.34, HST UV 1.10/0.97
# — UV[1700,2500] L/HST = 0.016-0.025 (1.6-2.5%) = catastrophic UV over-blanketing.
# CMFGEN has ~11x more lines than V3 (321k vs 28k) so V3 UV dampers f=0.3 ADD on top
# of inherent CMFGEN UV opacity → UV reprocessed into red (Red L/HST=3.3, NIR=4.6-5.0).
#
# Strategy: strip V3 dampers entirely. Test 4 cells on V3.1 champion (vi=10600, p+3.7d, hot1p45):
#  C0: BARE V4 — zero SCALE knobs, zero EPS_UV. Raw CMFGEN response.
#  C1: V4 + EPS_UV=0.7 RED_ONLY — pure UV escape, no opacity scaling.
#  C2: V4 + EPS_UV=0.3 RED_ONLY — stronger UV escape.
#  C3: V4 + SCALE3/4 INVERTED (Fe/Co UV f=2.0 BOOST instead of damp) — direction test.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
SRC_REF="$ROOT/data/tardis_reference_cmfgen"
BIN="$ROOT/lumina_cuda_h100_aulscale9"

N_PKT=800000
N_ITER=12

# All cells: p+3.7d, hot1p45, vi=10600 (V3.1 champion)
TAG=p+3.7d
T_EXP_S=1788480
L_BASE=1.1e+43
L_SCALE=1.45
SNIFS=sn2011fe_p4d2d.csv
HST_TAG=phase+03.7

SUFFIX=(bare epsuv07 epsuv03 fecoboost)

suffix=${SUFFIX[$SLURM_ARRAY_TASK_ID]}

L_inner=$(python3 -c "print(f'{$L_BASE * $L_SCALE:.6e}')")

vi=10600
ro=0.00
de=9.0
xfeo=0.05

label="v4_inv_${TAG}_${suffix}"
work_root="$ROOT/logs/v4_inv_${SLURM_ARRAY_JOB_ID}_${label}"
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
t_exp = $T_EXP_S; L_inner = $L_inner

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

echo "=== V4 INVERSION [${label}] L=${L_inner} N_PKT=${N_PKT} ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary: $BIN  Snifs: $SNIFS  HST tag: $HST_TAG  vi=$vi"
echo "REF: $SRC_REF (CMFGEN clean atomic data, 17.7k lev / 321.9k lines)"
echo "Time: $(date)"
echo "Probe cell: $suffix"

# Base env (common to all cells)
ENV_BASE=(
  "LUMINA_BF_OPACITY=1"
  "LUMINA_CMFGEN_SIGMA_BF=$ROOT/data/atomic/cmfgen_sigma_bf.bin"
  "LUMINA_DYNAMIC_TRANSPROB=1"
  "LUMINA_NLTE_SKIP_Z=14"
  "LUMINA_NLTE_START_ITER=2"
)

# Cell-specific env
case "$suffix" in
  bare)
    EXTRA_ENV=()
    ;;
  epsuv07)
    EXTRA_ENV=(
      "LUMINA_EPS_UV=0.7"
      "LUMINA_EPS_UV_RED_ONLY=1"
    )
    ;;
  epsuv03)
    EXTRA_ENV=(
      "LUMINA_EPS_UV=0.3"
      "LUMINA_EPS_UV_RED_ONLY=1"
    )
    ;;
  fecoboost)
    EXTRA_ENV=(
      "LUMINA_EPS_UV=0.7"
      "LUMINA_EPS_UV_RED_ONLY=1"
      "LUMINA_AUL_SCALE3_FACTOR=2.0"
      "LUMINA_AUL_SCALE3_LAMBDA_MAX=4000"
      "LUMINA_AUL_SCALE3_ZMASK=26"
      "LUMINA_AUL_SCALE3_IONMASK=1"
      "LUMINA_AUL_SCALE4_FACTOR=2.0"
      "LUMINA_AUL_SCALE4_LAMBDA_MAX=4000"
      "LUMINA_AUL_SCALE4_ZMASK=27"
      "LUMINA_AUL_SCALE4_IONMASK=1"
    )
    ;;
esac

env "${ENV_BASE[@]}" "${EXTRA_ENV[@]}" \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"

echo ""
echo "--- convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final|L_emitted' stdout.log | tail -10

echo ""
echo "=== score [${label}] vs Snifs $SNIFS + HST $HST_TAG ==="
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

# --- Snifs ---
o = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/$SNIFS", comment='#')
olam, oflu = o["wavelength_angstrom"].values, o["flux_erg_s_cm2_angstrom"].values
gO = band_int(olam, oflu, 4500, 5800)
r_snifs = rms_bn(mlam, mflu, olam, oflu, gO, wl_lo=3300., wl_hi=8000.)
print(f"  V4_inv ${suffix}  Snifs RMS_bn [3300,8000]  = {r_snifs:.4f}  (V3.1 vi10600 0.139)")

# --- HST stitched ---
hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]:
    pat = f"{hst_dir}/CCD_{grat}_*${HST_TAG}*sx1.csv"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"  ! no HST file for {grat} ${HST_TAG}")
        continue
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
        r_hst_can  = rms_bn(mlam, mflu, hl, hf, gH, wl_lo=3000., wl_hi=8000.)
        r_hst_uv   = rms_bn(mlam, mflu, hl, hf, gH, wl_lo=1700., wl_hi=3000.)
        print(f"  V4_inv ${suffix}  HST RMS_bn [3000,8000]   = {r_hst_can:.4f}  (V3.1 0.346)")
        print(f"  V4_inv ${suffix}  HST RMS_bn [1700,3000]   = {r_hst_uv:.4f}  (V3.1 0.346)")
        def br(lo, hi):
            return band_int(mlam, mflu, lo, hi), band_int(hl, hf, lo, hi)
        bands = [("UV",1700,2500),("Blue",3400,4500),("Ref",4500,5800),("Red",6500,10000),("NIR",8200,10000)]
        bm = {n:br(lo,hi) for (n,lo,hi) in bands}
        ref_m, ref_h = bm["Ref"]
        print(f"  V4_inv ${suffix}  --- SED band ratios (LUMINA/HST, normed to Ref) ---")
        for (n,lo,hi) in bands:
            mm, hh = bm[n]
            if ref_m>0 and ref_h>0 and hh>0:
                r = (mm/ref_m)/(hh/ref_h)
                print(f"  V4_inv ${suffix}    band {n:<6} [{lo:5d},{hi:5d}]  L/HST = {r:.3f}")
PYEOF

echo "Done: $(date)"
