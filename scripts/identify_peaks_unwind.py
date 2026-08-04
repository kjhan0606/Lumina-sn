#!/usr/bin/env python3
"""HST B-max 스펙트럼 → 알려진 SN Ia 진단선 trough 위치 → Doppler v → shell 매핑.

설계 변경 (v2): 데이터-driven brute force는 III/IV 가짜 매치 또는 neutral
공명선에 끌려가서 Si II 6355 같은 정통 진단선조차 놓침 (rare-species
A_ul = 1e8 vs Si II 5.8e7). 대신 **표준 SN Ia 진단선 카탈로그**를 사용:
잘 알려진 (ion, λ_rest)에 대해 데이터에서 트로프(흡수)와 피크(방출)를 찾고,
관측 위치로부터 Doppler v를 계산, 챔피언 geometry/abundance에 매핑.

출력:
  logs/feature_id_table.csv   — 큐레이트 진단선별 측정 trough/peak λ + v
  logs/zone_requirements.csv  — ion별 required shell + 챔피언 abundance 비교
  figures/identify_peaks_unwind.png — 스펙트럼 + 마커 시각화
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
REF  = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"
OUT_FEAT = ROOT / "logs/feature_id_table.csv"
OUT_ZONE = ROOT / "logs/zone_requirements.csv"
FIG_DIR  = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
FIG_OUT  = FIG_DIR / "identify_peaks_unwind.png"

C_KMS = 299792.458
T_EXP_DAY = 17.8

# ---------------- 표준 SN Ia 진단선 (Branch et al., Tanaka, Mazzali 등 표준) ----------------
# (label, Z, ion(0=I), λ_rest_Å, search_window_kms_blueshift_min, search_window_kms_max)
DIAGNOSTICS = [
    # UV (HST)
    ("Mg II 2796 (UV)",   12, 1, 2795.5,  5000, 22000),
    ("Mg II 2803 (UV)",   12, 1, 2802.7,  5000, 22000),
    ("Fe II 2382 (UV)",   26, 1, 2382.0,  5000, 22000),
    ("Fe II 2600 (UV)",   26, 1, 2600.2,  5000, 22000),
    ("Mn II 2576 (UV)",   25, 1, 2576.1,  5000, 22000),
    ("Mn II 2594 (UV)",   25, 1, 2594.5,  5000, 22000),
    # Optical
    ("Ca II H 3934",      20, 1, 3933.7,  6000, 22000),
    ("Ca II K 3968",      20, 1, 3968.5,  6000, 22000),
    ("Mg II 4481",        12, 1, 4481.3,  6000, 22000),
    ("Fe II 4924",        26, 1, 4923.9,  6000, 22000),
    ("Fe II 5018",        26, 1, 5018.4,  6000, 22000),
    ("Fe II 5169",        26, 1, 5169.0,  6000, 22000),
    ("Fe III 4404",       26, 2, 4404.8,  6000, 22000),
    ("Fe III 5129",       26, 2, 5129.2,  6000, 22000),
    ("S II 5454 (W blue)",16, 1, 5454.2,  6000, 18000),
    ("S II 5640 (W red)", 16, 1, 5640.0,  6000, 18000),
    ("Si II 5972",        14, 1, 5972.0,  6000, 18000),
    ("Si II 6355",        14, 1, 6355.0,  6000, 18000),
    ("Si III 4553",       14, 2, 4552.6,  6000, 22000),
    ("O I 7774",           8, 0, 7774.0,  6000, 18000),
    ("Ca II IR1 8498",    20, 1, 8498.0,  6000, 18000),
    ("Ca II IR2 8542",    20, 1, 8542.0,  6000, 18000),
    ("Ca II IR3 8662",    20, 1, 8662.0,  6000, 18000),
    # Inner Fe-peak III bump (the [3000,3100] residual we've been chasing)
    ("Co III ~3070",      27, 2, 3070.0,    -3000, 8000),  # pseudo-peak, allow small shift
    ("Fe III ~3070",      26, 2, 3070.0,    -3000, 8000),
]

ELEM = {1:"H",2:"He",6:"C",7:"N",8:"O",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",
        15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",
        24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn"}


def load_spectrum():
    df = pd.read_csv(HST)
    w = df.iloc[:, 0].values
    f = df.iloc[:, 1].values
    e = df.iloc[:, 2].values
    m = (w >= 1700) & (w <= 8800) & np.isfinite(f) & (f > 0)
    return w[m], f[m], e[m]


def pseudo_continuum(w, f, win_A=300.0):
    cont = np.zeros_like(f)
    for i in range(len(w)):
        sel = (w >= w[i] - win_A/2) & (w <= w[i] + win_A/2)
        cont[i] = np.percentile(f[sel], 90)
    if len(cont) > 51:
        cont = savgol_filter(cont, 51, 3)
    return cont


def measure_trough(w, fnorm, lam_rest, v_min, v_max):
    """Find deepest absorption inside the expected blueshift window.
    Returns (lam_obs, depth, v_kms) or (None, None, None) if window is too noisy."""
    lam_lo = lam_rest * (1 - v_max/C_KMS)
    lam_hi = lam_rest * (1 - v_min/C_KMS)
    if lam_hi < w.min() or lam_lo > w.max(): return None, None, None
    sel = (w >= lam_lo) & (w <= lam_hi)
    if sel.sum() < 5: return None, None, None
    ws = w[sel]; fs = fnorm[sel]
    # Smooth a tad
    if len(fs) > 11:
        fs_s = savgol_filter(fs, 11, 3)
    else:
        fs_s = fs
    j = int(np.argmin(fs_s))
    lam_obs = ws[j]
    depth = 1.0 - fs_s[j]
    if depth < 0.03:  # noise
        return None, None, None
    v_kms = (lam_rest - lam_obs) / lam_rest * C_KMS
    return lam_obs, depth, v_kms


def measure_peak(w, fnorm, lam_rest, v_min, v_max):
    """Find emission peak inside window. v_min can be negative (redshift allowed)."""
    lam_lo = lam_rest * (1 + v_min/C_KMS)  # for v_min<0 this shifts blueward
    lam_hi = lam_rest * (1 + v_max/C_KMS)
    if lam_hi < w.min() or lam_lo > w.max(): return None, None, None
    sel = (w >= lam_lo) & (w <= lam_hi)
    if sel.sum() < 5: return None, None, None
    ws = w[sel]; fs = fnorm[sel]
    if len(fs) > 11:
        fs_s = savgol_filter(fs, 11, 3)
    else:
        fs_s = fs
    j = int(np.argmax(fs_s))
    lam_obs = ws[j]
    height = fs_s[j] - 1.0
    if height < 0.02:
        return None, None, None
    v_kms = (lam_obs - lam_rest) / lam_rest * C_KMS
    return lam_obs, height, v_kms


def map_to_shell(v_kms, geom):
    v_cm_s = v_kms * 1e5
    m = (geom.v_inner <= v_cm_s) & (geom.v_outer >= v_cm_s)
    if m.any():
        return int(geom[m].shell_id.iloc[0])
    if v_cm_s < geom.v_inner.min(): return -1
    return int(geom.shell_id.max()) + 1


def main():
    w, f, e = load_spectrum()
    print(f"Loaded spectrum: {len(w)} points, {w[0]:.1f}–{w[-1]:.1f} Å")
    cont = pseudo_continuum(w, f, win_A=300.0)
    fnorm = f / cont

    geom = pd.read_csv(REF / "geometry.csv")
    ab = pd.read_csv(REF / "abundances.csv").set_index("atomic_number")
    ab.columns = ab.columns.astype(int)

    rows = []
    for label, Z, ion, lam_rest, v_min, v_max in DIAGNOSTICS:
        symbol = f"{ELEM.get(Z, str(Z))} {'I'*(ion+1) if ion<3 else 'IV'}"
        # trough (absorption)
        if v_min >= 0:
            lam_t, dep_t, v_t = measure_trough(w, fnorm, lam_rest, v_min, v_max)
        else:
            lam_t = None
        lam_p, h_p, v_p = measure_peak(w, fnorm, lam_rest, max(v_min, -3000), 3000)

        row = dict(label=label, species=symbol, Z=Z, ion=ion,
                   lam_rest=lam_rest,
                   abs_lam_obs=round(lam_t, 2) if lam_t else None,
                   abs_depth=round(dep_t, 3) if dep_t else None,
                   abs_v_kms=round(v_t, 0) if v_t else None,
                   em_lam_obs=round(lam_p, 2) if lam_p else None,
                   em_height=round(h_p, 3) if h_p else None,
                   em_v_kms=round(v_p, 0) if v_p else None)

        # Map abs v to shell + check abundance
        if v_t and v_t > 0:
            shell = map_to_shell(v_t, geom)
            if 0 <= shell < ab.shape[1]:
                ab_val = ab.loc[Z, shell] if Z in ab.index else 0.0
                row["abs_required_shell"] = shell
                row["abs_shell_v_kms"] = round(geom.v_inner.iloc[shell]/1e5, 0)
                row["abs_X_in_champion"] = float(ab_val)
                row["abs_status"] = ("PRESENT" if ab_val > 1e-4
                                      else "DEPLETED" if ab_val > 1e-7
                                      else "ABSENT")
            elif shell == -1:
                row["abs_status"] = f"v={v_t:.0f} < v_inner_min=10000"
            else:
                row["abs_status"] = f"v={v_t:.0f} > v_outer_max=25000"
        rows.append(row)

    feat = pd.DataFrame(rows)
    feat.to_csv(OUT_FEAT, index=False)

    # Confident "this ion is needed at this shell" table
    zone_rows = []
    for _, r in feat.iterrows():
        if pd.isna(r.abs_v_kms) or r.abs_v_kms is None: continue
        if r.abs_v_kms <= 0: continue
        shell = r.get("abs_required_shell")
        if pd.isna(shell): continue
        zone_rows.append(dict(
            line=r.label, species=r.species, Z=r.Z, ion=r.ion,
            v_kms=r.abs_v_kms, depth=r.abs_depth,
            required_shell=int(shell),
            shell_v_inner_kms=r.get("abs_shell_v_kms"),
            X_in_champion=r.get("abs_X_in_champion"),
            status=r.get("abs_status"),
        ))
    zone = pd.DataFrame(zone_rows)
    zone.to_csv(OUT_ZONE, index=False)

    print(f"\nWrote {OUT_FEAT}")
    print(f"Wrote {OUT_ZONE}")
    print()
    print("=== Detected diagnostic absorption troughs ===")
    cols = ["label","species","lam_rest","abs_lam_obs","abs_depth",
            "abs_v_kms","abs_required_shell","abs_X_in_champion","abs_status"]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_columns", None):
        det = feat[feat.abs_lam_obs.notna()][cols]
        print(det.to_string(index=False))

    # Group by ion → consensus v
    print("\n=== Consensus by ion (median trough v) ===")
    ion_grp = (feat[feat.abs_v_kms.notna() & (feat.abs_v_kms > 0)]
               .groupby(["species","Z","ion"])
               .agg(n_lines=("label","count"),
                    v_med=("abs_v_kms","median"),
                    v_std=("abs_v_kms","std"),
                    depth_med=("abs_depth","median"))
               .reset_index().sort_values("v_med"))
    print(ion_grp.to_string(index=False))

    # ----- Plot -----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharey=False)
    bands = [(1700, 3200, "UV"), (3200, 5800, "Blue/Optical"), (5800, 8800, "Red")]
    for ax, (lo, hi, name) in zip(axes, bands):
        m = (w >= lo) & (w <= hi)
        ax.plot(w[m], fnorm[m], 'k-', lw=0.6, label=f"HST norm flux ({name})")
        ax.axhline(1.0, color='gray', ls=':', lw=0.5)
        for _, r in feat.iterrows():
            if pd.notna(r.abs_lam_obs) and lo <= r.abs_lam_obs <= hi:
                ax.axvline(r.abs_lam_obs, color='red', ls='--', lw=0.5, alpha=0.7)
                ax.text(r.abs_lam_obs, 0.05 + 0.9*((hash(r.label)%5)/5),
                        f"{r.label}\nv={r.abs_v_kms:.0f}",
                        fontsize=6, color='red', ha='center', alpha=0.85)
            if pd.notna(r.em_lam_obs) and lo <= r.em_lam_obs <= hi:
                ax.axvline(r.em_lam_obs, color='blue', ls=':', lw=0.5, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("F / F_cont")
        ax.set_title(f"{name}: red = absorption trough ID, blue = emission peak")
    plt.tight_layout()
    fig.savefig(FIG_OUT, dpi=130)
    plt.close(fig)
    print(f"\nWrote {FIG_OUT}")


if __name__ == "__main__":
    main()
