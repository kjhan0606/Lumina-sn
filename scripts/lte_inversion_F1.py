#!/usr/bin/env python3
"""(F1) Analytical Sobolev/LTE inversion of HST B-max P-Cygni troughs.

Inputs
------
- HST stitched B-max + pseudo_cont
- 24 diagnostic line measurements (trough λ, depth)
- atomic data (line_list.csv, levels.csv from reference dir)
- t_exp = 17.8 d, distance = 6.4 Mpc (SN 2011fe)

Steps
-----
1. Fit B_λ(T) to pseudo_cont over [4500,7500] Å clean window → T_phot
2. v_phot from Si II 6355 trough → R_phot = v_phot × t_exp
3. Density profile ρ(v) from reference geometry/density (W7-like)
4. Initial T(v) = T_phot × (R_phot/r)^0.5 (radiative cooling)
5. Per-line Sobolev: τ_obs = −ln(1−d), solve for X_Z(v)
6. Refine T(v) from same-ion line ratios:
     - Si II 5972/6355 → T_exc (Branch ratio)
     - Fe II 4924/5018/5169 → Boltzmann between upper levels
7. Plot: T_phot BB fit, T(v), X_Z(v) per element
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ── constants (cgs) ──────────────────────────────────────────────────
H = 6.6260755e-27
KB = 1.380658e-16
C = 2.99792458e10
C_KMS = 299792.458
ME = 9.1093898e-28
EE = 4.80320425e-10  # esu
SIGMA_SB = 5.670400e-5
PI = np.pi
SOBOLEV_PREFAC = PI * EE**2 / (ME * C)  # cm² × Hz, for f-value
MPC = 3.0857e24

# SN 2011fe params
T_EXP_DAY = 17.8
T_EXP = T_EXP_DAY * 86400  # s
DIST = 6.4 * MPC

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
REF  = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"

# ── 24 + blend list (label, Z, ion, λ_rest, v_lo, v_hi) ──────────────
DIAG = [
    ("Fe II 2382",   26, 1, 2382.0, 12000, 20000),
    ("Fe II 2600",   26, 1, 2600.2, 12000, 18000),
    ("Mg II 2796",   12, 1, 2795.5, 12000, 19000),
    ("Mg II 2803",   12, 1, 2802.7, 12000, 19000),
    ("Mn II 2576",   25, 1, 2576.0, 11000, 17000),
    ("Mn II 2594",   25, 1, 2594.0, 11000, 17000),
    ("Ca II K 3934", 20, 1, 3933.7,  9000, 15000),
    ("Ca II H 3968", 20, 1, 3968.5,  9000, 15000),
    ("Fe III 4404",  26, 2, 4404.0, 10000, 17000),
    ("Mg II 4481",   12, 1, 4481.3,  9000, 15000),
    ("Si III 4553",  14, 2, 4552.6, 13000, 19000),
    ("Fe II 4924",   26, 1, 4923.9,  9000, 14000),
    ("Fe II 5018",   26, 1, 5018.4,  9000, 14000),
    ("Fe III 5129",  26, 2, 5129.2, 12000, 17000),
    ("Fe II 5169",   26, 1, 5169.0, 12000, 18000),
    ("S II W 5454",  16, 1, 5454.0,  8000, 13000),
    ("S II W 5640",  16, 1, 5640.0, 13000, 19000),
    ("Si II 5972",   14, 1, 5971.8,  9000, 13000),
    ("Si II 6355",   14, 1, 6355.0,  9000, 13000),
    ("O I 7774",      8, 0, 7773.4,  9000, 13000),
    ("Ca II 8498",   20, 1, 8498.0,  9000, 14000),
    ("Ca II 8542",   20, 1, 8542.0,  9000, 14000),
    ("Ca II 8662",   20, 1, 8662.0,  9000, 14000),
]

ATOM_M = {1:1.008, 8:16.0, 12:24.305, 14:28.086, 16:32.065,
          20:40.078, 22:47.867, 24:51.996, 25:54.938, 26:55.845,
          27:58.933, 28:58.693}
ION_COL = {("Fe", 1):"tab:blue", ("Fe", 2):"tab:cyan",
           ("Mg", 1):"tab:orange", ("Mn", 1):"tab:purple",
           ("Ca", 1):"tab:red", ("Si", 1):"tab:green",
           ("Si", 2):"darkgreen", ("S", 1):"olive",
           ("O", 0):"saddlebrown"}
ELEM_NAMES = {8:"O", 12:"Mg", 14:"Si", 16:"S", 20:"Ca", 25:"Mn", 26:"Fe"}


# ── 1. Pseudo-continuum & BB fit ─────────────────────────────────────
def pseudo_cont(lam, flu, win=300):
    cont = np.zeros_like(flu)
    for i in range(len(lam)):
        sel = (lam >= lam[i]-win/2) & (lam <= lam[i]+win/2)
        cont[i] = np.percentile(flu[sel], 90)
    if len(cont) > 51:
        cont = savgol_filter(cont, 51, 3)
    return cont


def bb_lam(lam_AA, T, scale):
    """Planck B_λ × scale, λ in Å, returns erg/s/cm²/Å (when scale absorbs (R/D)²)."""
    lam_cm = lam_AA * 1e-8
    x = H * C / (lam_cm * KB * T)
    x = np.clip(x, 0, 700)
    B = (2*H*C**2 / lam_cm**5) / (np.expm1(x))
    return scale * B * 1e-8  # convert /cm to /Å


def fit_bb(lam, fcont):
    """Fit BB to clean optical window."""
    m = (lam > 4500) & (lam < 7500)
    try:
        p, _ = curve_fit(bb_lam, lam[m], fcont[m], p0=[10000, 1e-22],
                         bounds=([4000, 1e-30], [30000, 1e-10]),
                         maxfev=5000)
        return p[0], p[1]
    except Exception as e:
        print(f"BB fit failed: {e}")
        return 10000, 1e-22


# ── 2. Trough measurement ────────────────────────────────────────────
def measure_trough(lam, fnorm, lam_rest, v_lo, v_hi):
    lo = lam_rest * (1 - v_hi/C_KMS)
    hi = lam_rest * (1 - v_lo/C_KMS)
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 5: return None
    sub_lam = lam[m]; sub_f = fnorm[m]
    if len(sub_f) > 11:
        fs = savgol_filter(sub_f, 11, 3)
    else:
        fs = sub_f
    j = int(np.argmin(fs))
    return dict(lam_obs=sub_lam[j], depth=1.0-float(fs[j]),
                v_blue=(lam_rest - sub_lam[j])/lam_rest * C_KMS)


# ── 3. Match line in atomic data ─────────────────────────────────────
def find_line(line_list, levels, Z, ion, lam_rest):
    cand = line_list[(line_list.atomic_number == Z) &
                     (line_list.ion_number == ion)]
    if cand.empty: return None
    diff = (cand.wavelength - lam_rest).abs()
    j = diff.idxmin()
    if diff.loc[j] > 5:  # 5Å tolerance
        return None
    row = cand.loc[j]
    lev = levels[(levels.atomic_number == Z) & (levels.ion_number == ion)]
    lev_l = lev[lev.level_number == row.level_number_lower]
    if lev_l.empty: return None
    return dict(lam_AA=float(row.wavelength), f_lu=float(row.f_lu),
                A_ul=float(row.A_ul),
                E_l_eV=float(lev_l.energy_eV.iloc[0]),
                g_l=float(lev_l.g.iloc[0]),
                level_l=int(row.level_number_lower),
                level_u=int(row.level_number_upper))


# ── 4. Sobolev τ ↔ X_Z ───────────────────────────────────────────────
def partition_func(levels, Z, ion, T):
    """Approximate U(T) = Σ g_i × exp(-E_i/kT_e)."""
    lev = levels[(levels.atomic_number == Z) & (levels.ion_number == ion)]
    E = lev.energy_eV.values
    g = lev.g.values
    return float(np.sum(g * np.exp(-E*1.602e-12/(KB*T))))


def tau_to_XZ(tau, lam_AA, f_lu, E_l_eV, g_l, U, T_e, rho, m_atom, t_exp):
    """Solve τ = (πe²/m_e c) f λ n_l t_exp, n_l = X_Z ρ/m × g_l/U × exp(-E_l/kT)."""
    boltz = (g_l / U) * np.exp(-E_l_eV * 1.602e-12 / (KB * T_e))
    if boltz < 1e-30 or rho <= 0:
        return None
    # n_l_per_X = ρ/m_atom × boltz
    n_l_per_X = (rho / (m_atom * 1.6605e-24)) * boltz
    tau_per_X = SOBOLEV_PREFAC * f_lu * (lam_AA * 1e-8) * n_l_per_X * t_exp
    if tau_per_X <= 0:
        return None
    return tau / tau_per_X


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    # Load HST
    h = pd.read_csv(HST)
    hlam = h.iloc[:,0].values; hflu = h.iloc[:,1].values
    m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
    hlam, hflu = hlam[m], hflu[m]
    hcont = pseudo_cont(hlam, hflu)

    # Step 1: BB fit to pseudo-cont
    T_phot, scale = fit_bb(hlam, hcont)
    # Implied (R/D)² from scale: B_lam scale = (R/D)²
    R_over_D = np.sqrt(scale)
    R_phot = R_over_D * DIST
    L_bol = 4 * PI * R_phot**2 * SIGMA_SB * T_phot**4
    print(f"=== Step 1: BB fit to pseudo-continuum ===")
    print(f"  T_phot   = {T_phot:.0f} K")
    print(f"  R_phot   = {R_phot:.3e} cm = {R_phot/1e15:.2f} × 10¹⁵ cm")
    print(f"  L_bol    = {L_bol:.3e} erg/s = {np.log10(L_bol):.2f} log10")

    # Step 2: Measure all troughs
    hnorm = hflu / hcont
    meas = []
    for label, Z, ion, lr, vlo, vhi in DIAG:
        m = measure_trough(hlam, hnorm, lr, vlo, vhi)
        if m is None: continue
        m.update(dict(label=label, Z=Z, ion=ion, lam_rest=lr))
        meas.append(m)
    print(f"\n=== Step 2: trough measurements ({len(meas)}) ===")

    # Step 3: load reference geometry/density + atomic
    geom = pd.read_csv(REF / "geometry.csv")
    dens = pd.read_csv(REF / "density.csv")
    line_list = pd.read_csv(REF / "line_list.csv")
    levels = pd.read_csv(REF / "levels.csv")
    print(f"  Geometry: {len(geom)} shells, v_inner = {geom.v_inner.iloc[0]/1e5:.0f}-"
          f"{geom.v_outer.iloc[-1]/1e5:.0f} km/s")

    # Step 4: photospheric anchors
    si6355 = next((x for x in meas if x["label"] == "Si II 6355"), None)
    if si6355:
        v_phot = si6355["v_blue"]
        R_phot_v = v_phot * 1e5 * T_EXP
        print(f"\n  v_phot (Si II 6355)  = {v_phot:.0f} km/s")
        print(f"  R_phot (= v_phot·t)  = {R_phot_v:.3e} cm = {R_phot_v/1e15:.2f} × 10¹⁵ cm")
        print(f"  R_phot (BB)/R_phot(v) = {R_phot/R_phot_v:.2f}")

    # Step 5: density profile interpolated at each v
    v_grid = geom.v_inner.values  # cm/s
    rho_grid = dens.iloc[:, 1].values if dens.shape[1]>=2 else None
    if rho_grid is None:
        # Try density.iloc[:, -1]
        rho_grid = dens.iloc[:, -1].values
    print(f"  Density grid: ρ(inner)={rho_grid[0]:.2e}, ρ(outer)={rho_grid[-1]:.2e} g/cm³")

    def rho_at(v_kms):
        v_cms = v_kms * 1e5
        return float(np.interp(v_cms, v_grid, rho_grid))

    def T_at(v_kms):
        """Initial T(v) = T_phot × (v_phot/v)^0.5."""
        v_cms = v_kms * 1e5
        v_phot_cms = v_phot * 1e5 if si6355 else 1e9
        return T_phot * np.sqrt(v_phot_cms / max(v_cms, 1e8))

    # Step 6: per-line Sobolev inversion
    print(f"\n=== Step 3: Sobolev inversion (LTE, T(v)=T_phot×√(v_phot/v)) ===")
    print(f"  {'line':<14} {'Z/ion':>5} {'v(km/s)':>8} {'T(v) K':>8} "
          f"{'depth':>6} {'τ':>6} {'ρ(v) g/cc':>11} {'X_Z':>9}")
    inversion = []
    for x in meas:
        atom = find_line(line_list, levels, x["Z"], x["ion"], x["lam_rest"])
        if atom is None:
            continue
        v = x["v_blue"]
        T = T_at(v)
        rho = rho_at(v)
        m_atom = ATOM_M.get(x["Z"], None)
        if m_atom is None: continue
        tau = -np.log(max(1.0 - x["depth"], 1e-3))
        U = partition_func(levels, x["Z"], x["ion"], T)
        XZ = tau_to_XZ(tau, atom["lam_AA"], atom["f_lu"], atom["E_l_eV"],
                       atom["g_l"], U, T, rho, m_atom, T_EXP)
        if XZ is None or XZ <= 0:
            continue
        XZ = min(XZ, 5.0)  # clip unphysical (saturation regime)
        x.update(dict(T=T, rho=rho, tau=tau, XZ=XZ, atom=atom))
        inversion.append(x)
        print(f"  {x['label']:<14} {x['Z']:>2}/{x['ion']:<2} {v:>8.0f} {T:>8.0f} "
              f"{x['depth']:>6.3f} {tau:>6.2f} {rho:>11.2e} {XZ:>9.2e}")

    # Step 7: T_exc from line ratios where multiple lines per ion
    # Si II 5972/6355 (Branch ratio)
    print(f"\n=== Step 4: T from line ratios ===")
    si_lines = [x for x in inversion if x["Z"]==14 and x["ion"]==1]
    if len(si_lines) >= 2:
        d5972 = next((x["depth"] for x in si_lines if "5972" in x["label"]), None)
        d6355 = next((x["depth"] for x in si_lines if "6355" in x["label"]), None)
        if d5972 and d6355:
            R = d5972 / d6355
            print(f"  Si II 5972/6355 depth ratio = {R:.3f}  (Branch low → hot, high → cool)")

    # Fe II 4924/5018/5169 (same ion, different upper levels)
    fe2 = [x for x in inversion if x["Z"]==26 and x["ion"]==1
           and any(w in x["label"] for w in ["4924","5018","5169"])]
    if len(fe2) >= 2:
        print(f"  Fe II Boltzmann excitation (upper levels):")
        for x in fe2:
            print(f"    {x['label']:<12} d={x['depth']:.3f} τ={x['tau']:.2f} X_Fe={x['XZ']:.2e}")

    # Fe II/Fe III at similar v (Saha-like)
    fe23 = [x for x in inversion if x["Z"]==26]
    print(f"  Fe II vs Fe III pairs (similar v → ionization):")
    for x2 in [y for y in fe23 if y["ion"]==1]:
        for x3 in [y for y in fe23 if y["ion"]==2]:
            if abs(x2["v_blue"] - x3["v_blue"]) < 1500:
                print(f"    {x2['label']:<12} (v={x2['v_blue']:.0f}) X_Fe⁺={x2['XZ']:.2e}  vs  "
                      f"{x3['label']:<12} (v={x3['v_blue']:.0f}) X_Fe²⁺={x3['XZ']:.2e}")

    # ── Plot: 4 panels ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # (a) BB fit on pseudo-cont
    ax = fig.add_subplot(gs[0,0])
    sel = (hlam > 1700) & (hlam < 9000)
    ax.plot(hlam[sel], hflu[sel]*1e14, "k-", lw=0.8, alpha=0.7, label="HST B-max")
    ax.plot(hlam[sel], hcont[sel]*1e14, "C0-", lw=1.2, label="pseudo-cont")
    bb = bb_lam(hlam[sel], T_phot, scale)
    ax.plot(hlam[sel], bb*1e14, "r--", lw=1.2,
            label=f"BB fit: T={T_phot:.0f}K\nR={R_phot/1e15:.2f}×10¹⁵ cm\nL={L_bol:.2e}")
    ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$)")
    ax.set_title("Step 1: 의사 연속체 → 광구 BB 온도")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) v vs depth (and T overlay)
    ax = fig.add_subplot(gs[0,1])
    for x in inversion:
        key = (ELEM_NAMES.get(x["Z"], f"Z{x['Z']}"), x["ion"])
        c = ION_COL.get(key, "tab:gray")
        ax.scatter(x["v_blue"], x["depth"], color=c, s=60,
                   edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(x["label"], (x["v_blue"], x["depth"]),
                    fontsize=6.5, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("v_blueshift (km/s)"); ax.set_ylabel("trough depth")
    ax.set_title("Step 2: 라인 v − depth 분포 (광구→외곽)")
    ax.axvline(v_phot, color="black", ls=":", alpha=0.5, label=f"v_phot={v_phot:.0f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) T(v) profile
    ax = fig.add_subplot(gs[1,0])
    vv = np.linspace(geom.v_inner.iloc[0]/1e5, geom.v_outer.iloc[-1]/1e5, 200)
    Tv = np.array([T_at(v) for v in vv])
    ax.plot(vv, Tv, "k-", lw=1.5, label=f"T(v) = {T_phot:.0f}×√(v_phot/v)")
    ax.axhline(T_phot, color="red", ls="--", alpha=0.5, label=f"T_phot={T_phot:.0f}")
    ax.set_xlabel("v (km/s)"); ax.set_ylabel("T (K)")
    ax.set_title("Step 3: T(v) 초기 추정 (Sobolev cooling 가정)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) X_Z(v) per element
    ax = fig.add_subplot(gs[1,1])
    seen_labels = set()
    by_elem = {}
    for x in inversion:
        Z = x["Z"]; ion = x["ion"]
        by_elem.setdefault((Z, ion), []).append(x)
    for (Z, ion), pts in sorted(by_elem.items()):
        elem = ELEM_NAMES.get(Z, f"Z{Z}")
        c = ION_COL.get((elem, ion), "tab:gray")
        vs = [p["v_blue"] for p in pts]
        xs = [p["XZ"] for p in pts]
        ax.scatter(vs, xs, color=c, s=80, edgecolor="black", lw=0.5,
                   label=f"{elem} {'I' if ion==0 else 'II' if ion==1 else 'III'}")
    ax.set_yscale("log")
    ax.set_xlabel("v (km/s)"); ax.set_ylabel("X_Z (mass fraction, 1차 추정)")
    ax.set_title("Step 4: X_Z(v) Sobolev/LTE 1차 inversion")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("(F1) HST P-Cygni → 광구/온도/원소 프로파일 1차 분석적 inversion\n"
                 "Sobolev + LTE Boltzmann/Saha 가정. NLTE/blend/saturation 보정 없음.",
                 fontsize=11)
    out = ROOT / "figures/lte_inversion_F1.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
