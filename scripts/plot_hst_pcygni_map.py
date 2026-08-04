#!/usr/bin/env python3
"""HST SN 2011fe B-max 위에 표준 P-Cygni 진단선 24개를 다음과 같이 표시:
   - 정지 파장 (verticle dashed) — rest λ
   - 흡수 trough 관측 위치 (verticle solid) — Doppler-shifted
   - 라인 라벨 + 측정 물리량: v_blueshift, depth, FWHM_v
   - depth = 1 - f/f_cont @ trough min
   - FWHM_v = (FWHM_λ / λ_rest) × c [km/s] (bulk velocity dispersion)
   - 열 도플러: T_e=10⁴K @ Fe → v_th=1.7 km/s ≪ bulk, 따라서 FWHM은 거의 전적으로 bulk
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

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP= ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"
C_KMS = 299792.458

# ---- single Fe II line P-Cygni overlay ----
SINGLE_LINE_LAM_REST = 5169.03      # Fe II 5169 (Multiplet 42)
SINGLE_LINE_LABEL    = "Fe II 5169"
V_PHOT = 11000.0   # km/s
V_MAX  = 22000.0   # km/s
TAU_OVERLAY = [2.0, 8.0]            # representative thin / thick τ
# 파장-비례 (Doppler-scaled) smoothing window
# Δλ_FWHM = (v_exp/c) × λ;  smoothing window = scale × Δλ_FWHM
# v_exp=10⁴ km/s, scale=2 → 200 Å @3000Å, 400 Å @6000Å, 600 Å @9000Å
V_SMOOTH    = 10000.0               # km/s — characteristic explosion velocity
PCONT_SCALE = 2.0                   # window / Doppler width ratio


def pcygni_homologous(lam, lam_rest, v_phot, v_max, tau0, F_c, n_p=400):
    """Sobolev P-Cygni in homologous expansion v(r)=(r/R_phot) v_phot.
    z_res in R_phot units; positive=blueshift. Pure scattering, S = W(r)·F_c.
    """
    F_obs   = np.zeros_like(F_c)
    p_max   = v_max / v_phot
    p_grid  = np.linspace(1e-6, p_max, n_p)
    dp      = p_grid[1] - p_grid[0]
    w_p     = 2.0 * np.pi * p_grid * dp
    A_total = np.pi
    for i, l in enumerate(lam):
        z_res = (lam_rest - l) / lam_rest * (C_KMS / v_phot)
        r_res = np.sqrt(p_grid**2 + z_res**2)
        in_atm     = (r_res >= 1.0) & (r_res <= p_max)
        in_core    = p_grid < 1.0
        z_phot_fr  = np.sqrt(np.maximum(0.0, 1.0 - p_grid**2))
        absorbed   = in_atm & in_core & (z_res > -z_phot_fr)
        emit_only  = in_atm & ~in_core
        clear_core = in_core & ~absorbed
        W = np.zeros_like(p_grid)
        W[in_atm] = 0.5 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (r_res[in_atm]**2))))
        S = F_c[i] * W
        I = np.zeros_like(p_grid)
        I[clear_core] = F_c[i]
        I[absorbed]   = F_c[i] * np.exp(-tau0) + S[absorbed]  * (1.0 - np.exp(-tau0))
        I[emit_only]  = S[emit_only] * (1.0 - np.exp(-tau0))
        F_obs[i] = np.sum(w_p * I) / A_total
    return F_obs

# BLEND features — multi-ion / multi-multiplet, 진단성은 떨어지지만 opacity 추적에 유용
# (label, λ_obs_target, v_typical for context)
BLENDS = [
    ("Fe II UV6 + Ti II",      3100, 3210),  # rest ~3200
    ("Fe II UV2 + Ni II",      3300, 3450),  # rest ~3450
    ("Si II 4128/4131 + Fe II",4000, 4135),  # rest ~4135
    ("Fe II m37/38 (W blend)", 4450, 4598),  # rest ~4600
    ("S II W valley + Fe II",  5400, 5535),  # rest ~5535 or W-shape valley
]


# (label, λ_rest, v_lo, v_hi, color_band)
# v window: 표준 SN Ia photospheric 10-25k km/s 외에는 cherry-picked
DIAG = [
    # UV iron forest
    ("Fe II 2382", 2382.0, 12000, 20000, "uv"),
    ("Fe II 2600", 2600.2, 12000, 18000, "uv"),
    ("Mg II 2796", 2795.5, 12000, 19000, "uv"),
    ("Mg II 2803", 2802.7, 12000, 19000, "uv"),
    ("Mn II 2576", 2576.0, 11000, 17000, "uv"),
    ("Mn II 2594", 2594.0, 11000, 17000, "uv"),
    ("Co/Fe III 3070", 3070.0, 10000, 17000, "uv"),
    # blue
    ("Ca II K 3934", 3933.7, 9000, 15000, "blue"),
    ("Ca II H 3968", 3968.5, 9000, 15000, "blue"),
    ("Fe III 4404", 4404.0, 10000, 17000, "blue"),
    ("Mg II 4481", 4481.3, 9000, 15000, "blue"),
    ("Si III 4553", 4552.6, 13000, 19000, "blue"),
    # green/yellow Fe-multiplet
    ("Fe II 4924", 4923.9, 9000, 14000, "grn"),
    ("Fe II 5018", 5018.4, 9000, 14000, "grn"),
    ("Fe III 5129", 5129.2, 12000, 17000, "grn"),
    ("Fe II 5169", 5169.0, 12000, 18000, "grn"),
    ("S II W 5454", 5454.0, 8000, 13000, "grn"),
    ("S II W 5640", 5640.0, 13000, 19000, "grn"),
    ("Si II 5972", 5971.8, 9000, 13000, "grn"),
    # red
    ("Si II 6355", 6355.0, 9000, 13000, "red"),
    ("O I 7774", 7773.4, 9000, 13000, "red"),
    ("Ca II 8498", 8498.0, 9000, 14000, "red"),
    ("Ca II 8542", 8542.0, 9000, 14000, "red"),
    ("Ca II 8662", 8662.0, 9000, 14000, "red"),
]

ION_COLOR = {
    "Fe II":  "tab:blue",
    "Fe III": "tab:cyan",
    "Mg II":  "tab:orange",
    "Mn II":  "tab:purple",
    "Co/Fe III": "tab:gray",
    "Ca II":  "tab:red",
    "Ca II K":"tab:red",
    "Ca II H":"tab:red",
    "Si II":  "tab:green",
    "Si III": "darkgreen",
    "S II W": "olive",
    "O I":    "saddlebrown",
}


def ion_of(label):
    for k in ION_COLOR:
        if label.startswith(k): return k
    return "Fe II"


def pseudo_cont(lam, flu, v_smooth=V_SMOOTH, scale=PCONT_SCALE):
    """Wavelength-dependent Gaussian smoothing.
    FWHM(λ) = scale·(v_smooth/c)·λ ;  σ = FWHM / 2.355
    """
    cont = np.zeros_like(flu)
    beta = v_smooth / C_KMS
    for i in range(len(lam)):
        sigma = scale * beta * lam[i] / 2.3548
        win = 4.0 * sigma                    # truncate kernel at ±4σ
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2:
            cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def measure(lam, fnorm, lam_rest, v_lo, v_hi):
    """trough λ_obs, depth, FWHM_kms"""
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
    lam_obs = sub_lam[j]
    fmin = float(fs[j])
    depth = 1.0 - fmin
    v_blue = (lam_rest - lam_obs)/lam_rest * C_KMS
    # FWHM: half-depth = 1 - depth/2
    half = 1.0 - depth/2.0
    # widen search to ±400 km/s of trough for FWHM
    wlo = lam_obs * (1 - 12000/C_KMS)
    whi = lam_obs * (1 + 12000/C_KMS)
    mw = (lam >= wlo) & (lam <= whi)
    if mw.sum() < 11:
        fwhm_kms = None
    else:
        wlam = lam[mw]; wf = fnorm[mw]
        if len(wf) > 11: wf = savgol_filter(wf, 11, 3)
        # Find left/right crossings of half level around trough
        idx_min = int(np.argmin(np.abs(wlam - lam_obs)))
        # left
        left_lam = None
        for k in range(idx_min, 0, -1):
            if wf[k] >= half:
                left_lam = wlam[k]; break
        right_lam = None
        for k in range(idx_min, len(wf)):
            if wf[k] >= half:
                right_lam = wlam[k]; break
        if left_lam is not None and right_lam is not None:
            fwhm_lam = right_lam - left_lam
            fwhm_kms = fwhm_lam / lam_rest * C_KMS
        else:
            fwhm_kms = None
    return dict(lam_obs=lam_obs, depth=depth, v_blue=v_blue, fwhm_kms=fwhm_kms,
                fmin=fmin)


# Load HST
h = pd.read_csv(HST)
hlam = h.iloc[:,0].values; hflu = h.iloc[:,1].values
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]
hcont = pseudo_cont(hlam, hflu)
hnorm = hflu / hcont

# ---- LUMINA pseudo-continuum + single Fe II line P-Cygni overlay ----
print(f"Loading LUMINA champion for pseudo-continuum: {CHAMP.name}")
ldf = pd.read_csv(CHAMP); llam = ldf.iloc[:,0].values
lcol = "flux" if "flux" in ldf.columns else ldf.columns[1]
lflu = ldf[lcol].values
# normalize champion to HST 4500-5800 band
band_h = np.trapezoid(hflu[(hlam>=4500)&(hlam<=5800)], hlam[(hlam>=4500)&(hlam<=5800)])
band_l = np.trapezoid(lflu[(llam>=4500)&(llam<=5800)], llam[(llam>=4500)&(llam<=5800)])
lflu *= band_h / band_l
F_c_lumina = pseudo_cont(llam, lflu)       # 300 Å 90-percentile envelope (same as HST)
print(f"Computing Fe II 5169 P-Cygni at τ ∈ {TAU_OVERLAY} (v_phot={V_PHOT:.0f}, v_max={V_MAX:.0f} km/s)...")
F_pcygni = {tau: pcygni_homologous(llam, SINGLE_LINE_LAM_REST, V_PHOT, V_MAX, tau, F_c_lumina)
            for tau in TAU_OVERLAY}

# Measure
measurements = []
for label, lr, vlo, vhi, band in DIAG:
    m = measure(hlam, hnorm, lr, vlo, vhi)
    if m is None: continue
    m["label"] = label; m["lam_rest"] = lr; m["band"] = band
    m["ion"] = ion_of(label)
    measurements.append(m)

# Print summary table
print(f"=== HST SN 2011fe B-max 진단선 측정 ({len(measurements)}/24) ===")
print(f"{'line':<14} {'λ_rest':>7} {'λ_obs':>7} {'v_blue':>7} {'depth':>6} {'FWHM_v':>8}")
for x in measurements:
    fw = f"{x['fwhm_kms']:6.0f}" if x['fwhm_kms'] else "  --"
    print(f"  {x['label']:<12} {x['lam_rest']:>7.1f} {x['lam_obs']:>7.1f} "
          f"{x['v_blue']:>7.0f} {x['depth']:>6.3f} {fw:>8}")

# === Plot ===
fig, axes = plt.subplots(4, 1, figsize=(17, 18))
panels = [(1700, 3200, "UV iron forest (Fe II / Mg II / Mn II / Co)"),
          (3200, 4800, "Blue (Ca II H&K, Fe III, Mg II 4481, Si III)"),
          (4800, 6800, "Green/Yellow (Fe II 4924/5018/5169, Fe III 5129, S II W, Si II)"),
          (6800, 9000, "Red/NIR (Si II 6355, O I 7774, Ca II IR triplet)")]

for ax, (lo, hi, title) in zip(axes, panels):
    sel = (hlam >= lo) & (hlam <= hi)
    ax.plot(hlam[sel], hflu[sel]*1e14, "k-", lw=0.9, alpha=0.85, label="HST B-max")
    ax.plot(hlam[sel], hcont[sel]*1e14, "-", color="darkorange", lw=1.6, alpha=0.95,
            label=f"HST smoothed (Gaussian FWHM={PCONT_SCALE:.0f}·(v/c)·λ, v={V_SMOOTH/1e3:.0f}k)")

    # ---- LUMINA pseudo-continuum + only Fe II 5169 line ----
    sel_l = (llam >= lo) & (llam <= hi)
    ax.plot(llam[sel_l], F_c_lumina[sel_l]*1e14, color="C0", ls=(0, (4, 2)), lw=1.3, alpha=0.85,
            label=f"LUMINA smoothed (Gaussian FWHM={PCONT_SCALE:.0f}·(v/c)·λ, champ 152761)")
    overlay_colors = {2.0: "tab:green", 8.0: "tab:red"}
    for tau in TAU_OVERLAY:
        ax.plot(llam[sel_l], F_pcygni[tau][sel_l]*1e14,
                color=overlay_colors[tau], lw=1.2, alpha=0.9,
                label=f"+ {SINGLE_LINE_LABEL} only (τ={tau})")
    # Mark Fe II 5169 rest λ + photospheric blueshift bound
    if lo <= SINGLE_LINE_LAM_REST <= hi:
        ax.axvline(SINGLE_LINE_LAM_REST, color="black", ls=":", lw=0.8, alpha=0.6)
    l_blue_phot = SINGLE_LINE_LAM_REST * (1 - V_PHOT/C_KMS)
    if lo <= l_blue_phot <= hi:
        ax.axvline(l_blue_phot, color="tab:red", ls=(0,(1,2)), lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("λ (Å)")
    ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # leave headroom for labels
    f_max = (hflu[sel]*1e14).max()
    ax.set_ylim(0, f_max * 1.55)

    in_band = sorted([x for x in measurements
                      if lo <= x["lam_rest"] <= hi or lo <= x["lam_obs"] <= hi],
                     key=lambda x: x["lam_obs"])

    # 5-level vertical stagger to avoid label collisions
    levels = [1.50, 1.38, 1.26, 1.14, 1.02]
    for k, x in enumerate(in_band):
        c = ION_COLOR[x["ion"]]
        # observed trough vertical line (solid colored)
        ax.axvline(x["lam_obs"], color=c, ls="-", lw=1.2, alpha=0.85)
        # rest λ (dotted) — only for visible separation
        if abs(x["lam_obs"] - x["lam_rest"]) > 5:
            ax.axvline(x["lam_rest"], color=c, ls=":", lw=0.6, alpha=0.4)
            # Doppler arrow rest -> obs at low y
            ax.annotate("", xy=(x["lam_obs"], f_max*0.05),
                        xytext=(x["lam_rest"], f_max*0.05),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.6))

        # Label at staggered y level
        y_lab = f_max * levels[k % len(levels)]
        # find spectrum y at trough for connector start
        idx = int(np.argmin(np.abs(hlam - x["lam_obs"])))
        y_anchor = hflu[idx]*1e14
        fwhm_str = f"\nFWHM={x['fwhm_kms']:.0f} km/s" if x['fwhm_kms'] else ""
        ax.annotate(
            f"{x['label']}\nv={x['v_blue']:.0f} km/s\nd={x['depth']:.2f}{fwhm_str}",
            xy=(x["lam_obs"], y_anchor),
            xytext=(x["lam_obs"], y_lab),
            fontsize=7.5, color=c, ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=0.5, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.7))
    # blend features in this band — gray dotted lines + small italic label at very top
    for label, lam_obs_blend, lam_rest_approx in BLENDS:
        if not (lo <= lam_obs_blend <= hi):
            continue
        ax.axvline(lam_obs_blend, color="dimgray", ls=(0,(2,2)), lw=0.9, alpha=0.6)
        # Doppler arrow at very low y for blend
        if abs(lam_obs_blend - lam_rest_approx) > 5:
            ax.annotate("", xy=(lam_obs_blend, f_max*0.02),
                        xytext=(lam_rest_approx, f_max*0.02),
                        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7, alpha=0.5))
        v_blend = (lam_rest_approx - lam_obs_blend) / lam_rest_approx * C_KMS
        ax.text(lam_obs_blend, f_max*0.92,
                f"[blend]\n{label}\nλ_obs≈{lam_obs_blend}\nv≈{v_blend:.0f}",
                fontsize=6.8, color="dimgray", ha="center", va="top",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow", ec="dimgray", lw=0.4, alpha=0.8))

    ax.legend(loc="upper right", fontsize=9)

# Overall caption
fig.suptitle("SN 2011fe HST B-max — P-Cygni 흡수 trough 진단 맵\n"
             "(실선=관측 trough λ, 점선=정지 λ, 화살표=Doppler 청색편이; "
             "v=blueshift 속도, d=trough 깊이 (1−f/f_cont), FWHM=속도분산\n"
             "열 도플러 @T_e=10⁴K Fe = 1.7 km/s ≪ FWHM 5000-13000 km/s 이므로 폭은 거의 전적으로 bulk 팽창)",
             fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "figures/hst_diagnostic_pcygni_map.png"
plt.savefig(out, dpi=140); plt.close()
print(f"\nWrote {out}")
