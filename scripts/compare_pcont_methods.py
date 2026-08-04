#!/usr/bin/env python3
"""4가지 pseudo-continuum 방법 비교 — HST SN 2011fe B-max.

(1) 90-pctl envelope, Doppler 윈도우  — 흡수 trough 전용 (방출 무시)
(2) Gaussian smoothing,   Doppler 윈도우  — 데이터 평균 추세
(3) Wide-window median 30,000 km/s        — pure scattering 광자수 보존 → 대칭
(4) BSNIP-II anchor 선형 보간             — line-free λ에서 anchor 잡고 직선
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
OUT  = ROOT / "figures/pcont_method_comparison.png"
C_KMS = 299792.458

V_TROUGH    = 10000.0   # km/s — Doppler scaling for trough-tracking methods (1)(2)
V_SYM       = 30000.0   # km/s — wide window for symmetric median (3) covers full P-Cygni
SCALE       = 2.0

# BSNIP-II 식 anchor — major SN Ia line 사이의 (대체로) line-free pivot
ANCHORS = np.array([
    1850, 2300, 2700, 3050, 3300,    # UV iron forest 사이
    3850,                             # Ca II HK 직후
    4070, 4540, 4640,                 # Si II "W" 양 옆 + Mg II 4481 옆
    4760, 5500,                       # Fe complex 양 옆
    5400, 5700,                       # S II W 양 옆
    5790, 6450,                       # Si II 5972/6355 양 옆
    6800, 7080, 7600,                 # O I 7774 양 옆
    8050, 8800                        # Ca II IR triplet 양 옆
], dtype=float)
ANCHOR_HALF = 25.0   # ±25 Å median


def pcont_envelope(lam, flu, v=V_TROUGH, scale=SCALE, pctl=90):
    cont = np.zeros_like(flu); beta = v / C_KMS
    for i in range(len(lam)):
        win = scale * beta * lam[i]
        sel = (lam >= lam[i]-win/2) & (lam <= lam[i]+win/2)
        cont[i] = np.percentile(flu[sel], pctl)
    return cont


def pcont_gaussian(lam, flu, v=V_TROUGH, scale=SCALE):
    cont = np.zeros_like(flu); beta = v / C_KMS
    for i in range(len(lam)):
        sigma = scale * beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def pcont_wide_median(lam, flu, v=V_SYM):
    """Wide-window median — pure scattering 광자수 보존을 이용한 대칭 추정."""
    cont = np.zeros_like(flu); beta = v / C_KMS
    for i in range(len(lam)):
        win = beta * lam[i]   # full window = (v/c)·λ (=1000 Å @6000Å for v=30k)
        sel = (lam >= lam[i]-win/2) & (lam <= lam[i]+win/2)
        if sel.sum() < 3: cont[i] = flu[i]; continue
        cont[i] = np.median(flu[sel])
    return cont


def pcont_anchor(lam, flu, anchors=ANCHORS, half=ANCHOR_HALF, pctl=80):
    """BSNIP-II 식: line-free anchor에서 high-flux를 잡고 선형 보간."""
    a_lam, a_flu = [], []
    for a in anchors:
        if a < lam.min() or a > lam.max(): continue
        sel = (lam >= a-half) & (lam <= a+half)
        if sel.sum() < 3: continue
        a_lam.append(a)
        a_flu.append(np.percentile(flu[sel], pctl))
    a_lam, a_flu = np.array(a_lam), np.array(a_flu)
    return np.interp(lam, a_lam, a_flu), a_lam, a_flu


def load(p):
    df = pd.read_csv(p); lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values


print("Loading HST...")
hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

print("(1) 90-pctl envelope (Doppler win, v=10k)...")
c_env = pcont_envelope(hlam, hflu)
print("(2) Gaussian smooth (Doppler win, v=10k)...")
c_gau = pcont_gaussian(hlam, hflu)
print("(3) Wide-window median (v=30k)...")
c_med = pcont_wide_median(hlam, hflu)
print("(4) BSNIP-II anchor interp (80-pctl in ±25Å)...")
c_anc, a_lam, a_flu = pcont_anchor(hlam, hflu)

# ---- Plot ----
fig = plt.figure(figsize=(16, 13))
gs  = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.0, 1.0],
                       hspace=0.30, wspace=0.18)

# (a) 전체 overlay
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu*1e14, "k-", lw=0.7, alpha=0.65, label="HST B-max")
ax.plot(hlam, c_env*1e14, color="tab:red",    ls="--", lw=1.4, alpha=0.9,
        label="(1) 90-pctl envelope (Doppler win, v=10k) — trough only")
ax.plot(hlam, c_gau*1e14, color="tab:blue",   ls="-",  lw=1.4, alpha=0.9,
        label="(2) Gaussian smooth (Doppler win, v=10k) — average trend")
ax.plot(hlam, c_med*1e14, color="tab:green",  ls="-",  lw=1.6, alpha=0.95,
        label="(3) Wide-window median (v=30k) — symmetric scatter recovery")
ax.plot(hlam, c_anc*1e14, color="tab:orange", ls="-",  lw=1.6, alpha=0.95,
        label="(4) BSNIP-II anchor interp — line-free pivots")
ax.scatter(a_lam, a_flu*1e14, c="tab:orange", marker="o", s=35,
           edgecolor="black", lw=0.7, zorder=5, label="(4) anchors")
ax.set_xlim(1700, 9000)
ax.set_ylim(0, (hflu*1e14).max() * 1.15)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title("HST SN 2011fe B-max — 4가지 pseudo-continuum 방법 비교 (전체)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)

# (b) Si II 6355 zoom — 흡수 trough + 방출 peak 모두 보임
axb = fig.add_subplot(gs[1, 0])
m_b = (hlam >= 5500) & (hlam <= 7000)
axb.plot(hlam[m_b], hflu[m_b]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
axb.plot(hlam[m_b], c_env[m_b]*1e14, color="tab:red",    ls="--", lw=1.4)
axb.plot(hlam[m_b], c_gau[m_b]*1e14, color="tab:blue",   ls="-",  lw=1.4)
axb.plot(hlam[m_b], c_med[m_b]*1e14, color="tab:green",  ls="-",  lw=1.7)
axb.plot(hlam[m_b], c_anc[m_b]*1e14, color="tab:orange", ls="-",  lw=1.7)
sel_a = (a_lam >= 5500) & (a_lam <= 7000)
axb.scatter(a_lam[sel_a], a_flu[sel_a]*1e14, c="tab:orange",
            marker="o", s=45, edgecolor="black", lw=0.7, zorder=5)
axb.axvline(6355*(1-10000/C_KMS), color="gray", ls=":", lw=0.7, alpha=0.7)
axb.text(6355*(1-10000/C_KMS), axb.get_ylim()[1]*0.9, "Si II 6355\ntrough",
         fontsize=8, color="dimgray", ha="center")
axb.axvline(6355, color="gray", ls=":", lw=0.7, alpha=0.5)
axb.text(6355, axb.get_ylim()[1]*0.9, "rest λ\n→ emission peak side",
         fontsize=8, color="dimgray", ha="left")
axb.set_xlim(5500, 7000)
axb.set_xlabel("λ (Å)"); axb.set_ylabel(r"Flux (10$^{-14}$)")
axb.set_title("(b) Si II 6355 zoom — 방출 peak에서 envelope/smooth/median이 어떻게 갈리나",
              fontsize=10, fontweight="bold")
axb.grid(alpha=0.3)

# (c) Ca II HK zoom — 강한 흡수 trough 단독
axc = fig.add_subplot(gs[1, 1])
m_c = (hlam >= 3500) & (hlam <= 4200)
axc.plot(hlam[m_c], hflu[m_c]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
axc.plot(hlam[m_c], c_env[m_c]*1e14, color="tab:red",    ls="--", lw=1.4)
axc.plot(hlam[m_c], c_gau[m_c]*1e14, color="tab:blue",   ls="-",  lw=1.4)
axc.plot(hlam[m_c], c_med[m_c]*1e14, color="tab:green",  ls="-",  lw=1.7)
axc.plot(hlam[m_c], c_anc[m_c]*1e14, color="tab:orange", ls="-",  lw=1.7)
sel_a = (a_lam >= 3500) & (a_lam <= 4200)
axc.scatter(a_lam[sel_a], a_flu[sel_a]*1e14, c="tab:orange",
            marker="o", s=45, edgecolor="black", lw=0.7, zorder=5)
axc.set_xlim(3500, 4200)
axc.set_xlabel("λ (Å)"); axc.set_ylabel(r"Flux (10$^{-14}$)")
axc.set_title("(c) Ca II H&K zoom — 깊은 흡수 trough", fontsize=10, fontweight="bold")
axc.grid(alpha=0.3)

# (d) UV iron forest zoom
axd = fig.add_subplot(gs[2, 0])
m_d = (hlam >= 2200) & (hlam <= 3200)
axd.plot(hlam[m_d], hflu[m_d]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
axd.plot(hlam[m_d], c_env[m_d]*1e14, color="tab:red",    ls="--", lw=1.4)
axd.plot(hlam[m_d], c_gau[m_d]*1e14, color="tab:blue",   ls="-",  lw=1.4)
axd.plot(hlam[m_d], c_med[m_d]*1e14, color="tab:green",  ls="-",  lw=1.7)
axd.plot(hlam[m_d], c_anc[m_d]*1e14, color="tab:orange", ls="-",  lw=1.7)
sel_a = (a_lam >= 2200) & (a_lam <= 3200)
axd.scatter(a_lam[sel_a], a_flu[sel_a]*1e14, c="tab:orange",
            marker="o", s=45, edgecolor="black", lw=0.7, zorder=5)
axd.set_xlim(2200, 3200)
axd.set_xlabel("λ (Å)"); axd.set_ylabel(r"Flux (10$^{-14}$)")
axd.set_title("(d) UV iron forest — 거의 모든 pixel 흡수, envelope 가정 깨짐",
              fontsize=10, fontweight="bold")
axd.grid(alpha=0.3)

# (e) Ca II IR triplet zoom
axe = fig.add_subplot(gs[2, 1])
m_e = (hlam >= 7800) & (hlam <= 9000)
axe.plot(hlam[m_e], hflu[m_e]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
axe.plot(hlam[m_e], c_env[m_e]*1e14, color="tab:red",    ls="--", lw=1.4)
axe.plot(hlam[m_e], c_gau[m_e]*1e14, color="tab:blue",   ls="-",  lw=1.4)
axe.plot(hlam[m_e], c_med[m_e]*1e14, color="tab:green",  ls="-",  lw=1.7)
axe.plot(hlam[m_e], c_anc[m_e]*1e14, color="tab:orange", ls="-",  lw=1.7)
sel_a = (a_lam >= 7800) & (a_lam <= 9000)
axe.scatter(a_lam[sel_a], a_flu[sel_a]*1e14, c="tab:orange",
            marker="o", s=45, edgecolor="black", lw=0.7, zorder=5)
axe.set_xlim(7800, 9000)
axe.set_xlabel("λ (Å)"); axe.set_ylabel(r"Flux (10$^{-14}$)")
axe.set_title("(e) Ca II IR triplet zoom — 적색 P-Cygni",
              fontsize=10, fontweight="bold")
axe.grid(alpha=0.3)

fig.suptitle(f"Pseudo-continuum 4 methods overlay — HST SN 2011fe B-max\n"
             f"(1) red dashed = 90-pctl envelope  ·  (2) blue = Gaussian smooth  ·  "
             f"(3) green = wide-median (v=30k)  ·  (4) orange = BSNIP-II anchors",
             fontsize=11, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")
