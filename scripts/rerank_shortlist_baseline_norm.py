#!/usr/bin/env python3
"""기존 LUMINA shortlist (174 runs) 재평가 — raw 6-band log-RMS vs baseline-norm RMS.

두 metric을 모두 계산하고 disagreement 산점도를 만든다.
- raw 6-band RMS: UVbl/CaK/UVtg/fluo/grn/red 정규화 후 log10 ratio RMS (canonical)
- baseline-norm RMS: Gauss FWHM=(40k/c)·λ baseline on both, residual on [3800, 8000]
  (NUV [<3800] 제외 — 거의 모든 pixel 흡수돼서 smoothing이 baseline을 회복하지 못함)
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
OUT_CSV = ROOT / "data/shortlist_baseline_rerank.csv"
OUT_PNG = ROOT / "figures/shortlist_baseline_rerank.png"
LOGS = ROOT / "logs"
C_KMS = 299792.458
FWHM_KMS = 40000.0

BANDS = [("UVbl",1700,2900),("CaK",3700,3950),("UVtg",2900,3700),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]
NORM_LO, NORM_HI = 4500, 5800
BNORM_LO, BNORM_HI = 3800, 8000


def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum() >= 2 else np.nan


def gauss_baseline(lam, flu, fwhm_kms=FWHM_KMS):
    cont = np.zeros_like(flu); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


# Load HST once, precompute baseline
print("Loading HST + computing baseline...")
df_h = pd.read_csv(HST)
hlam = df_h.iloc[:,0].values; hflu = df_h.iloc[:,1].values
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]
hcont = gauss_baseline(hlam, hflu)
hnorm = hflu / hcont
mask_bn = (hlam >= BNORM_LO) & (hlam <= BNORM_HI)
hnorm_band = hnorm[mask_bn]
hlam_band  = hlam[mask_bn]


def evaluate_one(path):
    try:
        df = pd.read_csv(path)
        lam = df.iloc[:,0].values
        col = "flux" if "flux" in df.columns else df.columns[1]
        flu = df[col].values
        if not np.isfinite(flu).all() or (flu < 0).any():
            flu = np.where(np.isfinite(flu) & (flu>=0), flu, 0.0)
        # Normalize to HST 4500-5800 (canonical)
        norm = band_int(hlam, hflu, NORM_LO, NORM_HI) / band_int(lam, flu, NORM_LO, NORM_HI)
        if not np.isfinite(norm) or norm <= 0:
            return None
        flu = flu * norm

        # 6-band log10 RMS
        log_r = []
        for n, lo, hi in BANDS:
            num = band_int(lam, flu, lo, hi)
            den = band_int(hlam, hflu, lo, hi)
            if num > 0 and den > 0:
                log_r.append(np.log10(num/den))
            else:
                log_r.append(-10.0)
        rms6 = float(np.sqrt(np.mean(np.array(log_r)**2)))

        # baseline-norm RMS on [3800, 8000]
        lcont = gauss_baseline(lam, flu)
        lnorm = flu / lcont
        # interpolate to HST band grid
        lnorm_on_h = np.interp(hlam_band, lam, lnorm)
        rms_bn = float(np.sqrt(np.mean((hnorm_band - lnorm_on_h)**2)))

        return (rms6, rms_bn, *log_r)
    except Exception as e:
        return None


paths = sorted(LOGS.glob("*/lumina_spectrum_formal.csv"))
print(f"Found {len(paths)} formal spectra. Evaluating...")

records = []
for k, p in enumerate(paths):
    if k % 20 == 0: print(f"  [{k+1}/{len(paths)}] {p.parent.name[:50]}")
    r = evaluate_one(p)
    if r is None: continue
    rms6, rms_bn, lUVbl, lCaK, lUVtg, lfluo, lgrn, lred = r
    records.append(dict(
        run = p.parent.name,
        rms6_log = rms6,
        rms_bnorm = rms_bn,
        log_UVbl = lUVbl, log_CaK = lCaK, log_UVtg = lUVtg,
        log_fluo = lfluo, log_grn  = lgrn, log_red  = lred,
    ))

dfr = pd.DataFrame(records).sort_values("rms6_log")
dfr.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}  ({len(dfr)} runs)")

print("\n=== Top-15 by raw 6-band RMS ===")
print(dfr.head(15)[["run","rms6_log","rms_bnorm"]].to_string(index=False))
print("\n=== Top-15 by baseline-norm RMS [3800,8000] ===")
print(dfr.sort_values("rms_bnorm").head(15)[["run","rms6_log","rms_bnorm"]].to_string(index=False))

# ===== Scatter =====
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.scatter(dfr["rms6_log"], dfr["rms_bnorm"], s=18, alpha=0.55,
           c="tab:blue", edgecolor="black", lw=0.3)
# top-10 each metric
top6 = dfr.nsmallest(10, "rms6_log")
topb = dfr.nsmallest(10, "rms_bnorm")
ax.scatter(top6["rms6_log"], top6["rms_bnorm"], s=70, c="tab:red",
           edgecolor="black", lw=0.5, label=f"Top-10 by raw 6-band ({len(top6)})", zorder=5)
ax.scatter(topb["rms6_log"], topb["rms_bnorm"], s=70, c="tab:green",
           edgecolor="black", lw=0.5, marker="^",
           label=f"Top-10 by baseline-norm ({len(topb)})", zorder=5)
common = set(top6["run"]) & set(topb["run"])
ax.text(0.02, 0.97, f"두 metric top-10 교집합: {len(common)} runs",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
ax.set_xlabel("raw 6-band log RMS (UVbl/CaK/UVtg/fluo/grn/red, log10 ratio)")
ax.set_ylabel("baseline-norm RMS [3800, 8000] Å")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title(f"Shortlist 재평가 — 두 metric 산점도 ({len(dfr)} runs)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3, which="both")

ax2 = axes[1]
# rank disagreement
dfr["rank6"]  = dfr["rms6_log"].rank()
dfr["rankBN"] = dfr["rms_bnorm"].rank()
dfr["drank"]  = dfr["rankBN"] - dfr["rank6"]
ax2.scatter(dfr["rank6"], dfr["rankBN"], s=18, alpha=0.55,
            c="tab:blue", edgecolor="black", lw=0.3)
n = len(dfr)
ax2.plot([1,n], [1,n], "k--", lw=0.7, alpha=0.5, label="rank-equal")
# Highlight large disagreements
big = dfr[(dfr["rank6"] <= 30) | (dfr["rankBN"] <= 30)].copy()
big["abs_drank"] = big["drank"].abs()
worst = big.nlargest(8, "abs_drank")
for _, r in worst.iterrows():
    short = r["run"][:35]
    ax2.annotate(short, (r["rank6"], r["rankBN"]), fontsize=6.5,
                 xytext=(4, 4), textcoords="offset points")
    ax2.scatter(r["rank6"], r["rankBN"], s=80, c="tab:red",
                edgecolor="black", lw=0.6, zorder=5)
ax2.set_xlabel("rank by raw 6-band RMS")
ax2.set_ylabel("rank by baseline-norm RMS")
ax2.set_title("Rank disagreement — top-30 후보 중 큰 swap", fontsize=11, fontweight="bold")
ax2.set_xlim(0, n+5); ax2.set_ylim(0, n+5)
ax2.grid(alpha=0.3); ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140); plt.close()
print(f"\nWrote {OUT_PNG}  ({OUT_PNG.stat().st_size//1024} KB)")

# Persist disagreement table for inspection
disagree = dfr[(dfr["rank6"]<=30) | (dfr["rankBN"]<=30)].copy()
disagree["abs_drank"] = disagree["drank"].abs()
disagree = disagree.sort_values("abs_drank", ascending=False).head(20)
print("\n=== Top-20 ranked-disagreement (top-30 region) ===")
print(disagree[["run","rms6_log","rms_bnorm","rank6","rankBN","drank"]].to_string(index=False))
