#!/usr/bin/env python3
"""173-run shortlist 재평가 — 5-band weighted RMS (UVbl drop, UVtg/CaK down-weight).

이미 계산된 data/shortlist_baseline_rerank.csv 의 6-band log10 ratio를 재사용해서
가중치만 다시 적용:

  UVbl  [1700,2900]: w=0.0 (drop, iron forest blanketing → flux 신뢰 불가)
  UVtg  [2900,3700]: w=0.5 (Fe forest, but flux non-zero, 진단 가치)
  CaK   [3700,3950]: w=0.7 (Ca II HK 강흡수 dominant)
  fluo  [3950,4500]: w=1.0
  grn   [4500,5800]: w=1.0
  red   [5800,7000]: w=1.0

baseline-norm RMS [3800, 8000] 는 그대로 사용 (이미 NUV 제외).

Output:
  data/shortlist_5band_weighted.csv
  figures/shortlist_5band_weighted.png
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
SRC  = ROOT / "data/shortlist_baseline_rerank.csv"
OUT_CSV = ROOT / "data/shortlist_5band_weighted.csv"
OUT_PNG = ROOT / "figures/shortlist_5band_weighted.png"

WEIGHTS = dict(UVbl=0.0, UVtg=0.5, CaK=0.7, fluo=1.0, grn=1.0, red=1.0)

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} runs from {SRC.name}")

bands = ["UVbl","CaK","UVtg","fluo","grn","red"]
W = np.array([WEIGHTS[b] for b in bands])
print(f"Weights: {dict(zip(bands, W))}")

# Weighted RMS
log_cols = [f"log_{b}" for b in bands]
LR = df[log_cols].to_numpy()         # (N, 6)
# Replace -10 sentinels (band integral failed) with 0 to avoid dominating weighted RMS;
# but if weight is 0 for that band, doesn't matter.
LR_clip = np.where(LR <= -9.0, 0.0, LR)
num = (W[None,:] * LR_clip**2).sum(axis=1)
den = W.sum()
df["rms5_weighted"] = np.sqrt(num / den)

# Pure rank
df["rank_raw6"]   = df["rms6_log"].rank()
df["rank_w5"]     = df["rms5_weighted"].rank()
df["rank_bnorm"]  = df["rms_bnorm"].rank()

df.sort_values("rms5_weighted", inplace=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

print("\n=== Top-15 by 5-band WEIGHTED RMS (UVbl drop, UVtg/CaK down-wt) ===")
print(df.head(15)[["run","rms6_log","rms5_weighted","rms_bnorm",
                   "rank_raw6","rank_w5","rank_bnorm"]].to_string(index=False))

print("\n=== Top-15 by raw 6-band RMS (for comparison) ===")
print(df.sort_values("rms6_log").head(15)[
    ["run","rms6_log","rms5_weighted","rms_bnorm","rank_raw6","rank_w5","rank_bnorm"]
].to_string(index=False))

print("\n=== Top-15 by baseline-norm RMS [3800,8000] ===")
print(df.sort_values("rms_bnorm").head(15)[
    ["run","rms6_log","rms5_weighted","rms_bnorm","rank_raw6","rank_w5","rank_bnorm"]
].to_string(index=False))

# Top-N intersections
top_raw  = set(df.nsmallest(10, "rms6_log")["run"])
top_w5   = set(df.nsmallest(10, "rms5_weighted")["run"])
top_bn   = set(df.nsmallest(10, "rms_bnorm")["run"])
print("\n=== Top-10 set intersections ===")
print(f"  raw6  & w5  : {len(top_raw & top_w5)} runs   {sorted(top_raw & top_w5)[:5]}")
print(f"  raw6  & bn  : {len(top_raw & top_bn)} runs   {sorted(top_raw & top_bn)[:5]}")
print(f"  w5    & bn  : {len(top_w5  & top_bn)} runs   {sorted(top_w5  & top_bn)[:5]}")
print(f"  triple    : {len(top_raw & top_w5 & top_bn)} runs   {sorted(top_raw & top_w5 & top_bn)[:5]}")

# Champion 152761 ranks
for tag in ["152761", "151197", "152629"]:
    sel = df[df["run"].str.contains(tag)]
    if len(sel):
        r = sel.iloc[0]
        print(f"\n  {r['run']:55s}  rms6={r['rms6_log']:.4f} (rk{int(r['rank_raw6']):3d})  "
              f"w5={r['rms5_weighted']:.4f} (rk{int(r['rank_w5']):3d})  "
              f"bn={r['rms_bnorm']:.4f} (rk{int(r['rank_bnorm']):3d})")

# ===== Figure =====
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# (a) raw vs weighted scatter — 가중치만 바꾼 효과
ax = axes[0,0]
ax.scatter(df["rms6_log"], df["rms5_weighted"], s=18, alpha=0.55,
           c="tab:blue", edgecolor="black", lw=0.3)
top_raw_df = df.nsmallest(10, "rms6_log")
top_w5_df  = df.nsmallest(10, "rms5_weighted")
ax.scatter(top_raw_df["rms6_log"], top_raw_df["rms5_weighted"], s=80, c="tab:red",
           edgecolor="black", lw=0.5, label=f"top-10 raw6", zorder=5)
ax.scatter(top_w5_df["rms6_log"], top_w5_df["rms5_weighted"], s=80, c="tab:orange",
           edgecolor="black", lw=0.5, marker="^", label="top-10 w5", zorder=5)
ax.text(0.02, 0.97, f"top-10 교집합: {len(top_raw & top_w5)} runs",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
ax.set_xlabel("raw 6-band log RMS")
ax.set_ylabel("5-band weighted RMS (UVbl drop, UVtg=0.5, CaK=0.7)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title(f"(a) raw vs weighted — UVbl drop 효과 ({len(df)} runs)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3, which="both")

# (b) weighted vs baseline-norm scatter — 두 직교 metric
ax = axes[0,1]
ax.scatter(df["rms5_weighted"], df["rms_bnorm"], s=18, alpha=0.55,
           c="tab:blue", edgecolor="black", lw=0.3)
top_w5_df = df.nsmallest(10, "rms5_weighted")
top_bn_df = df.nsmallest(10, "rms_bnorm")
ax.scatter(top_w5_df["rms5_weighted"], top_w5_df["rms_bnorm"], s=80, c="tab:orange",
           edgecolor="black", lw=0.5, label="top-10 w5", zorder=5)
ax.scatter(top_bn_df["rms5_weighted"], top_bn_df["rms_bnorm"], s=80, c="tab:green",
           edgecolor="black", lw=0.5, marker="^", label="top-10 bn", zorder=5)
ax.text(0.02, 0.97, f"top-10 교집합: {len(top_w5 & top_bn)} runs",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
ax.set_xlabel("5-band weighted RMS")
ax.set_ylabel("baseline-norm RMS [3800,8000]")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("(b) weighted vs baseline-norm — Pareto",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3, which="both")

# (c) rank swap: raw6 → w5 (UVbl drop 효과)
ax = axes[1,0]
ax.scatter(df["rank_raw6"], df["rank_w5"], s=18, alpha=0.55,
           c="tab:blue", edgecolor="black", lw=0.3)
n = len(df)
ax.plot([1,n],[1,n],"k--",lw=0.7,alpha=0.5)
df["drank_raw_w5"] = df["rank_w5"] - df["rank_raw6"]
big = df[(df["rank_raw6"] <= 30) | (df["rank_w5"] <= 30)].copy()
big["abs_d"] = big["drank_raw_w5"].abs()
worst = big.nlargest(8, "abs_d")
for _, r in worst.iterrows():
    ax.annotate(r["run"][:35], (r["rank_raw6"], r["rank_w5"]), fontsize=6.5,
                xytext=(4,4), textcoords="offset points")
    ax.scatter(r["rank_raw6"], r["rank_w5"], s=80, c="tab:red",
               edgecolor="black", lw=0.6, zorder=5)
ax.set_xlabel("rank by raw 6-band RMS")
ax.set_ylabel("rank by 5-band WEIGHTED RMS")
ax.set_title("(c) rank swap: raw6 → w5 (UVbl drop)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, n+5); ax.set_ylim(0, n+5); ax.grid(alpha=0.3)

# (d) rank swap: w5 vs bn (line shape vs SED, 가중치 적용 후)
ax = axes[1,1]
ax.scatter(df["rank_w5"], df["rank_bnorm"], s=18, alpha=0.55,
           c="tab:blue", edgecolor="black", lw=0.3)
ax.plot([1,n],[1,n],"k--",lw=0.7,alpha=0.5)
df["drank_w5_bn"] = df["rank_bnorm"] - df["rank_w5"]
big = df[(df["rank_w5"] <= 30) | (df["rank_bnorm"] <= 30)].copy()
big["abs_d"] = big["drank_w5_bn"].abs()
worst = big.nlargest(8, "abs_d")
for _, r in worst.iterrows():
    ax.annotate(r["run"][:35], (r["rank_w5"], r["rank_bnorm"]), fontsize=6.5,
                xytext=(4,4), textcoords="offset points")
    ax.scatter(r["rank_w5"], r["rank_bnorm"], s=80, c="tab:red",
               edgecolor="black", lw=0.6, zorder=5)
ax.set_xlabel("rank by 5-band weighted RMS")
ax.set_ylabel("rank by baseline-norm RMS")
ax.set_title("(d) rank swap: w5 vs bn (line shape vs SED)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, n+5); ax.set_ylim(0, n+5); ax.grid(alpha=0.3)

plt.suptitle(f"Best-of-best: 5-band weighted RMS (UVbl drop, UVtg=0.5, CaK=0.7) × baseline-norm",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=140); plt.close()
print(f"\nWrote {OUT_PNG}  ({OUT_PNG.stat().st_size//1024} KB)")
