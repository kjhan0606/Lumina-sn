#!/usr/bin/env python3
"""Lumina formal 스펙트럼 대 공개 StaNdaRT CMFGEN @19.48d 겹침 (진단용).

⚠이것은 **우회 진단**이다. 본선은 배선도/물리 검사다(user 2026-08-19).
그림이 예뻐 보여도 배선이 검증된 것이 아니다.
"""
import bisect, csv, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH = 19.48
CMF = "data/standart_data1/toy06/spectra_toy06_cmfgen.txt"
LUM = "lumina_spectrum_formal.csv"
OUT = "validation/spectrum_overlay/lumina_vs_cmfgen_19p48d.png"
BAND = (3000.0, 10000.0)          # 정규화 기준 대역
XLIM = (2500.0, 20000.0)


def read_cmfgen():
    with open(CMF) as f:
        f.readline(); f.readline()
        times = [float(x) for x in re.findall(r"[0-9.eE+-]+", f.readline().split(":", 1)[1])]
        f.readline()
        i = times.index(EPOCH)
        w, y = [], []
        for line in f:
            q = line.split()
            if len(q) > 1 + i:
                w.append(float(q[0])); y.append(float(q[1 + i]))
    return w, y


def read_lumina():
    w, y = [], []
    with open(LUM) as f:
        next(f)
        for r in csv.reader(f):
            w.append(float(r[0])); y.append(float(r[1]))
    return w, y


def integral(X, Y, a, b):
    s = 0.0
    for i in range(1, len(X)):
        if X[i - 1] >= a and X[i] <= b:
            s += 0.5 * (Y[i] + Y[i - 1]) * (X[i] - X[i - 1])
    return s


def interp(x, X, Y):
    i = bisect.bisect_left(X, x)
    if i <= 0 or i >= len(X):
        return None
    t = (x - X[i - 1]) / (X[i] - X[i - 1])
    return Y[i - 1] + t * (Y[i] - Y[i - 1])


def main():
    cw, cf = read_cmfgen()
    lw, lf = read_lumina()
    ic = integral(cw, cf, *BAND)
    il = integral(lw, lf, *BAND)
    scale = ic / il
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    ax[0].plot(cw, cf, lw=0.8, color="#1f77b4", label="CMFGEN (StaNdaRT, public) @19.48d")
    ax[0].plot(lw, lf, lw=0.8, color="#d62728", label="Lumina formal (2026-08-05)")
    ax[0].set_yscale("log"); ax[0].set_xlim(*XLIM)
    ax[0].set_ylim(1e36, 1e49)
    ax[0].set_ylabel("flux (as written, unscaled)")
    ax[0].set_title(f"① Absolute — declared units differ; band-integral ratio "
                    f"CMFGEN/Lumina = {scale:.3e}")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    ax[1].plot(cw, cf, lw=0.9, color="#1f77b4", label="CMFGEN")
    ax[1].plot(lw, [v * scale for v in lf], lw=0.9, color="#d62728",
               label=f"Lumina × {scale:.3e}  (scaled to match {BAND[0]:.0f}-{BAND[1]:.0f} Å integral)")
    ax[1].set_yscale("log"); ax[1].set_xlim(*XLIM)
    ax[1].set_ylim(1e36, 3e40)
    ax[1].set_ylabel("flux [erg/s/Ang]")
    ax[1].set_title("② Shape — single declared scale factor, no per-bin tuning")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

    rw, rr = [], []
    for x, y in zip(cw, cf):
        if XLIM[0] <= x <= min(XLIM[1], max(lw)) and y > 0:
            v = interp(x, lw, lf)
            if v and v > 0:
                rw.append(x); rr.append(v * scale / y)
    ax[2].plot(rw, rr, lw=0.7, color="#2ca02c")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].set_yscale("log"); ax[2].set_xlim(*XLIM)
    ax[2].set_xlabel("wavelength [Å]"); ax[2].set_ylabel("Lumina·scale / CMFGEN")
    ax[2].set_title("③ Ratio — deviation from 1 is the divergence map")
    ax[2].grid(alpha=.3)

    fig.suptitle("Lumina vs public CMFGEN @ 19.48 d — DIAGNOSTIC DETOUR "
                 "(main line remains wiring/physics verification)", fontsize=12)
    fig.savefig(OUT, dpi=140)
    rr.sort(); n = len(rr)
    print(f"scale(CMFGEN/Lumina, {BAND[0]:.0f}-{BAND[1]:.0f}A integral) = {scale:.6e}")
    print(f"ratio after scaling: q10={rr[n//10]:.3f} median={rr[n//2]:.3f} "
          f"q90={rr[9*n//10]:.3f}  min={rr[0]:.3e} max={rr[-1]:.3e}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
