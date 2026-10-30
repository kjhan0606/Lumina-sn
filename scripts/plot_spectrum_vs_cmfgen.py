#!/usr/bin/env python3
"""Lumina 스펙트럼과 공개 StaNdaRT CMFGEN @19.48d 비교.

고정 물질 관측 결과의 비교는 공동수렴 또는 물리적 동등성의 증거가 아니다.
"""
import argparse, bisect, csv, re, sys

EPOCH = 19.48
CMF = "data/standart_data1/toy06/spectra_toy06_cmfgen.txt"
LUM = "lumina_spectrum_formal.csv"
OUT = "validation/spectrum_overlay/lumina_vs_cmfgen_19p48d.png"
BAND = (3000.0, 10000.0)          # 정규화 기준 대역
XLIM = (2500.0, 20000.0)
CM_PER_ANGSTROM = 1e-8


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


def read_lumina(path=None):
    """생산 formal CSV의 erg/s/cm를 공개 참조와 같은 erg/s/Å로 변환한다."""
    w, y = [], []
    with open(LUM if path is None else path) as f:
        next(f)
        for r in csv.reader(f):
            w.append(float(r[0])); y.append(float(r[1]) * CM_PER_ANGSTROM)
    return w, y


def read_observer(path):
    """관측자 수송의 NPZ는 이미 erg/s/Å이며 파장이 역순일 수 있다."""
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        wave = np.asarray(data['wavelength_A'], dtype=float)
        luminosity = np.asarray(data['luminosity_erg_s_A'], dtype=float)
    if (wave.ndim != 1 or len(wave) < 2 or luminosity.shape != wave.shape or
            not np.isfinite(wave).all() or not np.isfinite(luminosity).all() or
            np.any(wave <= 0) or np.any(luminosity < 0)):
        raise ValueError('관측자 스펙트럼의 파장 또는 광도 오류')
    order = np.argsort(wave)
    wave, luminosity = wave[order], luminosity[order]
    if np.any(np.diff(wave) <= 0):
        raise ValueError('관측자 스펙트럼에 중복 파장 존재')
    return wave.tolist(), luminosity.tolist()


def integral(X, Y, a, b):
    s = 0.0
    for i in range(1, len(X)):
        if X[i - 1] >= a and X[i] <= b:
            s += 0.5 * (Y[i] + Y[i - 1]) * (X[i] - X[i - 1])
    return s


def integral_with_edges(X, Y, a, b):
    """겹치는 대역 양끝을 선형 보간한다. 관측 대역 밖 외삽은 하지 않는다."""
    import numpy as np
    x, y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or y.shape != x.shape or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(np.diff(x) <= 0) or not x[0] <= a < b <= x[-1]):
        raise ValueError('비교 대역이 스펙트럼에 완전히 포함되지 않음')
    inside = (x > a) & (x < b)
    grid = np.r_[a, x[inside], b]
    values = np.r_[np.interp(a, x, y), y[inside], np.interp(b, x, y)]
    return float(np.sum((values[:-1] + values[1:]) * np.diff(grid) * .5))


def interp(x, X, Y):
    i = bisect.bisect_left(X, x)
    if i <= 0 or i >= len(X):
        return None
    t = (x - X[i - 1]) / (X[i] - X[i - 1])
    return Y[i - 1] + t * (Y[i] - Y[i - 1])


def main(output=OUT, observer=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]

    cw, cf = read_cmfgen()
    lw, lf = read_lumina() if observer is None else read_observer(observer)
    integrate = integral if observer is None else integral_with_edges
    ic = integrate(cw, cf, *BAND)
    il = integrate(lw, lf, *BAND)
    if ic <= 0 or il <= 0:
        raise ValueError('형상 비교 기준 대역의 광도가 양수가 아님')
    scale = ic / il
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    ax[0].plot(cw, cf, lw=0.8, color="#1f77b4", label="CMFGEN (StaNdaRT, public) @19.48d")
    label = "Lumina formal" if observer is None else "Lumina observer (공동수렴 미판정)"
    ax[0].plot(lw, lf, lw=0.8, color="#d62728", label=label)
    ax[0].set_yscale("log"); ax[0].set_xlim(*XLIM)
    ax[0].set_ylabel("Lλ [erg/s/Å]")
    ax[0].set_title(f"절대 광도 — 단위 변환만 적용, 대역 적분비 CMFGEN/Lumina = {scale:.3e}")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    ax[1].plot(cw, cf, lw=0.9, color="#1f77b4", label="CMFGEN")
    ax[1].plot(lw, [v * scale for v in lf], lw=0.9, color="#d62728",
               label=f"Lumina × {scale:.3e}  (scaled to match {BAND[0]:.0f}-{BAND[1]:.0f} Å integral)")
    ax[1].set_yscale("log"); ax[1].set_xlim(*XLIM)
    ax[1].set_ylim(1e36, 3e40)
    ax[1].set_ylabel("Lλ [erg/s/Å]")
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

    fig.suptitle("Lumina vs public CMFGEN @ 19.48 d — 조건부 스펙트럼 비교", fontsize=12)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    rr.sort(); n = len(rr)
    print(f"scale(CMFGEN/Lumina, {BAND[0]:.0f}-{BAND[1]:.0f}A integral) = {scale:.6e}")
    print(f"ratio after scaling: q10={rr[n//10]:.3f} median={rr[n//2]:.3f} "
          f"q90={rr[9*n//10]:.3f}  min={rr[0]:.3e} max={rr[-1]:.3e}")
    print(f"-> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observer', help='관측자 수송의 observer-spectrum.npz')
    parser.add_argument('--output', default=OUT, help='출력 그림 경로')
    args = parser.parse_args()
    main(output=args.output, observer=args.observer)
