#!/usr/bin/env python3
"""Evaluate one or two LUMINA run dirs vs SN 2002bo (+0d): band ratios
(norm [3500,9000]), converged per-shell T_rad/W, and an estimator-collapse
check. Reusable per autonomous iteration.

Usage: eval_run_vs_2002bo.py <run_dir> [run_dir2] ...
Each run_dir must contain lumina_spectrum.csv and nlte_levels_iter*.csv.
"""
import sys, os, glob
import numpy as np, pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
LO, HI = 3500.0, 9000.0
BANDS = [("UV<3400", 3000, 3400), ("blue3500-4500", 3500, 4500),
         ("green4500-5500", 4500, 5500), ("red5500-7000", 5500, 7000),
         ("NIR7000-9000", 7000, 9000)]

obs = pd.read_csv(OBS, comment='#')
ow, of = obs.iloc[:, 0].values, obs.iloc[:, 1].values

def integ(w, f, a=LO, b=HI):
    s = (w >= a) & (w <= b)
    return np.trapezoid(f[s], w[s])

ofn = of / integ(ow, of)

def band_ratios(mw, mf):
    mfn = mf / integ(mw, mf)
    out = {}
    for nm, a, b in BANDS:
        sm = (mw >= a) & (mw <= b); so = (ow >= a) & (ow <= b)
        oi = np.trapezoid(ofn[so], ow[so])
        out[nm] = (np.trapezoid(mfn[sm], mw[sm]) / oi) if oi > 0 else float('nan')
    return out, mfn

def per_shell(run):
    fs = sorted(glob.glob(f"{run}/nlte_levels_iter*.csv"))
    if not fs:
        return None, []
    d = pd.read_csv(fs[-1])
    rows = []
    for s in [0, 5, 15, 29]:
        r = d[d.iloc[:, 2] == s]
        if len(r):
            r0 = r.iloc[0]
            rows.append((s, float(r0.iloc[9]), float(r0.iloc[10])))  # T_rad, W
    return os.path.basename(fs[-1]), rows

def score(r):
    """L1 distance of band ratios from 1.0 (lower = closer to obs)."""
    return sum(abs(v - 1.0) for v in r.values())

for run in sys.argv[1:]:
    run = run.rstrip('/')
    name = os.path.basename(run)
    try:
        m = pd.read_csv(f"{run}/lumina_spectrum.csv")
        mw, mf = m.iloc[:, 0].values, m.iloc[:, 1].values
    except Exception as e:
        print(f"== {name}: NO SPECTRUM ({e})"); continue
    r, _ = band_ratios(mw, mf)
    print(f"== {name}")
    print("   bands: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items())
          + f"   | L1(from 1.0)={score(r):.3f}")
    dumpf, rows = per_shell(run)
    if rows:
        collapse = any(T < 3500 for _, T, _ in rows)
        inflate = any(W > 1.05 for s, _, W in rows if s >= 5)  # outer W should be <1
        flags = []
        if collapse: flags.append("T_rad-COLLAPSE(<3500K)")
        if inflate:  flags.append("W-INFLATE(>1.05 outer)")
        tag = ("  [" + ", ".join(flags) + "]") if flags else "  [estimator sane]"
        print(f"   {dumpf}: " +
              " ".join(f"sh{s}(T={T:.0f},W={W:.2f})" for s, T, W in rows) + tag)
