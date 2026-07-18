#!/usr/bin/env python3
"""
Inter-code check ("audit the yardstick"): is the PUBLISHED StaNdaRT CMFGEN toy06
ionization consistent with the other radiative-transfer codes, at the Lumina
shell velocities?

For Fe and Co we tabulate f(IV) = IV / (III + IV) at ~19.48 d, interpolated to the
Lumina toy06 shell mid-velocities, for every code that has a photospheric-epoch
output. The published CMFGEN row is the one we adopt as the benchmark; the other
rows show whether CMFGEN sits inside the code spread (it does at the photosphere
and the deep end; the transition zone v~7-9 kK is where codes disagree most).

The last row overlays the Lumina offline no-floor baseline for reference.

Usage:  python cmp_ionfrac_codes_at_lumina_v.py
Reads:  data/standart_data1/toy06/ionfrac_{fe,co}_toy06_<code>.txt  (repo)
"""
import os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "data", "standart_data1", "toy06")

def parse(fn, tgt=19.48):
    L = open(fn).read().splitlines()
    times, st = [], []
    for i, l in enumerate(L):
        m = re.match(r'#TIME:\s*([\d.]+)', l)
        if m:
            times.append(float(m.group(1))); st.append(i)
    if not times:
        return None, None, None
    k = int(np.argmin(abs(np.array(times) - tgt))); ep = times[k]
    i0 = st[k]; i1 = st[k + 1] if k + 1 < len(st) else len(L)
    cols, v, d = None, [], []
    for l in L[i0:i1]:
        if l.startswith('#vel_mid'):
            cols = l.strip().lstrip('#').split()[1:]; continue
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) >= 2:
            try:
                vals = [float(x) for x in p]
            except ValueError:
                continue
            v.append(vals[0]); d.append(vals[1:])
    if not v:
        return ep, None, None
    v = np.array(v); d = np.array(d)
    return ep, v, {c: d[:, j] for j, c in enumerate(cols)}

def fiv_at(fn, elem, vt):
    ep, v, dd = parse(fn)
    if dd is None or f"{elem}2" not in dd:
        return None
    III = np.interp(vt, v, dd[f"{elem}2"]); IV = np.interp(vt, v, dd[f"{elem}3"])
    Vst = np.interp(vt, v, dd[f"{elem}4"]) if f"{elem}4" in dd else 0 * vt
    f = IV / np.where(III + IV > 0, III + IV, np.nan)
    return ep, f, Vst

# Lumina toy06 shell mid-velocities (km/s) — from
# data/tardis_reference_toy06_19p48d/geometry.csv (v_inner..v_outer, 728 km/s wide)
SHELLS = {0: 4264, 4: 7176, 6: 8632, 9: 10816, 10: 11544, 14: 14456}
VT = np.array(list(SHELLS.values()), float)
CODES = ['cmfgen', 'artis', 'sedona', 'tardis', 'supernu', 'crab']
# Lumina offline no-floor baseline f(IV) (r_nlte_baseline_realsigma.csv) for overlay
LUM = {'Fe': {0: 0.066, 4: 0.920, 6: 0.960, 10: 0.844},
       'Co': {0: 0.011, 4: 0.577, 6: 0.961, 10: 0.967}}

for elem, lab in [('fe', 'Fe'), ('co', 'Co')]:
    print(f"\n===== {lab}: f(IV)=IV/(III+IV) at Lumina shell velocities (19.48 d) =====")
    print(f"{'code':8s} " + "".join(f"{'s'+str(s):>7s}" for s in SHELLS))
    print(f"{'v[km/s]':8s} " + "".join(f"{v:7.0f}" for v in VT))
    for c in CODES:
        fn = os.path.join(DATA, f"ionfrac_{elem}_toy06_{c}.txt")
        if not os.path.exists(fn):
            continue
        r = fiv_at(fn, elem, VT)
        if r is None:
            print(f"{c:8s} n/a"); continue
        ep, f, Vst = r
        print(f"{c:8s} " + " ".join(f"{x:6.3f}" for x in f) +
              f"   (ep{ep:.1f}, Vmax={Vst.max():.2f})")
    lrow = " ".join((f"{LUM[lab][s]:6.3f}" if s in LUM[lab] else f"{'--':>6s}")
                    for s in SHELLS)
    print(f"{'LUMINA*':8s} " + lrow + "   (*offline no-floor baseline)")
