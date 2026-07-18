#!/usr/bin/env python3
"""
Compare our self-run CMFGEN structure T_e(v), n_e(v) against the PUBLISHED
StaNdaRT CMFGEN toy06 at 19.48 d.

This is the primary "is our run the same CMFGEN?" check. n_e is solved from the
ionization balance (not held fixed), so an n_e match at fixed T is a genuine
validation that our ionization physics reproduces the published CMFGEN.

Usage:
    python cmp_rvtj_T_ne_vs_published.py <path/to/RVTJ>

Inputs:
    argv[1]           : a CMFGEN RVTJ file from the run (contains V, n_e, T blocks)
    published phys    : data/standart_data1/toy06/phys_toy06_cmfgen.txt (repo)

The definitive comparison uses the CONVERGED RVTJ (after the FIX_T=T population
stint AND the FIX_T=F temperature-release stint). Run against an intermediate
RVTJ only for a trend check.
"""
import re, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PUB  = os.path.join(HERE, "..", "..", "..", "data", "standart_data1", "toy06",
                    "phys_toy06_cmfgen.txt")

def rvtj_block(lines, label, n):
    for i, l in enumerate(lines):
        if label in l:
            vals, j = [], i + 1
            while len(vals) < n and j < len(lines):
                p = re.findall(r'[-+]?\d+\.\d+E[-+]?\d+', lines[j])
                if not p and vals:
                    break
                vals += [float(x) for x in p]; j += 1
            return np.array(vals[:n])
    raise KeyError(label)

def read_rvtj(path):
    lines = open(path).read().splitlines()
    nd = None
    for l in lines:
        m = re.search(r'ND:\s*(\d+)', l)
        if m:
            nd = int(m.group(1)); break
    V  = rvtj_block(lines, "Velocity (km/s)", nd)
    Ne = rvtj_block(lines, "Electron density", nd)
    T  = rvtj_block(lines, "Temperature (10^4K)", nd) * 1e4
    return V, Ne, T

def read_published(path, tgt=19.480):
    L = open(path).read().splitlines()
    st = [i for i, l in enumerate(L) if re.match(rf'#TIME:\s*{tgt:.3f}', l)][0]
    pv, pt, pne = [], [], []
    for l in L[st + 1:]:
        if l.startswith('#TIME:'):
            break
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) >= 4:
            try:
                pv.append(float(p[0])); pt.append(float(p[1])); pne.append(float(p[3]))
            except ValueError:
                pass
    return map(np.array, (pv, pt, pne))

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    V, Ne, T = read_rvtj(sys.argv[1])
    pv, pt, pne = read_published(PUB)
    print(f"self-run RVTJ: {len(V)} depths, v {V.min():.0f}-{V.max():.0f} km/s")
    print(f"published CMFGEN: {len(pv)} shells, v {pv.min():.0f}-{pv.max():.0f} km/s\n")
    print(f"{'v[km/s]':>8s} | {'T_ours':>8s} {'T_pub':>8s} {'ratio':>6s} | "
          f"{'ne_ours':>10s} {'ne_pub':>10s} {'ratio':>6s}")
    for vt in [4264, 5213, 7176, 8632, 9900, 10816, 11544, 14456, 16601, 24460]:
        To = np.interp(vt, V[::-1], T[::-1]);  Tc = np.interp(vt, pv, pt)
        neo = np.interp(vt, V[::-1], Ne[::-1]); nec = np.interp(vt, pv, pne)
        print(f"{vt:8d} | {To:8.0f} {Tc:8.0f} {To/Tc:6.2f} | "
              f"{neo:10.3e} {nec:10.3e} {neo/nec:6.2f}")

if __name__ == "__main__":
    main()
