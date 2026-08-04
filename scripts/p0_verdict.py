#!/usr/bin/env python3
"""Batch A (fluorescence plan P0) falsifier verdict.

For baseline + each variant, print:
  - band fractions UV[2500-3000], [2500-2750], green[5000-6500] (of 2000-8000 A)
  - S II and Si II ionization fraction f(II) at line-forming shells s6,s9,s15,s25
CMFGEN targets (toy06 19.48d): green ~22%, UV[2500-3000] ~8.5%; S II f(II) 0.5-0.8.
Verdict: ionization is BINDING if any variant moves green UP + UV DOWN toward target.
"""
import csv, glob, os, sys

BASE = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUNS = [("baseline", "logs/coevolve_consume"),
        ("jbardump", "logs/coevolve_consume_jbardump"),
        ("frozenin", "logs/coevolve_consume_frozenin"),
        ("nophotoion", "logs/coevolve_consume_nophotoion")]

def band_fracs(spec):
    wl, fl = [], []
    with open(spec) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("wave"):
                continue
            try:
                w, x = float(row[0]), float(row[1])
            except ValueError:
                continue
            wl.append(w); fl.append(x)
    def integ(lo, hi):
        return sum(x for w, x in zip(wl, fl) if lo <= w < hi)
    tot = integ(2000, 8000)
    if tot <= 0:
        return None
    return dict(uv2530=integ(2500, 3000)/tot,
                uv2527=integ(2500, 2750)/tot,
                green=integ(5000, 6500)/tot)

def ion_fII(ionpops, Z, shells):
    # TRUE f(II) from ion_pops (all stages). stage is 0-based: 0=I,1=II,2=III.
    # (levelpop only holds I,II NLTE levels -> misses the over-ionized III reservoir.)
    tot = {}; sec = {}
    with open(ionpops) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                s, z, st, n = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            except (ValueError, IndexError):
                continue
            if z != Z or s not in shells:
                continue
            tot[s] = tot.get(s, 0.0) + n
            if st == 1:
                sec[s] = sec.get(s, 0.0) + n
    return {s: (sec.get(s, 0.0)/tot[s] if tot.get(s, 0.0) > 0 else float('nan')) for s in shells}

SH = [6, 9, 15, 25]
print(f"{'variant':11s} | {'green5-6.5':>10s} {'UV2.5-3':>8s} {'UV2.5-2.75':>10s} | "
      f"S II f(II) s6/9/15/25          | Si II f(II) s6/9/15/25")
print("-"*118)
print(f"{'CMFGEN tgt':11s} | {'~0.227':>10s} {'~0.085':>8s} {'~0.022':>10s} | "
      f"0.01/0.09/0.79/0.79 (dominant)     | ~0/0.003/0.09/0.14")
print("-"*118)
for tag, d in RUNS:
    sp = os.path.join(BASE, d, "lumina_spectrum_formal.csv")
    ip = os.path.join(BASE, d, "lumina_ion_pops.csv")
    if not os.path.exists(sp):
        print(f"{tag:11s} | (no output yet: {d})")
        continue
    b = band_fracs(sp)
    s2 = ion_fII(ip, 16, SH) if os.path.exists(ip) else {s: float('nan') for s in SH}
    si = ion_fII(ip, 14, SH) if os.path.exists(ip) else {s: float('nan') for s in SH}
    ss = "/".join(f"{s2[s]:.2f}" for s in SH)
    sis = "/".join(f"{si[s]:.3f}" for s in SH)
    if b:
        print(f"{tag:11s} | {b['green']:10.3f} {b['uv2530']:8.3f} {b['uv2527']:10.3f} | "
              f"{ss:33s} | {sis}")
