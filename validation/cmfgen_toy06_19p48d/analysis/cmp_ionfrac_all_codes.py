#!/usr/bin/env python3
"""
Broad inter-code ionization table for toy06 at ~19.48 d: Co and Fe ion fractions
(III/IV/V/VI) at an inner shell (v~1025) and a photospheric shell (v~11000) for
every StaNdaRT code that ships an ionfrac file.

Shows the consensus structure directly: all photospheric-epoch codes agree that
Fe/Co are IV-V deep and recombine to III at the photosphere. Codes whose nearest
epoch is nebular (sumo ~75 d, artisnebular ~199 d) or that lack a 19.48 d block
(urilight) are flagged with '*epoch!'.

Usage:  python cmp_ionfrac_all_codes.py
Reads:  data/standart_data1/toy06/ionfrac_{co,fe}_toy06_<code>.txt  (repo)
"""
import os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "data", "standart_data1", "toy06")

def parse_block(fn, tgt=19.480):
    lines = open(fn).read().splitlines()
    times, starts = [], []
    for i, l in enumerate(lines):
        m = re.match(r'#TIME:\s*([\d.]+)', l)
        if m:
            times.append(float(m.group(1))); starts.append(i)
    if not times:
        return None, None, None
    times = np.array(times)
    k = int(np.argmin(abs(times - tgt))); epoch = times[k]
    i0 = starts[k]; i1 = starts[k + 1] if k + 1 < len(starts) else len(lines)
    cols, vel, data = None, [], []
    for l in lines[i0:i1]:
        if l.startswith('#vel_mid'):
            cols = l.strip().lstrip('#').split()[1:]; continue
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) >= 2 and re.match(r'[\d.\-Ee+]+$', p[0]):
            try:
                vals = [float(x) for x in p]
            except ValueError:
                continue
            vel.append(vals[0]); data.append(vals[1:])
    if not vel:
        return epoch, None, None
    vel = np.array(vel); data = np.array(data)
    return epoch, vel, ({c: data[:, j] for j, c in enumerate(cols)} if cols else None)

def frac_at(fn, elem, tgt_vels):
    ep, vel, d = parse_block(fn)
    if d is None:
        return ep, None
    out = {}
    for tv in tgt_vels:
        j = int(np.argmin(abs(vel - tv)))
        row = {stage: d[f"{elem}{stage}"][j] for stage in range(6) if f"{elem}{stage}" in d}
        out[tv] = (vel[j], row)
    return ep, out

CODES = ['cmfgen', 'artis', 'sedona', 'tardis', 'urilight', 'supernu',
         'sumo', 'artisnebular', 'crab']
ROMAN = {2: 'III', 3: 'IV', 4: 'V', 5: 'VI'}
TGT = [1025., 11000.]

for elem, label in [('co', 'Co'), ('fe', 'Fe')]:
    print(f"\n================ {label} ionization @ ~19.48 d ================")
    print(f"{'code':13s} {'epoch':>6s}  {'v[km/s]':>8s}  " +
          "  ".join(f"{label} {ROMAN[s]:>3s}" for s in (2, 3, 4, 5)))
    for c in CODES:
        fn = os.path.join(DATA, f"ionfrac_{elem}_toy06_{c}.txt")
        if not os.path.exists(fn):
            continue
        ep, out = frac_at(fn, elem, TGT)
        if out is None:
            print(f"{c:13s} {ep:6.2f}  (no 19.48d / parse fail)"); continue
        for tv in TGT:
            vact, row = out[tv]
            cells = " ".join(f"{row.get(s, float('nan')):7.4f}" for s in (2, 3, 4, 5))
            tag = "" if abs(ep - 19.48) < 0.6 else " *epoch!"
            print(f"{c:13s} {ep:6.2f}  {vact:8.0f}  {cells}{tag}")
