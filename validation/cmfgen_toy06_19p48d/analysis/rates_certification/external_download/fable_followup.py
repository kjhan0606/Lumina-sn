#!/usr/bin/env python3
"""Follow-up probes: CaV f3 pattern, C6 ambiguity invariance, median dln chase,
3rd-largest raw line."""
import gzip
import os
import numpy as np

import fable_checks as fc

TOP = fc.TOP
NORAD = fc.NORAD
ATOMIC = fc.ATOMIC

# ---- 1. Ca V f3 tally by spin multiplicity
lines = gzip.open(os.path.join(TOP, 'p20.16.gz'), 'rt').read().split('\n')
i = 1
tally = {}
while i < len(lines):
    t = lines[i].split()
    if not t:
        i += 1
        continue
    slpi = tuple(int(x) for x in t)
    if slpi == (0, 0, 0, 0):
        break
    i += 1
    npn = int(lines[i].split()[1])
    i += 1
    f3 = float(lines[i].split()[0])
    i += 1 + npn
    key = (slpi[0], 0.0 if f3 == 0.0 else round(f3, 6))
    tally[key] = tally.get(key, 0) + 1
print('Ca V f3 tally by (2S+1, f3):')
for k in sorted(tally):
    print(f'  2S+1={k[0]}  f3={k[1]}  blocks={tally[k]}')

# ---- 2. C6 ambiguity: all fully-matching candidates, class invariance
ph = fc.parse_phot_entries(os.path.join(ATOMIC, 'NICK/II/19apr23/phot_data_A'))
t20 = [e for e in ph if e['typ'] == 20]
px = fc.parse_px_blocks(os.path.join(NORAD, 'ni2.px.txt'))
fp = {}
for bl in px:
    fp.setdefault(tuple(bl['S'][:8].tolist()), []).append(bl['idx'])

n_multi_full = 0
n_multi_fp = 0
invariant = True
for ei, e in enumerate(t20):
    key = tuple(e['s'][:8].tolist())
    fps = fp.get(key, [])
    if len(fps) > 1:
        n_multi_fp += 1
    full = [bi for bi in fps if px[bi]['ntot'] >= e['npts']
            and np.array_equal(px[bi]['S'][:e['npts']], e['s'])]
    if len(full) > 1:
        n_multi_full += 1
        classes = set()
        for bi in full:
            bl = px[bi]
            if e['npts'] == bl['ntot']:
                classes.add('IDENT')
            else:
                dE = np.diff(bl['E'])
                bad = np.nonzero(dE <= 0)[0]
                fni = int(bad[0] + 1) if bad.size else bl['ntot']
                classes.add('TRUNC_ok' if fni == e['npts'] else 'TRUNC_BAD')
        if len(classes) > 1:
            invariant = False
            print(f'  NON-INVARIANT entry {ei} {e["cfg"]}: candidates {full} classes {classes}')
print(f'entries with >1 fingerprint(first-8) candidates : {n_multi_fp}')
print(f'entries with >1 FULL preserved-range candidates : {n_multi_full}')
print(f'classification invariant across candidates      : {invariant}')

# duplicate blocks inside px.txt (identical sigma columns)
sig_fp = {}
for bl in px:
    sig_fp.setdefault((bl['ntot'],) + tuple(bl['S'][:8].tolist()), []).append(bl['idx'])
dups = {k: v for k, v in sig_fp.items() if len(v) > 1}
ndup_blocks = sum(len(v) for v in dups.values())
print(f'px.txt duplicate-fingerprint block groups: {len(dups)} covering {ndup_blocks} blocks')

# ---- 3. median dln chase
e0 = t20[0]
gd_lines = open(os.path.join(NORAD, 'ni2.px.gd.txt')).read().split('\n')
i = fc._dash_sep(gd_lines)
i = fc._skip_blank(gd_lines, i)
ntg = int(gd_lines[i].split()[2])
i += 1
ev = []
while len(ev) < ntg:
    ev += gd_lines[i].split()
    i += 1
i += 1  # slpi
i += 2  # BE/ntot + ac
gdE = np.array([float(gd_lines[i + k].split()[0]) for k in range(2166)])
gdS = np.array([float(gd_lines[i + k].split()[1]) for k in range(2166)])
for lab, g in [('CMFGEN x-grid', e0['x']), ('NORAD E-grid', gdE)]:
    d = np.diff(np.log(g))
    print(f'  {lab:14s}: median dln = {np.median(d):.6e}  mean {d.mean():.6e}')
# monotonic-sorted variant (Sigma class resorts non-ascending tables)
xs = np.sort(e0['x'])
print(f'  sorted x-grid : median dln = {np.median(np.diff(np.log(xs))):.6e}')
# positive-only variant
d = np.diff(np.log(e0['x']))
print(f'  x-grid, only increasing steps: median {np.median(d[d > 0]):.6e}  '
      f'(n_nonpos={int((d <= 0).sum())})')

# ---- 4. 3rd-largest raw line + CMFGEN x back-conversion
order = np.argsort(gdS)[::-1]
r3 = order[2]
print(f'  NORAD raw 3rd-largest line: {gd_lines[i + r3]!r}  (idx {r3})')
kmax = int(np.argmax(e0['s']))
print(f'  CMFGEN max at x={e0["x"][kmax]:.7f}; x*1.27605 = {e0["x"][kmax]*1.27605:.6f} '
      f'(report quoted 1.955590; raw NORAD E = {gdE[r3]:.6f})')
