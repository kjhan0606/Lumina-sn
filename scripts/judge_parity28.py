#!/usr/bin/env python3
"""parity28-binfield pre-registered judgment battery (ledger entry 2026-07-25).
Usage: judge_parity28.py <run_dir> [<ref_run_dir=logs/coevolve_consume_parity27>]
Reads only run-dir artifacts (post-preservation). Prints the battery table with
registered thresholds and PASS/FAIL; observation items print values only.
"""
import sys, os, csv, re
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'logs/coevolve_consume_parity28'
REF = sys.argv[2] if len(sys.argv) > 2 else 'logs/coevolve_consume_parity27'

def p(path, run=RUN): return os.path.join(run, path)

def bfac(run, fname, lev):
    with open(p(fname, run)) as f:
        for line in f:
            c = line.split(',')
            if c[0] == '8' and c[1] == '14' and c[2] == '2' and c[3] == str(lev):
                return float(c[8])
    return None

def flag(ok): return 'PASS' if ok else 'FAIL'

print(f"=== parity28 battery on {RUN} ===\n")

# (1) EUV bin J_bin, final iter, s8
th = {14: 3e7, 15: 1.4e6, 16: 3.6e4}
truth = {14: 2.4e6, 15: 5.2e4, 16: 4.6e4}
rows = {}
maxit = -1
with open(p('lumina_c1_bins.csv')) as f:
    r = csv.DictReader(f)
    for row in r:
        it = int(row['iter'])
        maxit = max(maxit, it)
        if row['shell'] == '8':
            rows[(it, int(row['bin']))] = row
print(f"(1) s8 EUV bin J_bin (final iter {maxit}); threshold = registered <, truth-target noted")
for b in (13, 14, 15, 16, 17):
    row = rows.get((maxit, b))
    if row is None: print(f"  bin{b}: MISSING"); continue
    J = float(row['J_bin']); mode = row['mode']; W = float(row['W']); TR = float(row['T_R'])
    if b in th:
        print(f"  bin{b}: J_bin={J:.4e} (th <{th[b]:.1e}, truth~{truth[b]:.1e}) W={W:.3e} T_R={TR:.1f} {mode}  [{flag(J < th[b])}]")
    else:
        print(f"  bin{b}: J_bin={J:.4e} W={W:.3e} T_R={TR:.1f} {mode}  [observation (5): truth color 10707K]")

# (2) forest emission 400-906A @s8
sys.path.insert(0, 'scripts')
from read_events import EVENT_DTYPE, LINE_DTYPE
ev = np.memmap(p('lumina_events.bin'), dtype=EVENT_DTYPE, mode='r', offset=32)
s8 = ev[ev['shell'] == 8]
lam = 2.99792458e18 / np.maximum(s8['nu_comov'], 1.0)
m = (s8['etype'] == 2) & (lam >= 400) & (lam < 906)
E = float(s8['energy'][m].sum())
print(f"\n(2) s8 emit 400-906A: E={E:.5f} (th <=0.036; parity27=0.180)  [{flag(E <= 0.036)}]")

# (9b) 0x16 routing share
n16 = int((s8['chan'] == 0x16).sum()); nact = int((s8['etype'] == 1).sum())
print(f"(9b) s8 0x16 router: {n16:,} = {n16/max(nact,1)*100:.2f}% of line-abs (parity27: 25,308 = 2.84%)  [observation]")

# (3)(4) resolve b-factors
for lev, th_, tag in ((9, 3.0, '(3) b9'), (4, 1.5, '(4) b4')):
    be = bfac(RUN, 'lumina_levelpop_resolve_ema.csv', lev)
    br = bfac(RUN, 'lumina_levelpop_resolve_raw.csv', lev)
    bi = bfac(RUN, 'lumina_levelpop.csv', lev)
    ok = (be is not None and be < th_) and (br is not None and br < th_)
    print(f"{tag}: resolve ema={be:.3f} raw={br:.3f} in-run={bi:.3f} (th <{th_})  [{flag(ok)}]")

# (6) banner M = X + Y
so = open(p('stdout.log')).read()
bf = re.findall(r'\[IUP-BINFIELD\] it\s*(\d+):\s*([\d,]+) lines.*?jblue ([\d,]+) / fallback ([\d,]+)\).*?bypass=([\d,]+)', so)
jb = re.findall(r'\[IUP-JBLUE\] it(\d+): up-rate lines using J_blue=(\d+)\s+fallback\(J_line\)=(\d+)', so)
if bf:
    it_, M, X, Y, Z = bf[-1]
    M, X, Y, Z = (int(x.replace(',', '')) for x in (M, X, Y, Z))
    print(f"\n(6) [IUP-BINFIELD] it{it_}: M={M:,} X={X:,} Y={Y:,} bypass={Z:,}  M==X+Y: {flag(M == X + Y)}  bypass~0: {flag(Z < M * 0.001)}")
else:
    print(f"\n(6) [IUP-BINFIELD] banner: NOT FOUND  [FAIL — gate not armed?]  ([IUP-JBLUE] lines: {len(jb)})")

# (7) footer env-diff vs ref
def footer_env(run):
    txt = open(p('stdout.log', run)).read()
    mm = re.search(r'=== RUN FOOTER.*?=== END RUN FOOTER', txt, re.S)
    return set(l.strip() for l in mm.group(0).splitlines() if l.strip().startswith('LUMINA_')) if mm else set()
a, b = footer_env(RUN), footer_env(REF)
print(f"\n(7) footer env-diff vs {REF}:")
for x in sorted(a - b): print(f"  + {x}")
for x in sorted(b - a): print(f"  - {x}")

# (8) plasma s8 + observation
with open(p('lumina_plasma_state.csv')) as f:
    for line in f:
        if line.startswith('8,'):
            _, W_, Tr_, ne_, Te_ = line.split(',')
            print(f"\n(8) s8: T_e={float(Te_):.1f} n_e={float(ne_):.3e} (parity27: 11293.7 / 7.522e8)  [observation]")

# (9a) 1113A jbar vs CMFGEN truth
TRUTH_J = {274798: 2.317e-6, 242336: 9.859e-7, 242412: 9.83e-7, 241268: 9.589e-7}
found = {}
with open(p('lumina_jbar_dump.csv')) as f:
    r = csv.DictReader(f)
    for row in r:
        if row['shell'] == '8' and int(row['line_idx']) in TRUTH_J and int(row['iter']) >= 11:
            found[int(row['line_idx'])] = float(row['jbar_line'])
print(f"(9a) s8 line jbar vs CMFGEN truth (parity26: 2.8x/70x/186x/25x):")
for li, jt in TRUTH_J.items():
    if li in found:
        print(f"  line {li}: jbar={found[li]:.3e} = {found[li]/jt:.1f}x truth")
