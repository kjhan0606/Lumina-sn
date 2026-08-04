#!/usr/bin/env python3
"""parity33-both pre-registered judgment battery (ledger entry 2026-07-27).

Usage: judge_parity33.py [<run_dir=logs/coevolve_consume_parity33>]

Registered items:
  (1) front A retained : pins hi == 0 ; [FORMAL-CONS] R <= 4x ; outer T_e monotone turn-up
  (2) front B retained : 4-band J_fine/truth each within 0.5-2.0x of parity31's solo values
  (3) NEW AUTHORITY    : forest E(400-906A, s8), b4, b9 -- direction registered in the ledger
  (4) interaction      : any (1)/(2) item outside 2x of its solo run => declare interaction

Reads run-dir artifacts only (post-preservation). Truth = CMFGEN jnu4 (single standard).
"""
import sys, os, csv, re
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'logs/coevolve_consume_parity33'
A_SOLO = 'logs/coevolve_consume_parity32'      # front A solo
B_SOLO = 'logs/coevolve_consume_parity31'      # front B solo
TRUTH = ('logs/coevolve_consume_parity27/analysis/jblue_yardstick_audit/'
         'truth_s8_fullrange.csv')

BANDS = [(1000, 1300), (1300, 2000), (2000, 3000), (3000, 4000)]
B_REF = [2.0966, 0.6227, 0.3276, 0.3580]       # parity31 solo J_fine/truth medians
A_REF_FORMAL = 3.484                            # parity32 solo [FORMAL-CONS] R
B30 = [0.1244, 0.1675, 0.0207, 0.0016]          # parity30 (broken det) for context

def p(f, run=RUN): return os.path.join(run, f)
def flag(ok): return 'PASS' if ok else 'FAIL'

def pins_and_formal(run):
    """last [SIMUL it=N] pins line + last [FORMAL-CONS] ratio in stdout."""
    hi = lo = None; R = None
    with open(p('stdout.log', run), errors='replace') as f:
        for line in f:
            m = re.search(r'\[SIMUL it=\d+\].*pins hi=(\d+)\s+lo=(\d+)', line)
            if m: hi, lo = int(m.group(1)), int(m.group(2))
            m = re.search(r'\[FORMAL-CONS\].*?([0-9.]+e?[-+]?\d*)\s*[xX]', line)
            if m:
                try: R = float(m.group(1))
                except ValueError: pass
    return hi, lo, R

def te_profile(run):
    T = {}
    with open(p('lumina_plasma_state.csv', run)) as f:
        for row in csv.DictReader(f):
            T[int(row['shell_id'])] = float(row['T_e'])
    return T

def bands(run):
    tr = np.genfromtxt(TRUTH, delimiter=',', names=True)
    tl, tj = tr['wavelength_A'], tr['J_nu']
    o = np.argsort(tl); tl, tj = tl[o], tj[o]
    d = np.genfromtxt(p('cmf_fine_linedump_s8.csv', run), delimiter=',', names=True)
    lam, jf = d['lambda_A'], d['J_fine']
    truth_at = np.interp(lam, tl, tj)
    out = []
    for lo, hi in BANDS:
        m = (lam >= lo) & (lam < hi)
        out.append(float(np.median(jf[m] / truth_at[m])) if m.sum() else float('nan'))
    return out

def forest_E(run, lo_A=400.0, hi_A=906.0, shell='8'):
    """sum of J_bin over c1 bins inside the forest window, final iteration, s8."""
    rows, maxit = [], -1
    with open(p('lumina_c1_bins.csv', run)) as f:
        for row in csv.DictReader(f):
            it = int(row['iter']); maxit = max(maxit, it)
            if row['shell'] == shell: rows.append((it, row))
    tot = 0.0; n = 0
    for it, row in rows:
        if it != maxit: continue
        a, b = float(row['lam_lo_A']), float(row['lam_hi_A'])
        if a >= lo_A and b <= hi_A:
            tot += float(row['J_bin']); n += 1
    return tot, n, maxit

def bfac(run, lev, fname='lumina_levelpop_resolve_raw.csv', shell='8', Z='14', ion='2'):
    with open(p(fname, run)) as f:
        for line in f:
            c = line.split(',')
            if len(c) > 8 and c[0] == shell and c[1] == Z and c[2] == ion and c[3] == str(lev):
                return float(c[8])
    return None

print(f"=== parity33 battery on {RUN} ===\n")

# ---- (1) front A retained -------------------------------------------------
hi, lo, R = pins_and_formal(RUN)
hiA, loA, RA = pins_and_formal(A_SOLO)
print("(1) FRONT A retained  [registered: pins hi == 0, FORMAL-CONS <= 4x]")
print(f"    pins hi/lo        : {hi}/{lo}      (parity32 solo {hiA}/{loA})   {flag(hi == 0)}")
if R is not None:
    print(f"    [FORMAL-CONS] R   : {R:.3f}x     (parity32 solo {RA:.3f}x)      {flag(R <= 4.0)}")
else:
    print("    [FORMAL-CONS] R   : NOT FOUND in stdout — gate wiring check needed  FAIL")
T, TA = te_profile(RUN), te_profile(A_SOLO)
outer = [s for s in range(39, 50) if s in T]
seq = [T[s] for s in outer]
mono = all(b >= a * 0.98 for a, b in zip(seq, seq[1:]))
print(f"    outer T_e s39..s49: " + " ".join(f"{v:.0f}" for v in seq))
print(f"    parity32 solo     : " + " ".join(f"{TA[s]:.0f}" for s in outer if s in TA))
print(f"    monotone turn-up  : {flag(mono)}   (truth 11979 -> 24599)")
print(f"    deep T_e[0]       : {T.get(0, float('nan')):.0f}  (parity32 solo {TA.get(0, float('nan')):.0f}; truth 18760)\n")

# ---- (2) front B retained -------------------------------------------------
print("(2) FRONT B retained  [registered: each band within 0.5-2.0x of parity31 solo]")
try:
    bb = bands(RUN)
    print(f"    {'band':<12} {'p33':>9} {'p31 solo':>9} {'ratio':>7} {'p30 broken':>11}  verdict")
    ok2 = True
    for (blo, bhi), v, ref, v30 in zip(BANDS, bb, B_REF, B30):
        r = v / ref if ref else float('nan')
        good = 0.5 <= r <= 2.0
        ok2 &= good
        print(f"    {blo:>4}-{bhi:<7} {v:9.4f} {ref:9.4f} {r:7.2f} {v30:11.4f}  {flag(good)}")
    print(f"    front B overall   : {flag(ok2)}\n")
except Exception as e:
    print(f"    band read failed: {e}\n")

# ---- (3) newly authoritative observables ----------------------------------
print("(3) NEW AUTHORITY (both yardsticks healthy for the first time)")
try:
    E, nb, it = forest_E(RUN)
    EA, nbA, _ = forest_E(A_SOLO)
    EB, nbB, _ = forest_E(B_SOLO)
    print(f"    forest E 400-906A s8 (iter {it}, {nb} bins): {E:.4e}")
    print(f"      parity32 solo {EA:.4e} ({nbA} bins) | parity31 solo {EB:.4e} ({nbB} bins)")
except Exception as e:
    print(f"    forest E failed: {e}")
for lev, ref in ((4, 3.3115), (9, None)):
    v = bfac(RUN, lev)
    va, vb = bfac(A_SOLO, lev), bfac(B_SOLO, lev)
    tag = f" (parity30 {ref})" if ref else ""
    print(f"    b{lev} Si III s8    : {v}   [p32 {va} | p31 {vb}]{tag}")
print("    registered reading: b4 -> 1 means much of front C was yardstick contamination;")
print("                        b4 unchanged means the forest engine defect is a real, separate object.\n")

# ---- (4) interaction watch ------------------------------------------------
print("(4) INTERACTION WATCH: any item above outside 2x of its solo run => declare")
print("    interaction between the two repairs and withhold attribution.")
