#!/usr/bin/env python3
"""SKIP_Z A/B judgment battery — REGISTERED 2026-07-27 before either run started.

Arms (single effective variable):
    parity36a  LUMINA_NLTE_SKIP_Z=14   (control; silicon keeps nebular tau)
    parity36b  LUMINA_NLTE_SKIP_Z=     (treatment; silicon is NLTE like the rest)
Everything else identical, same binary (withParityS = the parity33 binary), same
jbar dump ion list (14:1,14:2 — Si II added so the Si II half of the SKIP_Z
rationale finally has data).

WHY THIS A/B, in one line: tau and S_l must come from the same populations,
because tau's stim_corr = d/(1+d) cancels S_l = (2hv^3/c^2)/d EXACTLY
(S_l*tau = (2hv^3/c^2)*C*f*lambda*t*n_upper*(g_l/g_u); verified numerically to
1.000000). SKIP_Z breaks that pairing and parity35 showed what the broken pairing
costs: FORMAL-CONS 3.484 -> 5973.

REGISTERED ITEMS
  W1  WIRING (hard).  a: RESOLVED CONFIG contains LUMINA_NLTE_SKIP_Z=14 and the
      dump holds Si II rows.  b: RESOLVED CONFIG line is exactly
      "LUMINA_NLTE_SKIP_Z=", no "[NLTE-GPU] ... SKIP_Z active" banner anywhere,
      dump holds Si II rows.  Either arm failing -> that arm is void.

  F1  FALSIFIER of the central claim (hard, one number, registered direction).
      parity36b FORMAL-CONS R must satisfy  R < 35  (= 10x parity33's 3.484).
      Rationale: with the pairing restored, the S_l divergence is cancelled by
      construction, so no energy can be manufactured — whatever else changes,
      the scalar energy gate must stay in the same order of magnitude as the
      pre-repair baseline. parity35's broken pairing gave 5973. If b lands at
      1e2 or above, the cancellation argument is WRONG and the whole (A')
      framing must be re-opened. This is the item that can kill the analysis.

  M1  SKIP_Z RATIONALE, MEASURED (characterization, no threshold).
      The skip list exists because (lumina_plasma.c, above nlte_skip_z_load):
        "NLTE rate matrices can collapse populations of dominant ions (e.g.
         Si II) in inner shells ... producing tau values many orders of
         magnitude below the Saha-Boltzmann nebular estimate"
      beta in the jbar dump is radeq_beta_esc(tau_sobolev) at the consumption
      point, so arm a's beta inverts to tau_NEBULAR and arm b's to tau_NLTE, per
      (shell,line_idx) — the same internal index in both arms (same binary, same
      line list). Report per shell and per ion: median ratio tau_NLTE/tau_neb,
      the fraction below 1e-3 ("many orders below"), and — the part that decides
      whether opacity is actually lost — the same statistics restricted to lines
      that are NEBULAR-THICK (tau_neb > 1), plus how many of those stay thick.
      Already known for Si III from parity35: NO collapse, ratio 50-2000x in the
      photosphere, nebular-thick lines all stay thick. Si II is the open half.

  M2  Si II 6355 (characterization).  Silicon line blanketing is the point of
      this A/B, and Si II 6355 is the single most diagnostic SN Ia feature.
      Report the flux integral in 6000-6400 A normalised by a red-side
      pseudo-continuum (6600-6900 A) for both arms. A large move is expected and
      is NOT a failure by itself — it is the measurement.

  M3  Front-B producer bands (characterization) median J_fine/truth in the four
      bands vs parity33's 1.9006 / 0.5643 / 0.3083 / 0.3331, using the SAME truth
      table and the SAME estimator as judge_parity35.py's P1
      (parity27/analysis/jblue_yardstick_audit/truth_s8_fullrange.csv, interpolated
      onto the linedump grid). A first draft of this battery divided by the local
      Planck B instead and would have compared 0.73 against 1.90 — two different
      yardsticks reported as one trend. Caught in the pre-run dry run.

  M5  CONTROL forest E(400-906 A) at s8 vs parity33's 1.5403e8, +-10%. Same
      definition as judge_parity35.py's P4. A silicon-only change must not move
      the EUV forest energy by more than that.

  M4  Thermal/ionization state (characterization, report loudly if it moves):
      s8 T_e and n_e, Si and Fe ion fractions at s8, and the far-outer T_e run
      (s40-49) that front A is chasing.

  C1  CONTROL a vs parity33 (characterization, NOT a falsifier).  The dump is
      write-only observation, so physics should not move; but this arm runs on a
      different machine with OMP_NUM_THREADS=16 (slurm allocation) against
      parity33's 32, and thread count can reorder reductions. Report row-level
      differences in c1_bins and plasma_state; interpret a nonzero count as
      thread-order noise unless it is large.
"""
import csv, math, os, sys
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else 'logs'
A = os.path.join(BASE, 'coevolve_consume_parity36a')
B = os.path.join(BASE, 'coevolve_consume_parity36b')
P33 = os.path.join(BASE, 'coevolve_consume_parity33')
P33_BANDS = (1.9006, 0.5643, 0.3083, 0.3331)
P33_FORMAL = 3.484
P33_FOREST_E = 1.5403e8
F1_LIMIT = 35.0
BANDS = [(1000, 1300), (1300, 2000), (2000, 3000), (3000, 4000)]
# Same truth table judge_parity35.py P1 uses — do not substitute Planck here.
TRUTH = ('logs/coevolve_consume_parity27/analysis/jblue_yardstick_audit/'
         'truth_s8_fullrange.csv')

def rd(p):
    return open(p, errors='replace').read() if os.path.exists(p) else ''

def formal_R(run):
    for ln in rd(os.path.join(run, 'stdout.log')).splitlines()[::-1]:
        if '[FORMAL-CONS]' in ln and ' x L_inj' in ln:
            try: return float(ln.split('=')[2].split('x L_inj')[0].strip().split()[-1])
            except Exception: return None
    return None

def ok(b): return 'PASS' if b else 'FAIL'

print("=== parity36 SKIP_Z A/B battery ===\n")
for tag, run in (('a(SKIP_Z=14)', A), ('b(no SKIP_Z)', B)):
    print(f"  {tag:14s} {'present' if os.path.isdir(run) else 'MISSING':8s} {run}")
print()

# --- W1 ----------------------------------------------------------------------
print("W1 WIRING")
for tag, run, want_skip in (('a', A, True), ('b', B, False)):
    so = rd(os.path.join(run, 'stdout.log'))
    if not so:
        print(f"    {tag}: stdout missing"); continue
    lines = [l.rstrip() for l in so.splitlines()]
    has14 = '  LUMINA_NLTE_SKIP_Z=14' in lines
    hasempty = '  LUMINA_NLTE_SKIP_Z=' in lines
    banner = 'NLTE_SKIP_Z active' in so
    ions = '  LUMINA_JBAR_DUMP_IONS=14:1,14:2' in lines
    if want_skip:
        good = has14 and banner and ions
        print(f"    a: SKIP_Z=14 {has14}  banner {banner}  dump_ions {ions}   {ok(good)}")
    else:
        good = hasempty and (not has14) and (not banner) and ions
        print(f"    b: SKIP_Z empty {hasempty}  no-14 {not has14}  no-banner {not banner}"
              f"  dump_ions {ions}   {ok(good)}")

def dump_ions(run):
    p = os.path.join(run, 'lumina_jbar_dump.csv')
    if not os.path.exists(p): return {}
    seen = {}
    for r in csv.DictReader(open(p)):
        seen[(r['Z'], r['ion'])] = seen.get((r['Z'], r['ion']), 0) + 1
    return seen
for tag, run in (('a', A), ('b', B)):
    print(f"    {tag}: dump ions -> {dump_ions(run)}")

# --- F1 ----------------------------------------------------------------------
print("\nF1 FALSIFIER  [b's FORMAL-CONS must stay below 35 = 10x parity33]")
Ra, Rb = formal_R(A), formal_R(B)
print(f"    parity33 (reference)   : {P33_FORMAL}")
print(f"    parity35 (broken pair) : 5973   <- what a broken tau/S pairing costs")
print(f"    a (SKIP_Z=14)          : {Ra}")
print(f"    b (no SKIP_Z)          : {Rb}")
if Rb is None:
    print("    verdict: cannot judge — no FORMAL-CONS line in b")
else:
    good = Rb < F1_LIMIT
    print(f"    verdict: {ok(good)}"
          + ("" if good else "   <-- cancellation argument REFUTED; re-open (A') framing"))

# --- M1 ----------------------------------------------------------------------
print("\nM1 SKIP_Z RATIONALE MEASURED  [tau_NLTE (b) vs tau_nebular (a), per line+shell]")

def inv_beta(b):
    """radeq_beta_esc inverse: beta=(1-e^-t)/t, monotone decreasing."""
    if not (0.0 < b < 1.0): return None
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        bm = 1.0 / mid if mid > 700.0 else (1.0 - math.exp(-mid)) / mid
        if bm > b: lo = mid
        else: hi = mid
    return math.sqrt(lo * hi)

def taus(run):
    """(shell,line_idx,ion) -> tau, from the FINAL iter, mode-3 rows only."""
    p = os.path.join(run, 'lumina_jbar_dump.csv')
    if not os.path.exists(p): return {}
    rows = []
    mx = -1
    for r in csv.DictReader(open(p)):
        it = int(r['iter'])
        if it > mx: mx = it
        rows.append(r)
    out = {}
    for r in rows:
        if int(r['iter']) != mx or r['mode'] != '3': continue
        t = inv_beta(float(r['beta']))
        if t is None: continue
        out[(int(r['shell']), int(r['line_idx']), int(r['ion']))] = t
    return out

ta, tb = taus(A), taus(B)
common = sorted(set(ta) & set(tb))
print(f"    matched (shell,line,ion) cells with finite tau in BOTH arms: {len(common)}")
if common:
    for ion, name in ((1, 'Si II'), (2, 'Si III')):
        ks = [k for k in common if k[2] == ion]
        if not ks:
            print(f"    {name}: no matched cells"); continue
        print(f"\n    --- {name} ({len(ks)} cells) ---")
        print(f"    {'shells':>10} {'n':>6} {'med tau_neb':>12} {'med tau_NLTE':>13}"
              f" {'med ratio':>11} {'frac<1e-3':>10} {'thick_neb':>10} {'stay thick':>11}")
        for lo, hi, lab in ((0, 12, 's0-11'), (12, 25, 's12-24'),
                            (25, 40, 's25-39'), (40, 50, 's40-49')):
            sel = [k for k in ks if lo <= k[0] < hi]
            if not sel: continue
            n = np.array([ta[k] for k in sel]); m = np.array([tb[k] for k in sel])
            rt = m / np.maximum(n, 1e-300)
            th = n > 1.0
            stay = (m[th] > 1.0).mean() if th.any() else float('nan')
            print(f"    {lab:>10} {len(sel):6d} {np.median(n):12.3e} {np.median(m):13.3e}"
                  f" {np.median(rt):11.3e} {(rt < 1e-3).mean():10.3f}"
                  f" {int(th.sum()):10d} {stay:11.3f}")
    print("\n    reading: ratio << 1 with nebular-thick lines going thin = the stated")
    print("    SKIP_Z rationale is real and silicon opacity is lost without it.")
    print("    ratio >= 1 = the rationale is a fossil; SKIP_Z is only breaking the pairing.")

# --- M2 ----------------------------------------------------------------------
print("\nM2 Si II 6355 FEATURE  [flux in 6000-6400 A over a 6600-6900 A pseudo-continuum]")
def spec(run):
    p = os.path.join(run, 'lumina_spectrum_formal.csv')
    if not os.path.exists(p): return None
    d = np.genfromtxt(p, delimiter=',', names=True)
    return d['wavelength_angstrom'], d['flux']
for tag, run in (('parity33', P33), ('a', A), ('b', B)):
    s = spec(run)
    if s is None: print(f"    {tag:8s} spectrum missing"); continue
    wl, fx = s
    m1 = (wl >= 6000) & (wl < 6400); m2 = (wl >= 6600) & (wl < 6900)
    if m2.sum() == 0 or fx[m2].mean() <= 0:
        print(f"    {tag:8s} no continuum window"); continue
    print(f"    {tag:8s} feature/continuum = {fx[m1].mean() / fx[m2].mean():.4f}")

# --- M3 ----------------------------------------------------------------------
print("\nM3 FRONT-B PRODUCER BANDS  [J_fine/truth; parity33 = "
      f"{'/'.join(f'{x:.4f}' for x in P33_BANDS)}]")
def bands(run):
    p = os.path.join(run, 'cmf_fine_linedump_s8.csv')
    if not (os.path.exists(p) and os.path.exists(TRUTH)): return None
    tr = np.genfromtxt(TRUTH, delimiter=',', names=True)
    tl, tj = tr['wavelength_A'], tr['J_nu']
    o = np.argsort(tl); tl, tj = tl[o], tj[o]
    d = np.genfromtxt(p, delimiter=',', names=True)
    lam, jf = d['lambda_A'], d['J_fine']
    t = np.interp(lam, tl, tj)
    out = []
    for lo, hi in BANDS:
        m = (lam >= lo) & (lam < hi)
        out.append(float(np.median(jf[m] / t[m])) if m.sum() else float('nan'))
    return out
for tag, run in (('a', A), ('b', B)):
    v = bands(run)
    if v is None: print(f"    {tag}: linedump or truth table missing"); continue
    print(f"    {tag}: " + "  ".join(f"{x:.4f}({x/r:.3f}x)" for x, r in zip(v, P33_BANDS)))

# --- M5 ----------------------------------------------------------------------
print("\nM5 CONTROL forest E(400-906 A) at s8  [parity33 = 1.5403e8, +-10%]")
def forest_E(run, lo_A=400.0, hi_A=906.0, shell='8'):
    p = os.path.join(run, 'lumina_c1_bins.csv')
    if not os.path.exists(p): return None
    rows, maxit = [], -1
    for row in csv.DictReader(open(p)):
        it = int(row['iter']); maxit = max(maxit, it)
        if row['shell'] == shell: rows.append((it, row))
    return sum(float(r['J_bin']) for it, r in rows if it == maxit
               and float(r['lam_lo_A']) >= lo_A and float(r['lam_hi_A']) <= hi_A)
for tag, run in (('a', A), ('b', B)):
    E = forest_E(run)
    if E is None: print(f"    {tag}: c1_bins missing"); continue
    d = E / P33_FOREST_E - 1
    print(f"    {tag}: {E:.4e}  ({100*d:+.2f}%)   {ok(abs(d) <= 0.10)}")

# --- M4 ----------------------------------------------------------------------
print("\nM4 THERMAL / IONIZATION STATE")
def state(run):
    p = os.path.join(run, 'lumina_plasma_state.csv')
    if not os.path.exists(p): return None
    return {int(r['shell_id']): (float(r['T_e']), float(r['n_e']))
            for r in csv.DictReader(open(p))}
sa, sb = state(A), state(B)
if sa and sb:
    print(f"    {'shell':>6} {'T_e(a)':>10} {'T_e(b)':>10} {'ratio':>8}"
          f" {'n_e(a)':>12} {'n_e(b)':>12} {'ratio':>8}")
    for s in (8, 20, 30, 41, 45, 49):
        if s not in sa or s not in sb: continue
        (Ta, na), (Tb, nb) = sa[s], sb[s]
        print(f"    {s:6d} {Ta:10.1f} {Tb:10.1f} {Tb/max(Ta,1e-30):8.4f}"
              f" {na:12.4e} {nb:12.4e} {nb/max(na,1e-30):8.4f}")
def ionfrac(run, Z, shell=8):
    p = os.path.join(run, 'lumina_ion_pops.csv')
    if not os.path.exists(p): return {}
    tot, o = 0.0, {}
    for r in csv.DictReader(open(p)):
        if int(r['shell_id']) != shell or int(r['Z']) != Z: continue
        n = float(r['n_ion']); o[int(r['stage'])] = n; tot += n
    return {k: v / tot for k, v in o.items()} if tot > 0 else {}
for Z, nm in ((14, 'Si'), (26, 'Fe')):
    fa, fb = ionfrac(A, Z), ionfrac(B, Z)
    fmt = lambda f: '{' + ', '.join(f'{k}:{v:.4f}' for k, v in sorted(f.items()) if v > 1e-4) + '}'
    print(f"    {nm} s8  a {fmt(fa)}")
    print(f"    {nm} s8  b {fmt(fb)}")

# --- C1 ----------------------------------------------------------------------
print("\nC1 CONTROL  [a vs parity33 — expected equal; OMP 16 vs 32 + different host]")
def rowdiff(p, q):
    if not (os.path.exists(p) and os.path.exists(q)): return None
    la, lb = open(p).read().splitlines(), open(q).read().splitlines()
    if len(la) != len(lb): return (-1, len(la), len(lb))
    return (sum(1 for x, y in zip(la, lb) if x != y), len(la), len(lb))
for f in ('lumina_c1_bins.csv', 'lumina_plasma_state.csv', 'lumina_ion_pops.csv'):
    d = rowdiff(os.path.join(P33, f), os.path.join(A, f))
    print(f"    {f:28s} {'missing' if d is None else f'{d[0]} of {d[1]} rows differ'}")
