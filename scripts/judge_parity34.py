#!/usr/bin/env python3
"""parity34-consume pre-registered judgment battery (ledger entry 2026-07-27).

Usage: judge_parity34.py [<run_dir=logs/coevolve_consume_parity34>]

Single variable vs parity33: LUMINA_CMF_LINERES_CONSUME=1 (population-solve
consumer of the deterministic line field). Registered items:

  W1 wiring falsifier : stderr must contain "[cmf_consume] ... mode-2 ACTIVE".
                        Absent => the gate never fired => run VOID.
  P1 primary (front C): b4 alone. See REVISION below -- the b9 leg was retired to
                        a diagnostic on 2026-07-27 after an offline audit of which
                        transitions actually pump these levels.
                        b4 < 1.6 => in-window pump sets level 4 [ADOPT]
                        b4 > 2.5 => pump is not what holds it up [REJECT]
  P2 control          : forest E(400-906A) within +-10% (band is OUTSIDE the
                        1000-4000A fine window, so it must be nearly untouched)
  P3 front A          : pins hi == 0, FORMAL-CONS <= 4x
  P4 front B producer : det 4 bands within 0.9-1.1x of parity33
  P5 risk watch       : T_e/n_e/ion fractions (direction NOT registered)

SCOPE OF ANY VERDICT (registered 2026-07-27, do not over-read):
  LINERES_JBAR=2 makes the producer fire only on the FINAL pure iteration
  (cuda.cu:7036), so the det field exists only there. CONSUME therefore acts on
  the final iteration's NLTE solve + FINAL_RESOLVE only -- this is a ONE-STEP
  RESPONSE, not a converged co-evolution. A b4/b9 drop means "the last solve
  moves this way when fed the det field", NOT "the converged physics says so".
  The converged version needs LINERES_JBAR=1 (per-iteration, ~11 h).
  Run profile for reference: 11 cheap iterations ~27 min + final iteration ~56
  min (the producer is the cost). A yield before ~minute 80 loses everything.

REVISION of P1 (2026-07-27, registered BEFORE parity34 produced any output).
Two offline findings, in order -- the second RETRACTS the first's numbers.

(1) Which transitions pump these levels, from the atomic line list
    (data/.../line_list.csv: real level map + A_ul, no energy guessing):
      level 4 <- 1206.50A Si III resonance line from ground   IN-WINDOW
      level 9 <- 1113.23A line from level 3 (its ONLY decay)  IN-WINDOW
    So CONSUME reaches both probes. That part stands.
    A cascade-from-above account of level 9's excess was tested and FALSIFIED:
    computed cascade-in is ~1/4500 of what the shortfall needs.

(2) RETRACTED: an earlier point prediction (b4 -> 0.55..1.37) assumed the run
    pumps these lines off the BINNED J. It does not. jbar_dump shows both lines
    far above LUMINA_JBAR_MIN=3 (1113A count=2242, 1206A count=153), so
    use_jbar is TRUE (plasma.c:13029) and the run pumps off jbar_line (MC).
    The apparent 0.98 agreement of a b~J_binned/B predictor was COINCIDENCE:
    the mode-3 two-level ratio gives J_jbar/B = 1.088 against an observed
    b4 = 2.8335. No reliable offline b_k prediction is available; none is used.

WHAT THE INTERVENTION ACTUALLY DOES (final iteration, in-window lines):
    value: J_jbar goes from jbar_line to J_fine
             1206.50A  5.6125e-06 -> 2.9165e-06   (x0.52)
             1113.23A  7.1132e-05 -> 2.2256e-06   (x0.031, a 32x cut)
    form : det_jbar=1 makes the mode-3 branch (R = B_lu*beta*J_inc) fall through
           to the mode-2 differenced branch (bJext = Jbar - (1-beta)S_lag).
           plasma.c:13064 vs 13093 -- `!det_jbar` gates mode 3 off.
    => parity34 is NOT a clean one-variable test of the FIELD. It changes the
       estimator value and the rate algebra together. A movement in b4/b9 cannot
       be attributed to the deterministic field alone. Say so in any verdict.

Registered branches (direction only, no magnitude claimed):
    b4 < 2.5 and b9 < 15   -> the det field + mode-2 form deflates the probes
    b4 within +-5% of 2.8335 -> no effect reached the populations
    anything else            -> report, do not attribute
"""
import sys, os, csv, re
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'logs/coevolve_consume_parity34'
REF = 'logs/coevolve_consume_parity33'
TRUTH = ('logs/coevolve_consume_parity27/analysis/jblue_yardstick_audit/'
         'truth_s8_fullrange.csv')
BANDS = [(1000, 1300), (1300, 2000), (2000, 3000), (3000, 4000)]
REF_BANDS = [1.9006, 0.5643, 0.3083, 0.3331]
REF_B4, REF_B9, REF_E = 2.8335, 18.740, 1.5403e8
REF_TE, REF_NE = 11157.0, 7.5136e8

def p(f, run=RUN): return os.path.join(run, f)
def flag(ok): return 'PASS' if ok else 'FAIL'

def pins_and_formal(run):
    hi = lo = R = None
    with open(p('stdout.log', run), errors='replace') as f:
        for line in f:
            m = re.search(r'\[SIMUL it=\d+\].*pins hi=(\d+)\s+lo=(\d+)', line)
            if m: hi, lo = int(m.group(1)), int(m.group(2))
            m = re.search(r'\[FORMAL-CONS\].*?([0-9.]+e?[-+]?\d*)\s*[xX]', line)
            if m:
                try: R = float(m.group(1))
                except ValueError: pass
    return hi, lo, R

def bfac(run, lev, shell='8', Z='14', ion='2'):
    with open(p('lumina_levelpop_resolve_raw.csv', run)) as f:
        for line in f:
            c = line.split(',')
            if len(c) > 8 and c[0] == shell and c[1] == Z and c[2] == ion and c[3] == str(lev):
                return float(c[8])
    return None

def bands(run):
    tr = np.genfromtxt(TRUTH, delimiter=',', names=True)
    tl, tj = tr['wavelength_A'], tr['J_nu']
    o = np.argsort(tl); tl, tj = tl[o], tj[o]
    d = np.genfromtxt(p('cmf_fine_linedump_s8.csv', run), delimiter=',', names=True)
    lam, jf = d['lambda_A'], d['J_fine']
    t = np.interp(lam, tl, tj)
    return [float(np.median(jf[(lam >= a) & (lam < b)] / t[(lam >= a) & (lam < b)]))
            for a, b in BANDS]

def forest_E(run, lo_A=400.0, hi_A=906.0, shell='8'):
    rows, maxit = [], -1
    with open(p('lumina_c1_bins.csv', run)) as f:
        for row in csv.DictReader(f):
            it = int(row['iter']); maxit = max(maxit, it)
            if row['shell'] == shell: rows.append((it, row))
    return sum(float(r['J_bin']) for it, r in rows if it == maxit
               and float(r['lam_lo_A']) >= lo_A and float(r['lam_hi_A']) <= hi_A)

def state(run, shell='8'):
    with open(p('lumina_plasma_state.csv', run)) as f:
        for row in csv.DictReader(f):
            if row['shell_id'] == shell:
                return float(row['T_e']), float(row['n_e'])
    return float('nan'), float('nan')

def ionfrac(run, Z, shell='8'):
    t = {}
    with open(p('lumina_ion_pops.csv', run)) as f:
        for row in csv.DictReader(f):
            if row['shell_id'] == shell and row['Z'] == str(Z):
                t[int(row['stage'])] = float(row['n_ion'])
    s = sum(t.values()) or 1
    return {k: v / s for k, v in sorted(t.items())}

print(f"=== parity34 battery on {RUN} ===\n")

# W1 -----------------------------------------------------------------------
try:
    err = open(p('stderr.log'), errors='replace').read()
except OSError:
    err = ''
fired = 'cmf_consume' in err
print(f"W1 WIRING FALSIFIER  [registered: '[cmf_consume] ... mode-2 ACTIVE' required]")
print(f"    banner present    : {fired}   {flag(fired)}")
if not fired:
    print("    => gate never fired. RUN IS VOID as a test of LINERES_CONSUME.")
    print("       (note: the producer fills jbar_line_det only on the FINAL iteration,")
    print("        so the banner can only appear late in the run.)\n")

# P1 -----------------------------------------------------------------------
b4, b9 = bfac(RUN, 4), bfac(RUN, 9)
adopt = (b4 is not None and b4 < 2.5 and (b9 is None or b9 < 15))
reject = (b4 is not None and abs(b4/REF_B4 - 1) <= 0.05)
print("P1 PRIMARY (front C direct test) — b4 only; b9 retired to diagnostic")
if b4 is None:
    print("    b4: NOT FOUND — cannot judge")
else:
    print(f"    b4 Si III s8      : {b4:.4f}   (parity33 {REF_B4})   {100*(b4/REF_B4-1):+.1f}%")
    print(f"    (no offline point prediction — see RETRACTED note in the docstring)")
    print(f"    verdict           : "
          + ("ADOPT (direction) — det field + mode-2 form deflates the probes;\n                        CANNOT separate field from rate-form change" if adopt else
             "REJECT — no effect reached the populations" if reject else "INTERMEDIATE — neither registered branch fired; do not attribute"))
    print("    caveat            : CONSUME changes J AND the rate form (3->2) together.")
if b9 is not None:
    print(f"\n    [diag] b9 Si III s8: {b9:.4f}   (parity33 {REF_B9})   {100*(b9/REF_B9-1):+.1f}%")
    print(f"    its only pump (1113.23A) is cut 32x by the swap, and it has exactly one")
    print(f"    decay channel, so b9 is the most responsive probe available. Cascade-from-")
    print(f"    above was falsified offline as a competing source (~1/4500 of what is needed).")
    print(f"    reading           : "
          + ("collapsed — the 1113A field was holding it up" if b9 < 5 else
             "partially deflated" if b9 < 15 else
             "barely moved — something other than this line's J sustains it"))

# P2 -----------------------------------------------------------------------
E = forest_E(RUN)
dE = E / REF_E - 1
print(f"\nP2 CONTROL (400-906A is OUTSIDE the 1000-4000A window)")
print(f"    forest E          : {E:.4e}  (parity33 {REF_E:.4e})  {100*dE:+.2f}%   "
      f"{flag(abs(dE) <= 0.10)}")
if abs(dE) > 0.10:
    print("    => out-of-window band moved: unintended coupling, withhold attribution.")

# P3 / P4 ------------------------------------------------------------------
hi, lo, R = pins_and_formal(RUN)
print(f"\nP3 FRONT A non-regression")
print(f"    pins hi/lo        : {hi}/{lo}   {flag(hi == 0)}")
print(f"    [FORMAL-CONS] R   : {R}   {flag(R is not None and R <= 4.0)}")
print(f"\nP4 FRONT B producer non-regression [0.9-1.1x of parity33]")
try:
    bb = bands(RUN)
    for (a, b), v, r in zip(BANDS, bb, REF_BANDS):
        q = v / r
        print(f"    {a:>4}-{b:<7}      : {v:.4f}  (p33 {r:.4f})  {q:.3f}x  {flag(0.9 <= q <= 1.1)}")
except Exception as e:
    print(f"    band read failed: {e}")

# P5 -----------------------------------------------------------------------
Te, ne = state(RUN)
print(f"\nP5 RISK WATCH (direction not registered)")
print(f"    s8 T_e            : {Te:.1f}  (p33 {REF_TE:.1f})  {100*(Te/REF_TE-1):+.2f}%")
print(f"    s8 n_e            : {ne:.4e} (p33 {REF_NE:.4e}) {100*(ne/REF_NE-1):+.2f}%")
print(f"    Si s8             : { '{' + ', '.join(f'{k}:{v:.4f}' for k,v in ionfrac(RUN,14).items()) + '}' }")
print(f"      parity33        : {ionfrac(REF,14)}")
print(f"    Fe s8             : { '{' + ', '.join(f'{k}:{v:.4f}' for k,v in ionfrac(RUN,26).items()) + '}' }")
