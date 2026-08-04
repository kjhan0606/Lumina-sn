"""parity35-slwrite pre-registered battery (ledger 2026-07-27, written BEFORE the run).

Usage: judge_parity35.py [<run_dir=logs/coevolve_consume_parity35>]

Single effective variable vs parity33: LUMINA_SL_WRITE_SKIPZ=1 on withParityT
(the (A) repair: Si II/III get an NLTE line source instead of the silent LTE
B(T_e) fallback; nebular tau preserved). mode-2 CONSUME stays OFF (rejected).

Registered items:
  W1 wiring      : stdout contains "[SL-WRITE]" AND the final linedump has
                   Si II+III S_l>0 on ~1459 in-window lines (was 0 in parity33).
  P1 front-B producer bands (J_fine/truth, 4 bands) vs parity33
                   1.9006 / 0.5643 / 0.3083 / 0.3331.
                   CHARACTERIZATION, direction NOT registered: the Si source
                   change feeds the fine solve; movement TOWARD 1.0 = the LTE Si
                   source was part of the front-B transfer defect; away = it
                   was compensating other defects. Report, attribute only the
                   sign.
  P2 populations : b4, b9 (Si III s8, resolve_raw). CHARACTERIZATION — the pump
                   field at 1206.5/1113.2A changes with the Si source. No
                   pass/fail threshold; the parity34 lesson stands (the probe
                   moves only if the transported field at those lines moves).
  P3 front A     : pins hi == 0 (hard), FORMAL-CONS reported (parity33 3.484,
                   NOT a gate here: the Si emission change legitimately moves
                   the emitted luminosity; register the VALUE, judge later).
  P4 control     : forest E(400-906A) s8 within +-10% of parity33 1.5403e8
                   (Si lines are all in-window; EUV band should be untouched
                   to first order).
  P5 risk watch  : T_e / n_e / Si + Fe ion fractions (direction not registered;
                   ION_LOCK held T_e/n_e fixed in parity34 — if they move here,
                   say so loudly, it means the Si source reached the thermal
                   solve).
"""
import sys, os, csv, re
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'logs/coevolve_consume_parity35'
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

print(f"=== parity35 battery on {RUN} ===\n")

# W1 -----------------------------------------------------------------------
try:
    out = open(p('stdout.log'), errors='replace').read()
except OSError:
    out = ''
banner = 'SL-WRITE' in out
import csv as _csv
si = set()
for r in _csv.DictReader(open('data/tardis_reference_toy06_19p48d/line_list.csv')):
    if int(r['atomic_number']) == 14 and int(r['ion_number']) in (1, 2):
        si.add(int(r['line_id']))
sipos = sitot = 0
try:
    dd = np.genfromtxt(p('cmf_fine_linedump_s8.csv'), delimiter=',', names=True)
    ii = {int(v): i for i, v in enumerate(dd['line_id'].astype(np.int64))}
    sitot = sum(1 for l in si if l in ii)
    sipos = sum(1 for l in si if l in ii and dd['S_l'][ii[l]] > 0)
except OSError:
    pass
print("W1 WIRING  [banner + Si S_l>0 in the final linedump]")
print(f"    [SL-WRITE] banner : {banner}   {flag(banner)}")
print(f"    Si S_l>0          : {sipos}/{sitot}   {flag(sipos > 1000)}   (parity33: 0/1459)")
if not (banner and sipos > 1000):
    print("    => repair not wired through this run; characterization below is VOID.")

# P1 -----------------------------------------------------------------------
print("\nP1 FRONT-B PRODUCER BANDS  [J_fine/truth vs parity33; characterization]")
try:
    bb = bands(RUN)
    for (a, b), v, r in zip(BANDS, bb, REF_BANDS):
        q = v / r
        d_new, d_old = abs(v - 1.0), abs(r - 1.0)
        lbl = ("unchanged" if abs(q - 1.0) < 5e-3 else
               "toward truth" if d_new < d_old else "away from truth")
        print(f"    {a:>4}-{b:<7}      : {v:.4f}  (p33 {r:.4f})  {q:.3f}x   {lbl}")
except Exception as e:
    print(f"    band read failed: {e}")

# P2 -----------------------------------------------------------------------
b4, b9 = bfac(RUN, 4), bfac(RUN, 9)
print("\nP2 POPULATIONS  [characterization, no threshold]")
if b4 is not None: print(f"    b4 Si III s8      : {b4:.4f}   (parity33 {REF_B4})   {100*(b4/REF_B4-1):+.1f}%")
if b9 is not None: print(f"    b9 Si III s8      : {b9:.4f}   (parity33 {REF_B9})   {100*(b9/REF_B9-1):+.1f}%")

# P4 control: forest E
E = forest_E(RUN)
dE = E / REF_E - 1
print(f"\nP4 CONTROL forest E(400-906A): {E:.4e}  (p33 {REF_E:.4e})  {100*dE:+.2f}%   "
      f"{flag(abs(dE) <= 0.10)}")

# P3 -----------------------------------------------------------------------
hi, lo, R = pins_and_formal(RUN)
print(f"\nP3 FRONT A")
print(f"    pins hi/lo        : {hi}/{lo}   {flag(hi == 0)}")
print(f"    [FORMAL-CONS] R   : {R}   (p33 3.484; VALUE registered, not gated — "
      f"Si emission legitimately moves L)")

# P5 -----------------------------------------------------------------------
Te, ne = state(RUN)
print(f"\nP5 RISK WATCH (direction not registered)")
print(f"    s8 T_e            : {Te:.1f}  (p33 {REF_TE:.1f})  {100*(Te/REF_TE-1):+.2f}%")
print(f"    s8 n_e            : {ne:.4e} (p33 {REF_NE:.4e}) {100*(ne/REF_NE-1):+.2f}%")
print(f"    Si s8             : { '{' + ', '.join(f'{k}:{v:.4f}' for k,v in ionfrac(RUN,14).items()) + '}' }")
print(f"      parity33        : {ionfrac(REF,14)}")
print(f"    Fe s8             : { '{' + ', '.join(f'{k}:{v:.4f}' for k,v in ionfrac(RUN,26).items()) + '}' }")
