#!/usr/bin/env python3
"""LUMINA_SL_WRITE_SKIPZ smoke battery (registered 2026-07-27, before the runs).

Three short runs (PKTS=3000 NITER=3 FINE_ALI=200), all other gates equal:
    smoke_S_off  lumina_cuda.withParityS, gate absent      -- reference
    smoke_T_off  lumina_cuda.withParityT, gate absent      -- must equal reference
    smoke_T_on   lumina_cuda.withParityT, SL_WRITE_SKIPZ=1 -- the repair

Registered items
  R1 non-regression : T_off must differ from S_off by NO MORE than one run of
                      S_off differs from an identical repeat of itself.
                      REVISED 2026-07-27 after the first attempt: the original
                      criterion was plain md5 byte-identity, which FAILED on
                      artifacts that are not bit-reproducible for reasons
                      unrelated to the edit --
                        lumina_jbar_dump.csv  : identical as an unordered SET;
                          only the ROW ORDER differs (parallel writer).
                        lumina_levelpop_*.csv : 1 row of 1,051,901 differed in
                          one field by 1 ULP (9.570841e-07 -> 9.570842e-07),
                          b_k unchanged.
                      A criterion sensitive to writer thread order and printf
                      last-digit rounding cannot decide this question. It is
                      replaced by a control-referenced one -- NOT relaxed:
                      run 23_smoke_S_rep (same binary, same config) and require
                      diff(S_off,T_off) <= diff(S_off,S_rep) per artifact.
                      An artifact that is identical as an unordered MULTISET of
                      records passes on that ground alone (row order in a
                      parallel-written dump carries no physics).
                      Without the control the R1 verdict is HELD, not passed.
  R2 wiring         : T_on stdout contains "[SL-WRITE] LUMINA_SL_WRITE_SKIPZ=1".
  R3 repair effect  : Si II+III in-window lines with S_l>0 goes 0 -> ~1459.
                      INSTRUMENT REVISION (2026-07-27, after the NITER=3 attempt):
                      the linedump's S_l column is a LAGGED snapshot -- the fine
                      producer dumps BEFORE the same iteration's tau/S_l update.
                      With NITER=3 and NLTE starting on the final iteration, the
                      dump precedes the ONLY update: measured ALL 786,556 lines
                      S_l=0 in every NITER=3 smoke (vs 96% >0 in 12-iter
                      parity33). The NITER=3 "R3 FAIL" was instrument blindness,
                      not a repair failure. R3/R4/R5 are therefore judged on a
                      NITER=4 pair (smoke_T_off4 / smoke_T_on4), whose final
                      dump reflects iteration 3's write.
  R4 tau preserved  : Si III per-line beta in lumina_jbar_dump.csv IDENTICAL
                      between T_on4 and T_off4. SKIP_Z must still keep nebular
                      tau; if beta moves, the edit changed tau -- STOP.
  R5 no side effect : non-Si lines' S_l and non-Si ions' b_k unchanged
                      T_on4 vs T_off4. (Si b_k SHOULD move here -- the repaired
                      S_l feeds iteration 4's solve.)

R1 and R4 are falsifiers: failing either means do not proceed to a judgment run.
"""
import csv, hashlib, os, sys
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else 'logs'
S_OFF = os.path.join(BASE, 'coevolve_consume_smoke_S_off')
T_OFF = os.path.join(BASE, 'coevolve_consume_smoke_T_off')
T_ON  = os.path.join(BASE, 'coevolve_consume_smoke_T_on')
S_REP = os.path.join(BASE, 'coevolve_consume_smoke_S_rep')
T_OFF4 = os.path.join(BASE, 'coevolve_consume_smoke_T_off4')
T_ON4  = os.path.join(BASE, 'coevolve_consume_smoke_T_on4')
LL = 'data/tardis_reference_toy06_19p48d/line_list.csv'
NUMERIC = ['cmf_fine_linedump_s8.csv', 'lumina_levelpop_resolve_raw.csv',
           'lumina_levelpop.csv', 'lumina_plasma_state.csv', 'lumina_c1_bins.csv',
           'lumina_ion_pops.csv', 'lumina_jbar_dump.csv', 'lumina_spectrum_formal.csv']

def md5(p):
    if not os.path.exists(p): return None
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def ok(b): return 'PASS' if b else 'FAIL'

def si_lines():
    s = set()
    for r in csv.DictReader(open(LL)):
        if int(r['atomic_number']) == 14 and int(r['ion_number']) in (1, 2):
            s.add(int(r['line_id']))
    return s

def dump(run):
    p = os.path.join(run, 'cmf_fine_linedump_s8.csv')
    if not os.path.exists(p): return None
    d = np.genfromtxt(p, delimiter=',', names=True)
    return d, {int(v): i for i, v in enumerate(d['line_id'].astype(np.int64))}

def beta_si(run):
    p = os.path.join(run, 'lumina_jbar_dump.csv')
    if not os.path.exists(p): return None
    rows, mx = [], -1
    for r in csv.DictReader(open(p)):
        if r['shell'] != '8': continue
        it = int(r['iter']); mx = max(mx, it); rows.append(r)
    return {int(r['line_idx']): float(r['beta']) for r in rows if int(r['iter']) == mx}

def bk(run):
    p = os.path.join(run, 'lumina_levelpop_resolve_raw.csv')
    if not os.path.exists(p): return {}
    o = {}
    for line in open(p):
        c = line.rstrip().split(',')
        if len(c) < 9: continue
        try:
            if int(c[0]) != 8: continue
            o[(int(c[1]), int(c[2]), int(c[3]))] = float(c[8])
        except ValueError: continue
    return o

print("=== LUMINA_SL_WRITE_SKIPZ smoke battery ===\n")
for tag, run in [('S_off', S_OFF), ('T_off', T_OFF), ('T_on', T_ON),
                 ('S_rep', S_REP), ('T_off4', T_OFF4), ('T_on4', T_ON4)]:
    print(f"  {tag:6s} {'present' if os.path.isdir(run) else 'MISSING':8s} {run}")
print()

# R1 --------------------------------------------------------------------------
print("R1 NON-REGRESSION  [T_off vs S_off, referenced to the S_off<->S_rep control]")

def compare(p, q):
    """(n_rows_differing, max_rel_diff, set_identical) for two CSV artifacts."""
    if not (os.path.exists(p) and os.path.exists(q)): return None
    la, lb = open(p).read().splitlines(), open(q).read().splitlines()
    if len(la) != len(lb): return (-1, float('inf'), False)
    setsame = sorted(la) == sorted(lb)
    d = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    mx = 0.0
    for i in d[:20000]:
        for x, y in zip(la[i].split(','), lb[i].split(',')):
            if x == y: continue
            try: a_, b_ = float(x), float(y)
            except ValueError: return (len(d), float('inf'), setsame)
            if a_ != 0: mx = max(mx, abs(b_ - a_) / abs(a_))
    return (len(d), mx, setsame)

have_ctl = os.path.isdir(S_REP)
print(f"    control run 23_smoke_S_rep : {'present' if have_ctl else 'NOT YET RUN — verdict HELD'}")
print(f"    {'artifact':34s} {'md5':>9} {'T_off vs S_off':>22} {'S_rep vs S_off (noise)':>24}")
verdicts = []
for f in NUMERIC:
    same = md5(os.path.join(S_OFF, f)) == md5(os.path.join(T_OFF, f))
    ce = compare(os.path.join(S_OFF, f), os.path.join(T_OFF, f))
    cc = compare(os.path.join(S_OFF, f), os.path.join(S_REP, f)) if have_ctl else None
    def fmt(c):
        if c is None: return '  —'
        n, mx, ss = c
        return f"{n} rows, {mx:.1e}{' [set=]' if ss else ''}"
    print(f"    {f:34s} {'same' if same else 'DIFFERS':>9} {fmt(ce):>22} {fmt(cc):>24}")
    # An unordered dump that holds exactly the same records is not a regression:
    # row order in a parallel-written CSV carries no physics. Everything else
    # must fall within the measured run-to-run noise.
    if same or (ce and ce[2]): verdicts.append(True)
    elif cc is None: verdicts.append(None)
    else: verdicts.append(ce[0] <= cc[0] and ce[1] <= max(cc[1], 0.0))
if any(v is None for v in verdicts):
    print("    verdict           : HELD — run the control before judging R1")
else:
    print(f"    verdict           : {ok(all(verdicts))}"
          + ("" if all(verdicts) else "   <-- exceeds run-to-run noise; STOP"))

# R2 --------------------------------------------------------------------------
print("\nR2 WIRING  [T_on must announce the gate]")
so = os.path.join(T_ON, 'stdout.log')
txt = open(so, errors='replace').read() if os.path.exists(so) else ''
fired = 'SL_WRITE_SKIPZ=1' in txt
print(f"    [SL-WRITE] banner : {fired}   {ok(fired)}")

# R3 --------------------------------------------------------------------------
print("\nR3 REPAIR EFFECT  [Si S_l>0 : 0 -> ~1459 on the NITER=4 pair; see docstring]")
si = si_lines()
for tag, run in [('T_off4', T_OFF4), ('T_on4', T_ON4)]:
    r = dump(run)
    if r is None: print(f"    {tag:6s} run not present yet"); continue
    d, idx = r
    tot = sum(1 for l in si if l in idx)
    pos = sum(1 for l in si if l in idx and d['S_l'][idx[l]] > 0)
    print(f"    {tag:6s} Si in-window {tot:5d}   S_l>0: {pos:5d}  ({100*pos/max(tot,1):5.1f}%)")

# R4 --------------------------------------------------------------------------
print("\nR4 TAU PRESERVED  [Si III beta identical T_on vs T_off — nebular tau kept]")
bo, bn = beta_si(T_OFF4), beta_si(T_ON4)
if not bo or not bn:
    print("    jbar_dump missing — cannot judge")
else:
    common = set(bo) & set(bn)
    diff = [k for k in common if bo[k] != bn[k]]
    print(f"    lines compared    : {len(common)}   differing beta: {len(diff)}   {ok(not diff)}")
    if diff:
        k = diff[0]
        print(f"    e.g. line {k}: {bo[k]:.6e} -> {bn[k]:.6e}")
        print("    <-- SKIP_Z tau was NOT preserved; the gate leaked into tau. STOP")

# R5 --------------------------------------------------------------------------
print("\nR5 NO SIDE EFFECT  [non-Si S_l and b_k unchanged T_on vs T_off]")
ro, rn = dump(T_OFF4), dump(T_ON4)
if ro and rn:
    (do, io), (dn, iN) = ro, rn
    ks = [l for l in io if l in iN and l not in si]
    a = np.array([do['S_l'][io[l]] for l in ks]); b = np.array([dn['S_l'][iN[l]] for l in ks])
    nd = int((a != b).sum())
    print(f"    non-Si lines      : {len(ks)}   S_l differing: {nd}   {ok(nd == 0)}")
ko, kn = bk(T_OFF4), bk(T_ON4)
if ko and kn:
    ks = [k for k in ko if k in kn and k[0] != 14]
    nd = sum(1 for k in ks if ko[k] != kn[k])
    print(f"    non-Si levels     : {len(ks)}   b_k differing: {nd}   {ok(nd == 0)}")
    ks14 = [k for k in ko if k in kn and k[0] == 14]
    nd14 = sum(1 for k in ks14 if ko[k] != kn[k])
    print(f"    Si levels (SHOULD move — repaired S_l feeds iter 4): {len(ks14)}   b_k differing: {nd14}"
          + ("   <-- no movement: repair still not reaching physics" if nd14 == 0 else ""))
