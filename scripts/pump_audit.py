#!/usr/bin/env python3
"""Which transitions pump each NLTE level, how much of its departure coefficient
they explain, and what LUMINA_CMF_LINERES_CONSUME would do to it. Offline, from a
completed run's artifacts plus the atomic line list.

Usage: pump_audit.py [<run_dir>] [<Z>] [<ion>] [<shell>]
       defaults: logs/coevolve_consume_parity33 14 2 8   (Si III, shell 8)

Predictor
---------
Radiative steady state for level k, counting only pumps from below and A-decays
to below (no cascade-in from above, no collisions, no recombination):

    n_k SUM_j A_kj = SUM_j n_j B_jk Jbar_jk      (j < k)

With n = b n* and the LTE relation n_j* B_jk B_nu(T_e) = n_k* A_kj (hv/kT >> 1):

    b_k = SUM_j A_kj b_j (Jbar/B)_jk / SUM_j A_kj

i.e. an A_ul-WEIGHTED mean of b_lower * (Jbar/B) over the pump transitions. The
line->level map, A_ul and B_lu come from the atomic line list -- nothing is
inferred from energy matching.

Reading the output
------------------
pred(binned) uses the J the run actually pumped with, so "agree" = pred/observed
says how much of the level's population the pumps from below account for:
  agree ~ 1   level is pump-controlled -> the CONSUME swap is predictable
  agree < 1   an additional source (cascade from above, recombination,
              collisions) supplies the shortfall
  agree > 1   decays to below are not the only sink, or a pump is overcounted
pred(fine) is the same formula with the deterministic field substituted on
in-window lines only (out-of-window lines keep the binned J -- exactly what
CONSUME does), i.e. the prediction for a parity34-class run.

This predictor reproduces the mechanism the code itself uses; agreement licenses
predicting the swap, it is not independent evidence about nature.
"""
import csv, os, sys
import numpy as np

RUN   = sys.argv[1] if len(sys.argv) > 1 else 'logs/coevolve_consume_parity33'
Z     = int(sys.argv[2]) if len(sys.argv) > 2 else 14
ION   = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SHELL = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LINELIST = 'data/tardis_reference_toy06_19p48d/line_list.csv'
WIN = (1000.0, 4000.0)
H, C, KB = 6.62607015e-27, 2.99792458e10, 1.380649e-16   # cgs

def p(f): return os.path.join(RUN, f)

# ---- level table (E, g, b_k) ------------------------------------------------
lev = {}
with open(p('lumina_levelpop_resolve_raw.csv')) as f:
    for line in f:
        c = line.rstrip('\n').split(',')
        if len(c) < 9: continue
        try:
            if int(c[0]) != SHELL or int(c[1]) != Z or int(c[2]) != ION: continue
            lev[int(c[3])] = (float(c[4]), float(c[5]), float(c[8]))
        except ValueError: continue
if not lev: sys.exit(f"no levels for Z={Z} ion={ION} shell={SHELL} in {RUN}")
BK = {k: v[2] for k, v in lev.items()}
E  = {k: v[0] for k, v in lev.items()}
G  = {k: v[1] for k, v in lev.items()}

# ---- T_e ---------------------------------------------------------------------
Te = None
with open(p('lumina_plasma_state.csv')) as f:
    for r in csv.DictReader(f):
        if int(r['shell_id']) == SHELL: Te = float(r['T_e']); break

def planck_nu(nu, T):
    x = H * nu / (KB * T)
    return (2 * H * nu**3 / C**2) / np.expm1(x)

# ---- binned J(lambda) at this shell, final iteration -------------------------
bins, maxit = [], -1
with open(p('lumina_c1_bins.csv')) as f:
    for r in csv.DictReader(f):
        if int(r['shell']) != SHELL: continue
        it = int(r['iter']); maxit = max(maxit, it)
        bins.append((it, float(r['lam_lo_A']), float(r['lam_hi_A']), float(r['J_bin'])))
bins = sorted([(lo, hi, J) for it, lo, hi, J in bins if it == maxit])
blo = np.array([b[0] for b in bins]); bhi = np.array([b[1] for b in bins])
bJ  = np.array([b[2] for b in bins])
def J_binned(lam):
    i = np.searchsorted(blo, lam, side='right') - 1
    if 0 <= i < len(bJ) and lam < bhi[i]: return bJ[i]
    return np.nan

# ---- deterministic fine field per line (in-window only) ----------------------
d = np.genfromtxt(p('cmf_fine_linedump_s8.csv'), delimiter=',', names=True)
Jfine = {int(v): (d['J_fine'][i], d['J_binned'][i], d['B'][i])
         for i, v in enumerate(d['line_id'].astype(np.int64))}

# ---- atomic line list --------------------------------------------------------
pump = {}   # upper -> list of (lower, lam, A_ul, ratio_binned, ratio_fine, inwin)
with open(LINELIST) as f:
    for r in csv.DictReader(f):
        if int(r['atomic_number']) != Z or int(r['ion_number']) != ION: continue
        lo, up = int(r['level_number_lower']), int(r['level_number_upper'])
        if lo not in BK or up not in BK: continue
        lam = float(r['wavelength']); nu = float(r['nu']); A = float(r['A_ul'])
        lid = int(r['line_id'])
        B_nu = planck_nu(nu, Te)
        rec = Jfine.get(lid)
        if rec is not None and rec[2] > 0:
            jb, jf, Bl = rec[1], rec[0], rec[2]
        else:
            jb, jf, Bl = J_binned(lam), np.nan, B_nu
        if not np.isfinite(jb) or Bl <= 0: continue
        inwin = (WIN[0] <= lam < WIN[1]) and rec is not None and rec[0] >= 0
        rb = jb / Bl
        rf = (jf / Bl) if inwin else rb          # CONSUME leaves out-of-window alone
        pump.setdefault(up, []).append((lo, lam, A, rb, rf, inwin))

print(f"pump audit: {RUN}  Z={Z} ion={ION} shell={SHELL}  T_e={Te:.1f} K  (iter {maxit})")
print(f"line list: {LINELIST}   window {WIN[0]:.0f}-{WIN[1]:.0f} A")
print(f"predictor: b_k = SUM_j A_kj b_j (Jbar/B)_jk / SUM_j A_kj   (A_ul-weighted)\n")
print(f"{'lev':>4} {'E_eV':>8} {'g':>3} {'b_k obs':>9} {'nlines':>6} {'in-win':>6} "
      f"{'wA_in%':>7} {'pred(bin)':>10} {'agree':>6} {'pred(fine)':>11} {'drop':>7}")
print('-' * 100)
out = {}
for k in sorted(BK):
    ps = pump.get(k)
    if not ps:
        continue
    A = np.array([q[2] for q in ps]); rb = np.array([q[3] for q in ps])
    rf = np.array([q[4] for q in ps]); iw = np.array([q[5] for q in ps])
    bl = np.array([BK[q[0]] for q in ps])
    W = A.sum()
    if W <= 0: continue
    pb = float((A * bl * rb).sum() / W)
    pf = float((A * bl * rf).sum() / W)
    agree = pb / BK[k] if BK[k] > 0 else np.nan
    wA_in = 100.0 * A[iw].sum() / W
    drop = pb / pf if pf > 0 else np.inf
    out[k] = (pb, pf, agree)
    if pb > 1e3:
        # Pumped by a line where J >> B(T_e) -- i.e. an out-of-window EUV/FUV
        # transition whose local Planck function is exponentially tiny. The
        # radiative-only balance then predicts an absurd b_k, so something the
        # predictor omits (collisional de-excitation, line trapping/beta, cascade
        # sink) is what actually limits the level. Do not use these rows.
        print(f"{k:4d} {E[k]:8.3f} {G[k]:3.0f} {BK[k]:9.4f} {len(ps):6d} {int(iw.sum()):6d} "
              f"{wA_in:6.1f}%   predictor invalid (J/B={pb:.1e} on a super-thermal pump)")
        continue
    flag = '' if 0.7 <= agree <= 1.4 else '  <-- other source'
    print(f"{k:4d} {E[k]:8.3f} {G[k]:3.0f} {BK[k]:9.4f} {len(ps):6d} {int(iw.sum()):6d} "
          f"{wA_in:6.1f}% {pb:10.3f} {agree:6.2f} {pf:11.3f} {drop:6.2f}x{flag}")
print('-' * 100)
good = [k for k in out if 0.7 <= out[k][2] <= 1.4]
print(f"pump-controlled levels (0.7<=agree<=1.4): {len(good)} of {len(out)}  -> {sorted(good)[:20]}")
