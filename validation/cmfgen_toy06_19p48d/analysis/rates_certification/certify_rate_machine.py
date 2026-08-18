#!/usr/bin/env python3
"""certify_rate_machine.py -- Stage-2 rate-machine certification.

Question: with CMFGEN's OWN field (J_nu) and OWN populations (n_l) pinned, does
LUMINA's sigma-bake + integration layer reproduce CMFGEN's own converged
photoionization rate (*PRRR)?

    Gamma_ion(d) = SUM_l [n_l(d)/n_ion(d)] * 4pi INT sigma_l(nu) J_nu(d)/(h nu) dnu

Offline python only.  src/ is NOT touched.  Nothing from a LUMINA *run* enters;
the only LUMINA object is the baked cross-section table, which is under test.

Layer ladder (each row changes exactly one thing w.r.t. the row above):

  A    sigma = CMFGEN's own sub_phot_gen fit, evaluated on CMFGEN's own
       196185-point CMF grid, CMFGEN model levels    -> must reproduce *PRRR
  A|g  same integral truncated to LUMINA's grid [1.5e14, 3e16] Hz
                                                     -> grid truncation
  Bpt  exact sigma sampled at the 1000 log-bin CENTRES x bin-averaged J
                                                     -> LUMINA's quadrature
  Bav  bin-AVERAGED exact sigma x bin-averaged J     -> "ideal bake";
                                                        Bpt/Bav = resonance blur
  C    sigma from LUMINA's baker (expand_atomic_data_cmfgen.bake_sigma_bf_grid,
       math reproduced verbatim below) fed the SAME phot file toy06 used
                                                     -> baker fit-type / tail
                                                        approximations
  C2   the same baker fed LUMINA's OWN phot file (the atomic-data vintage the
       shipped binary was built from)                -> provenance control
  D    sigma read verbatim from the shipped binary, indexed by the ref
       levels.csv row order, CMFGEN populations mapped onto those rows
                                                     -> THE CERTIFICATION NUMBER

Pre-registered gate: D/PRRR in [0.5,2.0] at every shell AND [0.8,1.25] at s6-s8.

Truth   = <ion>PRRR   (PR(SL,d) = n_SL * R_SL, subs/prrr_sl_v6.f:126)
Field   = EDDFACTOR (+_INFO); populations = POP{COB,IRON,SUL}
Parsers reused from ../gamma_coiii_alllevel/gamma_coiii_alllevel.py (already
gate-verified there: POPCOB round-trip, EDDFACTOR direct read, PRRR block).
"""
from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import re
import struct
import sys
import time

import numpy as np

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/gamma_coiii_alllevel'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from gamma_coiii_alllevel import (            # noqa: E402  (verified parsers)
    parse_popcob, parse_f_to_s, parse_prrr, read_eddfactor, rvtj_block,
)
from cmfgen_parser import parse_osc, parse_phot   # noqa: E402

H = 6.62607015e-27
C_LIGHT = 2.99792458e10
EV = 1.602176634e-12
FOURPI = 4.0 * math.pi
CM2EV = 1.239841984e-4

# LUMINA bf/NLTE grid -- src/lumina.h:487-489, binning per lumina_nlte_gemm.cu:142
NU_MIN, NU_MAX, NBIN = 1.5e14, 3.0e16, 1000

RUN = '/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
REF = os.path.join(ROOT, 'data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv')
BIN = os.path.join(ROOT, 'data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin')

LABELS = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
TARGET_V = [4264, 5720, 7176, 7904, 8632, 9360, 10088, 10816, 11544]

IONS = [
    dict(lab='Co III', Z=27, ion=2, cmf='CoIII', pop='POPCOB',
         osc='CoIII_F_OSCDAT', fts='CoIII_F_TO_S', phot='PHOTCoIII_A',
         lum_osc='data/atomic/cmfgen/COB/III/19apr23/osc_data',
         lum_phot='data/atomic/cmfgen/COB/III/19apr23/phot_data_A'),
    dict(lab='Fe III', Z=26, ion=2, cmf='FeIII', pop='POPIRON',
         osc='FeIII_F_OSCDAT', fts='FeIII_F_TO_S', phot='PHOTFeIII_A',
         lum_osc='data/atomic/cmfgen/FE/III/19apr23/osc_data',
         lum_phot='data/atomic/cmfgen/FE/III/19apr23/phot_data_A'),
    dict(lab='S II', Z=16, ion=1, cmf='S2', pop='POPSUL',
         osc='S2_F_OSCDAT', fts='S2_F_TO_S', phot='PHOTS2_A',
         lum_osc='data/atomic/cmfgen/SUL/II/19apr23/osc_data',
         lum_phot='data/atomic/cmfgen/SUL/II/19apr23/phot_data_A'),
    dict(lab='S III', Z=16, ion=2, cmf='SIII', pop='POPSUL',
         osc='SIII_F_OSCDAT', fts='SIII_F_TO_S', phot='PHOTSIII_A',
         lum_osc='data/atomic/cmfgen/SUL/III/19apr23/osc_data',
         lum_phot='data/atomic/cmfgen/SUL/III/19apr23/phot_data_A'),
]

TERM_RE = re.compile(r'\[[^\[\]]*\]\s*$')


def norm_cfg(s):
    return str(s).strip().lower()


def term_cfg(s):
    return TERM_RE.sub('', norm_cfg(s))


# --------------------------------------------------------------- sigma models
class Sigma:
    """Faithful CMFGEN newsubs/sub_phot_gen.f evaluator for the fit types used
    by the four target ions.

      type 1  Seaton         l.412 : RU=EDGE/nu ; s=1e-18*A0*(A1+(1-A1)RU)*RU^A2
      type 7  mod. Seaton    l.505 : RU=(EDGE+A3)/nu, identically 0 unless RU<=1
                                     (A3 in CMFGEN frequency units = 1e15 Hz)
      type 20/21 tabulated   l.387 : linear interpolation in u=nu/EDGE over the
                                     (NU_NORM, CROSS_A) table; ABOVE the last
                                     node CMFGEN extrapolates CROSS_A[N]*(u_N/u)^3
    Types 2/3/8/9 need CMFGEN's hydrogenic gaunt tables (BF_L_CROSS/BF_N_GAUNT);
    flagged as non-evaluable and their population share is reported, never faked.
    """
    SUPPORTED = (1, 7, 20, 21, 22)

    def __init__(self, cs_type, energy, sigma_Mb, nu_th):
        self.cs = int(cs_type)
        self.nu_th = float(nu_th)
        self.ok = self.cs in self.SUPPORTED
        if self.cs in (20, 21, 22):
            self.u = np.asarray(energy, dtype=float)
            self.s = np.asarray(sigma_Mb, dtype=float) * 1e-18
            if self.u.size < 2:
                self.ok = False
            elif np.any(np.diff(self.u) < 0):        # np.interp needs ascending u
                o = np.argsort(self.u)
                self.u, self.s = self.u[o], self.s[o]
                self.resorted = True
        else:
            self.p = np.asarray(sigma_Mb, dtype=float)
            if self.cs == 1 and self.p.size < 3:
                self.ok = False
            if self.cs == 7 and self.p.size < 4:
                self.ok = False

    def __call__(self, nu):
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        out = np.zeros(nu.shape)
        if not self.ok or self.nu_th <= 0:
            return out
        if self.cs in (20, 21, 22):
            m = nu >= self.nu_th
            if not m.any():
                return out
            u = nu[m] / self.nu_th
            v = np.interp(u, self.u, self.s, left=self.s[0], right=0.0)
            tail = u >= self.u[-1]
            if tail.any():
                v[tail] = self.s[-1] * (self.u[-1] / u[tail]) ** 3
            out[m] = v
        elif self.cs == 1:
            if self.p[0] == 0.0:
                return out
            m = nu >= self.nu_th
            if not m.any():
                return out
            ru = self.nu_th / nu[m]
            out[m] = 1e-18 * self.p[0] * (self.p[1] + (1.0 - self.p[1]) * ru) * ru ** self.p[2]
        elif self.cs == 7:
            if self.p[0] == 0.0:
                return out
            edge = self.nu_th + self.p[3] * 1e15
            m = nu >= edge
            if not m.any():
                return out
            ru = edge / nu[m]
            out[m] = 1e-18 * self.p[0] * (self.p[1] + (1.0 - self.p[1]) * ru) * ru ** self.p[2]
        return out

    def nodes(self):
        """Abscissae where the exact sigma has structure (kinks / edges)."""
        if not self.ok:
            return np.empty(0)
        if self.cs in (20, 21, 22):
            return self.u * self.nu_th
        if self.cs == 7:
            return np.array([self.nu_th + self.p[3] * 1e15])
        return np.array([self.nu_th])


def bake_lumina(cs_type, energy, sigma_Mb, nu_th, nu_grid):
    """VERBATIM reproduction of the shipped baker's per-level math:
    scripts/expand_atomic_data_cmfgen.py:639-666 (identical to
    scripts/bake_cmfgen_sigma_bf_carsus.py:bake_one_level).

    Note the two deliberate approximations in that code:
      * tabulated: np.interp(..., right=sigma[-1])  -- HOLDS the last tabulated
        value for all nu above the table, whereas CMFGEN extrapolates (u_N/u)^3.
      * fit types 2,3,7,8,9: sigma_0 := params[0] (Mb) and a Kramers (nu_0/nu)^3
        shape.  For type 7 params[0] IS a cross-section, for types 2/3/8 it is a
        principal quantum number, for type 9 a Verner fit parameter.
    """
    nb = nu_grid.size
    sig = np.zeros(nb)
    if nu_th <= 0:
        return sig, False
    idx = int(np.searchsorted(nu_grid, nu_th))
    if idx >= nb:
        return sig, False
    if cs_type in (20, 21, 22):
        e = np.asarray(energy, float)
        s = np.asarray(sigma_Mb, float)
        if e.size == 0 or s.size == 0:
            return sig, False
        nu_pts = e * nu_th
        smb = s * 1.0e-18
        sig[idx:] = np.interp(nu_grid[idx:], nu_pts, smb, left=0.0, right=smb[-1])
        return sig, True
    p = np.asarray(sigma_Mb, float)
    if cs_type == 1 and p.size >= 3:
        y = nu_grid[idx:] / nu_th
        sig[idx:] = p[0] * (p[1] + (1.0 - p[1]) / y) * (y ** (-p[2])) * 1e-18
        return sig, True
    if cs_type in (2, 3, 7, 8, 9) and p.size >= 1:
        s0 = float(p[0]) * 1e-18
        if s0 <= 0:
            return sig, False
        y = nu_grid[idx:] / nu_th
        sig[idx:] = s0 / (y ** 3)
        return sig, True
    return sig, False


# ------------------------------------------------------------- binary reader
def read_bake_bin(path):
    with open(path, 'rb') as f:
        magic, ver = struct.unpack('<II', f.read(8))
        nlev, nbin = struct.unpack('<ii', f.read(8))
        numin, numax = struct.unpack('<dd', f.read(16))
        has = np.frombuffer(f.read(nlev), dtype='i1').astype(int)
        pad = (8 - (nlev % 8)) % 8
        f.read(pad)
        off = f.tell()
    sig = np.memmap(path, dtype='<f8', mode='r', offset=off, shape=(nlev, nbin))
    exp = off + nlev * nbin * 8
    return dict(magic=magic, ver=ver, nlev=nlev, nbin=nbin, numin=numin,
                numax=numax, has=has, sigma=sig, offset=off,
                size_ok=(os.path.getsize(path) == exp))


# ---------------------------------------------------------------- quadrature
def cum_from_below(nu, y):
    C = np.zeros(nu.size)
    C[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(nu))
    return C


def eval_cum(nu, y, C, x):
    x = np.clip(np.asarray(x, float), nu[0], nu[-1])
    i = np.clip(np.searchsorted(nu, x), 1, nu.size - 1)
    n0, n1 = nu[i - 1], nu[i]
    f = (x - n0) / (n1 - n0)
    yx = y[i - 1] + f * (y[i] - y[i - 1])
    return C[i - 1] + 0.5 * (y[i - 1] + yx) * (x - n0)


def bin_average(nu, y, edges):
    C = cum_from_below(nu, y)
    F = eval_cum(nu, y, C, edges)
    return np.diff(F) / np.diff(edges)


def trap_weights(x):
    """w such that INT y dx = w . y (trapezoid on a non-uniform grid)."""
    w = np.zeros(x.size)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def bin_integral_exact(sig_fn, edges, nu_th, n_sub=6):
    """INT_bin sigma dnu per bin: nodes = bin edges + the level's structure
    nodes + n_sub log-subdivisions, so every trapezoid segment sits in one bin."""
    nb = edges.size - 1
    out = np.zeros(nb)
    lo = max(nu_th, edges[0])
    if lo >= edges[-1]:
        return out
    parts = [np.array([lo, edges[-1]]), edges[(edges > lo) & (edges < edges[-1])]]
    ex = sig_fn.nodes()
    if ex.size:
        ex = ex[(ex > lo) & (ex < edges[-1])]
        if ex.size:
            parts.append(ex)
    k0 = max(int(np.searchsorted(edges, lo)) - 1, 0)
    le = np.log(edges[k0:])
    if le.size >= 2 and n_sub > 1:
        frac = (np.arange(1, n_sub) / n_sub)[None, :]
        sub = np.exp(le[:-1, None] + (le[1:, None] - le[:-1, None]) * frac).ravel()
        sub = sub[(sub > lo) & (sub < edges[-1])]
        if sub.size:
            parts.append(sub)
    x = np.unique(np.concatenate(parts))
    s = sig_fn(x)
    seg = 0.5 * (s[1:] + s[:-1]) * np.diff(x)
    mid = 0.5 * (x[1:] + x[:-1])
    b = np.clip(np.searchsorted(edges, mid) - 1, 0, nb - 1)
    np.add.at(out, b, seg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--ref', default=REF)
    ap.add_argument('--bin', default=BIN)
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--extra-depths', default='42,45,47,49,50',
                    help='extra 1-based CMFGEN depths to put through the SAME ladder '
                         '(diagnostic only; they are NOT part of the pre-registered gate)')
    a = ap.parse_args()
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    t0 = time.time()
    say('=' * 96)
    say('RATE-MACHINE CERTIFICATION   (LUMINA sigma-bake + integration layer vs CMFGEN *PRRR)')
    say(f'  run = {a.run}')
    say(f'  ref = {a.ref}')
    say(f'  bin = {a.bin}')
    say('=' * 96)

    # ---------------------------------------------------------------- field
    J, nu, ND, finish = read_eddfactor(f'{a.run}/EDDFACTOR')
    if not np.isfinite(finish) or finish == 0:
        raise SystemExit('EDDFACTOR incomplete (FINISH_REC=0)')
    rt = open(f'{a.run}/RVTJ').read()
    V = rvtj_block(rt, 'Velocity (km/s)', ND)
    Te = rvtj_block(rt, 'Temperature (10^4K)', ND) * 1e4
    say(f'[field] EDDFACTOR ND={ND} nfreq={J.shape[0]} nu=[{nu[0]:.4e},{nu[-1]:.4e}] Hz '
        f'FINISH={finish}')
    say(f'[field] RVTJ V=[{V[0]:.0f},{V[-1]:.0f}] km/s  Te=[{Te[0]:.0f},{Te[-1]:.0f}] K')

    shell_depth = {lb: int(np.argmin(np.abs(V - vt))) for lb, vt in zip(LABELS, TARGET_V)}
    GATE = [shell_depth[lb] for lb in LABELS]
    say('[shells] ' + '  '.join(f'{lb}:d{shell_depth[lb]+1}(v={V[shell_depth[lb]]:.0f})'
                                for lb in LABELS))
    XD = [int(t) - 1 for t in a.extra_depths.split(',') if t.strip()] if a.extra_depths else []
    XD = [d for d in XD if 0 <= d < ND and d not in GATE]
    XLAB = [f'd{d+1}' for d in XD]
    SD = GATE + XD
    ALAB = LABELS + XLAB
    NG = len(LABELS)
    if XD:
        say('[extra]  diagnostic depths (NOT gated): '
            + '  '.join(f'{lb}(v={V[d]:.0f})' for lb, d in zip(XLAB, XD)))

    dln = math.log(NU_MAX / NU_MIN) / NBIN
    edges = NU_MIN * np.exp(np.arange(NBIN + 1) * dln)
    nu_c = NU_MIN * np.exp((np.arange(NBIN) + 0.5) * dln)
    dnu = np.diff(edges)
    Wbin = FOURPI / (H * nu_c) * dnu       # LUMINA's per-bin weight, per unit sigma
    say(f'[grid]  LUMINA: {NBIN} log bins, nu=[{NU_MIN:.3e},{NU_MAX:.3e}] Hz  '
        f'lam=[{C_LIGHT*1e8/NU_MAX:.0f},{C_LIGHT*1e8/NU_MIN:.0f}] A  dlnnu={dln:.5e} '
        f'(={C_LIGHT*dln/1e5:.0f} km/s per bin)')

    Jbar = np.empty((NBIN, ND))
    for d in range(ND):
        Jbar[:, d] = bin_average(nu, J[:, d], edges)
    say(f'[grid]  bin-averaged J built for {ND} depths ({time.time()-t0:.1f}s)')
    # sanity: bin-average conserves INT J dnu over the LUMINA range
    d0 = SD[0]
    i_lo, i_hi = np.searchsorted(nu, NU_MIN), np.searchsorted(nu, NU_MAX)
    native_int = np.trapz(J[i_lo:i_hi, d0], nu[i_lo:i_hi])
    say(f'[grid]  quadrature check @s0: SUM Jbar*dnu = {np.sum(Jbar[:,d0]*dnu):.6e} vs '
        f'native INT J dnu over the same range = {native_int:.6e} '
        f'(ratio {np.sum(Jbar[:,d0]*dnu)/native_int:.6f})')

    # ------------------------------------------------------------ shipped bin
    bk = read_bake_bin(a.bin)
    say(f'[bin]   magic=0x{bk["magic"]:08X} ver={bk["ver"]} n_levels={bk["nlev"]} '
        f'n_bins={bk["nbin"]} nu=[{bk["numin"]:.4e},{bk["numax"]:.4e}] size_ok={bk["size_ok"]} '
        f'has_cmfgen={int(bk["has"].sum())} ({100*bk["has"].mean():.1f}%)')
    if bk['magic'] != 0x434D4644 or bk['nbin'] != NBIN or not bk['size_ok']:
        raise SystemExit('shipped binary header/size mismatch')
    if abs(bk['numin'] - NU_MIN) > 1 or abs(bk['numax'] - NU_MAX) > 1:
        raise SystemExit('shipped binary grid != lumina.h grid')

    ref_rows = []
    with open(os.path.join(a.ref, 'levels.csv')) as f:
        for r in _csv.DictReader(f):
            ref_rows.append((int(r['atomic_number']), int(r['ion_number']),
                             float(r['energy_eV']), float(r['g'])))
    ref_Z = np.array([r[0] for r in ref_rows])
    ref_i = np.array([r[1] for r in ref_rows])
    ref_E = np.array([r[2] for r in ref_rows])
    ref_g = np.array([r[3] for r in ref_rows])
    say(f'[ref]   levels.csv rows={len(ref_rows)} vs bin n_levels={bk["nlev"]}: '
        f'{"MATCH" if len(ref_rows)==bk["nlev"] else "MISMATCH"}')
    if len(ref_rows) != bk['nlev']:
        raise SystemExit('levels.csv row count != binary n_levels')
    say()

    popcache = {}
    rows_out = []
    summary = []

    for spec in IONS:
        lab, cmf = spec['lab'], spec['cmf']
        say('#' * 96)
        say(f'### {lab}   (CMFGEN species name "{cmf}")')
        say('#' * 96)
        ti = time.time()

        # --- populations -------------------------------------------------
        if spec['pop'] not in popcache:
            popcache[spec['pop']] = parse_popcob(f'{a.run}/{spec["pop"]}')
        pop_species, allions, order, nd2 = popcache[spec['pop']]
        if nd2 != ND:
            raise SystemExit('ND mismatch POP vs EDDFACTOR')
        CI = allions[cmf][0]
        NF = CI.shape[1]
        n_ion = CI.sum(axis=1)
        rt_bad = 0
        for k in range(len(order) - 1):
            dci = allions[order[k]][1]
            gnd = allions[order[k + 1]][0][:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(gnd > 0, dci / gnd, np.nan)
            rt_bad += int(np.nansum(np.abs(r - 1.0) > 1e-6))
        say(f'  [pop]  {spec["pop"]} ions={order}; {cmf} NF={NF} '
            f'n_ion=[{n_ion.min():.3e},{n_ion.max():.3e}] cm^-3; '
            f'DCI round-trip violations = {rt_bad} -> {"PASS" if rt_bad==0 else "FAIL"}')
        if rt_bad:
            raise SystemExit('POP round-trip failed -> parse is wrong, stop')

        # --- CMFGEN model levels ----------------------------------------
        oscp = os.path.join(a.run, spec['osc'])
        osc = parse_osc(oscp)
        if osc.n_levels < NF:
            raise SystemExit(f'{spec["osc"]}: {osc.n_levels} levels < NF={NF}')
        cfg = np.array([str(c) for c in osc.levels['config'][:NF]])
        Ecm = np.asarray(osc.levels['E_cm'][:NF], float)
        gl = np.asarray(osc.levels['g'][:NF], float)
        chi_eV = osc.ionization_eV
        E_eV = Ecm * CM2EV
        nu_th = (chi_eV - E_eV) * EV / H
        say(f'  [osc]  {spec["osc"]} -> {os.path.realpath(oscp)}')
        say(f'         nlev_file={osc.n_levels} used(NF)={NF} chi={chi_eV:.6f} eV; '
            f'lam_th: gnd={C_LIGHT*1e8/nu_th[0]:.1f} A, top={C_LIGHT*1e8/nu_th[NF-1]:.0f} A; '
            f'levels with nu_th<{NU_MIN:.1e}: {int((nu_th<NU_MIN).sum())}')

        SL, fts_names = parse_f_to_s(os.path.join(a.run, spec['fts']), osc.n_levels)
        order_ok = (list(fts_names[:NF]) == list(cfg))
        NS = int(SL[:NF].max())
        say(f'  [fts]  F_TO_S names identical to osc order over the first {NF}: {order_ok}; '
            f'NS={NS}')
        if not order_ok:
            raise SystemExit('level ordering could not be independently confirmed')

        # --- CMFGEN truth ------------------------------------------------
        PR, DI = parse_prrr(os.path.join(a.run, f'{cmf}PRRR'), cmf, ND, NS)
        G_truth = PR.sum(axis=0) / np.where(n_ion > 0, n_ion, np.nan)
        say(f'  [prrr] {cmf}PRRR rows={NS}; Gamma_truth(s0..s8) = '
            + ' '.join(f'{G_truth[d]:.3e}' for d in GATE))

        # SNAPSHOT SELF-CONSISTENCY: the PRRR file prints its own "Ion Density"
        # record.  It must track POP's next-ion ground population up to one fixed
        # scale.  Any depth where the ratio departs from that scale means PRRR and
        # POP were written from different states -> Gamma_PRRR is not a usable
        # truth there.  This is a property of the CMFGEN snapshot, not of LUMINA.
        ki = order.index(cmf)
        bandmask = np.zeros(ND, dtype=bool)
        if ki + 1 < len(order):
            gnext = allions[order[ki + 1]][0][:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                dir_ = np.where(gnext > 0, DI / gnext, np.nan)
            scale = float(np.nanmedian(dir_[GATE]))
            devi = dir_ / scale
            bandmask = np.isfinite(devi) & (np.abs(devi - 1.0) > 0.05)
            bd = np.where(bandmask)[0]
            say(f'  [snap] DI(PRRR)/ground({order[ki+1]},POP): reference scale over the 9 '
                f'gated shells = {scale:.4f}')
            if bd.size:
                say(f'         {bd.size}/{ND} depths deviate >5% from that scale: '
                    f'd{bd.min()+1}-d{bd.max()+1} (v={V[bd.max()]:.0f}-{V[bd.min()]:.0f} km/s), '
                    f'median deviation factor = {np.nanmedian(devi[bandmask]):.4f}')
                say('         -> PRRR and POP are NOT from the same state at those depths; '
                    'Gamma_PRRR is not a truth there.')
            else:
                say('         all 90 depths consistent within 5%.')
            gated_bad = [LABELS[i] for i, d in enumerate(GATE) if bandmask[d]]
            say(f'         gated shells inside the inconsistent band: '
                f'{gated_bad if gated_bad else "NONE"}')

        # --- CMFGEN sigma (the run's own phot file) ----------------------
        photp = os.path.join(a.run, spec['phot'])
        ph = parse_phot(photp)
        cfg_e, term_e = {}, {}
        for e in ph.entries:
            cfg_e.setdefault(norm_cfg(e.config), []).append(e)
            term_e.setdefault(term_cfg(e.config), []).append(e)
        sig_ex = [None] * NF
        cs_of = np.zeros(NF, dtype=int)
        ent_of = [None] * NF
        n_nomatch = 0
        for k in range(NF):
            ents = cfg_e.get(norm_cfg(cfg[k])) or term_e.get(term_cfg(cfg[k]))
            if not ents:
                n_nomatch += 1
                continue
            e = ents[0]
            ent_of[k] = e
            cs_of[k] = e.cs_type
            s = Sigma(e.cs_type, e.energy, e.sigma_Mb, nu_th[k])
            if s.ok:
                sig_ex[k] = s
        hist = {int(t): int((cs_of == t).sum()) for t in np.unique(cs_of)}
        npair = sum(e.n_points for e in ph.entries)
        say(f'  [phot] {spec["phot"]} -> {os.path.realpath(photp)}')
        say(f'         entries={len(ph.entries)} data_pairs={npair}; cs_type histogram over '
            f'the {NF} model levels = {hist}; unmatched levels = {n_nomatch}')
        nex = sum(s is not None for s in sig_ex)
        say(f'         levels with an EXACT CMFGEN evaluator: {nex}/{NF}')
        bad = np.array([s is None for s in sig_ex])
        share_bad = CI[:, bad].sum(axis=1) / np.where(n_ion > 0, n_ion, np.nan)
        say('         population share on non-evaluable levels, s0..s8: '
            + ' '.join(f'{share_bad[d]:.2e}' for d in SD))

        # --- layer A / A|grid --------------------------------------------
        keep = nu > 1e13
        nuA = np.ascontiguousarray(nu[keep])
        YA = np.ascontiguousarray(J[keep][:, SD] / nuA[:, None])   # J/nu at 9 depths
        nQ = len(SD)
        RA = np.zeros((NF, nQ))
        RAg = np.zeros((NF, nQ))
        jmax = int(np.searchsorted(nuA, NU_MAX))
        for k in range(NF):
            s = sig_ex[k]
            if s is None or nu_th[k] <= 0 or nu_th[k] >= nuA[-1]:
                continue
            i0 = int(np.searchsorted(nuA, nu_th[k]))
            x = np.empty(nuA.size - i0 + 1)
            x[0] = nu_th[k]
            x[1:] = nuA[i0:]
            Y = np.empty((x.size, nQ))
            Y[0] = [np.interp(nu_th[k], nuA, YA[:, q]) for q in range(nQ)]
            Y[1:] = YA[i0:]
            w = trap_weights(x) * s(x) * (FOURPI / H)
            RA[k] = w @ Y
            lo = max(nu_th[k], NU_MIN)
            if lo < NU_MAX:
                j0 = int(np.searchsorted(nuA, lo))
                xr = np.empty(jmax - j0 + 2)
                xr[0] = lo
                xr[1:-1] = nuA[j0:jmax]
                xr[-1] = NU_MAX
                Yr = np.empty((xr.size, nQ))
                for q in range(nQ):
                    Yr[0, q] = np.interp(lo, nuA, YA[:, q])
                    Yr[-1, q] = np.interp(NU_MAX, nuA, YA[:, q])
                Yr[1:-1] = YA[j0:jmax]
                wr = trap_weights(xr) * s(xr) * (FOURPI / H)
                RAg[k] = wr @ Yr
        say(f'  [A]    native-grid integrals done ({time.time()-ti:.1f}s; '
            f'{nuA.size} native points > 1e13 Hz)')

        # --- binned sigma flavours ---------------------------------------
        S_pt = np.zeros((NF, NBIN))
        S_av = np.zeros((NF, NBIN))
        S_ck = np.zeros((NF, NBIN))
        ck_ok = np.zeros(NF, dtype=bool)
        for k in range(NF):
            e = ent_of[k]
            if e is not None:
                sg, okc = bake_lumina(e.cs_type, e.energy, e.sigma_Mb, nu_th[k], nu_c)
                S_ck[k] = sg
                ck_ok[k] = okc
            s = sig_ex[k]
            if s is None:
                continue
            S_pt[k] = s(nu_c)
            S_av[k] = bin_integral_exact(s, edges, nu_th[k]) / dnu
        say(f'  [B/C]  binned sigma built ({time.time()-ti:.1f}s); baker replica covered '
            f'{int(ck_ok.sum())}/{NF} levels')

        # --- layer C2: baker on LUMINA's own atomic-data vintage ---------
        S_c2 = np.zeros((NF, NBIN))
        prov = 'n/a'
        lp_osc = os.path.join(ROOT, spec['lum_osc'])
        lp_pho = os.path.join(ROOT, spec['lum_phot'])
        if os.path.exists(lp_osc) and os.path.exists(lp_pho):
            losc = parse_osc(lp_osc)
            lph = parse_phot(lp_pho)
            lcfg_e, lterm_e = {}, {}
            for e in lph.entries:
                lcfg_e.setdefault(norm_cfg(e.config), []).append(e)
                lterm_e.setdefault(term_cfg(e.config), []).append(e)
            lE = np.asarray(losc.levels['E_cm'], float) * CM2EV
            lcfg = np.array([str(c) for c in losc.levels['config']])
            lchi = losc.ionization_eV
            nmatch2 = 0
            for k in range(min(NF, losc.n_levels)):
                nth = (lchi - lE[k]) * EV / H
                ents = lcfg_e.get(norm_cfg(lcfg[k])) or lterm_e.get(term_cfg(lcfg[k]))
                if not ents:
                    continue
                e = ents[0]
                sg, okc = bake_lumina(e.cs_type, e.energy, e.sigma_Mb, nth, nu_c)
                if okc:
                    S_c2[k] = sg
                    nmatch2 += 1
            say(f'  [C2]   LUMINA-side data: {spec["lum_osc"]} / {spec["lum_phot"]}'
                f'  (nlev={losc.n_levels}, chi={lchi:.6f} eV) baked {nmatch2}/{NF}')
        else:
            say(f'  [C2]   LUMINA-side data missing ({spec["lum_phot"]}) -- skipped')

        # --- layer D: shipped binary rows --------------------------------
        sel = np.where((ref_Z == spec['Z']) & (ref_i == spec['ion']))[0]
        rE, rg = ref_E[sel], ref_g[sel]
        row_of = np.full(NF, -1)
        ident = (sel.size >= NF and np.max(np.abs(rE[:NF] - E_eV)) < 1e-6
                 and np.array_equal(rg[:NF], gl))
        if ident:
            row_of[:] = sel[:NF]
            say(f'  [D]    ref rows for (Z={spec["Z"]},ion={spec["ion"]}) = {sel.size}; '
                f'first {NF} are 1:1 with the CMFGEN model levels '
                f'(max|dE|={np.max(np.abs(rE[:NF]-E_eV)):.2e} eV, g identical)')
        else:
            used = np.zeros(sel.size, dtype=bool)
            for k in range(NF):
                dE = np.abs(rE - E_eV[k])
                c = np.where((dE < 1e-3) & (np.abs(rg - gl[k]) < 0.5) & (~used))[0]
                if c.size == 0:
                    c = np.where((dE < 1e-3) & (~used))[0]
                if c.size == 0:
                    continue
                j = int(c[np.argmin(dE[c])])
                used[j] = True
                row_of[k] = sel[j]
            say(f'  [D]    ref rows = {sel.size}; matched by (E,g): '
                f'{int((row_of>=0).sum())}/{NF}')
        S_D = np.zeros((NF, NBIN))
        hasD = np.zeros(NF, dtype=bool)
        for k in range(NF):
            if row_of[k] >= 0:
                S_D[k] = np.asarray(bk['sigma'][row_of[k]])
                hasD[k] = bool(bk['has'][row_of[k]])
        posD = (S_D > 0).any(axis=1)
        say(f'         shipped rows has_cmfgen=1: {int(hasD.sum())}/{NF}; '
            f'sigma>0 anywhere: {int(posD.sum())}/{NF}')
        lostD = CI[:, ~posD].sum(axis=1) / np.where(n_ion > 0, n_ion, np.nan)
        say('         CMFGEN population share sitting on sigma==0 rows, s0..s8: '
            + ' '.join(f'{lostD[d]:.3f}' for d in SD))
        n_extra = sel.size - NF
        if n_extra > 0:
            ex_pos = int(((np.asarray(bk['sigma'][sel[NF:]]) > 0).any(axis=1)).sum())
            say(f'         SCOPE: the ref carries {n_extra} further levels for this ion that '
                f'the toy06 model atom does not have ({ex_pos} of them have sigma>0). They '
                f'cannot be tested here -- no CMFGEN population exists for them.')
        if S_c2.any():
            m = posD & (S_c2 > 0).any(axis=1)
            if m.any():
                num = np.abs(S_D[m] - S_c2[m]).sum()
                den = np.abs(S_D[m]).sum()
                prov = f'{num/den:.3e}'
            say(f'  [C2]   provenance: sum|D - C2| / sum|D| over co-covered levels = {prov} '
                f'(0 => the shipped binary IS the baker applied to that vintage)')

        # --- assemble ----------------------------------------------------
        p9 = np.ascontiguousarray(np.array([CI[d] / n_ion[d] for d in SD]).T)   # [NF,9]
        JS = np.ascontiguousarray(Jbar[:, SD])

        def binned(S):
            return ((S * Wbin[None, :]) @ JS * p9).sum(axis=0)

        GA = (RA * p9).sum(axis=0)
        GAg = (RAg * p9).sum(axis=0)
        GB = binned(S_pt)
        GBa = binned(S_av)
        GC = binned(S_ck)
        GC2 = binned(S_c2)
        GD = binned(S_D)
        GT = G_truth[SD]

        pall = CI / np.where(n_ion > 0, n_ion, np.nan)[:, None]      # [ND,NF]
        GD_all = (((S_D * Wbin[None, :]) @ Jbar) * pall.T).sum(axis=0)
        GC_all = (((S_ck * Wbin[None, :]) @ Jbar) * pall.T).sum(axis=0)

        def r(x, y):
            return x / y if (np.isfinite(y) and y != 0) else float('nan')

        say()
        say(f'  --- {lab}: LAYER LADDER (Gamma in s^-1; each ratio isolates ONE change) ---')
        say(f'  {"sh":>3} {"v":>6} {"Te":>6} | {"PRRR":>10} {"A":>10} {"A/PRRR":>7} '
            f'{"Ag/A":>6} {"Bpt/Ag":>7} {"Bav/Ag":>7} {"Bpt/Bav":>8} {"C/Bpt":>7} {"D/C":>7} '
            f'| {"D":>10} {"D/PRRR":>7}')
        for q, lb in enumerate(ALAB):
            d = SD[q]
            say(f'  {lb:>3} {V[d]:6.0f} {Te[d]:6.0f} | {GT[q]:10.3e} {GA[q]:10.3e} '
                f'{r(GA[q],GT[q]):7.3f} {r(GAg[q],GA[q]):6.3f} {r(GB[q],GAg[q]):7.3f} '
                f'{r(GBa[q],GAg[q]):7.3f} {r(GB[q],GBa[q]):8.3f} {r(GC[q],GB[q]):7.3f} '
                f'{r(GD[q],GC[q]):7.3f} | {GD[q]:10.3e} {r(GD[q],GT[q]):7.3f}')
            rows_out.append(dict(ion=lab, shell=lb, depth=d + 1, v_kms=V[d], Te_K=Te[d],
                                 n_ion=n_ion[d], G_prrr=GT[q], G_A=GA[q], G_Agrid=GAg[q],
                                 G_Bpt=GB[q], G_Bavg=GBa[q], G_C=GC[q], G_C2=GC2[q],
                                 G_D=GD[q],
                                 ratio_A_prrr=r(GA[q], GT[q]),
                                 ratio_B_prrr=r(GB[q], GT[q]),
                                 ratio_C_prrr=r(GC[q], GT[q]),
                                 ratio_D_prrr=r(GD[q], GT[q]),
                                 pop_share_sigma_zero=lostD[d],
                                 pop_share_no_exact_sigma=share_bad[d]))
        say()

        say(f'  --- {lab}: Gamma split by CMFGEN fit type (all shares normalised to '
            f'Gamma_Bpt, so they are directly comparable) ---')
        for q, lb in [(0, 's0'), (6, 's6')]:
            conB = ((S_pt * Wbin[None, :]) @ JS[:, q]) * p9[:, q]
            conC = ((S_ck * Wbin[None, :]) @ JS[:, q]) * p9[:, q]
            conD = ((S_D * Wbin[None, :]) @ JS[:, q]) * p9[:, q]
            tb = max(conB.sum(), 1e-300)
            for t in sorted(hist):
                m = cs_of == t
                say(f'   {lb} type{t:<2d} n={int(m.sum()):5d}  '
                    f'exact={conB[m].sum()/tb:9.5f}  baker(C)={conC[m].sum()/tb:9.5f}  '
                    f'shipped(D)={conD[m].sum()/tb:9.5f}   '
                    f'(D-exact) contributes {(conD[m].sum()-conB[m].sum())/tb:+.5f} of Gamma')
        say()

        # --- how well can a 1000-bin grid resolve these sigma tables? --------
        res = []
        for k in range(NF):
            s = sig_ex[k]
            if s is None or s.cs not in (20, 21, 22):
                continue
            u = s.u
            m = (u > 1.0) & (u < 5.0)
            if m.sum() > 3:
                res.append(np.median(np.diff(np.log(u[m]))))
        if res:
            res = np.array(res)
            say(f'  [res]  tabulated-sigma node spacing over 1<u<5: median dln(nu) = '
                f'{np.median(res):.2e} (min {res.min():.2e}, max {res.max():.2e}) vs LUMINA '
                f'bin dln(nu) = {dln:.2e}  ->  data/bin resolution ratio = '
                f'{dln/np.median(res):.2f}x')
            say('         (ratio > 1 means the bake grid is COARSER than the cross-section '
                'table, i.e. structure is being sampled, not resolved)')
        say()

        summary.append(dict(lab=lab, GT=GT, GA=GA, GAg=GAg, GB=GB, GBa=GBa, GC=GC,
                            GC2=GC2, GD=GD, lostD=lostD[SD], badshare=share_bad[SD],
                            hist=hist, GD_all=GD_all, GC_all=GC_all, G_truth=G_truth,
                            NF=NF, NS=NS, n_ion=n_ion, prov=prov, band=bandmask))
        say(f'  ({time.time()-ti:.1f}s for {lab})')
        say()

    # ------------------------------------------------------------ CSV output
    cols = ['ion', 'shell', 'depth', 'v_kms', 'Te_K', 'n_ion', 'G_prrr', 'G_A', 'G_Agrid',
            'G_Bpt', 'G_Bavg', 'G_C', 'G_C2', 'G_D', 'ratio_A_prrr', 'ratio_B_prrr',
            'ratio_C_prrr', 'ratio_D_prrr', 'pop_share_sigma_zero',
            'pop_share_no_exact_sigma']
    csvp = os.path.join(a.out, 'rates_certification.csv')
    with open(csvp, 'w') as f:
        f.write(','.join(cols) + '\n')
        for rw in rows_out:
            f.write(','.join(f'{rw[c]:.6e}' if isinstance(rw[c], float) else str(rw[c])
                             for c in cols) + '\n')
    say(f'[out] {csvp}')

    csv2 = os.path.join(a.out, 'rates_certification_alldepth.csv')
    with open(csv2, 'w') as f:
        f.write('ion,depth,v_kms,Te_K,n_ion,G_prrr,G_C,G_D,ratio_C_prrr,ratio_D_prrr,'
                'prrr_pop_inconsistent\n')
        for s in summary:
            for d in range(ND):
                gt = s['G_truth'][d]
                f.write(f'{s["lab"]},{d+1},{V[d]:.4f},{Te[d]:.2f},{s["n_ion"][d]:.6e},'
                        f'{gt:.6e},{s["GC_all"][d]:.6e},{s["GD_all"][d]:.6e},'
                        f'{(s["GC_all"][d]/gt if gt>0 else float("nan")):.6f},'
                        f'{(s["GD_all"][d]/gt if gt>0 else float("nan")):.6f},'
                        f'{int(s["band"][d])}\n')
    say(f'[out] {csv2}')

    # ------------------------------------------------------------ verdict
    say()
    say('=' * 96)
    say('PRE-REGISTERED GATE:  D/PRRR in [0.5,2.0] at every shell AND [0.8,1.25] at s6-s8')
    say('=' * 96)
    say(f'  {"ion":>7} ' + ' '.join(f'{lb:>8}' for lb in LABELS) + '   verdict')
    for s in summary:
        rr = (s['GD'] / s['GT'])[:NG]
        allok = bool(np.all((rr > 0.5) & (rr < 2.0)))
        formok = bool(np.all((rr[6:] > 0.8) & (rr[6:] < 1.25)))
        say(f'  {s["lab"]:>7} ' + ' '.join(f'{x:8.3f}' for x in rr)
            + f'   {"PASS" if (allok and formok) else "FAIL"}'
            + f'   [all-shell x2 {"ok" if allok else "NO"}; forming 0.8-1.25 '
              f'{"ok" if formok else "NO"}]')
    say()
    say('  CONTROL row -- A/PRRR (CMFGEN sigma, CMFGEN grid; must be ~1 or the whole')
    say('  ladder is meaningless):')
    for s in summary:
        say(f'  {s["lab"]:>7} ' + ' '.join(f'{x:8.3f}' for x in (s['GA'] / s['GT'])[:NG]))
    say()
    say('  Layer attribution at s6 (forming shell), factor introduced by each layer:')
    say(f'  {"ion":>7} {"A/PRRR":>8} {"Ag/A":>8} {"Bpt/Ag":>8} {"Bpt/Bav":>8} {"C/Bpt":>8} '
        f'{"D/C":>8} {"D/PRRR":>8}')
    for s in summary:
        q = 6
        say(f'  {s["lab"]:>7} {s["GA"][q]/s["GT"][q]:8.3f} {s["GAg"][q]/s["GA"][q]:8.3f} '
            f'{s["GB"][q]/s["GAg"][q]:8.3f} {s["GB"][q]/s["GBa"][q]:8.3f} '
            f'{s["GC"][q]/s["GB"][q]:8.3f} {s["GD"][q]/s["GC"][q]:8.3f} '
            f'{s["GD"][q]/s["GT"][q]:8.3f}')

    with open(os.path.join(a.out, 'run_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')
    say(f'\n[done] {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
