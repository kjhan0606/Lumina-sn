#!/usr/bin/env python3
"""external_anchors.py -- independent (non-CMFGEN) cross-checks for the ions the
rate certification cannot settle from CMFGEN data alone.

Everything is read from disk; no network.  Sources actually used:

  CMF-run   the phot file the toy06 reference run itself loaded (its vintage)
  CMF-23    CMFGEN 19apr23 (what LUMINA's baker picks)      -- same code family
  ARTIS     ../artis-ref/tests/toy06_whitebox_run/phixsdata.txt  (v1 tables)
  VFKY96    Verner, Ferland, Korista & Yakovlev 1996 ground-shell analytic fit,
            parameters from Cloudy's data/phfit.dat (independent code+data)
  NORAD     Nahar OSU unified RR+DR totals, data/atomic/dr_norad/raw_*.rrc.txt
  Badnell   Badnell (2006) RR fits, Cloudy data/badnell_rr.dat

Quantities:
  sigma_th   ground-level threshold cross-section [Mb]
  I_grid     INT sigma dnu over LUMINA's bf grid [1.5e14,3e16] Hz, ground level
  alpha_gnd  Milne recombination integral for the GROUND level at T, with the
             g-ratio held FIXED across sources so the number isolates sigma
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from certify_rate_machine import Sigma, norm_cfg, term_cfg   # noqa: E402
from cmfgen_parser import parse_osc, parse_phot              # noqa: E402

H = 6.62607015e-27
C = 2.99792458e10
EV = 1.602176634e-12
KB = 1.380649e-16
ME = 9.1093837015e-28
CM2EV = 1.239841984e-4
NU_MIN, NU_MAX = 1.5e14, 3.0e16

ATOMIC = '/gpfs/kjhan/cmfgen_21jun23/atomic'
ARTIS = '/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_whitebox_run'
CLOUDY = '/gpfs/kjhan/cloudy-master/data'
NORAD = os.path.join(ROOT, 'data/atomic/dr_norad')

# lab -> (Z, stage, elem_dir, run-vintage osc, run-vintage phot, artis ionstage)
IONS = [
    ('Ni II', 28, 2, 'NICK/II', '18oct00/nkii_osc.dat', '18oct00/phot_data.dat'),
    ('Ni III', 28, 3, 'NICK/III', '18oct00/nkiii_osc.dat', '18oct00/phot_data.dat'),
    ('Co II', 27, 2, 'COB/II', '18oct00/coii_osc.dat', '18oct00/phot_data.dat'),
    ('Co III', 27, 3, 'COB/III', '18oct00/coiii_osc.dat', '18oct00/phot_data.dat'),
    ('Fe II', 26, 2, 'FE/II', '19apr23/osc_data', '19apr23/phot_data_A'),
    ('Fe III', 26, 3, 'FE/III', '19apr23/osc_data', '19apr23/phot_data_A'),
    ('Ca III', 20, 3, 'CA/III', '10apr99/osc_op_sp.dat', '10apr99/phot_smooth.dat'),
    ('Ca IV', 20, 4, 'CA/IV', '10apr99/osc_op_sp.dat', '10apr99/phot_smooth.dat'),
    ('Ca V', 20, 5, 'CA/V', '10apr99/osc_op_sp.dat', '10apr99/phot_smooth.dat'),
    ('S IV', 16, 4, 'SUL/IV', '3oct00/sivosc_fin.dat', '3oct00/phot_sm_3000.dat'),
    ('S V', 16, 5, 'SUL/V', '3oct00/svosc_fin.dat', '3oct00/phot_sm_3000.dat'),
    ('Si IV', 14, 4, 'SIL/IV', '5dec96/osc_op_split.dat', '5dec96/phot_op.dat'),
]


# ----------------------------------------------------------------- generic
def milne_alpha(nu, sig, nu_th, T, gratio):
    """alpha_l = (g_l/2 g_i) (h^2/(2 pi me k T))^{3/2} (8 pi/c^2)
                 INT nu^2 sigma(nu) exp(-h(nu-nu_th)/kT) dnu
    The 1/2 is the electron spin factor in Saha; see known_answer_H()."""
    lam3 = (H * H / (2.0 * math.pi * ME * KB * T)) ** 1.5
    w = np.exp(-H * (nu - nu_th) / (KB * T))
    integ = np.trapz(nu * nu * sig * w, nu)
    return 0.5 * gratio * lam3 * (8.0 * math.pi / (C * C)) * integ


# CMFGEN fit type 9 (Verner) -- sub_phot_gen.f:518-530, ported verbatim.  The
# harness's Sigma class declines this type, so it is implemented here (anchors
# only, the certification numbers are untouched).
_EV_TO_HZ = 0.241798840766     # sub_phot_gen.f:129, eV -> 1e15 Hz
_PROG_TO_CM2 = 1.0e-10


class Sigma9:
    def __init__(self, params, nu_th):
        p = np.asarray(params, dtype=float)
        self.ok = p.size >= 8 and p.size % 8 == 0
        self.nu_th = nu_th
        self.sh = []
        if self.ok:
            for j in range(p.size // 8):
                q = p[8 * j:8 * j + 8]
                if q[3] <= 0 or q[5] <= 0:
                    self.ok = False
                    return
                e_j = nu_th if j == 0 else _EV_TO_HZ * q[2] * 1e15
                self.sh.append((e_j, q))

    def __call__(self, nu):
        nu = np.atleast_1d(np.asarray(nu, float))
        out = np.zeros(nu.shape)
        if not self.ok:
            return out
        glob = nu >= self.nu_th
        for e_j, q in self.sh:
            m = glob & (nu >= e_j)
            if not m.any():
                continue
            U = nu[m] * 1e-15 / q[3] / _EV_TO_HZ
            T1 = (U - 1.0) ** 2 + q[7] ** 2
            T2 = U ** (5.5 + q[1] - 0.5 * q[6])
            T3 = (1.0 + np.sqrt(U / q[5])) ** q[6]
            out[m] += 1e-08 * T1 * q[4] / T2 / T3 * _PROG_TO_CM2
        return out


def known_answer_H():
    """sanity: hydrogenic ground state, sigma = 6.30e-18 (nu_th/nu)^3."""
    nu_th = 13.5984 * EV / H
    nu = nu_th * np.exp(np.linspace(0, math.log(300.0), 40000))
    sig = 6.304e-18 * (nu_th / nu) ** 3
    return milne_alpha(nu, sig, nu_th, 1e4, 2.0 / 1.0)


def sigma_grid(nu_th, top=300.0, n=60000):
    return nu_th * np.exp(np.linspace(0, math.log(top), n))


# ------------------------------------------------------------------ CMFGEN
def cmfgen_ground(elem_dir, osc_rel, phot_rel):
    op = os.path.join(ATOMIC, elem_dir, osc_rel)
    pp = os.path.join(ATOMIC, elem_dir, phot_rel)
    osc = parse_osc(op)
    ph = parse_phot(pp)
    cfg = str(osc.levels['config'][0])
    g0 = float(osc.levels['g'][0])
    E0 = float(osc.levels['E_cm'][0]) * CM2EV
    nu_th = (osc.ionization_eV - E0) * EV / H
    cfg_e, term_e = {}, {}
    for e in ph.entries:
        cfg_e.setdefault(norm_cfg(e.config), []).append(e)
        term_e.setdefault(term_cfg(e.config), []).append(e)
    ents = cfg_e.get(norm_cfg(cfg)) or term_e.get(term_cfg(cfg))
    if not ents:
        return None
    e = ents[0]
    s = Sigma9(e.sigma_Mb, nu_th) if e.cs_type == 9 else \
        Sigma(e.cs_type, e.energy, e.sigma_Mb, nu_th)
    gion = None
    for k, v in ph.__dict__.items():
        if k == 'final_state_g':
            gion = v
    return dict(cfg=cfg, g=g0, nu_th=nu_th, sig=s, cs_type=e.cs_type,
                npts=e.n_points, gion=gion, chi=osc.ionization_eV,
                osc=op, phot=pp, nlev=osc.n_levels)


# ------------------------------------------------------------------- ARTIS
def artis_tables():
    """-> {(Z, lowerionstage, lowerlevel): (E_eV_above_th, sigma_Mb)} for level 1."""
    out = {}
    with open(os.path.join(ARTIS, 'phixsdata.txt')) as f:
        while True:
            ln = f.readline()
            if not ln:
                break
            t = ln.split()
            if len(t) != 6:
                continue
            Z, ui, ul, li, ll, n = map(int, t)
            rows = [f.readline().split() for _ in range(n)]
            if ll == 1:
                a = np.array([[float(r[0]), float(r[1])] for r in rows])
                out.setdefault((Z, li), a)
    return out


def artis_ionpot():
    """-> {(Z, ionstage): (ionpot_eV, g_ground, E_ground_eV)}"""
    out = {}
    with open(os.path.join(ARTIS, 'adata.txt')) as f:
        while True:
            ln = f.readline()
            if not ln:
                break
            t = ln.split()
            if len(t) != 4:
                continue
            Z, stage, nlev, ip = int(t[0]), int(t[1]), int(t[2]), float(t[3])
            first = f.readline().split()
            out[(Z, stage)] = (ip, float(first[2]), float(first[1]))
            for _ in range(nlev - 1):
                f.readline()
    return out


# ------------------------------------------------------------------ VFKY96
def load_vfky96():
    """Cloudy data/phfit.dat, section after the '-1 -1 -1' separator.
    columns: ne-1  Z-1  E0[eV] sigma0[Mb] ya P yw y0 y1"""
    par = {}
    seen = False
    for ln in open(os.path.join(CLOUDY, 'phfit.dat')):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('-1 -1 -1'):
            seen = True
            continue
        if not seen:
            continue
        t = s.split()
        if len(t) != 9:
            continue
        ne, Z = int(t[0]) + 1, int(t[1]) + 1
        par[(Z, ne)] = [float(x) for x in t[2:]]
    return par


def load_vy95():
    """Cloudy data/phfit.dat, section BEFORE the '-1 -1 -1' separator.
    columns: nshell-1 ne-1 Z-1 Eth[eV] E0[eV] sigma0[Mb] ya P yw
    plus the 3-line preamble: l per shell, 96-formula lower shell bound per ne,
    and the shell index each additional electron occupies.
    Formula (VY95 eq.1, identical to CMFGEN fit type 9):
      y = E/E0 ; F = ((y-1)^2 + yw^2) y^{-(5.5+l-0.5P)} (1+sqrt(y/ya))^{-P}"""
    rows, pre = {}, []
    for ln in open(os.path.join(CLOUDY, 'phfit.dat')):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        t = s.split()
        if t[:3] == ['-1', '-1', '-1']:
            break
        if len(pre) < 4:
            pre.append([float(x) for x in t])
            continue
        if len(t) != 9:
            continue
        rows[(int(t[2]) + 1, int(t[1]) + 1, int(t[0]) + 1)] = [float(x) for x in t[3:]]
    l_of_shell = [int(x) for x in pre[1]]
    shell_of_ne = [int(x) for x in pre[3]]
    return rows, l_of_shell, shell_of_ne


def vy95_sigma(p, l, E_eV):
    Eth, E0, s0, ya, P, yw = p
    y = E_eV / E0
    F = ((y - 1.0) ** 2 + yw * yw) * y ** (-(5.5 + l - 0.5 * P)) \
        * (1.0 + np.sqrt(y / ya)) ** (-P)
    return np.where(E_eV >= Eth, s0 * F, 0.0), Eth


def vfky96_sigma(p, E_eV, Eth_eV):
    E0, s0, ya, P, yw, y0, y1 = p
    x = E_eV / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    F = ((x - 1.0) ** 2 + yw * yw) * y ** (0.5 * P - 5.5) * (1.0 + np.sqrt(y / ya)) ** (-P)
    s = s0 * F
    return np.where(E_eV >= Eth_eV, s, 0.0)


# ----------------------------------------------------------------- Badnell
def badnell_rr(Z, N, T):
    """N = electrons in the RECOMBINING (initial) ion."""
    tot = 0.0
    hit = 0
    for ln in open(os.path.join(CLOUDY, 'badnell_rr.dat')):
        t = ln.split()
        if len(t) < 8:
            continue
        try:
            z, n = int(t[0]), int(t[1])
        except ValueError:
            continue
        if z != Z or n != N:
            continue
        A, B, T0, T1 = float(t[4]), float(t[5]), float(t[6]), float(t[7])
        if len(t) >= 10:
            Cc, T2 = float(t[8]), float(t[9])
            B = B + Cc * math.exp(-T2 / T)
        a = A / (math.sqrt(T / T0) * (1 + math.sqrt(T / T0)) ** (1 - B)
                 * (1 + math.sqrt(T / T1)) ** (1 + B))
        tot += a
        hit += 1
    return (tot if hit else float('nan')), hit


# ------------------------------------------------------------------- NORAD
def norad_total(raw, T):
    import re
    lines = open(os.path.join(NORAD, raw)).read().splitlines()
    hdr = re.compile(r'\blog\(T\)\s+RRC\(low n\)\s+RRC\(DR\)')
    st = next(i for i, l in enumerate(lines) if hdr.search(l))
    lt, al = [], []
    for l in lines[st + 1:]:
        t = l.split()
        if len(t) < 6:
            if lt:
                break
            continue
        try:
            lt.append(float(t[0]))
            al.append(float(t[-1]))
        except ValueError:
            if lt:
                break
    lt, al = np.array(lt), np.array(al)
    return float(np.interp(math.log10(T), lt, al))


def alpha_total_cmfgen(elem_dir, osc_rel, phot_rel, T, nmax=None):
    """Sum the Milne integral over EVERY level of the ion that the phot file
    covers.  Comparable (modulo DR and the missing high-n levels) to a published
    total recombination rate."""
    osc = parse_osc(os.path.join(ATOMIC, elem_dir, osc_rel))
    ph = parse_phot(os.path.join(ATOMIC, elem_dir, phot_rel))
    gion = ph.final_state_g
    cfg_e, term_e = {}, {}
    for e in ph.entries:
        cfg_e.setdefault(norm_cfg(e.config), []).append(e)
        term_e.setdefault(term_cfg(e.config), []).append(e)
    n = osc.n_levels if nmax is None else min(nmax, osc.n_levels)
    tot, nhit, nmiss = 0.0, 0, 0
    kT_hz = KB * T / H
    for k in range(n):
        cfg = str(osc.levels['config'][k])
        gl = float(osc.levels['g'][k])
        nu_th = (osc.ionization_eV - float(osc.levels['E_cm'][k]) * CM2EV) * EV / H
        if nu_th <= 0:
            continue
        ents = cfg_e.get(norm_cfg(cfg)) or term_e.get(term_cfg(cfg))
        if not ents:
            nmiss += 1
            continue
        e = ents[0]
        s = Sigma9(e.sigma_Mb, nu_th) if e.cs_type == 9 else \
            Sigma(e.cs_type, e.energy, e.sigma_Mb, nu_th)
        if not getattr(s, 'ok', False):
            nmiss += 1
            continue
        hi = nu_th + 60.0 * kT_hz
        grid = np.linspace(nu_th, hi, 3000)
        nd = s.nodes() if hasattr(s, 'nodes') else np.array([])
        if nd.size:
            nd = nd[(nd > nu_th) & (nd < hi)]
            if nd.size:
                grid = np.unique(np.concatenate([grid, nd]))
        tot += milne_alpha(grid, s(grid), nu_th, T, gl / gion)
        nhit += 1
    return tot, nhit, nmiss, gion


def main():
    T = 1e4
    print(f'[known-answer] Milne integral, H 1s, sigma=6.304e-18 (nu_th/nu)^3, T=1e4 K: '
          f'alpha = {known_answer_H():.3e} cm^3/s   (literature alpha_1s ~ 1.58e-13)')
    print()

    art = artis_tables()
    aip = artis_ionpot()
    vf = load_vfky96()
    vy95, l_of_shell, shell_of_ne = load_vy95()

    print('=' * 118)
    print('GROUND-LEVEL THRESHOLD CROSS-SECTION AND MILNE alpha(1e4 K), source by source')
    print('  alpha uses ONE fixed g_l/g_ion per ion (CMFGEN 19apr23 values) so it varies only with sigma.')
    print('=' * 118)
    hdr = (f'{"ion":8s} {"source":10s} {"cfg/shell":24s} {"lam_th(A)":>9s} {"type":>6s} '
           f'{"sig_th(Mb)":>11s} {"I_grid":>11s} {"alpha_gnd":>11s}')
    print(hdr)
    rows = []
    for lab, Z, stage, ed, osc_rel, phot_rel in IONS:
        cur = cmfgen_ground(ed, '19apr23/osc_data', '19apr23/phot_data_A')
        run = cmfgen_ground(ed, osc_rel, phot_rel)
        if cur is None:
            print(f'{lab:8s} -- 19apr23 ground entry not matched')
            continue
        gratio = (cur['g'] / cur['gion']) if cur['gion'] else float('nan')
        for tag, d in (('CMF-run', run), ('CMF-19apr23', cur)):
            if d is None:
                print(f'{lab:8s} {tag:10s} (ground entry not found in that vintage)')
                continue
            nu = sigma_grid(d['nu_th'])
            sg = d['sig'](nu)
            m = (nu >= NU_MIN) & (nu <= NU_MAX)
            Ig = np.trapz(sg[m], nu[m]) if m.sum() > 2 else 0.0
            al = milne_alpha(nu, sg, d['nu_th'], T, gratio)
            print(f'{lab:8s} {tag:11s} {d["cfg"][:23]:24s} {C*1e8/d["nu_th"]:9.1f} '
                  f'{d["cs_type"]:6d} {d["sig"](d["nu_th"]*1.0000001)[0]*1e18:11.4f} '
                  f'{Ig:11.4e} {al:11.4e}')
            rows.append((lab, tag, d['sig'](d['nu_th'] * 1.0000001)[0] * 1e18, Ig, al))
        # ARTIS
        key = (Z, stage)
        if key in art and key in aip:
            a = art[key]
            ip, gg, e0 = aip[key]
            nu_th = (ip - e0) * EV / H
            nu = nu_th + a[:, 0] * EV / H
            sg = a[:, 1] * 1e-18
            # extend with the (nu_max/nu)^3 tail ARTIS itself uses
            nx = np.exp(np.linspace(math.log(nu[-1]), math.log(nu_th * 300), 4000))[1:]
            nu = np.concatenate([nu, nx])
            sg = np.concatenate([sg, sg[-1] * (nu[len(a) - 1] / nx) ** 3])
            m = (nu >= NU_MIN) & (nu <= NU_MAX)
            Ig = np.trapz(sg[m], nu[m]) if m.sum() > 2 else 0.0
            al = milne_alpha(nu, sg, nu_th, T, gratio)
            print(f'{lab:8s} {"ARTIS":11s} {"level 1 (g=%.0f)"%gg:24s} {C*1e8/nu_th:9.1f} '
                  f'{len(a):6d} {a[0,1]:11.4f} {Ig:11.4e} {al:11.4e}')
            rows.append((lab, 'ARTIS', a[0, 1], Ig, al))
        # VY95 valence shell (covers the Fe-peak, which VFKY96 in Cloudy does not)
        ne = Z - (stage - 1)
        nsh = shell_of_ne[ne - 1] if ne <= len(shell_of_ne) else None
        p95 = vy95.get((Z, ne, nsh)) if nsh else None
        if p95:
            Eth95 = p95[0]
            E = np.linspace(cur['chi'], cur['chi'] * 300, 200000)
            sg, _ = vy95_sigma(p95, l_of_shell[nsh - 1], E)
            sg = np.where(E >= cur['chi'], sg, 0.0) * 1e-18
            nu = E * EV / H
            nu_th = cur['chi'] * EV / H
            m = (nu >= NU_MIN) & (nu <= NU_MAX)
            Ig = np.trapz(sg[m], nu[m]) if m.sum() > 2 else 0.0
            al = milne_alpha(nu, sg, nu_th, T, gratio)
            print(f'{lab:8s} {"VY95":11s} {"shell %d (Eth=%.1f eV)"%(nsh,Eth95):24s} '
                  f'{C*1e8/nu_th:9.1f} {0:6d} {sg[0]*1e18:11.4f} {Ig:11.4e} {al:11.4e}')
            rows.append((lab, 'VY95', sg[0] * 1e18, Ig, al))
        # VFKY96 ground shell
        p = vf.get((Z, ne))
        if p:
            Eth = cur['chi']
            E = np.linspace(Eth, Eth * 300, 200000)
            sg = vfky96_sigma(p, E, Eth) * 1e-18
            nu = E * EV / H
            nu_th = Eth * EV / H
            m = (nu >= NU_MIN) & (nu <= NU_MAX)
            Ig = np.trapz(sg[m], nu[m]) if m.sum() > 2 else 0.0
            al = milne_alpha(nu, sg, nu_th, T, gratio)
            print(f'{lab:8s} {"VFKY96":11s} {"valence shell (Cloudy)":24s} {C*1e8/nu_th:9.1f} '
                  f'{0:6d} {sg[0]*1e18:11.4f} {Ig:11.4e} {al:11.4e}')
            rows.append((lab, 'VFKY96', sg[0] * 1e18, Ig, al))
        print(f'{"":8s} {"(g_l/g_ion used = %.4f; CMFGEN 19apr23 g_l=%.0f g_ion=%.0f)"%(gratio,cur["g"],cur["gion"] or 0)}')
        print()

    print('=' * 118)
    print('TOTAL (all-level) RECOMBINATION ANCHORS  --  independent of the sigma tables above')
    print('=' * 118)
    print()
    print('  ALL-LEVEL Milne sum over the CMFGEN sigma tables themselves '
          '(no DR, no levels above the file):')
    for lab, Z, stage, ed, osc_rel, phot_rel in IONS:
        for tag, o, p in (('run-vintage', osc_rel, phot_rel),
                          ('19apr23', '19apr23/osc_data', '19apr23/phot_data_A')):
            try:
                a, nh, nm, gi = alpha_total_cmfgen(ed, o, p, T)
            except Exception as exc:                      # noqa: BLE001
                print(f'  CMF-sum {lab:7s} {tag:12s} FAILED: {exc}')
                continue
            print(f'  CMF-sum {lab:7s} {tag:12s} alpha_sum(1e4 K) = {a:.4e} cm^3/s '
                  f'({nh} levels summed, {nm} skipped, g_ion={gi:g})')
    print()
    for lab, Z, stage, raw in [('Ni II', 28, 2, 'raw_ni2.rrc.txt'),
                               ('Fe II', 26, 2, 'raw_fe2.rrc-ls.txt'),
                               ('Fe III', 26, 3, 'raw_fe3.rrc.ls.txt')]:
        a = norad_total(raw, T)
        print(f'  NORAD  {lab:7s}  X+1 + e -> {lab}   alpha_total(RR+DR, 1e4 K) = {a:.4e} cm^3/s'
              f'   [{raw}]')
    for lab, Z, stage in [('Si IV', 14, 4), ('S V', 16, 5), ('Ca III', 20, 3),
                          ('Ni II', 28, 2), ('Co II', 27, 2), ('Co III', 27, 3)]:
        N = Z - stage          # electrons in the recombining ion (stage+1 charge state)
        a, hit = badnell_rr(Z, N, T)
        print(f'  Badnell {lab:7s} Z={Z} N={N} (recombining ion) -> alpha_RR(1e4 K) = '
              f'{a if hit else float("nan"):.4e}  [{hit} shells/terms]')


if __name__ == '__main__':
    main()
