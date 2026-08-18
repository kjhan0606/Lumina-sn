#!/usr/bin/env python3
"""fable_checks.py -- adversarial re-verification of the Opus external-download
claims (C1-C10), all quantities recomputed from raw bytes with INDEPENDENT
parsers (no import of cmfgen_parser / certify_rate_machine).

Outputs: printed verdict material + fable_c6_classification.csv next to this file.
"""
import gzip
import hashlib
import math
import os
import re
import struct
import sys

import numpy as np

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
EXT = os.path.join(ROOT, 'data/atomic/external')
NORAD = os.path.join(EXT, 'norad_ni2')
TOP = os.path.join(EXT, 'topbase_ca')
ATOMIC = '/gpfs/kjhan/cmfgen_21jun23/atomic'
HERE = os.path.dirname(os.path.abspath(__file__))

H = 6.62607015e-27
C = 2.99792458e10
EV = 1.602176634e-12
KB = 1.380649e-16
ME = 9.1093837015e-28
RY_HZ = 3.2898419602508e15
RY_CM = 109737.31568160
CM2EV = 1.239841984e-4
NU_MIN, NU_MAX = 1.5e14, 3.0e16
LUMINA_DLN = 5.3e-3


def sec(t):
    print('\n' + '=' * 100)
    print(t)
    print('=' * 100)


# ------------------------------------------------------------ CMFGEN phot parser
def parse_phot_entries(path):
    """Independent minimal parser: returns (header_dict, [entry dicts])."""
    txt = open(path, errors='replace').read()
    lines = txt.split('\n')

    def fl(v):
        return float(v.replace('D', 'E').replace('d', 'e'))
    n = len(lines)
    entries = []
    i = 0
    while i < n:
        if 'Configuration name' in lines[i]:
            cfg = lines[i].split('!')[0].strip()
            j = i + 1
            while j < n and 'Type of cross-section' not in lines[j]:
                j += 1
            typ = int(lines[j].split('!')[0].strip())
            while j < n and 'Number of cross-section points' not in lines[j]:
                j += 1
            npts = int(lines[j].split('!')[0].strip())
            j += 1
            need = 2 * npts if typ in (20, 21, 22) else npts
            vals = []
            while j < n and len(vals) < need:
                if 'Configuration name' in lines[j]:
                    break
                vals += lines[j].split('!')[0].split()
                j += 1
            arr = np.array([fl(v) for v in vals[:need]])
            if typ in (20, 21, 22):
                arr = arr.reshape(npts, 2)
                entries.append(dict(cfg=cfg, typ=typ, npts=npts,
                                    x=arr[:, 0].copy(), s=arr[:, 1].copy()))
            else:
                entries.append(dict(cfg=cfg, typ=typ, npts=npts, par=arr))
            i = j
        else:
            i += 1
    return entries


def _dash_sep(lines):
    """Index just past the long dashed separator ending the NORAD preamble."""
    for i, ln in enumerate(lines):
        s = ln.strip()
        if len(s) >= 40 and set(s) == {'-'}:
            return i + 1
    raise ValueError('separator not found')


def _skip_blank(lines, i):
    while not lines[i].split():
        i += 1
    return i


# ------------------------------------------------------------ NORAD px parser
def parse_px_blocks(path):
    lines = open(path).read().split('\n')
    n = len(lines)
    i = _dash_sep(lines)
    blocks = []
    while i < n:
        t = lines[i].split()
        if not t:
            i += 1
            continue
        zz, nn, ntg = int(t[0]), int(t[1]), int(t[2])
        i += 1
        ev = []
        while len(ev) < ntg:
            ev += lines[i].split()
            i += 1
        slpi = tuple(int(x) for x in lines[i].split())
        i += 1
        t = lines[i].split()
        be, ntot = float(t[0]), int(t[1])
        i += 1
        i += 1  # ac line
        E = np.empty(ntot)
        S = np.empty(ntot)
        for k in range(ntot):
            t = lines[i].split()
            E[k] = float(t[0])
            S[k] = float(t[1])
            i += 1
        blocks.append(dict(idx=len(blocks), slpi=slpi, be=be, ntot=ntot, E=E, S=S))
    return blocks


# ------------------------------------------------- type-20 evaluator (method spec)
def sigma20(u_tab, s_Mb, nu_th):
    s_cm2 = np.asarray(s_Mb) * 1e-18
    u_tab = np.asarray(u_tab, float)

    def f(nu):
        nu = np.atleast_1d(np.asarray(nu, float))
        out = np.zeros(nu.shape)
        m = nu >= nu_th
        if not m.any():
            return out
        u = nu[m] / nu_th
        v = np.interp(u, u_tab, s_cm2, left=s_cm2[0], right=0.0)
        tail = u >= u_tab[-1]
        if tail.any():
            v[tail] = s_cm2[-1] * (u_tab[-1] / u[tail]) ** 3
        out[m] = v
        return out
    return f


def loggrid(nu_th, top=300.0, npt=60000):
    return nu_th * np.exp(np.linspace(0.0, math.log(top), npt))


def igrid(nu, sg):
    m = (nu >= NU_MIN) & (nu <= NU_MAX)
    return float(np.trapz(sg[m], nu[m])) if m.sum() > 2 else 0.0


def milne(nu, sg, nu_th, T, gratio):
    lam3 = (H * H / (2.0 * math.pi * ME * KB * T)) ** 1.5
    w = np.exp(-H * (nu - nu_th) / (KB * T))
    return 0.5 * gratio * lam3 * (8.0 * math.pi / (C * C)) * float(np.trapz(nu * nu * sg * w, nu))


# =============================================================================
def main():
    out_csv = []

    # ---------------------------------------------------------------- C1
    sec('C1  NORAD file inventory (sha256/bytes) + rrc round trip')
    claimed = {
        'ni2.px.gd.txt': (56019, '9ad2c6421d5da2788efccd2a4e1d96cfbaa55fed95ae0642b054945973cce328'),
        'ni2.px.txt': (53156229, 'b2a9d72ac01521267a257e8589ddd397342e443a64a37e3ce696650d8197247f'),
        'ni2.ptpx.txt': (77416568, '1c5f6aa66c97536c5e3370fc514be22e757f71e88ae9fab7cc054405f90304ca'),
        'ni2.en.ls.txt': (46237, '5e17090471fbdad947d0574fc7ccd9f62aef73366bd4ec3a5f2a81f81070689d'),
        'ni2.rrc.txt': (548605, '07627d787477a645bca28563b51f368d1082483a1ed9ccdadc8a47daa612cd27'),
    }
    ok = True
    for fn, (nb, sh) in claimed.items():
        p = os.path.join(NORAD, fn)
        b = open(p, 'rb').read()
        got = hashlib.sha256(b).hexdigest()
        match = (len(b) == nb and got == sh)
        ok &= match
        print(f'  {fn:16s} bytes {len(b):9d} (claim {nb:9d})  sha256 {"MATCH" if match else "MISMATCH " + got}')
    a = open(os.path.join(NORAD, 'ni2.rrc.txt'), 'rb').read()
    b = open(os.path.join(ROOT, 'data/atomic/dr_norad/raw_ni2.rrc.txt'), 'rb').read()
    print(f'  rrc vs repo raw_ni2.rrc.txt byte-identical: {a == b}')
    print(f'  C1 => {"CONFIRMED" if ok and a == b else "PROBLEM"}')

    # ---------------------------------------------------------------- parse main inputs
    sec('Parsing raw inputs')
    ph = parse_phot_entries(os.path.join(ATOMIC, 'NICK/II/19apr23/phot_data_A'))
    t20 = [e for e in ph if e['typ'] == 20]
    print(f'  phot_data_A: {len(ph)} entries total, {len(t20)} type-20, '
          f'other types {sorted(set(e["typ"] for e in ph if e["typ"] != 20))}')

    gd_lines = open(os.path.join(NORAD, 'ni2.px.gd.txt')).read().split('\n')
    i = _dash_sep(gd_lines)
    i = _skip_blank(gd_lines, i)
    t = gd_lines[i].split()
    zz, nn, ntg = int(t[0]), int(t[1]), int(t[2])
    i += 1
    ev = []
    while len(ev) < ntg:
        ev += gd_lines[i].split()
        i += 1
    gd_slpi = tuple(int(x) for x in gd_lines[i].split())
    i += 1
    t = gd_lines[i].split()
    gd_be, gd_ntot = float(t[0]), int(t[1])
    i += 2
    gdE = np.empty(gd_ntot)
    gdS = np.empty(gd_ntot)
    gd_raw = []
    for k in range(gd_ntot):
        gd_raw.append(gd_lines[i])
        t = gd_lines[i].split()
        gdE[k] = float(t[0])
        gdS[k] = float(t[1])
        i += 1
    print(f'  px.gd: zz={zz} nn={nn} ntg={ntg} SLpi={gd_slpi} BE={gd_be} ntot={gd_ntot}')

    px = parse_px_blocks(os.path.join(NORAD, 'ni2.px.txt'))
    print(f'  px.txt: {len(px)} blocks, total pairs {sum(bl["ntot"] for bl in px)}')

    # ---------------------------------------------------------------- C2
    sec('C2  CMFGEN entry 0 vs NORAD ground (x = E_Ry/|BE|)')
    e0 = t20[0] if ph[0]['typ'] == 20 else None
    e0 = ph[0]
    print(f'  CMFGEN entry 0: cfg={e0["cfg"]!r} type={e0["typ"]} npts={e0["npts"]}')
    print(f'  NORAD px.gd  : SLpi={gd_slpi} BE={gd_be} Ry, ntot={gd_ntot}')
    same_n = e0['npts'] == gd_ntot
    dx = e0['x'] - gdE / abs(gd_be)
    print(f'  npts equal: {same_n};  max|x_CMF - E/|BE|| = {np.abs(dx).max():.3e}')

    # ---------------------------------------------------------------- C3
    sec('C3  ground sigma diff: expect exactly 2 clipped points')
    mism = np.nonzero(gdS != e0['s'])[0]
    print(f'  differing sigma points (exact float !=): {len(mism)} at indices {mism.tolist()}')
    for k in mism:
        lo = max(0, k - 1)
        print(f'    idx {k}: E={gdE[k]:.6f} Ry  NORAD sig={gdS[k]:.4e}  CMFGEN sig={e0["s"][k]:.4e}  '
              f'ratio NORAD/CMF={gdS[k]/e0["s"][k]:.6g}')
        print(f'      neighbors NORAD: {gdS[max(0,k-1)]:.4e} | {gdS[min(gd_ntot-1,k+1)]:.4e}')
        print(f'      raw NORAD line : {gd_raw[k]!r}')
    order = np.argsort(gdS)[::-1]
    print('  NORAD ground top-5 sigma:')
    for r in order[:5]:
        print(f'    {gdS[r]:.4e} Mb @ {gdE[r]:.6f} Ry (idx {r})')
    kmax_c = int(np.argmax(e0['s']))
    print(f'  CMFGEN entry-0 max sigma = {e0["s"][kmax_c]:.4e} Mb @ x={e0["x"][kmax_c]:.7f} '
          f'(-> E = {e0["x"][kmax_c]*abs(gd_be):.6f} Ry)')
    r3 = order[2]
    print(f'  NORAD 3rd-largest = {gdS[r3]:.4e} @ {gdE[r3]:.6f} Ry ; '
          f'CMF max == NORAD 3rd (exact): {e0["s"][kmax_c] == gdS[r3]}')

    # mantissa/exponent form check from raw strings
    for k in mism:
        nor_tok = gd_raw[k].split()[1]
        print(f'    idx {k}: NORAD token {nor_tok!r}  CMFGEN value {e0["s"][k]:.4E}')

    # ---------------------------------------------------------------- C5 (threshold rescale)
    sec('C5  threshold rescale ratio')
    osc = open(os.path.join(ATOMIC, 'NICK/II/19apr23/osc_data')).read().split('\n')
    ion_cm = None
    for ln in osc[:40]:
        if 'Ionization energy' in ln:
            ion_cm = float(ln.split('!')[0].strip())
            break
    ion_eV = ion_cm * CM2EV
    nu_th_cmf = ion_eV * EV / H
    nu_th_op = abs(gd_be) * RY_HZ
    ratio = nu_th_cmf / nu_th_op
    lam_cmf = C * 1e8 / nu_th_cmf
    lam_op = C * 1e8 / nu_th_op
    print(f'  CMFGEN osc ionization = {ion_cm} cm^-1 = {ion_eV:.5f} eV -> nu_th {nu_th_cmf:.6e} Hz'
          f' = {lam_cmf:.2f} A')
    print(f'  OP |BE| = {abs(gd_be)} Ry -> nu_th {nu_th_op:.6e} Hz = {lam_op:.2f} A')
    print(f'  rescale ratio = {ratio:.6f}   (claim 1.046499; report lam_OP 714.15 A vs measured {lam_op:.2f})')
    print(f'  E(Ry) as eV: {abs(gd_be)*RY_CM*CM2EV:.5f} eV ; NIST-vintage claim 18.169 eV; '
          f'18.169/{abs(gd_be)*RY_CM*CM2EV:.5f} = {18.169/(abs(gd_be)*RY_CM*CM2EV):.6f}')

    # ---------------------------------------------------------------- C4 (integrals)
    sec('C4  I_grid / alpha reproductions')
    gratio = 6.0 / 21.0
    T = 1e4
    # (a) trapz on CMFGEN's own nodes
    nu_own = e0['x'] * nu_th_cmf
    s_own = e0['s'] * 1e-18
    m = (nu_own >= NU_MIN) & (nu_own <= NU_MAX) & (nu_own >= nu_th_cmf)
    I_own_above = float(np.trapz(s_own[m], nu_own[m]))
    m2 = (nu_own >= NU_MIN) & (nu_own <= NU_MAX)
    I_own_all = float(np.trapz(s_own[m2], nu_own[m2]))
    print(f'  (a) CMFGEN own-node trapz: {I_own_above:.4f} (nodes>=nu_th) / {I_own_all:.4f} (all nodes)   '
          f'[report 5.3750]')
    # (b) certification method: 60000-pt log grid + nu^-3 tail
    f_cmf = sigma20(e0['x'], e0['s'], nu_th_cmf)
    nu = loggrid(nu_th_cmf)
    sg = f_cmf(nu)
    I_cert = igrid(nu, sg)
    a_cert = milne(nu, sg, nu_th_cmf, T, gratio)
    sig_th = f_cmf(nu_th_cmf * 1.0000001)[0] * 1e18
    print(f'  (b) certification method: I_grid = {I_cert:.4f}  alpha_gnd = {a_cert:.4e}  sig_th = {sig_th:.4f}')
    print(f'      log line 9 claims     I_grid = 5.3941e+00 alpha_gnd = 3.0631e-14  sig_th = 2.4896')
    # (c) NORAD original, both nu_th conventions
    u_nor = gdE / abs(gd_be)
    f_nor_resc = sigma20(u_nor, gdS, nu_th_cmf)
    nu_r = loggrid(nu_th_cmf)
    sg_r = f_nor_resc(nu_r)
    I_nor_resc = igrid(nu_r, sg_r)
    a_nor_resc = milne(nu_r, sg_r, nu_th_cmf, T, gratio)
    f_nor_nat = sigma20(u_nor, gdS, nu_th_op)
    nu_n = loggrid(nu_th_op)
    sg_n = f_nor_nat(nu_n)
    I_nor_nat = igrid(nu_n, sg_n)
    a_nor_nat = milne(nu_n, sg_n, nu_th_op, T, gratio)
    # own-node trapz for NORAD
    nu_nor_own_r = u_nor * nu_th_cmf
    I_nor_own_r = igrid(nu_nor_own_r, gdS * 1e-18)
    nu_nor_own_n = gdE * RY_HZ
    I_nor_own_n = igrid(nu_nor_own_n, gdS * 1e-18)
    print(f'  (c) NORAD cert-method: rescaled {I_nor_resc:.4f} / native {I_nor_nat:.4f}   '
          f'[claims 25.49 / 24.36]')
    print(f'      NORAD own-node trapz: rescaled {I_nor_own_r:.4f} / native {I_nor_own_n:.4f}   '
          f'[report 25.8941 / 24.7435]')
    print(f'      alpha_gnd NORAD: rescaled {a_nor_resc:.4e} / native {a_nor_nat:.4e}  '
          f'ratio vs CMFGEN {a_nor_resc/a_cert:.3f} / {a_nor_nat/a_cert:.3f}   [claim x4.73]')
    # (d) restore the two clipped points only
    s_fix = e0['s'].copy()
    for k in mism:
        s_fix[k] = gdS[k]
    f_fix = sigma20(e0['x'], s_fix, nu_th_cmf)
    sg_f = f_fix(nu)
    I_fix = igrid(nu, sg_f)
    a_fix = milne(nu, sg_f, nu_th_cmf, T, gratio)
    print(f'  (d) CMFGEN with ONLY the {len(mism)} clipped points restored: I_grid = {I_fix:.4f} '
          f'alpha = {a_fix:.4e}   [claims 25.492 / 1.4495e-13]')
    print(f'      2-point share of restored I_grid: {(I_fix-I_cert)/I_fix*100:.1f}%   [claim 79%]')

    # ---------------------------------------------------------------- C6 full classification
    sec('C6  exhaustive classification of all type-20 entries vs px.txt blocks')
    fp = {}
    for bl in px:
        key = tuple(bl['S'][:8].tolist())
        fp.setdefault(key, []).append(bl['idx'])
    n_ident = n_trunc = n_mod = n_nomatch = 0
    ambiguous = []
    trunc_list = []
    mod_list = []
    for ei, e in enumerate(t20):
        key = tuple(e['s'][:8].tolist())
        cand = []
        for bi in fp.get(key, []):
            bl = px[bi]
            if bl['ntot'] >= e['npts'] and np.array_equal(bl['S'][:e['npts']], e['s']):
                cand.append(bi)
        # also allow the modified entry (ground): fingerprint matches, but full != ; retry
        if not cand:
            for bi in fp.get(key, []):
                bl = px[bi]
                if bl['ntot'] >= e['npts']:
                    nm = int(np.sum(bl['S'][:e['npts']] != e['s']))
                    cand.append(bi)
        if not cand:
            n_nomatch += 1
            out_csv.append((ei, e['cfg'], e['typ'], e['npts'], -1, '', 'NO_MATCH', '', ''))
            continue
        if len(cand) > 1:
            ambiguous.append((ei, e['cfg'], len(cand)))
        bi = cand[0]
        bl = px[bi]
        nm = int(np.sum(bl['S'][:e['npts']] != e['s']))
        if nm == 0 and e['npts'] == bl['ntot']:
            n_ident += 1
            cls = 'BIT_IDENTICAL'
            note = ''
        elif nm == 0 and e['npts'] < bl['ntot']:
            n_trunc += 1
            cls = 'TAIL_TRUNCATED'
            dE = np.diff(bl['E'])
            bad = np.nonzero(dE <= 0)[0]
            first_noninc = int(bad[0] + 1) if bad.size else bl['ntot']
            okc = (first_noninc == e['npts'])
            note = f'cut@{e["npts"]}/{bl["ntot"]} first_nonincreasing={first_noninc} match={okc}'
            trunc_list.append((ei, e['cfg'], e['npts'], bl['ntot'], first_noninc, okc))
        else:
            n_mod += 1
            cls = 'VALUE_MODIFIED'
            idxs = np.nonzero(bl['S'][:e['npts']] != e['s'])[0]
            note = f'{nm} differing points at {idxs.tolist()[:10]}'
            mod_list.append((ei, e['cfg'], nm, idxs.tolist()[:10]))
        out_csv.append((ei, e['cfg'], e['typ'], e['npts'], bi, str(bl['slpi']), cls,
                        len(cand), note))
    print(f'  type-20 entries: {len(t20)}  =>  bit-identical {n_ident} / tail-truncated {n_trunc}'
          f' / value-modified {n_mod} / no-match {n_nomatch}')
    print(f'  ambiguous fingerprint matches (>1 candidate block): {len(ambiguous)}')
    print('  truncated entries (all):')
    for row in trunc_list:
        print(f'    entry {row[0]:3d} {row[1]:28s} cut {row[2]:5d}/{row[3]:5d} '
              f'first_nonincreasing_idx {row[4]:5d} agrees={row[5]}')
    print('  value-modified entries:')
    for row in mod_list:
        print(f'    entry {row[0]:3d} {row[1]:28s} {row[2]} points differ at {row[3]}')
    trunc_ok = all(r[5] for r in trunc_list)
    print(f'  all truncation cuts == first non-increasing E index: {trunc_ok}')

    # px.gd vs px.txt duplicate-extract check
    same = [bl['idx'] for bl in px if bl['ntot'] == gd_ntot and np.array_equal(bl['S'], gdS)]
    print(f'  px.gd sigma column identical to px.txt block(s): {same}')

    # ---------------------------------------------------------------- C7
    sec('C7  file-wide max sigma')
    best = max(t20, key=lambda e: e['s'].max())
    vmax = best['s'].max()
    print(f'  CMFGEN max sigma over all type-20 tables = {vmax:.4e} Mb in {best["cfg"]!r}')
    hit = [bl['idx'] for bl in px if (bl['S'] == vmax).any()]
    print(f'  NORAD px.txt blocks containing this exact value: {hit}')

    # ---------------------------------------------------------------- C9 grid stats (Ni II part)
    sec('C9  grid statistics')
    dln = np.diff(np.log(e0['x']))
    print(f'  Ni II CMFGEN ground dln(nu): mean {dln.mean():.4e} median {np.median(dln):.4e} '
          f' -> LUMINA(5.3e-3)/median = {LUMINA_DLN/np.median(dln):.1f}x  [claims 4.4755e-4 / 6.691e-5 / 79.2x]')

    # ---------------------------------------------------------------- C8 + C9 TOPbase
    sec('C8  TOPbase: hashes, units round-trip, traps')
    prov = open(os.path.join(TOP, 'PROVENANCE.txt')).read()
    tab = re.findall(r'^([pef]20\.\d+\.gz)\s+(\d+)\s+([0-9a-f]{64})', prov, re.M)
    allok = True
    for fn, nb, sh in tab:
        b = open(os.path.join(TOP, fn), 'rb').read()
        got = hashlib.sha256(b).hexdigest()
        okf = (len(b) == int(nb) and got == sh)
        allok &= okf
        mt = struct.unpack('<I', b[4:8])[0]
        import datetime
        mts = datetime.datetime.utcfromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S')
        print(f'  {fn:10s} bytes {len(b):8d} sha {"MATCH" if okf else "MISMATCH"}  gz-mtime {mts}')
    print(f'  all 9 gz match PROVENANCE table: {allok}')

    def top_p_blocks(fn):
        lines = gzip.open(os.path.join(TOP, fn), 'rt').read().split('\n')
        assert lines[0].split()[2] == 'P'
        i = 1
        blocks = []
        while i < len(lines):
            t = lines[i].split()
            if not t:
                i += 1
                continue
            slpi = tuple(int(x) for x in t)
            if slpi == (0, 0, 0, 0):
                rest = [l for l in lines[i + 1:] if l.strip()]
                return blocks, len(rest)
            i += 1
            t = lines[i].split()
            ntot, npn = int(t[0]), int(t[1])
            i += 1
            t = lines[i].split()
            f3, ac = float(t[0]), float(t[1])
            i += 1
            E = np.empty(npn)
            S = np.empty(npn)
            for k in range(npn):
                t = lines[i].split()
                E[k] = float(t[0])
                S[k] = float(t[1])
                i += 1
            blocks.append(dict(slpi=slpi, ntot=ntot, np=npn, f3=f3, ac=ac, E=E, S=S))
        return blocks, -1  # no terminator found

    def efile_ground(fn, sym):
        lines = gzip.open(os.path.join(TOP, fn), 'rt').read().split('\n')
        cur = None
        for ln in lines[1:]:
            t = ln.split()
            if len(t) == 4 and all(x.lstrip('-').isdigit() for x in t):
                cur = tuple(int(x) for x in t)
                continue
            if cur is not None and cur[:3] == sym and len(t) >= 3 and t[1] in ('C', 'T'):
                if int(t[0]) == 1:
                    return float(t[-1])
        return None

    ions = [('Ca III', 'p20.18.gz', 'e20.16.gz', (1, 0, 0), 3),
            ('Ca IV', 'p20.17.gz', 'e20.17.gz', (2, 1, 1), 4),
            ('Ca V', 'p20.16.gz', 'e20.16.gz', (3, 1, 0), 5)]
    # fix e-file names: Ca III -> e20.18
    ions[0] = ('Ca III', 'p20.18.gz', 'e20.18.gz', (1, 0, 0), 3)
    claims = {'Ca III': (178, 19881, 3.68369, 2.00e-2),
              'Ca IV': (322, 109674, 4.99190, 2.39e-4),
              'Ca V': (442, 278060, 6.23865, 1.90e-4)}
    for lab, pfn, efn, sym, z in ions:
        blocks, rest = top_p_blocks(pfn)
        npair = sum(b['np'] for b in blocks)
        eg = efile_ground(efn, sym)
        eth = -eg if eg is not None else float('nan')
        g0 = next(b for b in blocks if b['slpi'][:3] == sym and b['slpi'][3] == 1)
        off = eth - g0['E'][0]
        dlnv = np.diff(np.log(g0['E']))
        f3set = sorted(set(round(b['f3'], 6) for b in blocks))
        cl = claims[lab]
        print(f'  {lab}: blocks {len(blocks)} (claim {cl[0]}), pairs {npair} (claim {cl[1]}), '
              f'terminator-at-EOF leftover lines {rest}')
        print(f'    e-file ground E = {eg} Ry -> E_th {eth} (claim {cl[2]});  '
              f'grid[0] {g0["E"][0]:.6f}, E_th-grid[0] = {off:.6f} vs z^2/100 = {z*z/100:.6f}')
        print(f'    ground NP {g0["np"]}, dln(nu) median {np.median(dlnv):.3e} (claim {cl[3]:.2e}); '
              f'f3 distinct count {len(f3set)}'
              + (f' (all == {f3set[0]})' if len(f3set) == 1 else f' first3 {f3set[:3]}'))
        if lab == 'Ca III':
            sig_at_th = float(np.interp(eth, g0['E'], g0['S']))
            print(f'    round-trip: sigma(p-file, linear @E_th) = {sig_at_th:.4f} Mb (claim 9.6121)')
            phc = parse_phot_entries(os.path.join(ATOMIC, 'CA/III/10apr99/phot_smooth.dat'))
            g = phc[0]
            print(f'    CMFGEN phot_smooth entry0 {g["cfg"]!r} type {g["typ"]} npts {g["npts"]}, '
                  f'sigma(u=1) = {g["s"][0]:.4f} Mb (claim 9.6080); ratio {sig_at_th/g["s"][0]:.4f}')
        if lab == 'Ca V':
            const = all(abs(b['f3'] - 0.246793) < 5e-7 for b in blocks)
            print(f'    f3 == 0.246793 for ALL {len(blocks)} blocks: {const}')

    # ---------------------------------------------------------------- C10b ptpx walk
    sec('C10b  ptpx nr-walk')
    with open(os.path.join(NORAD, 'ni2.ptpx.txt')) as f:
        lines = f.read().split('\n')
    i = 0
    while not (len(lines[i].split()) == 3 and lines[i].split()[2] == 'P'):
        i += 1
    i += 1
    nblocks = 0
    npairs = 0
    first_block = None
    term = False
    leftover = -1
    while i < len(lines):
        t = lines[i].split()
        if not t:
            i += 1
            continue
        vals = tuple(int(x) for x in t)
        if vals == (0, 0, 0, 0):
            term = True
            leftover = sum(1 for l in lines[i + 1:] if l.strip())
            break
        i += 1
        t = lines[i].split()
        ntot, nr = int(t[0]), int(t[1])
        i += 1
        i += 1  # BE ac
        if first_block is None:
            first_block = (vals, ntot, nr)
        # spot-verify first and last row of the block parse as (float,float)
        float(lines[i].split()[0]), float(lines[i].split()[1])
        float(lines[i + nr - 1].split()[0]), float(lines[i + nr - 1].split()[1])
        i += nr
        nblocks += 1
        npairs += nr
    print(f'  blocks {nblocks} (claim 533), pairs {npairs} (claim 3,095,334), '
          f'terminator 0 0 0 0 found: {term}, non-blank lines after terminator: {leftover}')
    print(f'  first block: SLpi={first_block[0]} ntot={first_block[1]} nr={first_block[2]} '
          f'(ntot != nr -> ntot-walk desyncs)')

    # ---------------------------------------------------------------- CSV
    import csv
    with open(os.path.join(HERE, 'fable_c6_classification.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['entry_idx', 'config', 'cs_type', 'npts', 'px_block', 'slpi',
                    'class', 'n_candidates', 'note'])
        for r in out_csv:
            w.writerow(r)
    print('\nCSV written: fable_c6_classification.csv')


if __name__ == '__main__':
    main()
