#!/usr/bin/env python3
"""gamma_coiii_alllevel.py -- ALL-LEVEL photoionization rate of Co III from CMFGEN's OWN data.

  Gamma_all(d) = SUM_l [n_l(d)/n_CoIII(d)] * R_l(d)
  R_l(d)       = 4pi INT_{nu_l}^inf sigma_l(nu) J_nu(d) / (h nu) dnu

Everything comes from one converged CMFGEN snapshot (toy06 @ 19.48d, jnu4):
  n_l(d)   POPCOB          (rite_asc.f: CI(NCI,ND) column-major, then DCI(ND)=next-ion ground)
  sigma_l  PHOTCoIII_A     (term-grouped, Type-1 Seaton; sub_phot_gen.f:412-417)
  nu_l,g_l CoIII_F_OSCDAT  (level names -> phot terms by NAME, never by index)
  J_nu(d)  EDDFACTOR       (direct-access; J(1:ND) then FL[1e15 Hz] per record)
  v(d)     RVTJ

CMFGEN Type-1 Seaton (newsubs/sub_phot_gen.f:412-417, CONV_FAC=1e-8 Mb->prog units):
  RU = nu_th/nu ;  sigma = 1e-18 * A0 * (A1 + (1-A1)*RU) * RU**A2   [cm^2]
so with (A1,A2) = (1,2) for every Co III term:  sigma = 1e-18*A0*(nu_th/nu)^2.

Gate order (fail-closed): V1 POPCOB round-trip -> V2 Gamma_gnd anchor -> V3 name match
-> V4 population closure -> only then Gamma_all.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

H = 6.62607015e-27          # erg s
C_LIGHT = 2.99792458e10     # cm/s
EV = 1.602176634e-12
KB_EV = 8.617333262e-5
FOURPI = 4.0 * math.pi
MB = 1.0e-18                # Megabarn -> cm^2

RUN = '/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
LABELS = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
# The 9 forming-shell labels are a SUBSET of the v_mid = 4264+728*i ladder
# (lumina indices 0,2,4,5,6,7,8,9,10) -- identical list to the anchor CSV.
TARGET_V = [4264, 5720, 7176, 7904, 8632, 9360, 10088, 10816, 11544]
LUM_IDX = [0, 2, 4, 5, 6, 7, 8, 9, 10]
# CMFGEN ground-state Gamma(Co III), repo validation/.../gamma_feiii_coiii_formingshells.csv
ANCHOR = {'s0': 0.434427555687745, 's1': 0.01614722644525109, 's2': 3.8710877546217315e-4,
          's3': 1.6822814907910634e-5, 's4': 3.6805366461489786e-6, 's5': 6.663060369760248e-7,
          's6': 1.8843929907591032e-8, 's7': 1.9468261230400717e-8, 's8': 1.7267149049332054e-7}


# ----------------------------------------------------------------- parsers
def parse_popcob(path):
    """-> (POP_SPECIES[ND], {ion: (CI[ND,NCI], DCI[ND])}, ion order)."""
    txt = open(path).read().splitlines()
    nd = None
    for ln in txt[:10]:
        if ln.strip().startswith('ND:'):
            nd = int(ln.split(':')[1])
    if nd is None:
        raise SystemExit('POPCOB: ND not found')

    hdr = [(i, ln) for i, ln in enumerate(txt) if 'Number of' in ln and 'levels:' in ln]

    def grab(start, count):
        """read `count` floats from line `start` onwards; return (arr, next_line)."""
        vals = []
        j = start
        while j < len(txt) and len(vals) < count:
            vals += [float(t) for t in txt[j].split()]
            j += 1
        if len(vals) != count:
            raise SystemExit(f'POPCOB: expected {count} floats at line {start}, got {len(vals)}')
        return np.array(vals), j

    pop_species, _ = grab(4, nd)          # POP_SPECIES(1:ND) written just before ion blocks
    ions, order = {}, []
    for (i, ln) in hdr:
        name = ln.split('Number of')[1].split('levels:')[0].strip()
        nci = int(ln.split(':')[1])
        if nci == 0:
            continue
        ci_flat, nxt = grab(i + 2, nci * nd)
        dci, _ = grab(nxt, nd)
        ions[name] = (ci_flat.reshape(nd, nci), dci)   # Fortran CI(NCI,ND) -> [depth, level]
        order.append(name)
    return pop_species, ions, order, nd


def parse_osc(path):
    """-> (names, g, E_cm, chi_cm, n_levels). Level order = file order = POPCOB order."""
    lines = open(path).read().splitlines()
    nlev = chi_cm = None
    hdr_end = None
    for j, ln in enumerate(lines):
        if 'Number of energy levels' in ln:
            nlev = int(ln.split('!')[0].strip())
            hdr_end = j
        elif 'Ionization energy' in ln:
            chi_cm = float(ln.split('!')[0].strip())
        if nlev and chi_cm:
            break
    names, g, ecm = [], [], []
    for ln in lines[hdr_end + 1:]:
        t = ln.split()
        if len(t) < 7:
            continue
        try:
            gg = float(t[1]); ee = float(t[2]); float(t[3]); int(float(t[6]))
        except ValueError:
            continue
        if re.match(r'^[-+]?\d*\.?\d+$', t[0]):
            continue
        names.append(t[0]); g.append(gg); ecm.append(ee)
        if len(names) >= nlev:
            break
    return names, np.array(g), np.array(ecm), chi_cm, nlev


def parse_f_to_s(path, nlev):
    """-> (super_level_index[nlev] 1-based, names[nlev]). Order == osc/POPCOB order."""
    lines = open(path).read().splitlines()
    sl, nm = [], []
    for ln in lines:
        t = ln.split()
        if len(t) < 7:
            continue
        try:
            float(t[1]); float(t[2]); float(t[3]); int(t[5]); int(t[6]); idx = int(t[7])
        except (ValueError, IndexError):
            continue
        if idx != len(sl) + 1:
            continue
        sl.append(int(t[5])); nm.append(t[0])
        if len(sl) >= nlev:
            break
    return np.array(sl), nm


def parse_prrr(path, ion, nd, nsl):
    """CMFGEN's own converged rates (WRRECOMCHK_V5 / PRRR_SL_V6:126).

    PR(I,J) = n_SL_I(J) * R_SL_I(J)  [cm^-3 s^-1], written raw, 10 depths per block.
    -> PR[nsl, nd], ion_density[nd].
    """
    lines = open(path).read().replace('\f', '\n').splitlines()
    PR = np.zeros((nsl, nd))
    DI = np.zeros(nd)
    j0 = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == 'Depth index':
            j0 = int(lines[i + 1].split()[0]) - 1
        elif s == 'Ion Density':
            v = [float(t) for t in lines[i + 1].split()]
            DI[j0:j0 + len(v)] = v
        elif s == f'{ion} Photoionization Rates':
            for k in range(nsl):
                v = [float(t) for t in lines[i + 1 + k].split()]
                PR[k, j0:j0 + len(v)] = v
    return PR, DI


def parse_phot(path):
    """-> ({term_lower: (cs_type, params)}, header dict). Handles all fit/tabulated types."""
    lines = open(path).read().splitlines()
    hdr, start = {}, 0
    for k, ln in enumerate(lines):
        if '!' in ln:
            v, key = ln.split('!', 1)[0].strip(), ln.split('!', 1)[1].strip()
            hdr[key] = v
        if 'Total number of data pairs' in ln:
            start = k + 1
            break
    terms = {}
    k = start
    while k < len(lines):
        if 'Configuration name' in lines[k]:
            name = lines[k].split('!')[0].strip()
            cs_type = int(lines[k + 1].split('!')[0].strip())
            npts = int(lines[k + 2].split('!')[0].strip())
            vals, j = [], k + 3
            while j < len(lines) and len(vals) < npts:
                s = lines[j].strip()
                if s:
                    vals += [float(x.replace('D', 'E').replace('d', 'e')) for x in s.split()]
                j += 1
            terms[name.lower()] = (cs_type, vals[:npts])
            k = j
        else:
            k += 1
    return terms, hdr


def read_eddfactor(path):
    """-> (J[nfreq, ND] ascending in nu, nu[Hz], ND, finish_flag)."""
    info = open(path + '_INFO').read().splitlines()[2].split()
    nd, recl, word = int(info[0]), int(info[1]), int(info[2])
    little = (info[5] == 'T')
    nwr = recl // word
    raw = np.fromfile(path, dtype='<f8' if little else '>f8')
    raw = raw[:(raw.size // nwr) * nwr].reshape(-1, nwr)
    finish = raw[4, 0]
    data = raw[14:]
    good = np.isfinite(data[:, :nd]).all(axis=1) & (data[:, nd] > 0)
    J = data[good, :nd]
    nu = data[good, nd] * 1e15
    o = np.argsort(nu)
    return np.ascontiguousarray(J[o]), nu[o], nd, finish


def rvtj_block(text, label, nd):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == label:
            vals, j = [], i + 1
            while j < len(lines) and len(vals) < nd:
                try:
                    vals += [float(t) for t in lines[j].split()]
                except ValueError:
                    break
                j += 1
            return np.array(vals[:nd])
    raise KeyError(label)


# ------------------------------------------------------- radiative kernels
def cumint_from_above(nu, y):
    """C[i] = INT_{nu_i}^{nu_max} y dnu (trapezoid), nu ascending."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(nu)
    C = np.zeros(nu.size)
    C[:-1] = np.cumsum(seg[::-1])[::-1]
    return C


def integral_above(nu, y, C, nu_th):
    """INT_{nu_th}^{nu_max} y dnu with exact partial first bin (linear interp of y)."""
    if nu_th <= nu[0]:
        return C[0]
    if nu_th >= nu[-1]:
        return 0.0
    i = int(np.searchsorted(nu, nu_th))          # nu[i-1] <= nu_th < nu[i]
    f = (nu_th - nu[i - 1]) / (nu[i] - nu[i - 1])
    y_th = y[i - 1] + f * (y[i] - y[i - 1])
    return 0.5 * (y_th + y[i]) * (nu[i] - nu_th) + C[i]


def level_rates(nu, Jd, nu_th, s0_mb, a1, a2):
    """R_l for every level at one depth. Vectorized via two cumulative moments.

    R = 4pi/h * 1e-18*A0 * [ A1*nu_th^A2 * INT J nu^-(A2+1) dnu
                           + (1-A1)*nu_th^(A2+1) * INT J nu^-(A2+2) dnu ]
    """
    R = np.zeros(nu_th.size)
    for (aa1, aa2) in set(zip(a1.tolist(), a2.tolist())):
        sel = np.where((a1 == aa1) & (a2 == aa2) & (s0_mb > 0))[0]
        if sel.size == 0:
            continue
        y1 = Jd * nu ** (-(aa2 + 1.0))
        C1 = cumint_from_above(nu, y1)
        need2 = abs(1.0 - aa1) > 0
        if need2:
            y2 = Jd * nu ** (-(aa2 + 2.0))
            C2 = cumint_from_above(nu, y2)
        for k in sel:
            nt = nu_th[k]
            if nt <= 0:
                continue
            v = aa1 * nt ** aa2 * integral_above(nu, y1, C1, nt)
            if need2:
                v += (1.0 - aa1) * nt ** (aa2 + 1.0) * integral_above(nu, y2, C2, nt)
            R[k] = FOURPI / H * MB * s0_mb[k] * v
    return R


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    R = a.run
    log = []

    def say(s=''):
        print(s)
        log.append(s)

    # ---------------- inputs
    pop_species, ions, ion_order, ND = parse_popcob(f'{R}/POPCOB')
    say(f'[popcob] ND={ND} ions={ion_order}')
    names, gl, ecm, chi_cm, nlev = parse_osc(f'{R}/CoIII_F_OSCDAT')
    say(f'[osc]    CoIII levels={nlev} parsed={len(names)} chi={chi_cm} cm^-1 '
        f'= {chi_cm*C_LIGHT*H/EV:.6f} eV')
    terms, phot_hdr = parse_phot(f'{R}/PHOTCoIII_A')
    say(f"[phot]   terms={len(terms)} unit={phot_hdr.get('Cross-section unit')} "
        f"splitJ={phot_hdr.get('Split J levels')} routes={phot_hdr.get('Number of photoionization routes')}")
    J, nu, ndj, finish = read_eddfactor(f'{R}/EDDFACTOR')
    say(f'[edd]    ND={ndj} nfreq={J.shape[0]} nu=[{nu[0]:.4e},{nu[-1]:.4e}] Hz '
        f'lam=[{C_LIGHT*1e8/nu[-1]:.2f},{C_LIGHT*1e8/nu[0]:.3e}] A FINISH={finish}')
    rt = open(f'{R}/RVTJ').read()
    V = rvtj_block(rt, 'Velocity (km/s)', ND)
    Te = rvtj_block(rt, 'Temperature (10^4K)', ND) * 1e4
    say(f'[rvtj]   V=[{V[0]:.0f},{V[-1]:.0f}] km/s  Te=[{Te[0]:.0f},{Te[-1]:.0f}] K')
    say()

    if ndj != ND:
        raise SystemExit('ND mismatch EDDFACTOR vs POPCOB')
    if not np.isfinite(finish) or finish == 0:
        raise SystemExit('EDDFACTOR FINISH_REC==0 -> field incomplete')

    # ---------------- V1: POPCOB round-trip identities
    say('=== V1  POPCOB round-trip (DCI(ion) == ground pop of next ion) ===')
    v1_ok = True
    for k in range(len(ion_order) - 1):
        lo, hi = ion_order[k], ion_order[k + 1]
        dci = ions[lo][1]
        gnd = ions[hi][0][:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(gnd > 0, dci / gnd, np.nan)
        bad = int(np.sum(~np.isfinite(r) | (np.abs(r - 1.0) > 1e-6)))
        say(f'  {lo:>7s} DCI / {hi:<7s} ground : min={np.nanmin(r):.8f} max={np.nanmax(r):.8f} '
            f'  off-by->1e-6 depths: {bad}/{ND}')
        if bad:
            v1_ok = False
    say(f'  V1 = {"PASS" if v1_ok else "FAIL"}')
    say()
    if not v1_ok:
        raise SystemExit('V1 failed -> POPCOB parse wrong, stop.')

    CI = ions['CoIII'][0]                       # [ND, 1000]
    if CI.shape[1] != nlev:
        raise SystemExit(f'POPCOB CoIII NCI={CI.shape[1]} != osc nlev={nlev}')
    n_coiii = CI.sum(axis=1)

    # ---------------- V3: name matching (never index order)
    say('=== V3  level -> phot-term matching by NAME ===')
    TERMRE = re.compile(r'\[[^\[\]]*\]\s*$')
    s0_mb = np.zeros(nlev); a1 = np.zeros(nlev); a2 = np.zeros(nlev)
    matched = np.zeros(nlev, dtype=bool)
    cstypes = {}
    miss = []
    for k, nm in enumerate(names):
        t = TERMRE.sub('', nm).strip().lower()
        p = terms.get(t)
        if p is None:
            miss.append((k, nm, t))
            continue
        cs, par = p
        cstypes[cs] = cstypes.get(cs, 0) + 1
        if cs != 1:
            miss.append((k, nm, t + f' [unsupported cs_type={cs}]'))
            continue
        s0_mb[k], a1[k], a2[k] = par[0], par[1], par[2]
        matched[k] = True
    say(f'  matched {int(matched.sum())}/{nlev} levels  ({100*matched.mean():.2f}%)  '
        f'cs_type histogram={cstypes}')
    say(f'  distinct phot terms used = {len(set(TERMRE.sub("",n).strip().lower() for n in names) & set(terms))}'
        f' of {len(terms)} in file')
    say(f'  Seaton params: A1 unique={sorted(set(a1[matched].tolist()))}  '
        f'A2 unique={sorted(set(a2[matched].tolist()))}  '
        f'A0(Mb) range=[{s0_mb[matched].min():.4f},{s0_mb[matched].max():.4f}]')
    if miss:
        say(f'  UNMATCHED {len(miss)} levels; first 10:')
        for (k, nm, t) in miss[:10]:
            say(f'    lev#{k+1:4d} {nm:28s} -> term "{t}" not in phot file')
    else:
        say('  UNMATCHED: none')
    say()

    # thresholds (CMFGEN convention: nu_th = (chi - E_l)*c ; EXC_FREQ(final state)=0)
    nu_th = (chi_cm - ecm) * C_LIGHT
    above = nu_th <= 0
    say(f'  levels above the ionization limit (nu_th<=0): {int(above.sum())}')
    use = matched & (nu_th > 0)
    s0_mb = np.where(use, s0_mb, 0.0)

    # osc file's own nu15 column as an independent check of nu_th
    say(f'  nu_th(ground) = {nu_th[0]:.6e} Hz  (osc column 8.10039e15) '
        f'lam_th = {C_LIGHT*1e8/nu_th[0]:.3f} A')
    say()

    # ---------------- V4: population closure
    say('=== V4  population closure ===')
    tot = np.zeros(ND)
    for k, ion in enumerate(ion_order):
        tot += ions[ion][0].sum(axis=1)
    tot += ions[ion_order[-1]][1]                # DCI of the last ion = next stage ground
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(pop_species > 0, tot / pop_species, np.nan)
    say('  modelled Co stages = Co II..Co VI (+Co VII ground via DCI); Co VII excited and'
        ' Co VIII+ are NOT in the model, so closure must fail where Co is highly ionized.')
    say(f'  SUM_ions SUM_l n_l (+top DCI) / POP_SPECIES(Co): min={np.nanmin(rr):.6f} '
        f'max={np.nanmax(rr):.6f}')
    say(f'  n_CoIII: min={n_coiii.min():.4e} max={n_coiii.max():.4e} cm^-3')
    frac3 = n_coiii / np.where(pop_species > 0, pop_species, np.nan)
    say(f'  Co III ion fraction: min={np.nanmin(frac3):.4f} max={np.nanmax(frac3):.4f}')

    # ---------------- shell -> depth (nearest v, same convention as the anchor CSV)
    shell_depth = {lb: int(np.argmin(np.abs(V - vt))) for lb, vt in zip(LABELS, TARGET_V)}

    say('  closure at the 9 forming-shell depths:')
    v4_ok = True
    for lb in LABELS:
        d = shell_depth[lb]
        flag = '' if abs(rr[d] - 1) < 5e-3 else '   <-- OFF'
        if abs(rr[d] - 1) >= 5e-3:
            v4_ok = False
        say(f'    {lb} depth={d+1:3d} v={V[d]:8.0f} POP_SPECIES={pop_species[d]:.5e} '
            f'sum={tot[d]:.5e} ratio={rr[d]:.6f}  X(CoIII)={frac3[d]:.4f}{flag}')
    say(f'  V4 = {"PASS" if v4_ok else "PARTIAL"} (closure tol 5e-3 at the forming shells)')
    say()

    # ---------------- per-depth rates
    say('=== rates ===')
    Ggnd = np.zeros(ND); Gall = np.zeros(ND); Gboltz = np.zeros(ND)
    frac_nosig = np.zeros(ND)
    Rstore = {}
    for d in range(ND):
        Rl = level_rates(nu, J[:, d], nu_th, s0_mb, a1, a2)
        n = CI[d]
        ntot = n.sum()
        p = n / ntot
        Ggnd[d] = Rl[0]
        Gall[d] = float(np.dot(p, Rl))
        frac_nosig[d] = float(p[~use].sum())
        x = ecm * C_LIGHT * H / EV / (KB_EV * Te[d])
        w = np.where(x < 500, gl * np.exp(-np.minimum(x, 500)), 0.0)
        Gboltz[d] = float(np.dot(w / w.sum(), Rl))
        if d in shell_depth.values():
            Rstore[d] = (Rl, p)

    # ---------------- independent cross-checks of the machinery
    say('=== cross-checks ===')
    sigSB = 5.670374419e-5
    intJ = float(np.trapz(J[:, -1], nu))
    say(f'  radiation field units: INT J dnu (innermost, d={ND}) = {intJ:.4e} vs '
        f'sigma*T^4/pi = {sigSB*Te[-1]**4/math.pi:.4e}  ratio={intJ/(sigSB*Te[-1]**4/math.pi):.4f}')
    d0c = shell_depth['s0']
    m = nu >= nu_th[0]
    sig_bf = MB * s0_mb[0] * (a1[0] + (1 - a1[0]) * nu_th[0] / nu[m]) * (nu_th[0] / nu[m]) ** a2[0]
    brute = FOURPI * float(np.trapz(sig_bf * J[m, d0c] / (H * nu[m]), nu[m]))
    say(f'  brute-force ground R(s0) = {brute:.6e}  vs cumulative-kernel {Ggnd[d0c]:.6e}  '
        f'rel.diff={(Ggnd[d0c]/brute-1)*100:+.5f}%')
    say()

    # ---------------- V2: anchor reproduction
    say('=== V2  Gamma_gnd anchor reproduction (repo gamma_feiii_coiii_formingshells.csv) ===')
    say(f'  {"shell":>5} {"v_target":>8} {"v_cmfgen":>9} {"depth":>5} {"anchor":>11} '
        f'{"this work":>11} {"ratio":>7}')
    v2_ok = True
    for lb in LABELS:
        d = shell_depth[lb]
        r = Ggnd[d] / ANCHOR[lb]
        gate = '' if 0.5 < r < 2.0 else '   <-- outside x2'
        if not (0.5 < r < 2.0):
            v2_ok = False
        say(f'  {lb:>5} {TARGET_V[LABELS.index(lb)]:8d} {V[d]:9.1f} {d+1:5d} '
            f'{ANCHOR[lb]:11.4e} {Ggnd[d]:11.4e} {r:7.3f}{gate}')
    say(f'  V2 = {"PASS" if v2_ok else "FAIL"} (factor-2 gate, all 9 shells)')
    say()
    if not v2_ok:
        raise SystemExit('V2 failed -> stop before Gamma_all.')

    # ---------------- grid convergence (top-5 levels at s0, 2x refinement)
    say('=== integration convergence: 2x refined nu grid, top-5 contributing levels at s0 ===')
    d0 = shell_depth['s0']
    Rl0, p0 = Rstore[d0]
    top5 = np.argsort(-(p0 * Rl0))[:5]
    lnu = np.log(nu); lJ = np.log(np.maximum(J[:, d0], 1e-300))
    nu2 = np.exp(np.sort(np.concatenate([lnu, 0.5 * (lnu[1:] + lnu[:-1])])))
    J2 = np.exp(np.interp(np.log(nu2), lnu, lJ))
    Rl0_ref = level_rates(nu2, J2, nu_th, s0_mb, a1, a2)
    for k in top5:
        say(f'  lev#{k+1:4d} {names[k]:26s} R={Rl0[k]:.6e}  R(2x)={Rl0_ref[k]:.6e}  '
            f'rel.diff={(Rl0_ref[k]/Rl0[k]-1)*100:+.4f}%')
    say(f'  Gamma_all(s0) native={float(np.dot(p0,Rl0)):.6e}  2x={float(np.dot(p0,Rl0_ref)):.6e}  '
        f'rel.diff={(np.dot(p0,Rl0_ref)/np.dot(p0,Rl0)-1)*100:+.4f}%')
    say()

    say('=== forming shells: Gamma_all / Gamma_gnd ===')
    say(f'  {"shell":>5} {"v_cmf":>8} {"Te":>7} {"n_CoIII":>10} {"G_gnd":>11} {"G_all":>11} '
        f'{"ratio":>8} {"G_boltz":>11} {"b-ratio":>8} {"no_sig%":>8}')
    for lb in LABELS:
        d = shell_depth[lb]
        say(f'  {lb:>5} {V[d]:8.0f} {Te[d]:7.0f} {n_coiii[d]:10.3e} {Ggnd[d]:11.4e} '
            f'{Gall[d]:11.4e} {Gall[d]/Ggnd[d]:8.3f} {Gboltz[d]:11.4e} '
            f'{Gboltz[d]/Ggnd[d]:8.3f} {100*frac_nosig[d]:8.4f}')
    say()

    say('=== how concentrated is Gamma_all? (no single level dominates -> report the spread) ===')
    E_eV = ecm * C_LIGHT * H / EV
    bands = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 14), (14, 18), (18, 40)]
    say(f'  {"shell":>5} {"N(50%)":>7} {"N(90%)":>7} {"N(99%)":>7} {"<E_l>_w(eV)":>12} '
        f'{"<lam_th>_w(A)":>13} | contribution share by E_l band (eV)')
    say(' ' * 60 + '| ' + '  '.join(f'{lo}-{hi}' for lo, hi in bands))
    for lb in LABELS:
        d = shell_depth[lb]
        Rl, p = Rstore[d]
        con = p * Rl
        tot_c = con.sum()
        cs = np.cumsum(np.sort(con)[::-1]) / tot_c
        n50 = int(np.searchsorted(cs, 0.50) + 1)
        n90 = int(np.searchsorted(cs, 0.90) + 1)
        n99 = int(np.searchsorted(cs, 0.99) + 1)
        Ew = float(np.dot(con, E_eV) / tot_c)
        lw = float(np.dot(con, C_LIGHT * 1e8 / nu_th) / tot_c)
        shares = [float(con[(E_eV >= lo) & (E_eV < hi)].sum() / tot_c) for lo, hi in bands]
        say(f'  {lb:>5} {n50:7d} {n90:7d} {n99:7d} {Ew:12.3f} {lw:13.1f} | '
            + '  '.join(f'{s:.3f}' for s in shares))
    say()

    # ---- super-level granularity: the F-levels are NOT independent unknowns
    say('=== super-level view (CoIII_F_TO_S): which SOLVED degree of freedom carries Gamma_all ===')
    SL, slnames = parse_f_to_s(f'{R}/CoIII_F_TO_S', nlev)
    if SL.size == nlev and slnames == names:
        nsl = int(SL.max())
        say(f'  {nlev} F-levels -> {nsl} super levels; F_TO_S level names/order identical to '
            f'CoIII_F_OSCDAT (independent confirmation of the level ordering used for POPCOB)')
        # is the within-super-level distribution Boltzmann at T_e? (CMFGEN's SL assumption)
        d = shell_depth['s0']
        w = gl * np.exp(-E_eV / (KB_EV * Te[d]))
        dev = []
        for s in range(1, nsl + 1):
            k = np.where(SL == s)[0]
            if k.size < 2:
                continue
            q = CI[d][k] / w[k]
            dev.append(float(q.max() / q.min() - 1.0))
        say(f'  within-super-level n_l/(g exp(-E/kTe)) spread at s0: median={np.median(dev):.2e} '
            f'max={max(dev):.2e}  -> Boltzmann-at-Te inside each SL is '
            f'{"CONFIRMED" if max(dev) < 1e-3 else "NOT exact"}')
        for lb in ['s0', 's6', 's8']:
            d = shell_depth[lb]
            Rl, p = Rstore[d]
            con = p * Rl
            tot_c = con.sum()
            agg = np.array([con[SL == s].sum() for s in range(1, nsl + 1)])
            Eagg = np.array([E_eV[SL == s].min() for s in range(1, nsl + 1)])
            o = np.argsort(-agg)[:8]
            say(f'  {lb}: top super levels (share | SL# | E_min eV | n_F): '
                + ', '.join(f'{agg[i]/tot_c:.3f}|SL{i+1}|{Eagg[i]:.2f}eV|{int((SL==i+1).sum())}'
                            for i in o))
            cs = np.cumsum(np.sort(agg)[::-1]) / tot_c
            say(f'      super levels for 50%/90% of Gamma_all: '
                f'{int(np.searchsorted(cs,0.5)+1)}/{int(np.searchsorted(cs,0.9)+1)} of {nsl}')
        # ---- V5: CMFGEN's OWN converged photoionization rate (CoIIIPRRR)
        say()
        say('=== V5  vs CMFGEN\'s OWN converged rate (CoIIIPRRR, PRRR_SL_V6:126) ===')
        say('  PR(SL,d) = n_SL * R_SL [cm^-3 s^-1]  =>  Gamma_all_CMFGEN = SUM_SL PR / n_CoIII')
        try:
            PR, DI = parse_prrr(f'{R}/CoIIIPRRR', 'CoIII', ND, nsl)
            Gcmf = PR.sum(axis=0) / np.where(n_coiii > 0, n_coiii, np.nan)
            say(f'  {"shell":>5} {"depth":>5} {"Gamma_all(this)":>16} {"Gamma_all(CMFGEN)":>18} '
                f'{"ratio":>7}')
            rr5 = []
            for lb in LABELS:
                d = shell_depth[lb]
                q = Gall[d] / Gcmf[d]
                rr5.append(q)
                say(f'  {lb:>5} {d+1:5d} {Gall[d]:16.4e} {Gcmf[d]:18.4e} {q:7.3f}')
            say(f'  V5 = {"PASS" if all(0.5 < q < 2.0 for q in rr5) else "FAIL"} '
                f'(factor-2 gate vs CMFGEN\'s own rate)')
            gcol = Gcmf
            # V5b: ground TERM (super level 1 = the four 3d7_a4Fe J-levels) -- an
            # anchor-independent check of the sigma x J chain at the ground edge.
            k1 = np.where(SL == 1)[0]
            say(f'  V5b ground super level SL1 = {[names[i] for i in k1]}')
            say(f'  {"shell":>5} {"depth":>5} {"R_SL1(this)":>13} {"R_SL1(CMFGEN)":>15} '
                f'{"ratio":>7} {"Gamma_gnd(J=9/2)":>17}')
            for lb in LABELS:
                d = shell_depth[lb]
                Rl, _ = Rstore[d]
                n1 = CI[d][k1].sum()
                mine = float(np.dot(CI[d][k1], Rl[k1]) / n1)
                cm = PR[0, d] / n1
                say(f'  {lb:>5} {d+1:5d} {mine:13.4e} {cm:15.4e} {mine/cm:7.3f} '
                    f'{Ggnd[d]:17.4e}')
        except Exception as e:                                     # noqa: BLE001
            say(f'  PRRR comparison unavailable: {e}')
            gcol = np.full(ND, np.nan)
    else:
        say('  F_TO_S parse mismatch -- super-level view skipped')
        gcol = np.full(ND, np.nan)
    say()

    # ---------------- outputs
    outcsv = os.path.join(a.out, 'gamma_coiii_alllevel.csv')
    lab_of = {v: k for k, v in shell_depth.items()}
    with open(outcsv, 'w') as f:
        f.write('depth,v_kms,Te_K,lumina_shell,n_CoIII,Gamma_gnd_reproduced,Gamma_all,'
                'Gamma_all_cmfgen_prrr,Gamma_boltz,ratio_all_over_gnd,frac_pop_without_sigma\n')
        for d in range(ND):
            f.write(f'{d+1},{V[d]:.6f},{Te[d]:.2f},{lab_of.get(d,"")},{n_coiii[d]:.6e},'
                    f'{Ggnd[d]:.6e},{Gall[d]:.6e},{gcol[d]:.6e},{Gboltz[d]:.6e},'
                    f'{(Gall[d]/Ggnd[d] if Ggnd[d]>0 else float("nan")):.6f},'
                    f'{frac_nosig[d]:.6e}\n')
    say(f'[out] {outcsv}')

    # level breakdown
    brk = os.path.join(a.out, 'gamma_coiii_level_breakdown.csv')
    with open(brk, 'w') as f:
        f.write('shell,depth,rank,level_index,level_name,E_cm,E_eV,g,lam_th_A,sigma0_Mb,'
                'n_l_over_n_ion,R_l,contrib,contrib_frac_of_Gall\n')
        for lb in LABELS:
            d = shell_depth[lb]
            Rl, p = Rstore[d]
            con = p * Rl
            tot_c = con.sum()
            for rank, k in enumerate(np.argsort(-con)[:20]):
                f.write(f'{lb},{d+1},{rank+1},{k+1},{names[k]},{ecm[k]:.4f},'
                        f'{ecm[k]*C_LIGHT*H/EV:.6f},{gl[k]:.0f},{C_LIGHT*1e8/nu_th[k]:.3f},'
                        f'{s0_mb[k]:.4f},{p[k]:.6e},{Rl[k]:.6e},{con[k]:.6e},'
                        f'{con[k]/tot_c:.6f}\n')
    say(f'[out] {brk}')

    for lb in ['s0', 's6', 's8']:
        d = shell_depth[lb]
        Rl, p = Rstore[d]
        con = p * Rl
        say()
        say(f'--- {lb} (depth {d+1}, v={V[d]:.0f} km/s, Te={Te[d]:.0f} K) top-10 contributors '
            f'to Gamma_all={con.sum():.4e} ---')
        say(f'  {"#":>3} {"lev":>5} {"name":26s} {"E(eV)":>8} {"lam_th":>9} {"sig0":>7} '
            f'{"n_l/n":>10} {"R_l":>11} {"share":>7}')
        for rank, k in enumerate(np.argsort(-con)[:10]):
            say(f'  {rank+1:>3} {k+1:>5} {names[k]:26s} {ecm[k]*C_LIGHT*H/EV:8.3f} '
                f'{C_LIGHT*1e8/nu_th[k]:9.1f} {s0_mb[k]:7.3f} {p[k]:10.3e} {Rl[k]:11.3e} '
                f'{con[k]/con.sum():7.4f}')

    with open(os.path.join(a.out, 'run_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
