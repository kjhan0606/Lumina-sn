#!/usr/bin/env python3
"""M1: Kramers (k-packet fb active path) vs real-sigma_bf Milne recombination alpha.

Offline only. Reproduces, byte-for-byte in formula, two production code paths:

 (K) src/lumina_plasma.c:4337   alpha = 2.6e-13 * stage^2 * (T_e/1e4)^-0.75
     (the alpha that builds C_fb -> p_kpacket_fb; single dominant edge)

 (M) src/lumina_plasma.c:5210   frozenin_alpha_rr(ip_prod, ip_recomb, T_e)
     = Milne integral over the LOADED CMFGEN sigma_bf of every level of the
       product ion (+ optional spin gate, + optional Badnell/ADAS DR)

Env chain reproduced = parity42 RUN FOOTER:
  LUMINA_ALPHA_SPINGATE=1, LUMINA_FROZENIN_DR=1, LUMINA_RATES_FIX unset,
  LUMINA_FB_COOL_KT=1, LUMINA_KPKT_FB_MULTI unset, LUMINA_SUPER_CUTOFF=100
"""
import csv, math, os, re, struct, sys
import numpy as np

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
REF = os.path.join(ROOT, 'data/tardis_reference_toy06_19p48d_sivcaiv')
BIN = os.path.join(REF, 'cmfgen_sigma_bf.bin')
PLASMA = os.path.join(ROOT, 'logs/coevolve_consume_parity42/lumina_plasma_state.csv')
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')

# --- constants exactly as src/lumina.h ---------------------------------------
H = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
ME = 9.1093837015e-28
EV = 1.602176634e-12
NU_MIN, NU_MAX, NBIN = 1.5e14, 3.0e16, 1000


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
    ok = (os.path.getsize(path) == off + nlev * nbin * 8)
    return dict(magic=magic, ver=ver, nlev=nlev, nbin=nbin, numin=numin,
                numax=numax, has=has, sigma=sig, size_ok=ok)


# --- data --------------------------------------------------------------------
lv_Z, lv_ion, lv_num, lv_E, lv_g, lv_sl = [], [], [], [], [], []
with open(os.path.join(REF, 'levels.csv')) as f:
    for r in csv.DictReader(f):
        lv_Z.append(int(r['atomic_number'])); lv_ion.append(int(r['ion_number']))
        lv_num.append(int(r['level_number'])); lv_E.append(float(r['energy_eV']))
        lv_g.append(float(r['g'])); lv_sl.append(int(r['super_level']))
lv_Z = np.array(lv_Z); lv_ion = np.array(lv_ion); lv_num = np.array(lv_num)
lv_E = np.array(lv_E); lv_g = np.array(lv_g); lv_sl = np.array(lv_sl)

chi = {}
with open(os.path.join(REF, 'ionization_energies.csv')) as f:
    for r in csv.DictReader(f):
        chi[(int(r['atomic_number']), int(r['ion_number']))] = \
            float(r['ionization_energy_eV'])

mult = {}
with open(os.path.join(REF, 'level_multiplicity.csv')) as f:
    for r in csv.DictReader(f):
        m = int(r['multiplicity'])
        if m > 0:
            mult[(int(r['atomic_number']), int(r['ion_number']),
                  int(r['level_number']))] = m

bk = read_bake_bin(BIN)
assert bk['nlev'] == len(lv_Z), (bk['nlev'], len(lv_Z))
assert bk['nbin'] == NBIN and bk['size_ok']
assert abs(bk['numin'] - NU_MIN) < 1 and abs(bk['numax'] - NU_MAX) < 1
SIG = bk['sigma']; HAS = bk['has']

Te = []
with open(PLASMA) as f:
    for r in csv.DictReader(f):
        Te.append(float(r['T_e']))
Te = np.array(Te)

# --- grid (identical to frozenin_alpha_rr) -----------------------------------
dln = (math.log(NU_MAX) - math.log(NU_MIN)) / NBIN
log_lo = math.log(NU_MIN) + np.arange(NBIN) * dln
nu_c = np.exp(log_lo + 0.5 * dln)
dnu = np.exp(log_lo + dln) - np.exp(log_lo)

# --- spingate core-multiplicity table (src/lumina_plasma.c:5174) -------------
CORE_MULT = {
    14: {0: 3, 1: 2, 2: 1, 3: 2},
    16: {0: 3, 1: 4, 2: 3, 3: 2},
    20: {0: 1, 1: 2, 2: 1, 3: 2},
    26: {0: 5, 1: 6, 2: 5, 3: 6, 4: 5, 5: 4},
    27: {0: 4, 1: 3, 2: 4, 3: 5, 4: 6},
    28: {0: 3, 1: 2, 2: 3, 3: 4, 4: 5},
}

# --- DR table entries needed (parsed from src/lumina_plasma.c DR_TABLE) ------
def parse_dr_table():
    src = open(os.path.join(ROOT, 'src/lumina_plasma.c')).read()
    i = src.index('static const DRCoefficient DR_TABLE[] = {')
    j = src.index('\n};', i)
    body = src[i:j]
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    out = {}
    for m in re.finditer(
            r'\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'
            r'\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*(\w+)\s*\}', body):
        Z, ir, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        c = [float(x) for x in m.group(4).replace('\n', ' ').split(',') if x.strip()]
        E = [float(x) for x in m.group(5).replace('\n', ' ').split(',') if x.strip()]
        if (Z, ir) in out:      # dr_lookup returns the FIRST match
            continue
        out[(Z, ir)] = (c[:n], E[:n], m.group(6))
    return out

DR = parse_dr_table()


def alpha_dr(Z, ion_recomb, T):
    e = DR.get((Z, ion_recomb))
    if not e:
        return 0.0
    c, E, _ = e
    s = sum(ci * math.exp(-Ei / max(T, 1.0)) for ci, Ei in zip(c, E)
            if -Ei / max(T, 1.0) >= -700.0)
    return s * max(T, 1.0) ** -1.5


# --- Milne alpha (verbatim port of frozenin_alpha_rr, non-RATES_FIX branch) --
class Cont:
    """One recombination continuum: product ion (Z, k) <- recombining (Z, k+1)."""

    def __init__(self, Z, k):
        self.Z, self.k = Z, k
        self.chi_eV = chi.get((Z, k), 1e10)
        sel = np.where((lv_Z == Z) & (lv_ion == k))[0]
        self.idx = sel
        self.nxt = np.where((lv_Z == Z) & (lv_ion == k + 1))[0]
        # levels that actually enter the Milne sum
        keep = sel[(HAS[sel] > 0)]
        chi_l = self.chi_eV * EV - lv_E[keep] * EV
        keep = keep[chi_l > 0]
        self.keep = keep
        self.chi_l = self.chi_eV * EV - lv_E[keep] * EV
        self.nu_th = self.chi_l / H
        self.g = lv_g[keep]
        self.sig = np.asarray(SIG[keep, :], dtype=float)
        self.i0 = np.searchsorted(nu_c, self.nu_th)   # first bin with nu_c>=nu_th
        # spin gate
        Mcore = 0
        if self.nxt.size:
            gnd = self.nxt[lv_num[self.nxt] == 0]
            if gnd.size:
                Mcore = mult.get((Z, k + 1, 0), 0)
        if Mcore == 0:
            Mcore = CORE_MULT.get(Z, {}).get(k + 1, 0)
        self.Mcore = Mcore
        Ml = np.array([mult.get((Z, k, int(n)), 0) for n in lv_num[keep]])
        self.spin_skip = ((Mcore > 0) & (Ml > 0) &
                          (Ml != Mcore - 1) & (Ml != Mcore + 1))

    def U_ion(self, T):
        if self.nxt.size == 0:
            return 1.0
        x = lv_E[self.nxt] * EV / (KB * T)
        u = float(np.sum(lv_g[self.nxt][x < 50.0] * np.exp(-x[x < 50.0])))
        if u >= 1.0:
            return u
        return float(lv_g[self.nxt][0]) if lv_g[self.nxt][0] >= 1 else 1.0

    def alpha(self, T, gated):
        if self.keep.size == 0 or self.chi_eV >= 1e9:
            return 0.0
        kT = KB * T
        x = H * nu_c / kT
        with np.errstate(over='ignore'):
            B = (2.0 * H * nu_c ** 3 / C ** 2) / np.expm1(x)
        B = np.where(x > 700.0, 0.0, B)           # code's x>700 skip
        w = 4.0 * math.pi * B * dnu / (H * nu_c)  # per unit sigma, per bin
        S = self.sig * w[None, :]
        Ssuf = np.concatenate([np.cumsum(S[:, ::-1], axis=1)[:, ::-1],
                               np.zeros((S.shape[0], 1))], axis=1)
        Rbf = Ssuf[np.arange(S.shape[0]), np.clip(self.i0, 0, NBIN)]
        lam3 = (H * H / (2.0 * math.pi * ME * KB * T)) ** 1.5
        a = Rbf * lam3 * self.g / (2.0 * self.U_ion(T)) * np.exp(self.chi_l / kT)
        if gated:
            a = np.where(self.spin_skip, 0.0, a)
        return float(np.sum(a))


# --- the continua asked for (both readings of the ion labels) ----------------
SPEC = [(26, 'Fe'), (27, 'Co'), (16, 'S'), (14, 'Si'), (20, 'Ca')]
ROM = ['I', 'II', 'III', 'IV', 'V', 'VI']


def name(Z, k):
    el = dict(SPEC)[Z]
    return f'{el} {ROM[k]}'


# recombining ion -> product ion (recombining charge z = k+1)
WANT = [(26, 2), (26, 3), (27, 2), (27, 3),
        (16, 1), (16, 2), (16, 3), (16, 4),
        (14, 1), (14, 2), (14, 3),
        (20, 1), (20, 2), (20, 3)]

conts = {}
for (Z, zrec) in WANT:
    conts[(Z, zrec)] = Cont(Z, zrec - 1)   # product ion index = zrec-1

os.makedirs(OUT, exist_ok=True)

# ---- self-check against the production banner ------------------------------
# stdout parity42:1310  "[ALPHA-SPINGATE] Fe III: alpha_gated/alpha_full = 0.09
#                        (skipped 764 of 1500 levels, M_core=6 [table])"
c = conts[(26, 3)]          # product = Fe III, recombining = Fe IV
print(f'[selfcheck] Fe III continuum: levels with sigma+bound = {c.keep.size} '
      f'(Fe III total {c.idx.size}), spin-skipped = {int(np.sum(c.spin_skip))}, '
      f'M_core = {c.Mcore}')
for Tc in (5000.0, 10000.0, 15000.0, 20000.0):
    af, ag = c.alpha(Tc, False), c.alpha(Tc, True)
    print(f'[selfcheck]   T={Tc:7.0f}  alpha_full={af:.4e}  alpha_gated={ag:.4e} '
          f' gated/full={ag / af:.4f}')

rows = []
for s, T in enumerate(Te):
    for (Z, zrec) in WANT:
        c = conts[(Z, zrec)]
        aK = 2.6e-13 * zrec * zrec * (T / 1e4) ** -0.75
        aF = c.alpha(T, gated=False)
        aG = c.alpha(T, gated=True)
        aD = alpha_dr(Z, zrec, T)
        rows.append(dict(shell=s, T_e=T, Z=Z, z_recomb=zrec,
                         recomb_ion=name(Z, zrec), product_ion=name(Z, zrec - 1),
                         n_lev_sigma=int(c.keep.size),
                         n_lev_spinskip=int(np.sum(c.spin_skip)),
                         M_core=c.Mcore,
                         alpha_kramers=aK, alpha_milne_full=aF,
                         alpha_milne_gated=aG, alpha_dr=aD,
                         alpha_production=aG + aD,
                         ratio_K_over_full=(aK / aF) if aF > 0 else float('inf'),
                         ratio_K_over_prod=(aK / (aG + aD)) if (aG + aD) > 0
                         else float('inf')))

with open(os.path.join(OUT, 'm1_alpha_ratio.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('wrote', os.path.join(OUT, 'm1_alpha_ratio.csv'), len(rows), 'rows')
