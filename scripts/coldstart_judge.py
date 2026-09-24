#!/usr/bin/env python3
"""DET-COLDSTART judge — docs/PREREG_DET_COLDSTART_2026-09-24.md (rev 7).

Modes (all fail-closed; exit 0 = the requested comparison PASSED):
  referents   : public toy06 19.480 d + geometry + base ion.csv iter3-5 -> coldstart_referents.csv, J rule
  inj-table   : G-INJ(a) on a selftest CSV or lumina_init_ionfrac_applied.csv (NC-INJ)
  g-cold      : G-COLD on two ion.csv files (NC-COLD)
  classify-only : r/d/per-shell class/branch on iter3-5 ion.csv, gates bypassed (NC-HOT)
  b-only      : code-faithful source witness rho/kappa/b_g/window S_bf/B_code (NC-B) + optional LEVELS tag check
  full        : G-ID, G-INJ, G-COLD, G-RUN (+PIN-FAIL), r table, map, branch, source table -> coldstart_result.json
  fixture     : build NC inputs (nc-inj copy dir, nc-hot synthetic ion.csv, nc-b ground x3 levels file)
  selftest-rules : the prereg-preflight branch rules against the registered adversarial fixtures
"""
import argparse, csv, glob, hashlib, json, math, os, re, shutil, struct, sys
import numpy as np

ELEMS = [(14, 'si'), (16, 's'), (20, 'ca'), (26, 'fe'), (27, 'co'), (28, 'ni')]
J_REGISTERED = list(range(10, 16)) + list(range(19, 31))
J_CANDIDATES = list(range(9, 31))
N_PIN = 2
N_ITER = 6
CHANGESET_C = ['src/lumina_plasma.c', 'src/lumina.h', 'src/env_universe.h', 'tests/a2_10_radeq_selftest.c']
EXPORT_DELTA = {'LUMINA_INIT_IONFRAC_DIR', 'LUMINA_INIT_IONFRAC_PIN_ITERS'}
# §5 referent (x_pub, x_hot, gap dex), author values — regenerated and compared at 3 significant figures
REFERENT = {10: (.241, 15.4, 1.80), 11: (.181, 20.1, 2.05), 12: (.186, 5.27, 1.45), 13: (.197, 3.74, 1.28),
            14: (.210, 3.00, 1.15), 15: (.233, 2.50, 1.03), 19: (.349, 18.9, 1.73), 20: (.392, 24.8, 1.80),
            21: (.490, 45.6, 1.97), 22: (.572, 33.6, 1.77), 23: (.696, 42.8, 1.79), 24: (.744, 130, 2.24),
            25: (.880, 369, 2.62), 26: (.959, 1.51e3, 3.20), 27: (1.10, 2.52e3, 3.36), 28: (1.19, 3.74e3, 3.50),
            29: (1.30, 8.75e3, 3.83), 30: (1.43, 1.33e4, 3.97)}
# §4 NC-B referents (sh24, 398939 gen 8)
NCB_BASE = {'rho': (33.66696, 1e-5), 'kappa': (0.9549447, 1e-6), 'b_g': (2.836444e-2, 1e-5), 'window': (22.20462, 1e-4)}
NCB_X3 = {'rho': (33.66696 / 3, 1e-6), 'b_g': (3 * 2.836444e-2, 1e-6), 'window': (19.63796, 1e-4)}

# code constants (lumina.h:26-39, lumina_frequency_grid.h:13-15), contract §5 CODATA 2018
H = 6.62607015e-27; K = 1.380649e-16; C = 2.99792458e10; ME = 9.1093837015e-28; EV = 1.602176634e-12
NFB = 1234; NU_MIN = 5.8412785919616062e13; NU_MAX = 4.0362581455823112e16
DLN = math.log(NU_MAX / NU_MIN) / NFB
SIGMA0_KRAMERS_S2 = 2.00e-18
G_GROUND_S2 = 4


class Fail(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b):
    if b == 0.0:
        return 0.0 if a == 0.0 else math.inf
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------- public fractions (§2, independent)
def read_block(path):
    lines = open(path).read().splitlines()
    starts = [i for i, l in enumerate(lines)
              if l.startswith('#TIME:') and abs(float(l[6:].split()[0]) - 19.48) < 1e-9]
    if len(starts) != 1:
        raise Fail(f'BLOCK_19.480 count={len(starts)} {path}')
    i = starts[0]
    m = re.match(r'#NVEL:\s*(\d+)\s*$', lines[i + 1])
    if not m or not lines[i + 2].startswith('#vel_mid'):
        raise Fail(f'BLOCK_HEADER {path}')
    nv, ncol = int(m.group(1)), len(lines[i + 2].split()) - 1
    rows = []
    for j in range(nv):
        t = lines[i + 3 + j].split()
        if len(t) != ncol + 1 or t[0].startswith('#'):
            raise Fail(f'BLOCK_ROW {path} row={j}')
        rows.append([float(x) for x in t])
    after = i + 3 + nv
    if after < len(lines) and lines[after].strip() and not lines[after].startswith('#TIME:'):
        raise Fail(f'BLOCK_NVEL_MISMATCH {path}')
    a = np.array(rows)
    v, f = a[:, 0], a[:, 1:]
    if not (np.all(np.isfinite(a)) and np.all(np.diff(v) > 0) and np.all(v > 0) and np.all(f > 0)):
        raise Fail(f'BLOCK_VALUES {path}')
    return v, f


def load_public(public_dir):
    return {Z: read_block(os.path.join(public_dir, f'ionfrac_{sym}_toy06_cmfgen.txt')) for Z, sym in ELEMS}


def lumina_fracs(block, v_kms):
    v, f = block
    if not (v[0] <= v_kms <= v[-1]):
        raise Fail(f'V_OUT_OF_RANGE {v_kms}')
    i = min(int(np.searchsorted(v, v_kms, side='right')) - 1, len(v) - 2)
    w = (math.log(v_kms) - math.log(v[i])) / (math.log(v[i + 1]) - math.log(v[i]))
    lf = np.log(f[i]) + w * (np.log(f[i + 1]) - np.log(f[i]))
    fp = np.exp(lf)
    fp = fp / fp.sum()
    # rev 6: K public stages -> K Lumina stages; II <- I+II, k <- p_k, top two <- p_{K-1}
    return np.concatenate([[fp[0] + fp[1]], fp[2:-1], [fp[-1], fp[-1]]])


def x_pub_at(pub, v_kms):
    fs = lumina_fracs(pub[16], v_kms)
    return fs[1] / fs[0]


def read_geometry(path):
    rows = list(csv.DictReader(open(path)))
    vi = np.array([float(r['v_inner']) for r in rows]) / 1e5
    vo = np.array([float(r['v_outer']) for r in rows]) / 1e5
    if not (np.all(vo > vi) and len(rows) > 0):
        raise Fail('GEOMETRY')
    return vi, vo


# ---------------------------------------------------------------- run readers
def ion_path(ion_dir, k):
    hits = sorted(glob.glob(os.path.join(ion_dir, f'physics_*_iter{k:04d}.ion.csv')))
    if len(hits) != 1:
        raise Fail(f'ION_CSV iter={k} hits={len(hits)} dir={ion_dir}')
    return hits[0]


def read_ion(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r['shell_id'])] = {'gen': int(r['population_generation']), 'el': float(r['Z16_element_total_cm3']),
                                   'q1': float(r['Z16_stage_q1_cm3']), 'q2': float(r['Z16_stage_q2_cm3']),
                                   'n1': float(r['Z16_nlte_q1_cm3']), 'n2': float(r['Z16_nlte_q2_cm3'])}
    return out


def read_referents(path):
    out = {}
    for r in csv.DictReader(open(path)):
        # x_hot/gap_dex are '' (undefined) on shells whose base run has no sulfur (x_undefined_reason)
        out[int(r['shell'])] = {k: (float(r[k]) if r[k] != '' else None) for k in ('v_mid', 'x_pub', 'x_hot', 'gap_dex')} | \
                               {'in_J': r['in_J'] == '1'}
    return out


def s_ratio(c, where):
    """S III/S II from one ion.csv row; element-absent shell -> None, any other non-positive/non-finite -> Fail."""
    if c['el'] == 0.0 and c['q1'] == 0.0 and c['q2'] == 0.0:
        return None
    if not (all(math.isfinite(c[q]) for q in ('el', 'q1', 'q2')) and c['el'] > 0 and c['q1'] > 0 and c['q2'] > 0):
        raise Fail(f"NONPOSITIVE_Q {where} el={c['el']!r} q1={c['q1']!r} q2={c['q2']!r}")
    return c['q2'] / c['q1']


# ---------------------------------------------------------------- referents
def make_referents(public_dir, geometry, base_ion_dir, out):
    pub = load_public(public_dir)
    vi, vo = read_geometry(geometry)
    hot = [read_ion(ion_path(base_ion_dir, k)) for k in (3, 4, 5)]
    rows, problems, undefined = [], [], []
    for s in range(len(vi)):
        vm = 0.5 * (vi[s] + vo[s])
        xp = x_pub_at(pub, vm)
        band = abs(math.log10(x_pub_at(pub, vo[s]) / x_pub_at(pub, vi[s])))
        xs = [s_ratio(h[s], f'hot iter{k} sh{s}') for k, h in zip((3, 4, 5), hot)]
        if all(x is None for x in xs):
            # §5 J rule: gap undefined without sulfur -> rule false, shell outside J; x columns left undefined
            undefined.append(s)
            xh = gap = spread = None
            rule, reason = False, 'ELEMENT_ABSENT_Z16'
        elif any(x is None for x in xs):
            raise Fail(f'ELEMENT_ABSENT_PARTIAL sh{s} hot iter3-5 {xs}')
        else:
            xh = math.exp(sum(math.log(x) for x in xs) / 3)
            gap = math.log10(xh / xp)
            spread = math.log10(max(xs) / min(xs))
            rule, reason = gap >= 1.0 and spread <= gap / 3 and band <= gap / 3, ''
        rows.append(dict(shell=s, v_in=vi[s], v_out=vo[s], v_mid=vm, x_pub=xp, x_hot=xh, gap_dex=gap,
                         hot_spread_dex=spread, cold_band_dex=band, rule=int(rule),
                         in_J=int(rule and s in J_CANDIDATES), x_undefined_reason=reason))
    J = [r['shell'] for r in rows if r['in_J']]
    if J != J_REGISTERED:
        problems.append(f'J_RULE {J} != registered {J_REGISTERED}')
    for r in rows:
        ref = REFERENT.get(r['shell'])
        if ref and r['x_hot'] is None:
            problems.append(f"REFERENT sh{r['shell']} x undefined ({r['x_undefined_reason']})")
        elif ref and (rel(r['x_pub'], ref[0]) > 5.1e-3 or rel(r['x_hot'], ref[1]) > 5.1e-3 or abs(r['gap_dex'] - ref[2]) > 5.1e-3):
            problems.append(f"REFERENT sh{r['shell']} x_pub={r['x_pub']:.4g}/{ref[0]} x_hot={r['x_hot']:.4g}/{ref[1]} gap={r['gap_dex']:.3f}/{ref[2]}")
    with open(out, 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if v is None else repr(float(v)) if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})
    return {'J': J, 'x_undefined_shells': undefined, 'problems': problems, 'pass': not problems, 'sha256': sha256(out)}


# ---------------------------------------------------------------- G-INJ
def g_inj_a(table, public_dir, geometry):
    pub = load_public(public_dir)
    vi, vo = read_geometry(geometry)
    rows = list(csv.DictReader(open(table)))
    if not rows:
        raise Fail(f'EMPTY_TABLE {table}')
    has_n = 'n_ion' in rows[0]
    if has_n and 'n_ion_pre' not in rows[0]:
        raise Fail(f'APPLIED_SCHEMA n_ion_pre missing {table}')
    seen, problems, worst = set(), [], {'f': 0.0, 'ratio': 0.0, 'top_sum': 0.0, 'top_cross': 0.0, 'v_mid': 0.0}
    by_cell = {}
    for r in rows:
        s, Z, st = int(r['shell']), int(r['Z']), int(r['stage'])
        if (s, Z, st) in seen:
            problems.append(f'DUPLICATE {s},{Z},{st}')
        seen.add((s, Z, st))
        vm = 0.5 * (vi[s] + vo[s])
        worst['v_mid'] = max(worst['v_mid'], rel(float(r['v_mid']), vm))
        fr = lumina_fracs(pub[Z], vm)
        if not 1 <= st <= len(fr):
            problems.append(f'STAGE_RANGE sh{s} Z{Z} st{st} K={len(fr)}')
            continue
        f_ref, f_app = fr[st - 1], float(r['f_pub'])
        e = rel(f_app, f_ref)
        worst['f'] = max(worst['f'], e)
        if e > 1e-6:
            problems.append(f'F sh{s} Z{Z} st{st} applied={f_app:.10g} indep={f_ref:.10g} rel={e:.3g}')
        if has_n:
            by_cell.setdefault((s, Z), {})[st] = (float(r['n_ion']), float(r['n_element']), f_app,
                                                  float(r['n_ion_pre']))
    for s in range(len(vi)):
        for Z, _ in ELEMS:
            for st in range(1, len(lumina_fracs(pub[Z], 0.5 * (vi[s] + vo[s]))) + 1):
                if (s, Z, st) not in seen:
                    problems.append(f'MISSING {s},{Z},{st}')
    # rev 6 G-INJ(a): k <= K-2: n = f*nel; top two: sum = f*nel and cross product
    # n(K)*pre(K-1) = n(K-1)*pre(K) (no division); nel == 0: every n exactly 0.
    for (s, Z), cells in by_cell.items():
        ks = max(cells)
        if sorted(cells) != list(range(1, ks + 1)) or ks < 3:
            problems.append(f'CELL_STAGES sh{s} Z{Z} {sorted(cells)}')
            continue
        nel = cells[1][1]
        if any(c[1] != nel for c in cells.values()):
            problems.append(f'N_ELEMENT_INCONSISTENT sh{s} Z{Z}')
        if nel > 0:
            for st in range(1, ks - 1):
                n, _, f, _ = cells[st]
                e = rel(n, f * nel)
                worst['ratio'] = max(worst['ratio'], e)
                if e > 1e-9:
                    problems.append(f'N_RATIO sh{s} Z{Z} st{st} n={n:.12g} f*nel={f * nel:.12g}')
            (n1, _, f1, p1), (n2, _, f2, p2) = cells[ks - 1], cells[ks]
            e = rel(n1 + n2, f1 * nel)
            worst['top_sum'] = max(worst['top_sum'], e)
            if e > 1e-9:
                problems.append(f'TOP_SUM sh{s} Z{Z} n={n1 + n2:.12g} f*nel={f1 * nel:.12g}')
            if p1 == 0.0 and p2 == 0.0:
                problems.append(f'TOP_SPLIT_UNDEFINED_PRESENT sh{s} Z{Z}')
            else:
                a_, b_ = n2 * p1, n1 * p2
                m = max(abs(a_), abs(b_))
                e = 0.0 if m == 0.0 else abs(a_ - b_) / m
                worst['top_cross'] = max(worst['top_cross'], e)
                if e > 1e-9:
                    problems.append(f'TOP_SPLIT sh{s} Z{Z} n(K)*pre(K-1)={a_:.12g} n(K-1)*pre(K)={b_:.12g}')
        elif nel == 0.0:
            if any(c[0] != 0.0 for c in cells.values()):
                problems.append(f'ZERO_ELEMENT_NONZERO_ION sh{s} Z{Z}')
        else:
            problems.append(f'N_ELEMENT_INVALID sh{s} Z{Z}')
    if worst['v_mid'] > 1e-9:
        problems.append(f"V_MID rel={worst['v_mid']:.3g}")
    return {'pass': not problems, 'rows': len(rows), 'has_n_columns': has_n, 'worst_rel': worst,
            'problems': problems[:40], 'n_problems': len(problems)}


def g_inj_b(djdn, refs):
    last = {}
    for r in csv.DictReader(open(djdn)):
        if int(r['epoch']) == 2 and int(r['Z']) == 16 and int(r['stage']) == 1:
            last[int(r['shell'])] = float(r['x_lag'])
    problems, table = [], {}
    for s in J_REGISTERED:
        if s not in last:
            problems.append(f'MISSING sh{s}')
            continue
        e = rel(last[s], refs[s]['x_pub'])
        table[s] = {'x_lag': last[s], 'x_pub': refs[s]['x_pub'], 'rel': e}
        if e > 1e-6:
            problems.append(f'X_LAG sh{s} {last[s]:.9g} vs {refs[s]["x_pub"]:.9g}')
    return {'pass': not problems, 'table': table, 'problems': problems}


# ---------------------------------------------------------------- G-COLD
def g_cold(ion0, ion1, refs):
    problems, table = [], {}
    for k, path, gen in ((0, ion0, 3), (1, ion1, 4)):
        ion = read_ion(path)
        for s in J_REGISTERED:
            c, xp = ion[s], refs[s]['x_pub']
            if not (c['q1'] > 0 and c['q2'] > 0 and c['n1'] > 0 and c['n2'] > 0):
                problems.append(f"iter{k} sh{s} NONPOSITIVE_Q el={c['el']!r} q=({c['q1']!r},{c['q2']!r}) nlte=({c['n1']!r},{c['n2']!r})")
                continue
            xs, xn = c['q2'] / c['q1'], c['n2'] / c['n1']
            e = (rel(xs, xp), rel(xn, xp), rel(c['n1'], c['q1']))
            table[f'{k}:{s}'] = {'x_stage': xs, 'x_nlte': xn, 'x_pub': xp, 'rel': e, 'gen': c['gen']}
            if e[0] > 1e-6 or e[1] > 1e-6 or e[2] > 1e-9:
                problems.append(f'iter{k} sh{s} stage={xs:.6g} nlte={xn:.6g} x_pub={xp:.6g} nlte_q1/stage_q1-1={e[2]:.3g}')
            if c['gen'] != gen:
                problems.append(f'iter{k} sh{s} population_generation={c["gen"]} expected {gen}')
    return {'pass': not problems, 'problems': problems, 'table': table}


# ---------------------------------------------------------------- r table, classification, branch
def r_table(ion_dir, refs, ks=(2, 3, 4, 5), paths=None):
    out = {}
    for k in ks:
        ion = read_ion(paths[k] if paths else ion_path(ion_dir, k))
        for s in J_REGISTERED:
            c = ion[s]
            if not (math.isfinite(c['q1']) and math.isfinite(c['q2']) and c['q1'] > 0 and c['q2'] > 0):
                raise Fail(f'NONPOSITIVE_Q iter{k} sh{s}')
            x, xp, xh = c['q2'] / c['q1'], refs[s]['x_pub'], refs[s]['x_hot']
            if xh is None:
                raise Fail(f'REFERENT_X_UNDEFINED sh{s}')
            out.setdefault(s, {})[k] = {'x': x, 'r': math.log(x / xp) / math.log(xh / xp), 'd': abs(math.log10(x / xp))}
    return out


def classify(rt):
    cls = {}
    for s, row in rt.items():
        rs, ds = [row[k]['r'] for k in (3, 4, 5)], [row[k]['d'] for k in (3, 4, 5)]
        c = 'COLD' if all(r <= 1 / 3 for r in rs) and all(d <= 0.5 for d in ds) else \
            'HOT' if all(r >= 2 / 3 for r in rs) else 'MID'
        dr = row[5]['r'] - row[3]['r']
        cls[s] = {'class': c, 'direction': ('UP' if dr > 0 else 'DOWN') if abs(dr) > 0.1 else 'FLAT',
                  'group': 'inner' if s <= 15 else 'outer'}
    return cls


RULES = [('PIN-FAIL', lambda m: m['count'] > 0 and m['pin_fail'] > 0 and m['gate_fail'] == 0),
         ('X', lambda m: m['count'] == 0 or m['gate_fail'] > 0),
         ('COLD-STAY', lambda m: m['count'] > 0 and m['pin_fail'] == 0 and m['gate_fail'] == 0 and m['cold'] == m['count']),
         ('HOT-RETURN', lambda m: m['count'] > 0 and m['pin_fail'] == 0 and m['gate_fail'] == 0 and m['hot'] == m['count']),
         ('DRIFT', lambda m: m['count'] > 0 and m['pin_fail'] == 0 and m['gate_fail'] == 0 and m['cold'] < m['count'] and m['hot'] < m['count'])]


def branch_of(values):
    m = {'count': len(values), 'gate_fail': sum(1 for x in values if x < 0.5),
         'cold': sum(1 for x in values if 0.5 <= x < 1.5), 'hot': sum(1 for x in values if 1.5 <= x < 2.5),
         'pin_fail': sum(1 for x in values if 3.5 <= x < 4.5)}
    hits = [n for n, p in RULES if p(m)]
    if len(hits) != 1:
        raise Fail(f'RULES_NOT_EXCLUSIVE {hits} {m}')
    return hits[0]


def branch(gate_fail, pin_fail, cls):
    code = {'COLD': 1.0, 'HOT': 2.0, 'MID': 3.0}
    values = [] if pin_fail else [code[c['class']] for c in cls.values()]
    if pin_fail:
        values.append(4.0)
    if gate_fail:
        values.append(0.0)
    return branch_of(values)


def selftest_rules():
    fixtures = {'empty': ([], 'X'), 'gate-fail': ([0.0, 1.0], 'X'), 'all-cold': ([1.0], 'COLD-STAY'),
                'all-hot': ([2.0], 'HOT-RETURN'), 'split': ([1.0, 2.0], 'DRIFT'), 'band': ([3.0], 'DRIFT'),
                'pin-fail': ([4.0], 'PIN-FAIL'), 'pin-fail-id-fail': ([4.0, 0.0], 'X')}
    res = {n: branch_of(v) for n, (v, _) in fixtures.items()}
    bad = {n: (res[n], want) for n, (_, want) in fixtures.items() if res[n] != want}
    return {'pass': not bad, 'results': res, 'mismatch': bad}


# ---------------------------------------------------------------- source witness (§5, code-faithful)
class Deck:
    def __init__(self, model_dir):
        self.chi = {(int(r['atomic_number']), int(r['ion_number'])): float(r['ionization_energy_eV'])
                    for r in csv.DictReader(open(os.path.join(model_dir, 'ionization_energies.csv')))}
        lev = list(csv.DictReader(open(os.path.join(model_dir, 'levels.csv'))))
        self.n_levels = {ion: sum(1 for r in lev if int(r['atomic_number']) == 16 and int(r['ion_number']) == ion) for ion in (1, 2)}
        self.E3 = np.array([float(r['energy_eV']) for r in lev if int(r['atomic_number']) == 16 and int(r['ion_number']) == 2])
        self.g3 = np.array([float(r['g']) for r in lev if int(r['atomic_number']) == 16 and int(r['ion_number']) == 2])
        path = os.path.join(model_dir, 'cmfgen_sigma_bf.bin')
        with open(path, 'rb') as f:
            magic, ver, nlev, nfr = struct.unpack('<IIii', f.read(16))
            numin, numax = struct.unpack('<dd', f.read(16))
            if magic != 0x434D4644 or ver != 1 or nfr != NFB or numin != NU_MIN or numax != NU_MAX:
                raise Fail(f'SIGMA_HEADER magic={magic:#x} v={ver} nfr={nfr} nu=({numin!r},{numax!r})')
            self.has = np.frombuffer(f.read(nlev), dtype=np.int8)
        off = 32 + nlev + (8 - nlev % 8) % 8
        self.sig = np.memmap(path, dtype='<f8', mode='r', offset=off, shape=(nlev, nfr))
        self.nu_c = NU_MIN * np.exp((np.arange(NFB) + 0.5) * DLN)
        self.nu_hi = np.exp(math.log(NU_MIN) + (np.arange(NFB) + 1) * DLN)
        lam = C * 1e8 / self.nu_c
        self.win = np.nonzero((lam >= 500) & (lam < 611))[0]
        self.win_set = set(self.win.tolist())
        self._rows = {}

    def sigma_row(self, gi):
        if gi not in self._rows:
            self._rows[gi] = np.array(self.sig[gi])
        return self._rows[gi]

    def U3(self, kT):
        return float(np.sum(self.g3 * np.exp(-(self.E3 - self.E3.min()) * EV / kT)))


def read_levels(path, shells):
    """-> {shell: {'T','ne'(or None),'n3','gen', 'lev': [(idx,gidx,E,g,n)]} , ion-2 levels count}"""
    out = {s: {'lev': [], 'n3': None, 'ne': None, 'T': None, 'gen': None, 'n3_levels': 0} for s in shells}
    for r in csv.DictReader(open(path)):
        s = int(r['shell'])
        if s not in out or int(r['Z']) != 16:
            continue
        d, ion = out[s], int(r['ion'])
        if ion == 1:
            d['lev'].append((int(r['level_idx']), int(r['global_idx']), float(r['E_eV']), int(r['g']), float(r['n_pop'])))
            d['T'] = float(r['T_e'])
            if 'n_e' in r:
                d['ne'] = float(r['n_e'])
            d['gen'] = int(r['population_generation'])
        elif ion == 2:
            d['n3'] = float(r['n_ion_total'])
            d['n3_levels'] += 1
    return out


def source_witness(deck, cell, ground_x3=False):
    T, ne, n3 = cell['T'], cell['ne'], cell['n3']
    if not (n3 is not None and n3 > 0 and ne > 0 and T > 0):
        raise Fail(f'SOURCE_UNDEFINED n_III={n3!r} n_e={ne!r} T_e={T!r} (element absent or bad state)')
    kT = K * T
    U3 = deck.U3(kT)
    lam3 = (H * H / (2 * math.pi * ME * kT)) ** 1.5
    saha_m = 2.0 * U3 / (n3 * ne * lam3)
    chi = deck.chi[(16, 1)] * EV
    levs = sorted(cell['lev'])
    if [l[0] for l in levs] != list(range(len(levs))):
        raise Fail('LEVEL_INDEX')
    B = lambda nu: 0.0 if H * nu / kT > 500 else 2 * H * nu ** 3 / C ** 2 / math.expm1(H * nu / kT)
    chi_w = np.zeros(NFB); eta_w = np.zeros(NFB); ground = None
    for li, gi, E, g, n in levs:
        if li == 0 and ground_x3:
            n *= 3.0
        if n < 1e-30:
            continue
        nu_e = (chi - E * EV) / H
        if nu_e <= NU_MIN:
            continue
        b0 = int(math.log(nu_e / NU_MIN) / DLN)
        tab = bool(deck.has[gi])
        Cinv = saha_m * n / g
        bins = sorted(set(deck.win.tolist()) | ({b0} if li == 0 else set()))
        for b in bins:
            if b < b0:
                continue
            nc = deck.nu_c[b]
            if tab:
                s = float(deck.sigma_row(gi)[b])
                if s <= 0.0:
                    continue
                nm = 0.5 * (nu_e + deck.nu_hi[b]) if (nc < nu_e < deck.nu_hi[b]) else nc
            else:
                if nc < nu_e:
                    continue
                s, nm = SIGMA0_KRAMERS_S2 * (nu_e / nc) ** 3, nc
            x = H * (nm - nu_e) / kT
            c_ = n * s
            e_ = c_ * (0.0 if x > 600 else 2 * H * nm ** 3 / C ** 2 * math.exp(-x) / Cinv)
            if li == 0 and b == b0:
                ground = dict(bin=b0, nu_c=nc, nu_m=nm, chi=c_, eta=e_, B_c=B(nc), n_g=n, g=g)
            if b in deck.win_set:
                chi_w[b] += c_
                eta_w[b] += e_
    if ground is None:
        raise Fail('GROUND_EDGE_BIN')
    rho = ground['eta'] / (ground['chi'] * ground['B_c'])
    kappa = (ground['nu_m'] / ground['nu_c']) ** 3 * math.expm1(H * ground['nu_c'] / kT) * math.exp(-H * ground['nu_m'] / kT)
    nstar = ne * n3 * (G_GROUND_S2 / (2 * U3)) * lam3 * math.exp(chi / kT)
    Bw = np.array([B(nu) for nu in deck.nu_c[deck.win]])
    window = float(eta_w[deck.win].sum() / (chi_w[deck.win] * Bw).sum())
    return {'rho': rho, 'kappa': kappa, 'b_g': ground['n_g'] / nstar, 'window': window, 'edge_bin': ground['bin'],
            'T_e': T, 'n_e': ne, 'n_III_stage': n3, 'U_III': U3}


def plasma_state_ne(path):
    return {int(r['shell_id']): float(r['n_e']) for r in csv.DictReader(open(path))}


def b_only(levels, model_dir, shells, plasma_state=None, ground_x3=False):
    deck = Deck(model_dir)
    cells = read_levels(levels, shells)
    ne_ps = plasma_state_ne(plasma_state) if plasma_state else {}
    out = {}
    for s in shells:
        c = cells[s]
        if c['ne'] is None:
            if s not in ne_ps:
                raise Fail(f'N_E sh{s}: levels file has no n_e and no plasma_state row')
            c['ne'] = ne_ps[s]
        if len(c['lev']) != deck.n_levels[1] or c['n3_levels'] != deck.n_levels[2]:
            raise Fail(f'LEVEL_COUNT sh{s} {len(c["lev"])}+{c["n3_levels"]} deck {deck.n_levels}')
        out[s] = source_witness(deck, c, ground_x3)
    return out, deck


def compare_ncb(res, ref):
    return {k: {'value': res[k], 'ref': v, 'rel': rel(res[k], v), 'tol': t, 'pass': rel(res[k], v) <= t}
            for k, (v, t) in ref.items()}


# ---------------------------------------------------------------- G-RUN
TAG_ARM = re.compile(r'^\[COLDSTART\] current_iter=(-?\d+) armed=([01]) n_pin=(\d+)\s*$')
TAG_LEV = re.compile(r'^\[COLDSTART\]\[LEVELS\] gen=(\d+) rows=(\d+) rc=(-?\d+)\s*$')
FATAL_SEED = re.compile(r'^\[A2-INIT\]\[FATAL\] .*event=SEED_MATERIAL_PREDICTOR')
FATAL_R7 = re.compile(r'^\[R7\]\[FATAL\] lane=DET iter=(\d+) ')
BLOCKED = re.compile(r'^\[(A2-INIT\]\[SEED|A2-10\]\[TRIAL)-BUNDLE-BLOCKED\].*\bsolve_stage=(\S+)')


def read_footer(path):
    out = {}
    for line in open(path):
        if '=' in line:
            k, v = line.rstrip('\n').split('=', 1)
            out[k.strip()] = v.strip()
    return out


def levels_file_check(path, gen, n_expect):
    rows, bad = 0, []
    for r in csv.DictReader(open(path)):
        rows += 1
        if int(r['population_generation']) != gen:
            bad.append('GEN_COLUMN')
        n, ne, T = float(r['n_pop']), float(r['n_e']), float(r['T_e'])
        if not (n >= 0 and math.isfinite(n) and math.isfinite(ne) and math.isfinite(T)):
            bad.append('VALUE')
        if bad:
            break
    if rows != n_expect:
        bad.append(f'ROWS {rows} != {n_expect}')
    return rows, bad


def g_run(run_dir, work_dir, ion_dir, stderr_path, deck):
    problems = []
    footer = read_footer(os.path.join(run_dir, 'RUN_FOOTER.txt')) if os.path.isfile(os.path.join(run_dir, 'RUN_FOOTER.txt')) else {}
    lines = open(stderr_path, errors='replace').read().splitlines()
    arms = [(i, TAG_ARM.match(l)) for i, l in enumerate(lines) if TAG_ARM.match(l)]
    lev_tags = [TAG_LEV.match(l) for l in lines if TAG_LEV.match(l)]
    seed_fatal = [i for i, l in enumerate(lines) if FATAL_SEED.match(l)]
    r7_fatal = [(i, int(FATAL_R7.match(l).group(1))) for i, l in enumerate(lines) if FATAL_R7.match(l)]
    success = footer.get('terminal_state') == 'SUCCESS'
    if success:
        if footer.get('model_rc') != '0' or footer.get('outer_iterations') != str(N_ITER) or seed_fatal or r7_fatal:
            problems.append(f"FOOTER model_rc={footer.get('model_rc')} outer_iterations={footer.get('outer_iterations')}")
        G, k, path = set(range(2, 9)), None, 'SUCCESS'
    elif seed_fatal and not r7_fatal:
        G, k, path = set(), 0, 'SEED_FATAL'
    elif r7_fatal and not seed_fatal and len(r7_fatal) == 1:
        k = r7_fatal[0][1]
        G, path = {2} | {i + 3 for i in range(k)}, 'R7_FATAL'
    else:
        return {'code': 'FAIL', 'problems': [f'TERMINAL_PATH footer={footer.get("terminal_state")} seed={len(seed_fatal)} r7={len(r7_fatal)}'], 'G': None}
    n_expect = (deck.n_levels[1] + deck.n_levels[2]) * 50
    files = {int(m.group(1)): p for p in glob.glob(os.path.join(work_dir, 'coldstart_levels_gen*.csv'))
             for m in [re.search(r'gen(\d+)\.csv$', p)] if m}
    if set(files) != G:
        problems.append(f'LEVEL_FILES {sorted(files)} != G {sorted(G)}')
    tag_gens = [int(m.group(1)) for m in lev_tags]
    if sorted(tag_gens) != sorted(G):
        problems.append(f'LEVEL_TAGS {sorted(tag_gens)} != G {sorted(G)}')
    for m in lev_tags:
        g, rows, rc = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if rc != 0:
            problems.append(f'LEVEL_TAG_RC gen={g} rc={rc}')
        if g in files:
            actual, bad = levels_file_check(files[g], g, n_expect)
            if rows != actual or bad:
                problems.append(f'LEVEL_FILE gen={g} tag_rows={rows} file_rows={actual} {bad}')
    ions = sorted(int(m.group(1)) for p in glob.glob(os.path.join(ion_dir, 'physics_*_iter*.ion.csv'))
                  for m in [re.search(r'iter(\d{4})\.ion\.csv$', p)] if m)
    want_ions = list(range(N_ITER)) if success else list(range(k))
    if ions != want_ions:
        problems.append(f'ION_CSV {ions} != {want_ions}')
    if success:
        for i in range(N_ITER):
            ion = read_ion(ion_path(ion_dir, i))
            for s in J_REGISTERED:
                c = ion[s]
                if c['gen'] != i + 3 or not all(math.isfinite(c[q]) and c[q] > 0 for q in ('q1', 'q2')):
                    problems.append(f'ION iter{i} sh{s} gen={c["gen"]}')
        # recorded only (not a contract criterion, audit §3): armed=1 <=> current_iter < n_pin, n_pin = N_PIN
        arm_bad = [f'line{i + 1}' for i, m in arms
                   if int(m.group(3)) != N_PIN or (m.group(2) == '1') != (int(m.group(1)) < int(m.group(3)))]
        return {'code': 'PASS' if not problems else 'FAIL', 'path': path, 'G': sorted(G), 'problems': problems, 'footer': footer,
                'arm_tags': {'n': len(arms), 'n_armed': sum(1 for _, m in arms if m.group(2) == '1'), 'inconsistent': arm_bad[:20]}}
    # MODEL_FAILED: PIN-FAIL identification (§4 (1)(2)(3)) — only if the G equality holds
    pin = []
    fatal_line = seed_fatal[0] if path == 'SEED_FATAL' else r7_fatal[0][0]
    if not arms:
        pin.append('NO_COLDSTART_TAG')
    else:
        i_last, m = arms[-1]
        cur, armed, npin = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (armed == 1 and cur < npin and cur == k and k < npin):
            pin.append(f'ARM_TAG current_iter={cur} armed={armed} n_pin={npin} k={k}')
        stages = [BLOCKED.match(l).group(2) for l in lines[i_last:fatal_line] if BLOCKED.match(l)]
        if not stages or stages[-1] not in ('PAIR_SOLVE', 'CE_NONCONVERGED'):
            pin.append(f'SOLVE_STAGE {stages[-1:] or None}')
    code = 'PIN-FAIL' if not problems and not pin else 'FAIL'
    return {'code': code, 'path': path, 'k': k, 'G': sorted(G), 'problems': problems + pin, 'footer': footer}


# ---------------------------------------------------------------- G-ID
def read_exports(path):
    out = {}
    for line in open(path):
        m = re.match(r'^(?:declare -x |export )?([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def source_members(path):
    """source_files {path: sha256} of a CMFGEN_TWIN_RUN_BUILD_V1 identity; the top-level binary is not a source."""
    d = json.load(open(path))
    if d.get('schema') != 'CMFGEN_TWIN_RUN_BUILD_V1' or not isinstance(d.get('source_files'), list) or not d['source_files']:
        raise Fail(f"BUILD_IDENTITY_SCHEMA {path} schema={d.get('schema')!r}")
    out = {}
    for e in d['source_files']:
        if e['path'] in out:
            raise Fail(f"BUILD_IDENTITY_DUPLICATE {e['path']}")
        out[e['path']] = e['sha256']
    return out


def read_manifest(path):
    out = {}
    for line in open(path):
        if line.strip():
            h, name = line.rstrip('\n').split(None, 1)
            out[name.lstrip('*')] = h
    return out


def sealed_inputs(a, problems, info):
    """§4 G-ID seals deck/sigma/topion/fixed_te: base (398939) manifests, re-hashed on the run's staged files."""
    for man, sub in (('deck.sha256', 'input/model'), ('topion.sha256', 'input/global_atomic')):
        try:
            mb, mr = read_manifest(os.path.join(a.base, 'input', man)), read_manifest(os.path.join(a.run, 'input', man))
        except OSError as e:
            problems.append(f'SEALED {man} {e}')
            continue
        if not mb or mb != mr:
            problems.append(f'SEALED {man} manifest run != base ({len(mr)} vs {len(mb)} entries)')
        bad = []
        for name, h in sorted(mb.items()):
            try:
                if sha256(os.path.join(a.run, sub, name)) != h:
                    bad.append(name)
            except OSError:
                bad.append(f'{name} (absent)')
        if bad:
            problems.append(f'SEALED {man} files {bad}')
        info[man] = len(mb)
    try:
        want = open(os.path.join(a.base, 'input/sigma.sha256')).read().split()[0]
        info['cmfgen_sigma_bf.bin'] = sha256(os.path.join(a.run, 'input/model/cmfgen_sigma_bf.bin'))
        if info['cmfgen_sigma_bf.bin'] != want:
            problems.append('SEALED sigma')
    except (OSError, IndexError) as e:
        problems.append(f'SEALED sigma {e}')
    try:
        hb, hr = sha256(os.path.join(a.base, 'input/fixed_te_profile.txt')), sha256(os.path.join(a.run, 'input/fixed_te_profile.txt'))
        info['fixed_te_profile.txt'] = hr
        if hb != hr:
            problems.append('SEALED fixed_te_profile.txt')
    except OSError as e:
        problems.append(f'SEALED fixed_te_profile.txt {e}')


def g_id(a):
    problems, info = [], {}
    sealed_inputs(a, problems, info)
    for rel_path in a.sealed:
        hb, hr = sha256(os.path.join(a.base, rel_path)), sha256(os.path.join(a.run, rel_path))
        info[rel_path] = hr
        if hb != hr:
            problems.append(f'SEALED {rel_path}')
    eb, er = read_exports(os.path.join(a.base, a.exports)), read_exports(os.path.join(a.run, a.exports))
    norm = lambda v: v.replace(os.path.abspath(a.base), '<ROOT>').replace(os.path.abspath(a.run), '<ROOT>')
    added, removed = set(er) - set(eb), set(eb) - set(er)
    changed = {k for k in set(eb) & set(er) if norm(eb[k]) != norm(er[k])}
    if added != EXPORT_DELTA or removed or changed:
        problems.append(f'EXPORTS added={sorted(added)} removed={sorted(removed)} changed={sorted(changed)}')
    if er.get('LUMINA_INIT_IONFRAC_PIN_ITERS') != str(N_PIN):
        problems.append(f"PIN_ITERS={er.get('LUMINA_INIT_IONFRAC_PIN_ITERS')}")
    for key, want in (('LUMINA_PURE_CMFGEN_ITER', str(N_ITER)), ('LUMINA_XSHELL_RESPONSE', '1')):
        if er.get(key) != want:
            problems.append(f'{key}={er.get(key)} want {want}')
    for key in ('LUMINA_NLTE_ION_LOCK', 'LUMINA_NLTE_LOCK_START_ITER', 'LUMINA_NLTE_PER_ION_RESCALE'):
        if er.get(key, '0') != eb.get(key, '0') or er.get(key, '0') not in ('0', ''):
            problems.append(f'{key}={er.get(key)}')
    pdir = er.get('LUMINA_INIT_IONFRAC_DIR', '')
    for _, sym in ELEMS:
        name = f'ionfrac_{sym}_toy06_cmfgen.txt'
        try:
            if sha256(os.path.join(pdir, name)) != sha256(os.path.join(a.public_dir, name)):
                problems.append(f'PUBLIC_COPY {name}')
        except OSError as e:
            problems.append(f'PUBLIC_COPY {name} {e}')
    rbi = a.run_build_identity or os.path.join(a.run, 'input/run_build_identity.json')
    bbi = a.base_build_identity or os.path.join(a.base, 'input/run_build_identity.json')
    try:
        rb, bb = source_members(rbi), source_members(bbi)
        diff = sorted(p for p in set(rb) | set(bb) if rb.get(p) != bb.get(p))
        allowed = {p for p in diff if p in CHANGESET_C}
        if set(diff) - allowed:
            problems.append(f'SOURCE_DIFF_OUTSIDE_CHANGESET {sorted(set(diff) - allowed)}')
        for p in sorted(allowed):
            if rb.get(p) != sha256(os.path.join(a.frozen_tree, p)):
                problems.append(f'SOURCE_NOT_TREE {p}')
        info['source_diff'] = diff
        info['build_identity'] = {'run': rbi, 'base': bbi, 'n_source_files': len(rb)}
    except (Fail, OSError, KeyError, ValueError, TypeError) as e:
        problems.append(f'BUILD_IDENTITY {type(e).__name__}: {e}')
    footer = read_footer(os.path.join(a.run, 'RUN_FOOTER.txt')) if os.path.isfile(os.path.join(a.run, 'RUN_FOOTER.txt')) else {}
    info['binary_sha256'] = footer.get('binary_sha256')
    binp = os.path.join(a.run, 'input/lumina_cuda')
    if os.path.isfile(binp):
        info['binary_sha256_file'] = sha256(binp)
        if footer.get('binary_sha256') and footer['binary_sha256'] != info['binary_sha256_file']:
            problems.append('BINARY_SHA_FOOTER_MISMATCH')
    if not info['binary_sha256']:
        problems.append('BINARY_SHA_NOT_RECORDED')
    if any('h200' in v.lower() for v in footer.values()):
        problems.append('H200')
    return {'pass': not problems, 'problems': problems, 'info': info}


# ---------------------------------------------------------------- fixtures
def fixture(a):
    if a.kind == 'nc-inj':
        os.makedirs(a.out, exist_ok=False)
        for _, sym in ELEMS:
            shutil.copy2(os.path.join(a.public_dir, f'ionfrac_{sym}_toy06_cmfgen.txt'), a.out)
        path = os.path.join(a.out, 'ionfrac_s_toy06_cmfgen.txt')
        lines = open(path).read().split('\n')
        i = next(i for i, l in enumerate(lines) if l.startswith('#TIME:') and abs(float(l[6:].split()[0]) - 19.48) < 1e-9)
        nv = int(lines[i + 1].split(':')[1])
        for j in range(i + 3, i + 3 + nv):
            t = lines[j].split()
            t[2] = f'{2.0 * float(t[2]):.4E}'
            lines[j] = '   ' + '   '.join(t)
        open(path, 'w').write('\n'.join(lines))
    elif a.kind == 'nc-hot':
        refs = read_referents(a.referents)
        os.makedirs(a.out, exist_ok=False)
        for k in (3, 4, 5):
            src = ion_path(a.ion_dir, k)
            rows = list(csv.reader(open(src)))
            hdr = rows[0]
            iq1, iq2 = hdr.index('Z16_stage_q1_cm3'), hdr.index('Z16_stage_q2_cm3')
            in1, in2 = hdr.index('Z16_nlte_q1_cm3'), hdr.index('Z16_nlte_q2_cm3')
            for r in rows[1:]:
                s = int(r[hdr.index('shell_id')])
                if s in refs:
                    q1 = float(r[iq1])
                    r[iq2] = repr(q1 * refs[s]['x_pub'])
                    r[in1], r[in2] = repr(q1), repr(q1 * refs[s]['x_pub'])
            with open(os.path.join(a.out, os.path.basename(src)), 'x', newline='') as f:
                csv.writer(f, lineterminator='\n').writerows(rows)
    elif a.kind in ('applied-ok', 'applied-bad-split'):
        # rev 6 G-INJ(a) formula check without a run: selftest f_pub table -> applied.csv with
        # n_element 1e7, n_pre top pair (3, 1) (sh0 Z26: both 0 with n_element 0); bad-split swaps the pair.
        rows = list(csv.DictReader(open(a.table)))
        by = {}
        for r in rows:
            by.setdefault((int(r['shell']), int(r['Z'])), []).append(r)
        with open(a.out, 'x', newline='') as f:
            f.write('shell,Z,stage,v_mid,f_pub,n_ion_pre,n_ion,n_element\n')
            for (s, Z), cell in by.items():
                ks = len(cell)
                nel = 0.0 if (s, Z) == (0, 26) else 1e7
                pre = [1.0] * (ks - 2) + [3.0, 1.0] if nel > 0 else [0.0] * ks
                top = float(cell[-1]['f_pub']) * nel
                w = (3.0, 1.0) if a.kind == 'applied-ok' else (1.0, 3.0)
                for i, r in enumerate(cell):
                    n = float(r['f_pub']) * nel if i < ks - 2 else top * w[i - ks + 2] / 4.0
                    f.write(f"{s},{Z},{r['stage']},{r['v_mid']},{r['f_pub']},{pre[i]!r},{n!r},{nel!r}\n")
    elif a.kind == 'nc-b-x3':
        with open(a.levels) as fi, open(a.out, 'x') as fo:
            hdr = fi.readline()
            fo.write(hdr)
            cols = hdr.strip().split(',')
            iz, ii, isl, il, ip = (cols.index(c) for c in ('Z', 'ion', 'shell', 'level_idx', 'n_pop'))
            for line in fi:
                t = line.rstrip('\n').split(',')
                if t[iz] == '16' and t[ii] == '1' and t[isl] == str(a.shell) and t[il] == '0':
                    t[ip] = repr(3.0 * float(t[ip]))
                    line = ','.join(t) + '\n'
                fo.write(line)
    return {'fixture': a.kind, 'out': a.out}


# ---------------------------------------------------------------- full
def full(a):
    res = {'contract': 'docs/PREREG_DET_COLDSTART_2026-09-24.md', 'rev': 7, 'inputs': {}}
    refs = read_referents(a.referents)
    res['inputs']['referents_sha256'] = sha256(a.referents)
    deck = Deck(os.path.join(a.run, 'input/model'))
    gates = {}
    gates['G-ID'] = g_id(a)
    gates['G-RUN'] = g_run(a.run, a.work, a.ion_dir, a.stderr, deck)
    pin_fail = gates['G-RUN']['code'] == 'PIN-FAIL'
    applied = os.path.join(a.work, 'lumina_init_ionfrac_applied.csv')
    djdn = os.path.join(a.work, 'lumina_pop_djdn.csv')
    for name, fn in (('G-INJ(a)', lambda: g_inj_a(applied, a.public_dir, a.geometry)),
                     ('G-INJ(b)', lambda: g_inj_b(djdn, refs)),
                     ('G-COLD', lambda: g_cold(ion_path(a.ion_dir, 0), ion_path(a.ion_dir, 1), refs))):
        try:
            gates[name] = fn()
        except (Fail, OSError, KeyError, ValueError) as e:
            gates[name] = {'pass': False, 'problems': [f'{type(e).__name__}: {e}']}
    for f in (applied, djdn):
        if os.path.isfile(f):
            res['inputs'][os.path.basename(f)] = sha256(f)
    id_fail = not gates['G-ID']['pass']
    if pin_fail:
        gate_fail = id_fail
    else:
        gate_fail = id_fail or gates['G-RUN']['code'] != 'PASS' or not all(
            gates[g]['pass'] for g in ('G-INJ(a)', 'G-INJ(b)', 'G-COLD'))
    cls, rt = {}, {}
    if not pin_fail and gates['G-RUN']['code'] == 'PASS':
        rt = r_table(a.ion_dir, refs)
        cls = classify(rt)
    res['gates'] = gates
    res['r_table'] = rt
    res['shell_map'] = cls
    res['branch'] = branch(gate_fail, pin_fail, cls)
    # §5 source witness = interpretation layer, not a gate: a Fail is recorded per cell and never stops the verdict
    src = {}
    for g in range(2, 9):
        p = os.path.join(a.work, f'coldstart_levels_gen{g:03d}.csv')
        if os.path.isfile(p):
            try:
                cells = read_levels(p, J_REGISTERED)
            except (OSError, KeyError, ValueError) as e:
                src[g] = {'undefined': f'{type(e).__name__}: {e}'}
                continue
            src[g] = {}
            for s in J_REGISTERED:
                try:
                    src[g][s] = source_witness(deck, cells[s])
                except (Fail, KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
                    src[g][s] = {'undefined': f'{type(e).__name__}: {e}'}
    res['source_table'] = src
    if 4 in src and 'undefined' not in src[4]:
        res['source_start'] = {s: ('UNDEFINED' if 'rho' not in v else 'COLD_SOURCE' if v['rho'] < 1
                                   else 'HOT_SOURCE' if v['rho'] > 1 else 'EQUAL') for s, v in src[4].items()}
    with open(a.out, 'x') as f:
        json.dump(res, f, indent=1, default=float)
    return {'branch': res['branch'], 'gates': {k: (v.get('code') or ('PASS' if v.get('pass') else 'FAIL')) for k, v in gates.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('referents'); q.add_argument('--public-dir', required=True); q.add_argument('--geometry', required=True)
    q.add_argument('--base-ion-dir', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('inj-table'); q.add_argument('table'); q.add_argument('--public-dir', required=True); q.add_argument('--geometry', required=True)
    q = sub.add_parser('g-cold'); q.add_argument('ion0'); q.add_argument('ion1'); q.add_argument('--referents', required=True)
    q = sub.add_parser('classify-only'); q.add_argument('--ion-dir', required=True); q.add_argument('--referents', required=True)
    q = sub.add_parser('b-only'); q.add_argument('levels'); q.add_argument('--model-dir', required=True)
    q.add_argument('--plasma-state'); q.add_argument('--shells', default='24'); q.add_argument('--ground-x3', action='store_true')
    q.add_argument('--expect', choices=['base', 'x3']); q.add_argument('--stderr', help='check [COLDSTART][LEVELS] tag rows/rc for this file')
    q = sub.add_parser('fixture'); q.add_argument('kind', choices=['nc-inj', 'nc-hot', 'nc-b-x3', 'applied-ok', 'applied-bad-split']); q.add_argument('--out', required=True)
    q.add_argument('--table'); q.add_argument('--public-dir'); q.add_argument('--ion-dir'); q.add_argument('--referents'); q.add_argument('--levels'); q.add_argument('--shell', type=int, default=24)
    sub.add_parser('selftest-rules')
    q = sub.add_parser('full')
    for k in ('--run', '--base', '--work', '--ion-dir', '--stderr', '--referents', '--public-dir', '--geometry', '--out', '--frozen-tree'):
        q.add_argument(k, required=True)
    q.add_argument('--exports', default='input/resolved_lumina.exports')
    q.add_argument('--sealed', nargs='*', default=[], help='extra run-relative sealed inputs beyond deck/sigma/topion/fixed_te')
    q.add_argument('--run-build-identity', help='default <run>/input/run_build_identity.json')
    q.add_argument('--base-build-identity', help='default <base>/input/run_build_identity.json')
    argv = sys.argv[1:]
    if argv and argv[0] in ('--classify-only', '--b-only'):
        argv[0] = argv[0][2:]
    a = p.parse_args(argv)
    try:
        if a.mode == 'referents':
            r = make_referents(a.public_dir, a.geometry, a.base_ion_dir, a.out)
        elif a.mode == 'inj-table':
            r = g_inj_a(a.table, a.public_dir, a.geometry)
        elif a.mode == 'g-cold':
            r = g_cold(a.ion0, a.ion1, read_referents(a.referents))
        elif a.mode == 'classify-only':
            rt = r_table(a.ion_dir, read_referents(a.referents), ks=(3, 4, 5))
            cls = classify(rt)
            r = {'r_table': rt, 'shell_map': cls, 'branch': branch(False, False, cls)}
            r['pass'] = r['branch'] == 'COLD-STAY'
        elif a.mode == 'b-only':
            shells = [int(x) for x in a.shells.split(',')]
            res, deck = b_only(a.levels, a.model_dir, shells, a.plasma_state, a.ground_x3)
            r = {'table': res, 'pass': True}
            if a.expect:
                cmp = compare_ncb(res[24], NCB_BASE if a.expect == 'base' else NCB_X3)
                r['compare'] = cmp
                r['pass'] = all(c['pass'] for c in cmp.values())
            if a.stderr:
                m = re.search(r'gen(\d+)\.csv$', a.levels)
                tags = [TAG_LEV.match(l) for l in open(a.stderr) if TAG_LEV.match(l)]
                tags = [t for t in tags if m and int(t.group(1)) == int(m.group(1))]
                rows, bad = levels_file_check(a.levels, int(m.group(1)), (deck.n_levels[1] + deck.n_levels[2]) * 50) if m else (0, ['NAME'])
                ok = len(tags) == 1 and int(tags[0].group(3)) == 0 and int(tags[0].group(2)) == rows and not bad
                r['levels_tag'] = {'tags': len(tags), 'rows': rows, 'bad': bad, 'pass': ok}
                r['pass'] = r['pass'] and ok
        elif a.mode == 'fixture':
            r = fixture(a) | {'pass': True}
        elif a.mode == 'selftest-rules':
            r = selftest_rules()
        elif a.mode == 'full':
            r = full(a)
            r['pass'] = True
    except Fail as e:
        r = {'pass': False, 'fail_closed': str(e)}
    print(json.dumps(r, indent=1, default=float))
    return 0 if r.get('pass') else 1


if __name__ == '__main__':
    sys.exit(main())
