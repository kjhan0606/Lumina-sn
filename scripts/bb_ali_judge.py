#!/usr/bin/env python3
"""DET-BB-ALI reader/judge (prereg docs/PREREG_DET_BB_ALI_2026-09-25.md, rev 1).

Modes
  referents  build bb_ali_referents.csv from the 398939 control products
  selftest   NC-ID, NC-FIRE, NC-CLASS (§4)
  full       G-ID / G-FIRE / G-RUN, delta table, classify (§4, §5)

G-ID identity-chain helpers are imported from the predecessor reader
(coldstart_judge); nothing there is modified.
"""
import argparse, csv, json, math, os, re, shutil, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coldstart_judge as cj  # noqa: E402

J = list(range(10, 16)) + list(range(19, 31))
CORE = list(range(10, 16)) + list(range(19, 25))
KS = (2, 3, 4, 5)
KS_S = (3, 4, 5)
N_ITER = 6
CHANGESET_SRC = ['src/line_net_rate.h', 'src/line_net_rate.c', 'src/lumina_plasma.c', 'src/env_universe.h']
EXPORTS = 'input/resolved_lumina.exports'
EXPORT_ADDED = {'LUMINA_BB_NRB': '1'}
EXPORT_CHANGED = {'LUMINA_A210_SPRODUCER_CAPTURE': ('0', '1')}
BANDS = {'100_2000': (100, 2000), 'lt912': (100, 912), '912_1400': (912, 1400), '1400_2000': (1400, 2000)}
TH_G, TH_X, TH_S, N_BOTH = 0.3, 0.5, 0.15, 6
H, K, C = 6.62607015e-27, 1.380649e-16, 2.99792458e10

NE_CLOSURE = re.compile(r'^\[A2-07\]\[NE-CLOSURE\] commit=\S+ generation=\d+ iterations=(\d+)\b')
TAG = re.compile(r'^\[BB-NRB\] iter=(-?\d+) operator=(\d+) armed=([01]) captured=([01]) n_shells=(\d+) skip_z=(\S+)\s*$')
FATAL_NRB = '[BB-NRB][FATAL]'
CAPTURED = '[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED'
FORBIDDEN = ('[cmf_fine][BLOCKED]', '[R6][BLOCKED]', '[R6][FATAL]', '[R7][FATAL]', '[RADIATION-FIELD][FATAL]')
SROP = re.compile(r'^\[cmf_fine\]\[SOBOLEV-LINE-OPERATOR\] .*\bsrce_chk_applied=(\d+)')


# ---------------------------------------------------------------- readers
def stderr_lines(root):
    return open(os.path.join(root, 'stderr.log'), errors='replace').read().splitlines()


def sweep_ends(lines):
    ends, c = [], 0
    for l in lines:
        m = NE_CLOSURE.match(l)
        if m:
            c += int(m.group(1))
            ends.append(c)
    return ends


def gammas(root, lines, ks=KS, shells=J):
    """S II gamma_phot at the last sweep of logical iteration k (ends[0] = SEED)."""
    ends = sweep_ends(lines)
    if len(ends) < max(ks) + 2:
        raise cj.Fail(f'NE_CLOSURE ends={len(ends)}')
    want = {ends[k + 1]: k for k in ks}
    rows = {}
    for r in csv.DictReader(open(os.path.join(root, 'work/lumina_ion_channels.csv'))):
        if r['Z'] == '16' and r['stage'] == '1' and int(r['gen']) in want and int(r['shell']) in shells:
            key = (want[int(r['gen'])], int(r['shell']))
            rows.setdefault(key, []).append(float(r['gamma_phot']))
    out, problems = {}, []
    for k in ks:
        for s in shells:
            v = rows.get((k, s), [])
            if len(v) != 1 or not (math.isfinite(v[0]) and v[0] > 0):
                problems.append(f'GAMMA k={k} sh{s} rows={v[:3]}')
            else:
                out[(k, s)] = v[0]
    return out, problems, ends


def ion_x(root, k):
    ion = cj.read_ion(cj.ion_path(os.path.join(root, 'work/physics_comparison'), k))
    return {s: (ion[s]['q2'] / ion[s]['q1'] if ion[s]['q1'] > 0 else math.nan) for s in ion}


def planck(nu, T):
    x = H * nu / (K * T)
    return 2 * H * nu ** 3 / C ** 2 / math.expm1(x) if x < 700 else 0.0


def s_bands(root, k, shells):
    base = os.path.join(root, f'work/physics_comparison/physics_DET_iter{k:04d}')
    te = {int(r['shell_id']): float(r['T_e_K']) for r in csv.DictReader(open(base + '.shell.csv'))}
    acc = {(s, b): [0.0, 0.0] for s in shells for b in BANDS}
    for r in csv.DictReader(open(base + '.spectral.csv')):
        s = int(r['shell_id'])
        if s not in shells:
            continue
        nlo, nhi = float(r['nu_lo_Hz']), float(r['nu_hi_Hz'])
        nu = 0.5 * (nlo + nhi)
        lam = C * 1e8 / nu
        dn = nhi - nlo
        ca = float(r['chi_bb_cm1']) + float(r['chi_bf_cm1']) + float(r['chi_ff_cm1'])
        bn = planck(nu, te[s])
        for b, (lo, hi) in BANDS.items():
            if lo <= lam < hi:
                acc[(s, b)][0] += float(r['eta_true_total']) * dn
                acc[(s, b)][1] += ca * bn * dn
    return {key: (a[0] / a[1] if a[1] > 0 else math.nan) for key, a in acc.items()}


# ---------------------------------------------------------------- referents (§3)
def make_referents(base, x_pub_csv, out):
    xpub = {int(r['shell']): float(r['x_pub']) for r in csv.DictReader(open(x_pub_csv)) if r['x_pub']}
    lines = stderr_lines(base)
    g, problems, _ = gammas(base, lines)
    if problems:
        raise cj.Fail(f'REFERENT_GAMMA {problems[:3]}')
    xs = {k: ion_x(base, k) for k in KS}
    ss = {k: s_bands(base, k, J) for k in KS_S}
    cols = ['shell', 'in_core', 'x_pub'] + [f'x_k{k}' for k in KS] + [f'g_k{k}' for k in KS] + \
           [f'S_{b}_k{k}' for b in BANDS for k in KS_S]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in J:
            w.writerow([s, int(s in CORE), repr(xpub[s])] + [repr(xs[k][s]) for k in KS] +
                       [repr(g[(k, s)]) for k in KS] + [repr(ss[k][(s, b)]) for b in BANDS for k in KS_S])
    return cj.sha256(out)


def read_refs(path):
    return {int(r['shell']): {k: float(v) for k, v in r.items() if k != 'shell'} for r in csv.DictReader(open(path))}


# ---------------------------------------------------------------- classify (§5)
def classify(table):
    """table: {shell: (dG, dx)} over J_core. Returns (branch, stats)."""
    dg = np.array([table[s][0] for s in CORE], dtype=float)
    dx = np.array([table[s][1] for s in CORE], dtype=float)
    if not (np.all(np.isfinite(dg)) and np.all(np.isfinite(dx))):
        return 'X', {'reason': 'NONFINITE_DELTA'}
    mg, mx = float(np.median(dg)), float(np.median(dx))
    n_both = int(np.sum((dg <= -TH_G) & (dx <= -TH_X)))
    fm, xd = mg <= -TH_G, mx <= -TH_X
    both = n_both >= N_BOTH
    up = mg >= TH_G or mx >= TH_X
    if fm and xd and both:
        br = 'NRB-DOWN'
    elif (fm != xd) or (fm and xd and not both):
        br = 'NRB-PARTIAL'
    elif not up:
        br = 'NRB-FLAT'
    else:
        br = 'NRB-UP'
    return br, {'med_dG': mg, 'med_dx': mx, 'n_both': n_both, 'field_moved': fm, 'x_dropped': xd,
                'both_shells': both, 'up': up, 'opposite_sign_shells': int(np.sum((dg <= -TH_G) & (dx >= TH_X)))}


def gmean45(v4, v5):
    return math.sqrt(v4 * v5) if v4 > 0 and v5 > 0 else math.nan


def delta_table(refs, g_run, x_run):
    table = {}
    for s in J:
        c = refs[s]
        dG = math.log10(gmean45(g_run.get((4, s), math.nan), g_run.get((5, s), math.nan)) / gmean45(c['g_k4'], c['g_k5']))
        dx = math.log10(gmean45(x_run[4].get(s, math.nan), x_run[5].get(s, math.nan)) / gmean45(c['x_k4'], c['x_k5']))
        table[s] = (dG, dx)
    return table


# ---------------------------------------------------------------- gates (§4)
def export_delta(eb, er, base_root, run_root):
    norm = lambda v: v.replace(os.path.abspath(base_root), '<ROOT>').replace(os.path.abspath(run_root), '<ROOT>')
    added = {k: er[k] for k in set(er) - set(eb)}
    removed = set(eb) - set(er)
    changed = {k: (eb[k], er[k]) for k in set(eb) & set(er) if norm(eb[k]) != norm(er[k])}
    ok = added == EXPORT_ADDED and not removed and changed == EXPORT_CHANGED
    return ok, {'added': added, 'removed': sorted(removed), 'changed': changed}


def g_fire(lines):
    problems = []
    tags = [TAG.match(l) for l in lines if TAG.match(l)]
    armed = [int(m.group(1)) for m in tags if m.group(2) == '2' and m.group(3) == '1' and m.group(4) == '1']
    if sorted(armed) != list(range(N_ITER)):
        problems.append(f'ARMED_TAGS {sorted(armed)}')
    if any(m.group(2) == '2' and m.group(3) == '0' for m in tags):
        problems.append('OP2_UNARMED')
    if sum(1 for m in tags if m.group(1) == '0' and m.group(2) == '1' and m.group(3) == '0') > 1:
        problems.append('SEED_TAG_DUP')
    n_cap = sum(1 for l in lines if l.startswith(CAPTURED))
    if n_cap != N_ITER:
        problems.append(f'CAPTURED {n_cap}')
    n_fatal = sum(1 for l in lines if l.startswith(FATAL_NRB))
    if n_fatal:
        problems.append(f'NRB_FATAL {n_fatal}')
    skip = sorted({m.group(6) for m in tags})
    return {'pass': not problems, 'problems': problems, 'n_tags': len(tags), 'skip_z': skip}


def g_run(root, lines):
    problems = []
    fp = os.path.join(root, 'RUN_FOOTER.txt')
    footer = cj.read_footer(fp) if os.path.isfile(fp) else {}
    if footer.get('terminal_state') != 'SUCCESS' or footer.get('model_rc') != '0' or \
            footer.get('outer_iterations') != str(N_ITER):
        problems.append(f"FOOTER {footer.get('terminal_state')} rc={footer.get('model_rc')} outer={footer.get('outer_iterations')}")
    x = {}
    for k in range(N_ITER):
        try:
            xk = ion_x(root, k)
            bad = [s for s in J if not (math.isfinite(xk.get(s, math.nan)) and xk[s] > 0)]
            if bad:
                problems.append(f'ION_X k={k} {bad}')
            x[k] = xk
        except (cj.Fail, OSError, KeyError) as e:
            problems.append(f'ION k={k} {e}')
    for k in KS_S:
        for ext in ('spectral', 'shell'):
            if not os.path.isfile(os.path.join(root, f'work/physics_comparison/physics_DET_iter{k:04d}.{ext}.csv')):
                problems.append(f'{ext.upper()} k={k}')
    for tag in FORBIDDEN:
        n = sum(1 for l in lines if l.startswith(tag))
        if n:
            problems.append(f'FORBIDDEN {tag} {n}')
    g, gp, ends = {}, [], []
    try:
        g, gp, ends = gammas(root, lines)
        gmax = max(int(r['gen']) for r in csv.DictReader(open(os.path.join(root, 'work/lumina_ion_channels.csv'))))
        if not ends or gmax != ends[-1]:
            problems.append(f'CHANNEL_GEN max={gmax} sigma={ends[-1] if ends else None}')
    except (cj.Fail, OSError, KeyError, ValueError) as e:
        gp = [f'GAMMA {e}']
    problems += gp
    srce = [int(m.group(1)) for m in (SROP.match(l) for l in lines) if m]
    return {'pass': not problems, 'problems': problems[:40], 'footer': footer, 'sweep_ends': ends,
            'srce_chk_applied': srce}, g, x


def g_id(a):
    problems, info = [], {}
    cj.sealed_inputs(a, problems, info)
    eb, er = cj.read_exports(os.path.join(a.base, EXPORTS)), cj.read_exports(os.path.join(a.run, EXPORTS))
    ok, d = export_delta(eb, er, a.base, a.run)
    info['exports_delta'] = d
    if not ok:
        problems.append(f'EXPORTS {d}')
    try:
        if cj.sha256(er['LUMINA_DEPOSITION_FILE']) != cj.sha256(eb['LUMINA_DEPOSITION_FILE']):
            problems.append('SEALED deposition')
    except (OSError, KeyError) as e:
        problems.append(f'SEALED deposition {e}')
    manifest_binary = None
    try:
        rb = cj.source_members(os.path.join(a.run, 'input/run_build_identity.json'))
        manifest_binary = json.load(open(os.path.join(a.run, 'input/run_build_identity.json'))).get('binary', {}).get('sha256')
        build_rec = cj.read_manifest(os.path.join(a.run, 'input/source_sha_at_build.txt'))
        prev = cj.read_manifest(a.prev_build_record)
        diff = sorted(p for p in set(build_rec) | set(prev) if build_rec.get(p) != prev.get(p))
        if diff != sorted(CHANGESET_SRC):
            problems.append(f'SOURCE_DIFF {diff}')
        approved = dict(kv.split('=', 1) for kv in (a.approved_sha or '').split(',') if '=' in kv)
        if set(approved) != set(CHANGESET_SRC):
            problems.append(f'APPROVED_SHA keys {sorted(approved)}')
        for p in CHANGESET_SRC:
            vals = {'approved': approved.get(p), 'build': build_rec.get(p), 'manifest': rb.get(p),
                    'tree': cj.sha256(os.path.join(a.frozen_tree, p))}
            if len(set(vals.values())) != 1 or None in vals.values():
                problems.append(f'SOURCE_CHAIN {p} ' + ' '.join(f'{k}={(v or "-")[:12]}' for k, v in vals.items()))
        if set(build_rec) != set(rb) or any(build_rec[p] != rb[p] for p in rb):
            problems.append('MANIFEST_NOT_BUILD_RECORD')
        info['source_diff'] = diff
    except (cj.Fail, OSError, KeyError, ValueError, TypeError) as e:
        problems.append(f'BUILD_IDENTITY {type(e).__name__}: {e}')
    fp = os.path.join(a.run, 'RUN_FOOTER.txt')
    footer = cj.read_footer(fp) if os.path.isfile(fp) else {}
    try:
        build_bin = open(os.path.join(a.run, 'input/binary.sha256')).read().split()[0]
    except (OSError, IndexError):
        build_bin = None
    binp = os.path.join(a.run, 'input/lumina_cuda')
    chain = {'build': build_bin, 'manifest': manifest_binary, 'footer': footer.get('binary_sha256'),
             'file': cj.sha256(binp) if os.path.isfile(binp) else None}
    info['binary_chain'] = chain
    if len(set(chain.values())) != 1 or None in chain.values():
        problems.append('BINARY_CHAIN ' + ' '.join(f'{k}={(v or "-")[:12]}' for k, v in chain.items()))
    if any('h200' in v.lower() for v in footer.values()):
        problems.append('H200')
    ref_sha = cj.sha256(a.referents)
    info['referents_sha256'] = ref_sha
    if ref_sha != a.referents_sha:
        problems.append(f'REFERENTS_SHA {ref_sha[:12]}')
    return {'pass': not problems, 'problems': problems, 'info': info}


# ---------------------------------------------------------------- full
def full(a):
    lines = stderr_lines(a.run)
    gid = g_id(a)
    gfire = g_fire(lines)
    grun, g, x = g_run(a.run, lines)
    res = {'G-ID': gid, 'G-FIRE': gfire, 'G-RUN': grun}
    if not (gid['pass'] and gfire['pass'] and grun['pass']):
        res['branch'] = 'X'
    else:
        refs = read_refs(a.referents)
        table = delta_table(refs, g, x)
        br, stats = classify({s: table[s] for s in CORE})
        res.update({'branch': br, 'stats': stats,
                    'delta': {s: {'dG': table[s][0], 'dx': table[s][1]} for s in J}})
        traj = {}
        for s in J:
            c = refs[s]
            traj[s] = {f'k{k}': {'dG': math.log10(g[(k, s)] / c[f'g_k{k}']), 'dx': math.log10(x[k][s] / c[f'x_k{k}'])}
                       for k in KS}
        res['trajectory'] = traj
        dS = {}
        for k in KS_S:
            sb = s_bands(a.run, k, J)
            dS[k] = {s: {b: math.log10(sb[(s, b)] / refs[s][f'S_{b}_k{k}']) for b in BANDS} for s in J}
        med_dS = float(np.median([0.5 * (dS[4][s]['100_2000'] + dS[5][s]['100_2000']) for s in CORE]))
        res['support'] = {'dS_band': dS, 'med_dS_100_2000_k45': med_dS, 'source_moved': med_dS <= -TH_S}
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v)
                      for k, v in res.items() if k in ('G-ID', 'G-FIRE', 'G-RUN', 'branch', 'stats')}, default=str))
    return 0


# ---------------------------------------------------------------- selftest (§4 NC)
def selftest(a):
    fails = []

    def expect(cond, name):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # NC-ID
    eb = cj.read_exports(os.path.join(a.base, EXPORTS))
    ok, _ = export_delta(eb, dict(eb), a.base, a.base)
    expect(not ok, 'NC-ID(1) base exports verbatim -> FAIL')
    er = dict(eb, LUMINA_BB_NRB='1')
    ok, _ = export_delta(eb, er, a.base, a.base)
    expect(not ok, 'NC-ID(2) +LUMINA_BB_NRB only, capture 0 -> FAIL')
    er['LUMINA_A210_SPRODUCER_CAPTURE'] = '1'
    ok, _ = export_delta(eb, er, a.base, a.base)
    expect(ok, 'NC-ID(0) exact two-key delta -> PASS')
    with tempfile.TemporaryDirectory() as td:
        t = os.path.join(td, 'ref.csv')
        shutil.copy(a.referents, t)
        with open(t, 'r+b') as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 1]))
        expect(cj.sha256(t) != a.referents_sha and cj.sha256(a.referents) == a.referents_sha,
               'NC-ID(3) referents 1-byte flip -> sha mismatch')
    # NC-FIRE
    lines = stderr_lines(a.base)
    expect(not g_fire(lines)['pass'], 'NC-FIRE(1) 398939 stderr -> FAIL')
    arm = [f'[BB-NRB] iter={k} operator=2 armed=1 captured=1 n_shells=50 skip_z=14' for k in range(N_ITER)]
    cap = [CAPTURED + ' fixture'] * N_ITER
    good = lines + ['[BB-NRB] iter=0 operator=1 armed=0 captured=0 n_shells=50 skip_z=14'] + arm + cap
    expect(g_fire(good)['pass'], 'NC-FIRE(2) +6 armed +6 CAPTURED -> PASS')
    expect(not g_fire([l for l in good if not l.startswith('[BB-NRB] iter=3 ')])['pass'], 'NC-FIRE(3) drop k=3 -> FAIL')
    expect(not g_fire(good + ['[BB-NRB] iter=2 operator=2 armed=0 captured=1 n_shells=50 skip_z=14'])['pass'],
           'NC-FIRE(4) +op2 armed=0 -> FAIL')
    # NC-CLASS
    half = lambda p, q: {s: (p if i < 6 else q) for i, s in enumerate(CORE)}
    allc = lambda v: {s: v for s in CORE}
    fx = [('D', allc((-0.5, -0.8)), 'NRB-DOWN'), ('D0', allc((-0.3, -0.5)), 'NRB-DOWN'),
          ('M', half((-1.0, 0.0), (0.0, -2.0)), 'NRB-PARTIAL'), ('P1', allc((-0.5, -0.2)), 'NRB-PARTIAL'),
          ('P2', allc((0.0, -0.8)), 'NRB-PARTIAL'), ('P3', half((-0.6, -0.7), (-0.4, 0.9)), 'NRB-PARTIAL'),
          ('F', allc((0.0, 0.0)), 'NRB-FLAT'), ('F2', allc((-0.29, -0.49)), 'NRB-FLAT'),
          ('U', allc((0.5, 0.0)), 'NRB-UP'), ('U2', allc((0.1, 0.6)), 'NRB-UP'),
          ('X1', {**allc((-0.5, -0.8)), CORE[0]: (math.nan, -0.8)}, 'X')]
    for name, t, want in fx:
        got, _ = classify(t)
        expect(got == want, f'NC-CLASS({name}) -> {want} (got {got})')
    print(f'[BB-ALI-SELFTEST] {"FAIL" if fails else "PASS"} failures={len(fails)}')
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('referents')
    q.add_argument('--base', required=True); q.add_argument('--x-pub-csv', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('selftest')
    q.add_argument('--base', required=True); q.add_argument('--referents', required=True); q.add_argument('--referents-sha', required=True)
    q = sub.add_parser('full')
    for k in ('--base', '--run', '--referents', '--referents-sha', '--approved-sha', '--frozen-tree',
              '--prev-build-record', '--out'):
        q.add_argument(k, required=True)
    a = p.parse_args()
    if a.mode == 'referents':
        print(make_referents(a.base, a.x_pub_csv, a.out))
        return 0
    if a.mode == 'selftest':
        return selftest(a)
    return full(a)


if __name__ == '__main__':
    sys.exit(main())
