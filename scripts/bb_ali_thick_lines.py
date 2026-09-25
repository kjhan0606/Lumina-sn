#!/usr/bin/env python3
"""DET-BB-ALI attribution support (prereg §3, not a branch term).

Fixed line set: mapped S II/III and Si II/III lines, 900-2000 A, whose Sobolev
beta < 1e-2 in the BASE final populations, per shell (15, 24).  For that same
set, compare base vs run S_l/B_nu(T_e) from each run's final NLTE populations,
opacity-weighted with the base weights (1 - e^-tau_base).  No cap on S_l/B;
non-finite ratios (inversion / ratio == 1) are counted, not clipped.

  bb_ali_thick_lines.py --base <root> --run <root> [--out json]
"""
import argparse, collections, csv, json, math

H, K, C = 6.62607015e-27, 1.380649e-16, 2.99792458e10
SOB = 0.02654008
T_EXP = 19.48 * 86400.0
SH = (15, 24)
IONS = {(16, 1), (16, 2), (14, 1), (14, 2)}


def populations(root):
    pop, te = collections.defaultdict(dict), {}
    for r in csv.DictReader(open(root + '/work/nlte_levels_final.csv')):
        s = int(r['shell'])
        key = (int(r['Z']), int(r['ion']))
        if s in SH and key in IONS:
            pop[(s,) + key][int(r['level_idx'])] = (float(r['n_pop']), float(r['g']))
            te[s] = float(r['T_e'])
    return pop, te


def line_state(P, lo, up, flu, lamcm, nu, T):
    nl, gl = P[lo]
    nu_pop, gu = P[up]
    if nl <= 0:
        return None
    ratio = (gu * nl) / (gl * nu_pop) if nu_pop > 0 else math.inf
    tau = SOB * flu * lamcm * T_EXP * nl * (1.0 - 1.0 / ratio)
    x = H * nu / (K * T)
    sl_b = math.expm1(x) / (ratio - 1.0) if ratio != 1.0 else math.inf
    return tau, sl_b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out')
    a = p.parse_args()
    pb, tb = populations(a.base)
    pr, tr = populations(a.run)
    acc = {s: collections.defaultdict(lambda: {'n': 0, 'w': 0.0, 'wb': 0.0, 'wr': 0.0, 'nonfinite_run': 0})
           for s in SH}
    with open(a.base + '/input/model/line_list.csv') as f:
        for r in csv.DictReader(f):
            lam = float(r['wavelength'])
            if not (900.0 <= lam < 2000.0):
                continue
            key = (int(r['atomic_number']), int(r['ion_number']))
            if key not in IONS:
                continue
            lo, up = int(r['level_number_lower']), int(r['level_number_upper'])
            nu, flu, lamcm = float(r['nu']), float(r['f_lu']), float(r['wavelength_cm'])
            for s in SH:
                P, Q = pb.get((s,) + key), pr.get((s,) + key)
                if not P or not Q or lo not in P or up not in P or lo not in Q or up not in Q:
                    continue
                b = line_state(P, lo, up, flu, lamcm, nu, tb[s])
                if b is None or b[0] <= 0:
                    continue
                tau = b[0]
                beta = (1 - math.exp(-tau)) / tau if tau < 700 else 1 / tau
                if beta >= 1e-2:
                    continue
                w = 1.0 - math.exp(-min(tau, 700.0))
                rr = line_state(Q, lo, up, flu, lamcm, nu, tr[s])
                e = acc[s][key]
                e['n'] += 1
                if rr is None or not math.isfinite(rr[1]) or not math.isfinite(b[1]):
                    e['nonfinite_run'] += 1
                    continue
                e['w'] += w
                e['wb'] += w * b[1]
                e['wr'] += w * rr[1]
    out = {}
    for s in SH:
        tot = {'n': 0, 'w': 0.0, 'wb': 0.0, 'wr': 0.0, 'nonfinite_run': 0}
        rows = {}
        for key, e in sorted(acc[s].items()):
            for k in tot:
                tot[k] += e[k]
            rows[f'Z{key[0]}_ion{key[1]}'] = {'n_lines': e['n'], 'nonfinite': e['nonfinite_run'],
                                              'SlB_base': e['wb'] / e['w'] if e['w'] else None,
                                              'SlB_run': e['wr'] / e['w'] if e['w'] else None}
        rows['ALL'] = {'n_lines': tot['n'], 'nonfinite': tot['nonfinite_run'],
                       'SlB_base': tot['wb'] / tot['w'] if tot['w'] else None,
                       'SlB_run': tot['wr'] / tot['w'] if tot['w'] else None,
                       'T_e_base': tb[s], 'T_e_run': tr[s]}
        out[f'sh{s}'] = rows
        for name, v in rows.items():
            ch = (v['SlB_run'] / v['SlB_base'] - 1) * 100 if v['SlB_base'] and v['SlB_run'] is not None else None
            print(f"sh{s} {name:10s} n={v['n_lines']:6d} nonfinite={v['nonfinite']:4d} "
                  f"S_l/B base={v['SlB_base']} run={v['SlB_run']} change%={ch}")
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
