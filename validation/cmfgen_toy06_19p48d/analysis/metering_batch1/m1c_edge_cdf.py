#!/usr/bin/env python3
"""M1c: the fb EMISSION FREQUENCY half of H1.

Active path (lumina_plasma.c:4342-4346): every free-bound r-packet in a shell is
emitted at ONE frequency = the ionization edge of the single most abundant ion.
Repair path [FB-MULTI] (4366-4430): edge CDF over all recombining continua with
weight w = n_e n_ion alpha_RR (kTe under LUMINA_FB_COOL_KT=1).
This prints what that CDF would be, from parity42's own pops.
"""
import csv, collections, math, os
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('m1', os.path.join(HERE, 'm1_kramers_vs_milne.py'))
src = open(spec.origin).read().split('os.makedirs(OUT, exist_ok=True)')[0]
ns = {'__name__': 'm1mod'}
exec(compile(src, spec.origin, 'exec'), ns)
Cont, alpha_dr, Te, lv_Z, lv_ion, chi = (ns['Cont'], ns['alpha_dr'], ns['Te'],
                                         ns['lv_Z'], ns['lv_ion'], ns['chi'])
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
RUN = os.path.join(ROOT, 'logs/coevolve_consume_parity42')
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')
ROM = ['I', 'II', 'III', 'IV', 'V', 'VI']
EL = {6: 'C', 8: 'O', 12: 'Mg', 13: 'Al', 14: 'Si', 16: 'S', 20: 'Ca', 21: 'Sc',
      22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni'}

nion = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(RUN, 'lumina_ion_pops.csv'))):
    s, Z, k, n = int(r['shell_id']), int(r['Z']), int(r['stage']), float(r['n_ion'])
    if k >= 1 and n > 0:
        nion[s][(Z, k)] = n

pops = sorted({(int(z), int(i)) for z, i in zip(lv_Z, lv_ion) if i >= 1})
conts = {p: Cont(p[0], p[1] - 1) for p in pops}

rows = []
for s in (0, 5, 8, 20, 30, 45, 49):
    T = Te[s]
    w = {}
    for (Z, k), n in nion[s].items():
        if (Z, k) not in conts:
            continue
        a = conts[(Z, k)].alpha(T, gated=True) + alpha_dr(Z, k, T)
        c = chi.get((Z, k - 1), 0.0)
        if a <= 0 or c <= 0 or c > 1e9:
            continue
        w[(Z, k)] = (n * a, 12398.419843320026 / c)
    tot = sum(v[0] for v in w.values())
    print(f'--- s{s}  T_e={T:.0f} K   (single-edge path emits 100% at '
          f'{max(nion[s].items(), key=lambda kv: kv[1])[0]} edge) ---')
    for (Z, k), (ww, lam) in sorted(w.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f'    {EL[Z]+" "+ROM[k]:<8s} -> {EL[Z]+" "+ROM[k-1]:<8s} '
              f'share {ww/tot*100:6.2f}%   edge {lam:8.1f} A')
        rows.append(dict(shell=s, T_e=T, recomb_ion=EL[Z] + ' ' + ROM[k],
                         share=ww / tot, edge_A=lam))
    # share by wavelength band
    bands = [(0, 500), (500, 912), (912, 1500), (1500, 3000), (3000, 1e9)]
    for lo, hi in bands:
        sh = sum(v[0] for v in w.values() if lo <= v[1] < hi) / tot
        if sh > 1e-4:
            print(f'      band {lo:>5.0f}-{hi if hi < 1e8 else 0:<6.0f} A : {sh*100:6.2f}%')

with open(os.path.join(OUT, 'm1c_fb_edge_cdf.csv'), 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    for r in rows:
        wtr.writerow(r)
