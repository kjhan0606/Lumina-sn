#!/usr/bin/env python3
"""hot-band falsifier judgment: bfdarkA/B vs repro22 vs CMFGEN.

Usage: python3 scripts/analyze_bfdark.py [rundir ...]
Prints outer T_e table + hot-band verdict; plots J(35eV) shell profile
from lumina_cmfgen_jnu.csv when present -> figures/bfdark_j35_profile.png
"""
import csv, sys, os
import numpy as np

H = 6.62607015e-27
EV = 1.602176634e-12

def load_te(rundir):
    path = os.path.join(rundir, 'lumina_plasma_state.csv')
    te = {}
    for r in csv.DictReader(open(path)):
        te[int(r['shell_id'])] = (float(r['T_e']), float(r['n_e']))
    return te

def load_cmfgen_T(epoch=19.48):
    lines = open('data/standart_data1/toy06/phys_toy06_cmfgen.txt').readlines()
    blocks, cur = {}, None
    for L in lines:
        if L.startswith('#TIME:'):
            cur = float(L.split()[1]); blocks[cur] = []
        elif not L.startswith('#') and cur is not None and L.strip():
            blocks[cur].append([float(x) for x in L.split()])
    d = np.array(blocks[epoch])
    return d[:, 0], d[:, 1]   # v[km/s], T[K]

def shell_velocity():
    # lumina shell velocities from the model dir
    mdir = 'data/tardis_reference_toy06_19p48d'
    for cand in ('model.csv', 'densities.csv'):
        p = os.path.join(mdir, cand)
        if os.path.exists(p):
            try:
                rows = [l.split(',') for l in open(p) if l.strip() and not l[0].isalpha()]
                v = [float(r[0]) for r in rows if len(r) > 1]
                if len(v) >= 50: return np.array(v[:51])
            except Exception:
                pass
    return None

def main():
    runs = sys.argv[1:] or ['logs/stage1_toy06_repro22',
                            'logs/stage1_toy06_bfdarkA',
                            'logs/stage1_toy06_bfdarkB']
    runs = [r for r in runs if os.path.exists(os.path.join(r, 'lumina_plasma_state.csv'))]
    tes = {os.path.basename(r).replace('stage1_toy06_', ''): load_te(r) for r in runs}
    names = list(tes.keys())
    print('shell  ' + '  '.join(f'{n:>9s}' for n in names))
    for s in range(30, 50):
        vals = '  '.join(f'{tes[n].get(s, (float("nan"),))[0]:9.0f}' for n in names)
        print(f'{s:5d}  {vals}')
    # verdict
    for n in names:
        band = [tes[n][s][0] for s in range(36, 41) if s in tes[n]]
        print(f'{n:>10s}: hot-band s36-40 max={max(band):.0f}  mean={np.mean(band):.0f}'
              f'  {"HOT (>30kK)" if max(band) > 30000 else "cold/CMFGEN-like"}')

    # J(35eV) profiles
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        nu35 = 35.0 * EV / H
        for r in runs:
            p = os.path.join(r, 'lumina_cmfgen_jnu.csv')
            if not os.path.exists(p): continue
            d = np.genfromtxt(p, delimiter=',', names=True)
            m = np.abs(d['nu'] - nu35) == np.min(np.abs(d['nu'] - nu35))
            nu_pick = d['nu'][m][0]
            sel = d['nu'] == nu_pick
            ax.semilogy(d['shell'][sel], np.maximum(d['J'][sel], 1e-40), '-o', ms=3,
                        label=os.path.basename(r).replace('stage1_toy06_', ''))
        ax.set_xlabel('shell'); ax.set_ylabel('J_nu @ ~35 eV')
        ax.axvspan(36, 40, alpha=0.15, color='red', label='hot band')
        ax.legend(); ax.set_title('35 eV lamp field vs shell')
        fig.tight_layout()
        os.makedirs('figures', exist_ok=True)
        fig.savefig('figures/bfdark_j35_profile.png', dpi=130)
        print('wrote figures/bfdark_j35_profile.png')
    except Exception as e:
        print('J-profile plot skipped:', e)

if __name__ == '__main__':
    main()
