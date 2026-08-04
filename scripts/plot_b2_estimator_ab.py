#!/usr/bin/env python
"""A3(B) fork-B2 A/B: binned-J vs first-moment T_rad estimator.
Compares two a3lstar runs against the DDC15 CMFGEN ref:
  panel A  per-shell n_e dex vs CMFGEN
  panel B  T_rad and T_e per shell (both arms)
  panel C  dilution factor W per shell (W>1 = unphysical, flagged)
  panel D  carbon C II fraction per shell (transition-zone over-ionization)
Usage: python plot_b2_estimator_ab.py <jidA>:<labelA> <jidB>:<labelB>
"""
import csv, math, os, sys, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG = '#140D44'

def load_plasma(wr):
    d = {}
    with open(os.path.join(wr, 'lumina_plasma_state.csv')) as f:
        for r in csv.DictReader(f):
            s = int(r['shell_id'])
            d[s] = dict(W=float(r['W']), T_rad=float(r['T_rad']),
                        n_e=float(r['n_e']), T_e=float(r['T_e']))
    return d

def load_ref_ne(wr):
    d = {}
    with open(os.path.join(wr, 'ref', 'electron_densities.csv')) as f:
        for r in csv.DictReader(f):
            d[int(r['shell_id'])] = float(r['n_e'])
    return d

def carbon_cii(wr):
    dumps = sorted(glob.glob(os.path.join(wr, 'nlte_levels_iter*.csv')))
    frac = {}
    if not dumps:
        return frac
    dump = dumps[-1]
    tot = defaultdict(float); ion1 = defaultdict(float); seen = set()
    with open(dump) as f:
        for row in csv.DictReader(f):
            if int(row['Z']) != 6:
                continue
            sh = int(row['shell']); ion = int(row['ion'])
            key = (sh, ion)
            if key in seen:
                continue
            seen.add(key)
            n = float(row['n_ion_total'])
            tot[sh] += n
            if ion >= 1:
                ion1[sh] += n
    for sh in tot:
        frac[sh] = ion1[sh] / tot[sh] if tot[sh] > 0 else float('nan')
    return frac

def main(args):
    arms = []
    for a in args:
        jid, lab = a.split(':', 1) if ':' in a else (a, a)
        wr = os.path.join('logs', f'ddc15_a3lstar_{jid}')
        arms.append((jid, lab, wr))

    ref = load_ref_ne(arms[0][2])
    shells = sorted(ref)
    colors = ['#3898EC', '#D97757']

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('DDC15 0.976d — fork-B2 A/B: T_rad estimator (binned vs first-moment)',
                 fontsize=14, weight='bold')

    for i, (jid, lab, wr) in enumerate(arms):
        ps = load_plasma(wr)
        cf = carbon_cii(wr)
        s = [x for x in shells if x in ps]
        c = colors[i % 2]
        dex = [math.log10(ps[x]['n_e'] / ref[x]) if ps[x]['n_e'] > 0 and ref[x] > 0 else float('nan') for x in s]
        ax[0, 0].plot(s, dex, '-o', ms=3, color=c, label=f'{lab} ({jid})')
        ax[0, 1].plot(s, [ps[x]['T_rad'] for x in s], '-o', ms=3, color=c, label=f'{lab} T_rad')
        ax[0, 1].plot(s, [ps[x]['T_e'] for x in s], '--', color=c, alpha=0.6, label=f'{lab} T_e')
        ax[1, 0].plot(s, [ps[x]['W'] for x in s], '-o', ms=3, color=c, label=lab)
        if cf:
            sc = sorted(cf)
            ax[1, 1].plot(sc, [cf[x] for x in sc], '-o', ms=3, color=c, label=lab)
        # RMS print
        allx = [d for d in dex if d == d]
        trn = [math.log10(ps[x]['n_e'] / ref[x]) for x in s if 13 <= x <= 27 and ps[x]['n_e'] > 0 and ref[x] > 0]
        rms = lambda q: math.sqrt(sum(v*v for v in q)/len(q)) if q else float('nan')
        print(f"[{jid} {lab}] all-RMS={rms(allx):.3f}  trans13-27 RMS={rms(trn):.3f}")

    ax[0, 0].axhline(0, color='k', lw=0.8, ls=':')
    ax[0, 0].set_title('A. n_e dex vs CMFGEN (0=perfect)')
    ax[0, 0].set_xlabel('shell'); ax[0, 0].set_ylabel('log10(n_e/ref)')
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
    ax[0, 0].axvspan(13, 27, color='orange', alpha=0.08)

    ax[0, 1].set_title('B. T_rad (solid) vs T_e (dashed)')
    ax[0, 1].set_xlabel('shell'); ax[0, 1].set_ylabel('T [K]')
    ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=0.3)
    ax[0, 1].axvspan(13, 27, color='orange', alpha=0.08)

    ax[1, 0].axhline(1.0, color='r', lw=0.8, ls=':', label='W=1 (physical max)')
    ax[1, 0].set_title('C. dilution factor W  (W>1 unphysical)')
    ax[1, 0].set_xlabel('shell'); ax[1, 0].set_ylabel('W')
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].set_title('D. carbon C II fraction (1=fully singly-ionized)')
    ax[1, 1].set_xlabel('shell'); ax[1, 1].set_ylabel('C II / C_tot')
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
    ax[1, 1].axvspan(13, 27, color='orange', alpha=0.08)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = 'figures/2026-06-09_ddc15_b2_estimator_ab.png'
    fig.savefig(out, dpi=130)
    print('wrote', out)

if __name__ == '__main__':
    main(sys.argv[1:])
