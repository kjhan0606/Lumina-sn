#!/usr/bin/env python3
"""M2: super-level lumping census + line_list join.

Two mappings are metered:
  (S) SHIPPED  = levels.csv `super_level` column (the CMFGEN f_to_s mapping)
  (P) PRODUCTION = the parity42 override, lumina_atomic.c:733-743
        LUMINA_SUPER_CUTOFF=100  =>  super = min(level_number, 100)
      so per ion: levels 0..99 explicit, ALL levels >=100 in ONE lump SL.
Within any SL the populations are forced to Boltzmann(T_e) about the SL anchor
(lumina_plasma.c:15578 nlte_precompute_within_sl_frac) => every level inside a
lump shares one departure coefficient; relative b_i/b_j == 1 by construction.
"""
import os
import numpy as np
import pandas as pd

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
REF = os.path.join(ROOT, 'data/tardis_reference_toy06_19p48d_sivcaiv')
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')
K = 100
ROM = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
EL = {6: 'C', 8: 'O', 12: 'Mg', 13: 'Al', 14: 'Si', 16: 'S', 20: 'Ca', 21: 'Sc',
      22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni'}


def ion_name(Z, k):
    return f'{EL.get(Z, "Z%d" % Z)} {ROM[k] if k < len(ROM) else k}'


lev = pd.read_csv(os.path.join(REF, 'levels.csv'))
print(f'levels.csv rows = {len(lev)}')

# --- verify level_number is the per-ion 0-based energy rank ------------------
bad = 0
for (Z, i), g in lev.groupby(['atomic_number', 'ion_number']):
    if not (g['level_number'].values == np.arange(len(g))).all():
        bad += 1
    if not np.all(np.diff(g['energy_eV'].values) >= -1e-9):
        print(f'  [warn] energy not monotone in level_number for {ion_name(Z, i)}')
print(f'  level_number == 0..N-1 per ion: {"YES" if bad == 0 else "NO (%d ions)" % bad}')

lev['lump_P'] = lev['level_number'] >= K
# shipped mapping: an SL is a "lump" if it holds >1 level
grp = lev.groupby(['atomic_number', 'ion_number', 'super_level']).size()
sl_size = grp.rename('n').reset_index()
lev = lev.merge(sl_size, on=['atomic_number', 'ion_number', 'super_level'])
lev['lump_S'] = lev['n'] > 1

tot = len(lev)
print(f'\n=== (P) PRODUCTION mapping, SUPER_CUTOFF={K} ===')
print(f'  levels in a lump SL (level_number >= {K}): {lev["lump_P"].sum()} '
      f'= {100 * lev["lump_P"].mean():.1f}% of {tot}')
print(f'  explicit levels                          : {(~lev["lump_P"]).sum()} '
      f'= {100 * (~lev["lump_P"]).mean():.1f}%')
nion_tot = lev.groupby(['atomic_number', 'ion_number']).ngroups
n_sl_P = sum(min(len(g), K) + (1 if len(g) > K else 0)
             for _, g in lev.groupby(['atomic_number', 'ion_number']))
print(f'  ions = {nion_tot};  solved unknowns (SLs) = {n_sl_P} '
      f'(vs {tot} full levels; compression {tot / n_sl_P:.2f}x)')

print(f'\n=== (S) SHIPPED levels.csv super_level mapping ===')
print(f'  levels in an SL holding >1 level : {lev["lump_S"].sum()} '
      f'= {100 * lev["lump_S"].mean():.1f}%')
n_sl_S = lev.groupby(['atomic_number', 'ion_number'])['super_level'].nunique().sum()
print(f'  distinct SLs = {n_sl_S} (compression {tot / n_sl_S:.2f}x)')

# --- per-ion table ----------------------------------------------------------
rows = []
for (Z, i), g in lev.groupby(['atomic_number', 'ion_number']):
    n = len(g)
    rows.append(dict(Z=Z, ion=i, ion_name=ion_name(Z, i), n_levels=n,
                     n_SL_shipped=g['super_level'].nunique(),
                     n_lumped_shipped=int(g['lump_S'].sum()),
                     n_SL_prod=min(n, K) + (1 if n > K else 0),
                     n_lumped_prod=int(g['lump_P'].sum()),
                     frac_lumped_prod=g['lump_P'].mean()))
ion_tab = pd.DataFrame(rows)

# --- line list join ---------------------------------------------------------
ll = pd.read_csv(os.path.join(REF, 'line_list.csv'),
                 usecols=['atomic_number', 'ion_number', 'level_number_lower',
                          'level_number_upper', 'wavelength'])
print(f'\nline_list.csv rows = {len(ll)}')
ll['up_lump'] = ll['level_number_upper'] >= K
ll['lo_lump'] = ll['level_number_lower'] >= K
win = (ll['wavelength'] >= 1000) & (ll['wavelength'] <= 4000)
ll['in_win'] = win

print(f'  wavelength span {ll["wavelength"].min():.1f} - '
      f'{ll["wavelength"].max():.1f} A;  in 1000-4000 A: {win.sum()} '
      f'({100 * win.mean():.1f}%)')


def pct(a, b):
    return 100.0 * a / b if b else float('nan')


print(f'\n=== (P) lines by (lower, upper) lump status, SUPER_CUTOFF={K} ===')
for lab, sub in (('ALL', ll), ('1000-4000 A', ll[win])):
    n = len(sub)
    ee = ((~sub['lo_lump']) & (~sub['up_lump'])).sum()
    el = ((~sub['lo_lump']) & (sub['up_lump'])).sum()
    le = ((sub['lo_lump']) & (~sub['up_lump'])).sum()
    llp = ((sub['lo_lump']) & (sub['up_lump'])).sum()
    print(f'  {lab:<12s} n={n:>9d}   explicit->explicit {ee:>9d} ({pct(ee,n):5.1f}%)'
          f'   MIXED explicit->lump {el:>9d} ({pct(el,n):5.1f}%)'
          f'   MIXED lump->explicit {le:>7d} ({pct(le,n):5.1f}%)'
          f'   lump->lump {llp:>9d} ({pct(llp,n):5.1f}%)')
    print(f'  {"":12s}  upper level in a lump: {sub["up_lump"].sum():>9d} '
          f'({pct(sub["up_lump"].sum(), n):.1f}%)')

# per-ion line stats
lr = []
for (Z, i), g in ll.groupby(['atomic_number', 'ion_number']):
    gw = g[g['in_win']]
    lr.append(dict(Z=Z, ion=i, ion_name=ion_name(Z, i),
                   n_lines=len(g), n_lines_upper_lump=int(g['up_lump'].sum()),
                   frac_upper_lump=g['up_lump'].mean(),
                   n_lines_win=len(gw),
                   n_lines_win_upper_lump=int(gw['up_lump'].sum()),
                   frac_win_upper_lump=gw['up_lump'].mean() if len(gw) else np.nan,
                   n_win_mixed=int(((~gw['lo_lump']) & gw['up_lump']).sum())))
line_tab = pd.DataFrame(lr)

tab = ion_tab.merge(line_tab, on=['Z', 'ion', 'ion_name'], how='outer')
tab = tab.sort_values('n_lines_win', ascending=False)
os.makedirs(OUT, exist_ok=True)
tab.to_csv(os.path.join(OUT, 'm2_sl_lumping_by_ion.csv'), index=False)

print('\n=== per-ion (top 20 by number of lines in 1000-4000 A) ===')
print(f'{"ion":<9s} {"nlev":>6s} {"SL_ship":>8s} {"SL_prod":>8s} {"lumped%":>8s} '
      f'{"lines_win":>10s} {"up_lump%":>9s} {"mixed_win":>10s}')
for _, r in tab.head(20).iterrows():
    print(f'{r["ion_name"]:<9s} {int(r["n_levels"]):>6d} '
          f'{int(r["n_SL_shipped"]):>8d} {int(r["n_SL_prod"]):>8d} '
          f'{100 * r["frac_lumped_prod"]:>7.1f}% {int(r["n_lines_win"]):>10d} '
          f'{100 * r["frac_win_upper_lump"]:>8.1f}% {int(r["n_win_mixed"]):>10d}')
print(f'\nwrote {os.path.join(OUT, "m2_sl_lumping_by_ion.csv")}')
