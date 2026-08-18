#!/usr/bin/env python3
"""M2b: tau-weighted lumped-upper share, from parity42's own fine line dumps."""
import os
import numpy as np
import pandas as pd

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
REF = os.path.join(ROOT, 'data/tardis_reference_toy06_19p48d_sivcaiv')
RUN = os.path.join(ROOT, 'logs/coevolve_consume_parity42')
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')
K = 100

ll = pd.read_csv(os.path.join(REF, 'line_list.csv'),
                 usecols=['atomic_number', 'ion_number', 'level_number_lower',
                          'level_number_upper', 'line_id']).set_index('line_id')
rows = []
for sh in (8, 45, 49):
    d = pd.read_csv(os.path.join(RUN, f'cmf_fine_linedump_s{sh}.csv'),
                    usecols=['line_id', 'lambda_A', 'tau_sob'])
    j = d.join(ll, on='line_id')
    j['up_lump'] = j['level_number_upper'] >= K
    j['lo_lump'] = j['level_number_lower'] >= K
    blk_all = 1 - np.exp(-j['tau_sob'])
    for lab, m in (('all', np.ones(len(j), bool)),
                   ('tau>1e-5 (formal cut)', j['tau_sob'] > 1e-5),
                   ('tau>0.1', j['tau_sob'] > 0.1),
                   ('tau>1', j['tau_sob'] > 1)):
        s = j[m]
        n = len(s)
        blk = 1 - np.exp(-s['tau_sob'])
        rows.append(dict(
            shell=sh, cut=lab, n_lines=n,
            frac_upper_lumped=s['up_lump'].mean() if n else np.nan,
            frac_mixed_explicit_to_lump=((~s['lo_lump']) & s['up_lump']).mean()
            if n else np.nan,
            blocking_share_upper_lumped=(blk[s['up_lump'].values].sum() / blk.sum())
            if n and blk.sum() > 0 else np.nan))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'm2b_tau_weighted_lumping.csv'),
                          index=False)
print(pd.DataFrame(rows).to_string(index=False))
