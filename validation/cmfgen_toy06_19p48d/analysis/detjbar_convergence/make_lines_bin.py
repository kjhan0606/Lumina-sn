#!/usr/bin/env python3
"""Build the harness's line-forest inputs.  Binary format: int64 n, then n*4 f8
rows (lambda_A, tau_sob@s8, tau_sob@s45, tau_sob@s49), lambda-ascending.

  lines3.bin  (default)   the 793505 IN-WINDOW (1000-4000 A) lines, taken
                          verbatim from the preserved parity42 fine linedumps.
  lines4.bin  (--blue)    lines3.bin PLUS the 900-1000 A lines that production's
                          window truncates away.  Those tau_sob are not in any
                          dump (the dump only writes in-window lines), so they
                          are recomputed from the SAME parity42 state:
                             tau = SOBOLEV_COEFF*f_lu*lam_cm*t_exp*n_lo*stim
                          (lumina_plasma.c:15534 nlte_update_tau_sobolev) with
                          n_lo,n_up,g from lumina_levelpop_resolve_ema.csv and
                          f_lu,lam from the model dir's line_list.csv.
                          ROUND-TRIP GATE: the identical recipe applied to the
                          in-window 1000-1120 A lines reproduces the dumped
                          tau_sob with median ratio 1.0006 (p10/p90 0.74/1.16,
                          n=28438) -- unbiased in the median, +-25% per line.
                          Only that statistical fidelity is claimed.
"""
import pandas as pd, numpy as np, struct, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '../../../../logs/coevolve_consume_parity42/')
MODEL = os.path.join(HERE, '../../../../data/tardis_reference_toy06_19p48d_sivcaiv/')
SOBOLEV_COEFF = 2.6540281e-02          # src/lumina.h:29
T_EXP = 1683072.0                      # model config.json time_explosion_s
SHELLS = (8, 45, 49)


def in_window():
    d8 = pd.read_csv(BASE + 'cmf_fine_linedump_s8.csv', usecols=['line_id', 'lambda_A', 'tau_sob'])
    d45 = pd.read_csv(BASE + 'cmf_fine_linedump_s45.csv', usecols=['line_id', 'tau_sob'])
    d49 = pd.read_csv(BASE + 'cmf_fine_linedump_s49.csv', usecols=['line_id', 'tau_sob'])
    m = d8.merge(d45, on='line_id', suffixes=('_8', '_45')).merge(d49, on='line_id')
    return np.stack([m['lambda_A'].values, m['tau_sob_8'].values,
                     m['tau_sob_45'].values, m['tau_sob'].values], axis=1)


def tau_from_levelpop(sub, lp):
    """tau_sob[nlines, len(SHELLS)] for the given line rows, from the level pops."""
    out = np.zeros((len(sub), len(SHELLS)))
    for j, s in enumerate(SHELLS):
        key = lp[lp.shell == s].set_index(['Z', 'ion', 'level_num'])
        lo = key.reindex(pd.MultiIndex.from_arrays(
            [sub.atomic_number, sub.ion_number, sub.level_number_lower]))
        up = key.reindex(pd.MultiIndex.from_arrays(
            [sub.atomic_number, sub.ion_number, sub.level_number_upper]))
        n_lo, g_lo = lo.n_k.values, lo.g.values
        n_up, g_up = up.n_k.values, up.g.values
        ok = np.isfinite(n_lo) & np.isfinite(n_up)
        stim = np.ones(len(sub))
        good = ok & (n_lo > 0) & (n_up > 0)
        stim[good] = np.clip(1.0 - (g_lo[good] * n_up[good]) / (g_up[good] * n_lo[good]), 0.0, None)
        tau = np.zeros(len(sub))
        tau[ok] = (SOBOLEV_COEFF * sub.f_lu.values[ok] * sub.wavelength.values[ok] * 1e-8
                   * T_EXP * n_lo[ok] * stim[ok])
        out[:, j] = np.nan_to_num(tau)
    return out


def blue_extension(lam_lo=900.0, lam_hi=1000.0):
    ll = pd.read_csv(MODEL + 'line_list.csv',
                     usecols=['atomic_number', 'ion_number', 'level_number_lower',
                              'level_number_upper', 'wavelength', 'f_lu'])
    lp = pd.read_csv(BASE + 'lumina_levelpop_resolve_ema.csv',
                     usecols=['shell', 'Z', 'ion', 'level_num', 'g', 'n_k'])
    sub = ll[(ll.wavelength >= lam_lo) & (ll.wavelength < lam_hi)].reset_index(drop=True)
    tau = tau_from_levelpop(sub, lp)
    keep = tau.max(axis=1) > 1e-12
    M = np.column_stack([sub.wavelength.values[keep], tau[keep]])
    print('blue extension %.0f-%.0f A: %d lines in list, %d with tau>1e-12'
          % (lam_lo, lam_hi, len(sub), keep.sum()))
    return M


def write(M, fn):
    o = np.argsort(M[:, 0])
    M = np.ascontiguousarray(M[o]).astype('<f8')
    with open(os.path.join(HERE, fn), 'wb') as f:
        f.write(struct.pack('<q', M.shape[0]))
        M.tofile(f)
    print('wrote', fn, M.shape, 'lambda %.2f - %.2f' % (M[0, 0], M[-1, 0]))


if __name__ == '__main__':
    if '--blue' in sys.argv:
        write(np.vstack([in_window(), blue_extension()]), 'lines4.bin')
    elif '--blue2' in sys.argv:          # 800 A: is the 900 A reference converged?
        write(np.vstack([in_window(), blue_extension(800.0, 1000.0)]), 'lines5.bin')
    else:
        write(in_window(), 'lines3.bin')
