"""detjbar harness statistics.

  python3 stats.py detjbar_TAG.csv [...]        LAG-vs-REF (convergence probe)
                                                + EXACT-vs-REF whenever the run
                                                  carried --exact (discretisation)
  python3 stats.py --pair NARROW.csv WIDE.csv   narrow-vs-wide window truncation.
                                                Both files must come from --outsel 1
                                                runs, so line_idx is the
                                                run-invariant key.
"""
import pandas as pd, numpy as np, sys

HDR = (f"{'group':<18s}{'n':>8s}{'median':>10s}{'p10':>10s}{'p90':>10s}"
       f"{'|max|':>9s}{'frac<0':>8s}{'mean':>10s}")


def _row(a, lbl):
    a = np.asarray(a)
    if len(a) == 0:
        return
    print(f"{lbl:<18s}{len(a):>8d}{np.median(a):>+10.4f}{np.percentile(a,10):>+10.4f}"
          f"{np.percentile(a,90):>+10.4f}{np.max(np.abs(a)):>9.3f}{np.mean(a<0):>8.3f}{np.mean(a):>+10.4f}")


def _breakdown(d, err, title):
    v = np.asarray(err)
    print(f"=== {title}   (n={len(d)}) ===")
    print(HDR)
    _row(v, 'ALL')
    for lo, hi in [(1000, 1120), (1000, 1300), (1300, 2000), (2000, 4000)]:
        _row(v[((d.lambda_A >= lo) & (d.lambda_A < hi)).values], f'{lo}-{hi}A')
    for lo, hi in [(0, 10), (10, 30), (30, 50)]:
        _row(v[((d.shell >= lo) & (d.shell < hi)).values], f'shell {lo}-{hi}')
    for lo, hi in [(1000, 1400), (2000, 4000)]:
        for slo, shi in [(0, 10), (10, 30), (30, 50)]:
            m = ((d.lambda_A >= lo) & (d.lambda_A < hi) &
                 (d.shell >= slo) & (d.shell < shi)).values
            _row(v[m], f'{lo}-{hi} s{slo}-{shi}')
    _row(v[(d.tau_sob > 0.01).values], 'tau>0.01')
    _row(v[(d.tau_sob > 1.0).values], 'tau>1')
    print()


def report(fn, label=None):
    d = pd.read_csv(fn)
    d = d[d.jbar_ref > 0].reset_index(drop=True)
    if 'jbar_lag' in d and d.jbar_lag.max() > 0:
        _breakdown(d, (d.jbar_lag - d.jbar_ref) / d.jbar_ref,
                   f'{label or fn}  LAG-vs-REF (ALI convergence)')
    if 'jbar_exact' in d and d.jbar_exact.max() > 0:
        _breakdown(d, (d.jbar_exact - d.jbar_ref) / d.jbar_ref,
                   f'{label or fn}  EXACT-vs-REF (frequency-advection discretisation)')
    if 'jbar_lagconv' in d and d.jbar_lagconv.max() > 0:
        e = (d.jbar_lagconv - d.jbar_ref) / d.jbar_ref
        print(f"  [gate] LAG-converged vs REF: max|rel|={np.abs(e).max():.3e}\n")


def pair(fn_narrow, fn_wide):
    """(narrow - wide)/wide: what production's 1000 A blue cut does to jbar."""
    a = pd.read_csv(fn_narrow)
    b = pd.read_csv(fn_wide)
    m = a.merge(b, on=['line_idx', 'shell'], suffixes=('_n', '_w'))
    m = m[(m.jbar_ref_n > 0) & (m.jbar_ref_w > 0)].reset_index(drop=True)
    m['lambda_A'] = m.lambda_A_n
    m['tau_sob'] = m.tau_sob_n
    _breakdown(m, (m.jbar_ref_n - m.jbar_ref_w) / m.jbar_ref_w,
               f'{fn_narrow} vs {fn_wide}  NARROW-vs-WIDE (window truncation)')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--pair':
        pair(sys.argv[2], sys.argv[3])
    else:
        for fn in sys.argv[1:]:
            report(fn)
