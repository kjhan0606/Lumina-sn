#!/usr/bin/env python3
"""Parse AUTOSTRUCTURE adasdr adf09 file, fit Burgess multi-term form,
emit DRCoefficient C-code stanza for LUMINA's DR_TABLE.

LUMINA evaluates α(T) = T^(-1.5) · Σ_i c_i · exp(-E_i/T)
(see dr_alpha_eval in src/lumina_plasma.c). c_i units cm³/s · K^(3/2),
E_i units K, n_terms ≤ DR_MAX_TERMS=10.
"""
import argparse
import re
import sys
import numpy as np
from scipy.optimize import nnls, minimize

ISOELECTRONIC_N = {
    'H': 1,  'HE': 2,  'LI': 3,  'BE': 4,  'B':  5,  'C':  6,
    'N': 7,  'O':  8,  'F':  9,  'NE': 10, 'NA': 11, 'MG': 12,
    'AL': 13,'SI': 14, 'P':  15, 'S':  16, 'CL': 17, 'AR': 18,
    'K':  19,'CA': 20, 'SC': 21, 'TI': 22, 'V':  23, 'CR': 24,
    'MN': 25,'FE': 26, 'CO': 27, 'NI': 28,
}

ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']


def parse_adf09(path):
    """Parse adf09 → dict with Z, ion_recomb, NPRNTI, T grid, ALFT array."""
    with open(path) as f:
        text = f.read()
    m = re.search(r"SEQ='(\w+)\s*'\s+NUCCHG=\s*(\d+)", text)
    if not m:
        raise ValueError("Cannot find SEQ/NUCCHG header")
    seq = m.group(1).upper()
    Z = int(m.group(2))
    if seq not in ISOELECTRONIC_N:
        raise ValueError(f"Unknown isoelectronic sequence '{seq}'")
    N_recomb = ISOELECTRONIC_N[seq]
    ion_recomb = Z - N_recomb + 1

    m = re.search(r"NPRNTI=\s*(\d+)\s+NPRNTF=\s*(\d+)", text)
    if not m:
        raise ValueError("Cannot find NPRNTI/NPRNTF")
    nprnti = int(m.group(1))

    # Find T(K) ALFT table
    lines = text.split('\n')
    T_arr = []
    alft = [[] for _ in range(nprnti)]
    in_table = False
    for ln in lines:
        if 'T(K)' in ln and 'ALFT' in ln:
            in_table = True
            continue
        if in_table and re.match(r'^\s*-+\s+-+', ln):
            continue
        if in_table:
            stripped = ln.strip()
            if not stripped or stripped.startswith('C') or 'CPU' in stripped:
                break
            parts = stripped.split()
            try:
                T = float(parts[0])
                vals = [float(x) for x in parts[1:1 + nprnti]]
                if len(vals) != nprnti:
                    break
                T_arr.append(T)
                for i, v in enumerate(vals):
                    alft[i].append(v)
            except (ValueError, IndexError):
                break

    return {
        'Z': Z, 'ion_recomb': ion_recomb, 'seq': seq, 'nprnti': nprnti,
        'T': np.array(T_arr),
        'alpha': [np.array(a) for a in alft],
    }


def burgess_fit_nnls(T, alpha, E_grid, weights=None):
    """Weighted NNLS: solve diag(w) K c ≈ diag(w) y, y = α·T^1.5, K[i,j]=exp(-E_j/T_i).
    Default weights = 1/y (relative-error weighting), so high-T tail is not swamped."""
    y = alpha * T ** 1.5
    K = np.exp(-np.outer(1.0 / T, E_grid))
    if weights is None:
        weights = 1.0 / np.maximum(y, 1e-300)
    Kw = K * weights[:, None]
    yw = y * weights
    c, _ = nnls(Kw, yw)
    return c


def burgess_fit_log(T, alpha, n_terms=5, T_gate=(4000, 100000), E_seed=None):
    """Fit α(T) = T^(-1.5)·Σ c_i exp(-E_i/T) by minimizing log-residual in gate region.
    Strategy: dense weighted-NNLS seed over a wide E grid → keep top n_terms by
    contribution → Nelder-Mead refinement of (log c_i, log E_i) on log-residuals."""
    mask = alpha > 0
    T_pos = T[mask]
    alpha_pos = alpha[mask]
    in_gate = (T_pos >= T_gate[0]) & (T_pos <= T_gate[1])
    refine_weights = np.where(in_gate, 10.0, 1.0)

    if E_seed is None:
        n_seed = max(4 * n_terms, 24)
        E_lo = max(T_pos.min() * 0.3, 10.0)
        E_hi = T_pos.max() * 2.0
        E_seed = np.logspace(np.log10(E_lo), np.log10(E_hi), n_seed)
    nnls_w = (1.0 / np.maximum(alpha_pos * T_pos ** 1.5, 1e-300)) * \
             np.where(in_gate, 3.0, 1.0)
    c0 = burgess_fit_nnls(T_pos, alpha_pos, E_seed, weights=nnls_w)
    nz_mask = c0 > 0
    if nz_mask.sum() == 0:
        return np.array([alpha.max() * T[np.argmax(alpha)] ** 1.5]), \
               np.array([T[np.argmax(alpha)] * 1.5])

    contrib = c0 * np.exp(-E_seed / max(T_gate[0], 1.0)) / max(T_gate[0], 1.0) ** 1.5 \
            + c0 * np.exp(-E_seed / max(T_gate[1], 1.0)) / max(T_gate[1], 1.0) ** 1.5
    contrib[~nz_mask] = 0.0
    keep_idx = np.argsort(contrib)[::-1][:n_terms]
    keep_idx = keep_idx[contrib[keep_idx] > 0]
    c0_keep = c0[keep_idx]
    E_keep = E_seed[keep_idx]

    def predict(params, T_eval):
        n = len(params) // 2
        log_c = params[:n]
        log_E = params[n:]
        c = np.exp(log_c)
        E = np.exp(log_E)
        K = np.exp(-np.outer(1.0 / T_eval, E))
        return (K @ c) / T_eval ** 1.5

    def loss(params):
        af = predict(params, T_pos)
        af = np.maximum(af, 1e-300)
        return float(np.sum(refine_weights * (np.log(af) - np.log(alpha_pos)) ** 2))

    p0 = np.concatenate([np.log(np.maximum(c0_keep, 1e-30)),
                         np.log(E_keep)])
    try:
        res = minimize(loss, p0, method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 20000})
        n = len(res.x) // 2
        c = np.exp(res.x[:n])
        E = np.exp(res.x[n:])
        order = np.argsort(E)
        return c[order], E[order]
    except Exception:
        order = np.argsort(E_keep)
        return c0_keep[order], E_keep[order]


def fit_residuals(T, alpha, c, E):
    K = np.exp(-np.outer(1.0 / T, E))
    alpha_fit = (K @ c) / T ** 1.5
    rel = np.where(alpha > 0, (alpha_fit - alpha) / alpha, np.nan)
    return alpha_fit, rel


def emit_c_entry(Z, ion_recomb, c, E, source='DR_SOURCE_AUTOSTRUCT', comment=''):
    n = len(c)
    c_str = ', '.join([f'{ci:.4e}' for ci in c])
    e_str = ', '.join([f'{ei:.4e}' for ei in E])
    return (f'    /* {comment} */\n'
            f'    {{{Z}, {ion_recomb}, {n},\n'
            f'     {{{c_str}}},\n'
            f'     {{{e_str}}},\n'
            f'     {source}}},')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('adf09', help='Path to adf09 file')
    p.add_argument('--metastable', type=int, default=1,
                   help='Initial parent index (1-based, default ground)')
    p.add_argument('--n-terms', type=int, default=5)
    p.add_argument('--gate-T-min', type=float, default=4000)
    p.add_argument('--gate-T-max', type=float, default=100000)
    p.add_argument('--gate-tol', type=float, default=0.02)
    p.add_argument('--plot', help='Write residual plot to PATH.png')
    p.add_argument('--label', default='', help='Label string for comment')
    args = p.parse_args()

    d = parse_adf09(args.adf09)
    print(f"Parsed: Z={d['Z']}, SEQ={d['seq']} → recombining "
          f"{ROMAN[d['ion_recomb'] + 1]} → {ROMAN[d['ion_recomb']]} "
          f"(LUMINA ion_recomb={d['ion_recomb']}), NPRNTI={d['nprnti']}")
    print(f"  T grid: {len(d['T'])} points, range "
          f"[{d['T'][0]:.1e}, {d['T'][-1]:.1e}] K")

    idx = args.metastable - 1
    if idx >= d['nprnti']:
        print(f"ERROR: metastable {args.metastable} exceeds NPRNTI={d['nprnti']}",
              file=sys.stderr)
        sys.exit(1)
    T = d['T']
    alpha = d['alpha'][idx]
    print(f"\nMetastable {args.metastable}: α(T) range "
          f"[{alpha.min():.2e}, {alpha.max():.2e}] cm³/s")

    c, E = burgess_fit_log(T, alpha, n_terms=args.n_terms,
                           T_gate=(args.gate_T_min, args.gate_T_max))
    print(f"\nBurgess fit ({len(c)} term{'s' if len(c) != 1 else ''}):")
    for i, (ci, ei) in enumerate(zip(c, E)):
        print(f"  c[{i}]={ci:.4e},  E[{i}]={ei:.4e} K")

    alpha_fit, rel_err = fit_residuals(T, alpha, c, E)
    gate_mask = (T >= args.gate_T_min) & (T <= args.gate_T_max)
    in_gate_err = np.abs(rel_err[gate_mask & np.isfinite(rel_err)])
    max_err = float(in_gate_err.max()) if in_gate_err.size else float('nan')
    verdict = 'PASS' if max_err < args.gate_tol else 'FAIL'

    print(f"\n=== GATE A: max |rel err| in "
          f"T∈[{args.gate_T_min:.0f}, {args.gate_T_max:.0f}] K = "
          f"{max_err * 100:.2f}% (tol {args.gate_tol * 100:.1f}%)  → {verdict} ===")

    print(f"\n{'T(K)':>12s}  {'α_data':>11s}  {'α_fit':>11s}  {'rel err':>10s}")
    for Ti, ai, afi, err in zip(T, alpha, alpha_fit, rel_err):
        flag = '*' if (args.gate_T_min <= Ti <= args.gate_T_max) else ' '
        err_str = f'{err * 100:+8.2f}%' if np.isfinite(err) else '       NA'
        print(f"  {Ti:>10.2e}  {ai:>11.3e}  {afi:>11.3e}   {err_str} {flag}")

    label = args.label if args.label else \
        f"Z={d['Z']} {d['seq']}-like recomb, AS pilot, metastable {args.metastable}"
    print('\n=== C-code stanza for DR_TABLE ===')
    print(emit_c_entry(d['Z'], d['ion_recomb'], c, E, comment=label))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.loglog(T, alpha, 'o', mfc='none', label='adf09')
        T_dense = np.logspace(np.log10(T[0]), np.log10(T[-1]), 200)
        K = np.exp(-np.outer(1.0 / T_dense, E))
        ax1.loglog(T_dense, (K @ c) / T_dense ** 1.5, '-',
                   label=f'Burgess {len(c)}-term')
        ax1.axvspan(args.gate_T_min, args.gate_T_max, alpha=0.1, color='green',
                    label='Gate A')
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel(r'$\alpha_\mathrm{DR}$ (cm$^3$/s)')
        ax1.set_title(f"Z={d['Z']}, ion_recomb={d['ion_recomb']}, "
                      f"meta {args.metastable}")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.semilogx(T, rel_err * 100, 'o-')
        ax2.axhline(args.gate_tol * 100, color='r', ls='--')
        ax2.axhline(-args.gate_tol * 100, color='r', ls='--')
        ax2.axvspan(args.gate_T_min, args.gate_T_max, alpha=0.1, color='green')
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel('relative error (%)')
        ax2.set_title(f'Residual (gate max {max_err * 100:.2f}%)')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"\nResidual plot saved: {args.plot}")

    sys.exit(0 if max_err < args.gate_tol else 2)


if __name__ == '__main__':
    main()
