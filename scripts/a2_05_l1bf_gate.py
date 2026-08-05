#!/usr/bin/env python3
"""A2-05 L-1bf gate: canonical-view bf photoionization rate vs CMFGEN *PRRR.

ORACLE_INPUT lane (deterministic, the judgment lane): CMFGEN EDDFACTOR J at the
nine gated depths -> conservative rebin to the canonical 4000-bin grid ->
deterministic commit inside the C fixture -> per-level Gamma through the SAME
shared entry point production uses (bf_rate_gamma_legacy_grid) -> population
weighting -> Gamma_ion vs PRRR truth.

Pre-registered PASS limits = the rates-certification's own registered gate
(VERDICT.md section 1): Gamma/Gamma_PRRR in [0.5, 2.0] at every gated shell AND
in [0.8, 1.25] at s6-s8; the ion x shell exclusion set is the LIVE snapshot
self-consistency bandmask (R2), not a hardcoded list.

Also produced (SPEC_A2_05_V2):
  * migration delta: Gamma_view / Gamma_legacy1000 (the certification's own
    1000-bin quadrature on the same field, sigma, populations) -- the campaign's
    first physics number for A2-05.
  * negative controls (--controls): (a) J -> W*B_nu(14172.549 K) => E_1 FAIL,
    (b) witness (Fe III, Co III) threshold one legacy bin up => E_sym FAIL,
    (c) alpha density round-trip poison => registration FAIL.  Runner exits 0
    iff every requested control shows its expected physical FAIL.
  * alpha channel: dimensional registration only (PRRR RR is a coefficient in
    cm^3 s^-1; Lumina-side coefficient comparison stays
    BLOCKED_MISSING_RATE_EXPORT until the recombination migration).

Exit codes: 0 gate verdict complete (PASS recorded, controls as expected),
1 gate FAIL or control did not fail as required, 2 input/schema failure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CERT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/rates_certification')
sys.path.insert(0, CERT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import certify_rate_machine as crm                       # noqa: E402
from certify_rate_machine import (                       # noqa: E402
    parse_popcob, parse_prrr, read_eddfactor, rvtj_block, parse_osc,
    read_bake_bin, bin_average, H, C_LIGHT, EV, FOURPI, CM2EV,
    NU_MIN, NU_MAX, NBIN, LABELS, TARGET_V, IONS,
)
from pathlib import Path as _Path                        # noqa: E402
from oracle_compare_cmfgen import parse_prrr as parse_prrr_full  # noqa: E402

# Canonical grid (radiation_field.h A2-02 authority).
CANON_N_BINS = 4000
CANON_NU_MIN = 1.4402928950097124e12
CANON_NU_MAX = 4.032418413741097e16
VALID, EXACT_ZERO, UNSAMPLED, OUT_OF_GRID, STALE = 1, 2, 3, 4, 5
STATE_NAME = {1: 'VALID', 2: 'EXACT_ZERO', 3: 'UNSAMPLED', 4: 'OUT_OF_GRID', 5: 'STALE'}

# Pre-registered limits (rates certification VERDICT.md section 1).
LIMIT_ALL = (0.5, 2.0)
LIMIT_FORMING = (0.8, 1.25)
FORMING = {'s6', 's7', 's8'}
# Negative-control pre-registration (SPEC_A2_05_V2 gate contract 5).
PLANCK_T = 14172.549
PLANCK_W = 0.5
ESYM_MIN = 0.005          # threshold one-bin shift must move Gamma_ion(s0) by >0.5%
# E_1 FAIL threshold for the Planck-substitute poison.  E_1 saturates at 1
# from below when the poison KILLS the rate (|0-G|/G = 1), so a >1.0 limit
# only detects overshoots.  0.5 = 3.7x the worst certified main-lane E_1
# (Co III 0.134) and is reached by any dead or wildly re-colored field.
E1_FAIL_LIMIT = 0.5
WITNESS = {'Fe III', 'Co III'}


def canonical_edges() -> np.ndarray:
    edges = np.geomspace(CANON_NU_MIN, CANON_NU_MAX, CANON_N_BINS + 1)
    edges[0], edges[-1] = CANON_NU_MIN, CANON_NU_MAX
    return edges


def conservative_rebin(nu: np.ndarray, y: np.ndarray,
                       edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid-conservative bin average + validity (A2-04 replay semantics:
    bins outside the native support are OUT_OF_GRID, never zero-VALID)."""
    dx = np.diff(nu)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)))

    def primitive(q):
        i = np.clip(np.searchsorted(nu, q, side='right') - 1, 0, nu.size - 2)
        d = q - nu[i]
        slope = (y[i + 1] - y[i]) / (nu[i + 1] - nu[i])
        return cum[i] + y[i] * d + 0.5 * slope * d * d

    inside = (edges[:-1] >= nu[0]) & (edges[1:] <= nu[-1])
    avg = np.zeros(edges.size - 1)
    avg[inside] = (primitive(edges[1:][inside]) - primitive(edges[:-1][inside])) \
        / np.diff(edges)[inside]
    state = np.full(edges.size - 1, OUT_OF_GRID, dtype=np.int32)
    state[inside & (avg > 0.0)] = VALID
    state[inside & (avg == 0.0)] = EXACT_ZERO
    if np.any(avg < 0.0):
        raise SystemExit('conservative rebin produced negative J')
    return avg, state


def planck_bnu(T: float, nu: np.ndarray) -> np.ndarray:
    KB = 1.380649e-16
    x = H * nu / (KB * T)
    out = np.zeros_like(nu)
    ok = x < 700.0
    out[ok] = 2.0 * H * nu[ok] ** 3 / C_LIGHT ** 2 / np.expm1(x[ok])
    return out


def run_fixture(fixture: str, tag: str, out_dir: str, generation: int,
                epoch: float, Jrows: np.ndarray, states: np.ndarray,
                nfb: int, nu_min: float, dln: float,
                sig_rows: np.ndarray, nu_th: np.ndarray) -> dict:
    """Write the A205IN01 input, run the C fixture, parse GAMMA lines."""
    n_shells = Jrows.shape[0]
    path = os.path.join(out_dir, f'l1bf_input_{tag}.bin')
    with open(path, 'wb') as f:
        f.write(b'A205IN01')
        f.write(struct.pack('<QQQd', 1, n_shells, generation, epoch))
        v = 1.0e8 + 1.0e7 * np.arange(n_shells + 1)
        f.write(v[:-1].astype('<f8').tobytes())          # v_inner
        f.write(v[1:].astype('<f8').tobytes())           # v_outer
        f.write(np.ascontiguousarray(Jrows, dtype='<f8').tobytes())
        f.write(np.ascontiguousarray(states, dtype='<i4').tobytes())
        f.write(struct.pack('<Qdd', nfb, nu_min, dln))
        f.write(struct.pack('<Q', sig_rows.shape[0]))
        for k in range(sig_rows.shape[0]):
            f.write(struct.pack('<d', float(nu_th[k])))
            f.write(np.ascontiguousarray(sig_rows[k], dtype='<f8').tobytes())
    proc = subprocess.run([fixture, path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'fixture rc={proc.returncode}: {proc.stderr[:400]}')
    n_lev = sig_rows.shape[0]
    gamma = np.zeros((n_lev, n_shells))
    state = np.zeros((n_lev, n_shells), dtype=np.int32)
    wmiss = np.zeros((n_lev, n_shells))
    done = False
    for line in proc.stdout.splitlines():
        if line.startswith('GAMMA '):
            t = line.split()
            lev, sh, st = int(t[1]), int(t[2]), int(t[3])
            gamma[lev, sh] = float(t[4])
            state[lev, sh] = st
            wmiss[lev, sh] = float(t[5])
        elif line.strip() == 'A2_05_L1BF_FIXTURE DONE':
            done = True
    if not done:
        raise SystemExit('fixture output truncated (no DONE marker)')
    return {'gamma': gamma, 'state': state, 'wmiss': wmiss}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4')
    ap.add_argument('--ref', default=os.path.join(
        ROOT, 'data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv'))
    ap.add_argument('--bin', default=os.path.join(
        ROOT, 'data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin'))
    ap.add_argument('--fixture', default=os.path.join(ROOT, 'l1bf_fixture'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'validation/a2_05'))
    ap.add_argument('--controls', action='store_true',
                    help='also run the three pre-registered negative controls')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    report = []

    def say(s=''):
        print(s, flush=True)
        report.append(s)

    say('A2-05 L-1bf GATE (ORACLE_INPUT lane, deterministic)')
    say(f'  run={a.run}')
    say(f'  fixture={a.fixture}')

    # ---- field ----------------------------------------------------------
    J, nu, ND, finish = read_eddfactor(f'{a.run}/EDDFACTOR')
    if finish != 1.0:
        raise SystemExit('EDDFACTOR FINISH_REC != 1')
    rt = open(f'{a.run}/RVTJ').read()
    V = rvtj_block(rt, 'Velocity (km/s)', ND)
    GATE = [int(np.argmin(np.abs(V - vt))) for vt in TARGET_V]
    NG = len(GATE)
    say(f'  shells: ' + ' '.join(f'{lb}:d{d+1}(v={V[d]:.0f})'
                                 for lb, d in zip(LABELS, GATE)))

    edges = canonical_edges()
    Jrows = np.zeros((NG, CANON_N_BINS))
    Srows = np.zeros((NG, CANON_N_BINS), dtype=np.int32)
    for q, d in enumerate(GATE):
        Jrows[q], Srows[q] = conservative_rebin(nu, J[:, d], edges)
    say(f'  canonical rebin: {CANON_N_BINS} bins x {NG} shells; '
        f'validity counts VALID={int((Srows==VALID).sum())} '
        f'EXACT_ZERO={int((Srows==EXACT_ZERO).sum())} '
        f'OUT_OF_GRID={int((Srows==OUT_OF_GRID).sum())} ({time.time()-t0:.1f}s)')

    # legacy 1000-bin J for the migration-delta channel (certification formula)
    dln_leg = math.log(NU_MAX / NU_MIN) / NBIN
    edges_leg = NU_MIN * np.exp(np.arange(NBIN + 1) * dln_leg)
    nu_c = NU_MIN * np.exp((np.arange(NBIN) + 0.5) * dln_leg)
    dnu = np.diff(edges_leg)
    Wbin = FOURPI / (H * nu_c) * dnu
    Jbar = np.zeros((NBIN, NG))
    for q, d in enumerate(GATE):
        Jbar[:, q] = bin_average(nu, J[:, d], edges_leg)

    # ---- shipped sigma binary + ref rows --------------------------------
    bk = read_bake_bin(a.bin)
    ref_Z, ref_i, ref_E, ref_g = [], [], [], []
    import csv as _csv
    with open(os.path.join(a.ref, 'levels.csv')) as f:
        for r in _csv.DictReader(f):
            ref_Z.append(int(r['atomic_number']))
            ref_i.append(int(r['ion_number']))
            ref_E.append(float(r['energy_eV']))
            ref_g.append(float(r['g']))
    ref_Z, ref_i = np.array(ref_Z), np.array(ref_i)
    ref_E, ref_g = np.array(ref_E), np.array(ref_g)

    # ---- per-ion assembly ------------------------------------------------
    ions = []
    popcache = {}
    for spec in IONS:
        pop = spec['pop']
        if pop not in popcache:
            popcache[pop] = parse_popcob(f'{a.run}/{pop}')
        _, allions, order, nd2 = popcache[pop]
        assert nd2 == ND
        CI = allions[spec['cmf']][0]
        NF = CI.shape[1]
        n_ion = CI.sum(axis=1)
        osc = parse_osc(os.path.join(a.run, spec['osc']))
        E_eV = np.asarray(osc.levels['E_cm'][:NF], float) * CM2EV
        gl = np.asarray(osc.levels['g'][:NF], float)
        nu_th = (osc.ionization_eV - E_eV) * EV / H
        # PRRR truth + live snapshot bandmask (R2)
        from certify_rate_machine import parse_f_to_s
        SL, _ = parse_f_to_s(os.path.join(a.run, spec['fts']), osc.n_levels)
        NS = int(SL[:NF].max())
        PR, DI = parse_prrr(os.path.join(a.run, f'{spec["cmf"]}PRRR'),
                            spec['cmf'], ND, NS)
        G_truth = PR.sum(axis=0) / np.where(n_ion > 0, n_ion, np.nan)
        ki = order.index(spec['cmf'])
        excluded = set()
        if ki + 1 < len(order):
            gnext = allions[order[ki + 1]][0][:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                dir_ = np.where(gnext > 0, DI / gnext, np.nan)
            scale = float(np.nanmedian(dir_[GATE]))
            devi = dir_ / scale
            for q, d in enumerate(GATE):
                if np.isfinite(devi[d]) and abs(devi[d] - 1.0) > 0.05:
                    excluded.add(LABELS[q])
        # shipped sigma rows (D mapping; certified 1:1)
        sel = np.where((ref_Z == spec['Z']) & (ref_i == spec['ion']))[0]
        ident = (sel.size >= NF and np.max(np.abs(ref_E[sel][:NF] - E_eV)) < 1e-6
                 and np.array_equal(ref_g[sel][:NF], gl))
        if not ident:
            raise SystemExit(f'{spec["lab"]}: ref row identity failed '
                             '(certified 1:1 no longer holds)')
        S_D = np.asarray(bk['sigma'][sel[:NF]], dtype=float)
        ions.append(dict(spec=spec, NF=NF, NS=NS, CI=CI, n_ion=n_ion,
                         nu_th=nu_th, S_D=S_D, G_truth=G_truth,
                         excluded=excluded))
        say(f'  [{spec["lab"]}] NF={NF} sigma rows shipped; '
            f'bandmask excluded shells: {sorted(excluded) or "NONE"}')

    all_sigma = np.concatenate([io['S_D'] for io in ions], axis=0)
    all_nuth = np.concatenate([io['nu_th'] for io in ions])
    offsets = np.cumsum([0] + [io['NF'] for io in ions])

    def ion_gamma(fix: dict) -> list[np.ndarray]:
        """Population-weighted Gamma_ion per gated shell from fixture output."""
        out = []
        for j, io in enumerate(ions):
            g = fix['gamma'][offsets[j]:offsets[j + 1]]         # [NF, NG]
            st = fix['state'][offsets[j]:offsets[j + 1]]
            usable = (st == VALID) | (st == EXACT_ZERO)
            gi = np.zeros(NG)
            blocked_share = np.zeros(NG)
            for q, d in enumerate(GATE):
                p = io['CI'][d] / io['n_ion'][d]
                gi[q] = float((p * np.where(usable[:, q], g[:, q], 0.0)).sum())
                blocked_share[q] = float(p[~usable[:, q]].sum())
            io['blocked_share'] = blocked_share
            out.append(gi)
        return out

    # ---- main lane -------------------------------------------------------
    epoch = 19.48 * 86400.0
    fix = run_fixture(a.fixture, 'main', a.out, 1, epoch, Jrows, Srows,
                      NBIN, NU_MIN, dln_leg, all_sigma, all_nuth)
    G_view = ion_gamma(fix)
    say(f'  fixture main lane done ({time.time()-t0:.1f}s)')

    verdict_rows = []
    all_pass = True
    say()
    say('  --- ORACLE_INPUT verdict: Gamma_view / Gamma_PRRR '
        f'(limits {LIMIT_ALL} all, {LIMIT_FORMING} s6-s8; bandmask live) ---')
    for j, io in enumerate(ions):
        lab = io['spec']['lab']
        Gt = io['G_truth'][GATE]
        Gl = np.array([(io['S_D'] * Wbin[None, :] @ Jbar[:, q]
                        * (io['CI'][GATE[q]] / io['n_ion'][GATE[q]])).sum()
                       for q in range(NG)])
        ratios, cells = [], []
        for q, lb in enumerate(LABELS):
            r = G_view[j][q] / Gt[q] if Gt[q] > 0 else float('nan')
            rl = G_view[j][q] / Gl[q] if Gl[q] > 0 else float('nan')
            if lb in io['excluded']:
                cells.append(f'{lb}=EXCL({r:.3f})')
                verdict_rows.append(dict(ion=lab, shell=lb, ratio_prrr=r,
                                         ratio_legacy=rl, ok=None,
                                         excluded=True,
                                         gamma_view=float(G_view[j][q]),
                                         gamma_truth=float(Gt[q])))
                continue
            lo, hi = LIMIT_FORMING if lb in FORMING else LIMIT_ALL
            ok = lo <= r <= hi
            all_pass &= ok
            cells.append(f'{lb}={r:.3f}{"" if ok else "!FAIL"}')
            ratios.append(r)
            verdict_rows.append(dict(ion=lab, shell=lb, ratio_prrr=r,
                                     ratio_legacy=rl, ok=bool(ok),
                                     gamma_view=float(G_view[j][q]),
                                     gamma_truth=float(Gt[q])))
        delta = [G_view[j][q] / Gl[q] if Gl[q] > 0 else float('nan')
                 for q in range(NG)]
        # E_1 (A2-04 definition carried to rate space): weighted L1 relative
        # error over the judged (non-excluded) gated cells, weights = 1.
        jm = [q for q, lb in enumerate(LABELS)
              if lb not in io['excluded'] and Gt[q] > 0]
        e1 = float(sum(abs(G_view[j][q] - Gt[q]) for q in jm) /
                   sum(Gt[q] for q in jm))
        io['E1_main'] = e1
        # f_cov (gate contract 3 / ORDER 6.3): the active set and the
        # denominator are built from the TRUTH-side contribution -- the
        # certification's own 1000-bin CMFGEN-field quadrature p*Gamma_lev,
        # with no view state anywhere in the construction (a state-filtered
        # denominator made f_cov=1 tautological; 1st re-review finding).
        st_lev = fix['state'][offsets[j]:offsets[j + 1]]
        truth_contrib = np.zeros((io['NF'], NG))
        for q, d in enumerate(GATE):
            p_lev = io['CI'][d] / io['n_ion'][d]
            truth_contrib[:, q] = p_lev * (io['S_D'] * Wbin[None, :]
                                           @ Jbar[:, q])
        io['truth_contrib'] = truth_contrib
        fcov_min = 1.0
        fcov_by_shell = []
        for q in range(NG):
            contrib = truth_contrib[:, q]
            total = contrib.sum()
            if not total > 0:
                fcov_by_shell.append(float('nan'))
                continue
            order_c = np.argsort(contrib)[::-1]
            csum = np.cumsum(contrib[order_c])
            n_active = int(np.searchsorted(csum, 0.999 * total) + 1)
            active = order_c[:n_active]
            usable = (st_lev[active, q] == VALID) | \
                (st_lev[active, q] == EXACT_ZERO)
            fcov = float(contrib[active][usable].sum() /
                         contrib[active].sum())
            fcov_by_shell.append(fcov)
            fcov_min = min(fcov_min, fcov)
        io['fcov_min'] = fcov_min
        io['fcov_by_shell'] = fcov_by_shell
        say(f'  {lab:7s} vs PRRR : ' + ' '.join(cells)
            + f'  | E_1={e1:.4f} f_cov(99.9% active,min)={fcov_min:.4f}')
        say(f'  {lab:7s} view/legacy1000 (migration delta): '
            + ' '.join(f'{LABELS[q]}={delta[q]:.4f}' for q in range(NG)))
        bs = io['blocked_share']
        if bs.max() > 0:
            say(f'  {lab:7s} blocked population share (R6, excluded from the '
                'numerator): ' + ' '.join(f'{LABELS[q]}={bs[q]:.2e}'
                                          for q in range(NG) if bs[q] > 0))
    say()
    say(f'  ORACLE_INPUT lane: deterministic commit => Poisson CI = 0 for every '
        'term; ORDER 6.3 CI qualification is trivially met (recorded, not waived).')
    say(f'  GATE verdict: {"PASS" if all_pass else "FAIL"}')

    control_results = {}
    if a.controls:
        say()
        say('  --- negative controls (each must FAIL physically; runner rc=0 '
            'iff observed) ---')
        # (a) dilute-Planck substitute field -> E_1
        nu_cc = np.sqrt(edges[:-1] * edges[1:])
        Jp = np.tile(PLANCK_W * planck_bnu(PLANCK_T, nu_cc), (NG, 1))
        Sp = np.where(Jp > 0, VALID, EXACT_ZERO).astype(np.int32)
        fa = run_fixture(a.fixture, 'ctl_planck', a.out, 1, epoch, Jp, Sp,
                         NBIN, NU_MIN, dln_leg, all_sigma, all_nuth)
        Gp = ion_gamma(fa)
        broke = 0
        e1_poison = {}
        for j, io in enumerate(ions):
            Gt = io['G_truth'][GATE]
            jm = [q for q, lb in enumerate(LABELS)
                  if lb not in io['excluded'] and Gt[q] > 0]
            e1_poison[io['spec']['lab']] = float(
                sum(abs(Gp[j][q] - Gt[q]) for q in jm) / sum(Gt[q] for q in jm))
            for q, lb in enumerate(LABELS):
                if lb in io['excluded'] or not Gt[q] > 0:
                    continue
                r = Gp[j][q] / Gt[q]
                lo, hi = LIMIT_FORMING if lb in FORMING else LIMIT_ALL
                if not (lo <= r <= hi):
                    broke += 1
        # pre-registered FAIL: weighted E_1 > E1_FAIL_LIMIT for every ion
        # (vs certified main-lane E_1 <= 0.134) AND at least one gated cell
        # out of ratio limit.
        e1_fail = all(v > E1_FAIL_LIMIT for v in e1_poison.values())
        obs_a = e1_fail and broke > 0
        control_results['planck_E1'] = dict(
            E1_poison={k: round(v, 3) for k, v in e1_poison.items()},
            E1_main={io['spec']['lab']: round(io['E1_main'], 4) for io in ions},
            E1_fail_limit=E1_FAIL_LIMIT, cells_out_of_limit=broke, observed_fail=obs_a)
        say(f'  (a) W*B_nu({PLANCK_T}) injection: E_1 = '
            + ' '.join(f'{k}:{v:.1f}' for k, v in e1_poison.items())
            + f' (main lane E_1 <= {max(io["E1_main"] for io in ions):.3f}; '
            + f'limit {E1_FAIL_LIMIT}), {broke} cells out of ratio limit -> '
            + ('FAIL observed (control PASS)' if obs_a else 'NO FAIL (control BROKEN)'))
        # (b) threshold one-bin shift on the witness ions -> E_sym
        nuth_shift = all_nuth.copy()
        for j, io in enumerate(ions):
            if io['spec']['lab'] in WITNESS:
                nuth_shift[offsets[j]:offsets[j + 1]] *= math.exp(dln_leg)
        fb = run_fixture(a.fixture, 'ctl_thresh', a.out, 1, epoch, Jrows, Srows,
                         NBIN, NU_MIN, dln_leg, all_sigma, nuth_shift)
        Gs = ion_gamma(fb)
        esym = {}
        for j, io in enumerate(ions):
            if io['spec']['lab'] in WITNESS and G_view[j][0] > 0:
                ga, gb = Gs[j][0], G_view[j][0]
                esym[io['spec']['lab']] = 2.0 * abs(ga - gb) / (abs(ga) + abs(gb))
        moved = {k: v for k, v in esym.items() if v > ESYM_MIN}
        control_results['threshold_Esym'] = dict(
            esym={k: float(v) for k, v in esym.items()},
            limit=ESYM_MIN, observed_fail=len(moved) == len(WITNESS))
        say(f'  (b) threshold +1 legacy bin (witness {sorted(WITNESS)}): '
            + ' '.join(f'{k}: dGamma/Gamma(s0)={v:.4f}' for k, v in esym.items())
            + f' (must each exceed {ESYM_MIN}) -> '
            + ('FAIL observed (control PASS)' if len(moved) == len(WITNESS)
               else 'NO FAIL (control BROKEN)'))
        # (c) alpha density round-trip poison (R7 registration)
        # registration identity on REAL PRRR data (R7): the file's RR
        # coefficient alpha [cm^3 s^-1] against its own ion/electron
        # densities.  rate_density = alpha*n_e*n_ion; recovering alpha
        # divides by (n_e*n_ion) exactly once; multiplying one extra
        # density is the pre-registered poison.
        io0 = ions[0]
        pr_full = parse_prrr_full(
            _Path(a.run) / f"{io0['spec']['cmf']}PRRR", io0['spec']['cmf'],
            ND, int(io0.get('NS', 0)) or 52)
        d0 = GATE[0]
        alpha = float(pr_full['alpha'][d0])
        ne_t = float(pr_full['electron_density'][d0])
        ni_t = float(pr_full['ion_density'][d0])
        if not (alpha > 0 and ne_t > 0 and ni_t > 0):
            raise SystemExit('alpha channel: non-positive PRRR data at s0')
        rate_density = alpha * ne_t * ni_t
        rt_alpha = rate_density / (ne_t * ni_t)
        poisoned = rate_density * ne_t        # density multiplied ONCE more
        rt_poison = poisoned / (ne_t * ni_t)
        ident_ok = abs(rt_alpha / alpha - 1.0) < 1e-12
        poison_fails = abs(rt_poison / alpha - 1.0) > 1e3
        control_results['alpha_registration'] = dict(
            identity_ok=bool(ident_ok), poison_detected=bool(poison_fails),
            ion=io0['spec']['lab'], alpha_cm3_s=alpha,
            n_e_cm3=ne_t, n_ion_cm3=ni_t, depth_1based=int(d0 + 1),
            note='Lumina-side alpha coefficient comparison '
                 'BLOCKED_MISSING_RATE_EXPORT until recombination migration')
        say(f'  (c) alpha registration ({io0["spec"]["lab"]} PRRR s0: '
            f'alpha={alpha:.3e} cm^3/s, n_e={ne_t:.3e}, n_ion={ni_t:.3e}): '
            f'round-trip identity {"OK" if ident_ok else "BROKEN"}; '
            f'extra density multiply detected: {poison_fails} '
            '-> Lumina alpha comparison BLOCKED_MISSING_RATE_EXPORT (registered)')
        alpha_reg_ok = ident_ok and poison_fails
        controls_ok = (broke > 0 and len(moved) == len(WITNESS) and alpha_reg_ok)
    else:
        controls_ok = True

    np.savez_compressed(
        os.path.join(a.out, 'oracle_truth_contrib.npz'),
        offsets=offsets,
        ions=np.array([io['spec']['lab'] for io in ions]),
        truth_contrib=np.concatenate(
            [io['truth_contrib'] for io in ions], axis=0),
        gamma_view=fix['gamma'], state=fix['state'])

    ledger = dict(
        schema='lumina-a2-05-l1bf-gate-v2',
        lane='ORACLE_INPUT',
        run=a.run, bin=a.bin,
        limits=dict(all=LIMIT_ALL, forming=sorted(FORMING),
                    forming_limit=LIMIT_FORMING),
        bandmask={io['spec']['lab']: sorted(io['excluded']) for io in ions},
        E1_main={io['spec']['lab']: round(io.get('E1_main', float('nan')), 5)
                 for io in ions},
        fcov_min={io['spec']['lab']: round(io.get('fcov_min', float('nan')), 5)
                  for io in ions},
        verdict='PASS' if all_pass else 'FAIL',
        rows=verdict_rows,
        controls=control_results,
        elapsed_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(a.out, 'L1BF_GATE_LEDGER.json'), 'w') as f:
        json.dump(ledger, f, indent=2)
    with open(os.path.join(a.out, 'L1BF_GATE_REPORT.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    say(f'\n[done] {time.time()-t0:.1f}s -> {a.out}/L1BF_GATE_LEDGER.json')
    return 0 if (all_pass and controls_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
