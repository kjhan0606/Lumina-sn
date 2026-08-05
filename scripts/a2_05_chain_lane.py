#!/usr/bin/env python3
"""A2-05 L-1bf CHAIN lane: fixed-seed MC segment capture -> production MC
commit -> canonical view -> per-level Gamma with ORDER 6.3 CI qualification.

Mechanism lane (SPEC_A2_05_V2 gate contract 1): the full MC -> commit -> view
-> rate chain runs on the A2-02C fixed-seed raw segment capture (generation 2).
Judgment against PRRR stays with the deterministic ORACLE_INPUT lane; here the
gated observables are (i) every judged term carries an honest validity state,
(ii) Poisson CIs are computed and each (ion, shell) is either CI-QUALIFIED
(1.96*sigma/Gamma <= ln(1.25)/3) or reported UNDERPOWERED, (iii) no legacy
fallback exists anywhere in the chain.  Gamma_chain/Gamma_oracle is recorded
(NOT gated -- it measures J-transport, which is physics, not machinery).

Heavy scan (55 GB capture): run on lageunha, NOT a login node.
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
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CERT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/rates_certification')
sys.path.insert(0, CERT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from a2_02c_segment_replay import (            # noqa: E402
    RECORD_DTYPE, HEADER_BYTES, capture_layout, _accumulate_hist_chunk,
)
from certify_rate_machine import (             # noqa: E402
    parse_popcob, parse_prrr, parse_f_to_s, parse_osc, read_bake_bin,
    EV, H, CM2EV, LABELS, TARGET_V, IONS,
)

CANON_N_BINS = 4000
CANON_NU_MIN = 1.4402928950097124e12
CANON_NU_MAX = 4.032418413741097e16
VALID, EXACT_ZERO = 1, 2
CI_REL_LIMIT = math.log(1.25) / 3.0     # ORDER 6.3: half-width <= limit/3
CHUNK = 2_000_000

_G = {}


def canonical_edges() -> np.ndarray:
    edges = np.geomspace(CANON_NU_MIN, CANON_NU_MAX, CANON_N_BINS + 1)
    edges[0], edges[-1] = CANON_NU_MIN, CANON_NU_MAX
    return edges


def count_hist_chunk(counts: np.ndarray, edges: np.ndarray,
                     records: np.ndarray, slots: np.ndarray) -> None:
    """Per-bin contribution counts: +1 for every bin a segment overlaps."""
    if records.size == 0:
        return
    nbin = edges.size - 1
    n0 = np.asarray(records["nu0"]); n1 = np.asarray(records["nu1"])
    low = np.minimum(n0, n1); high = np.maximum(n0, n1)
    first = np.searchsorted(edges, low, side="right") - 1
    last = np.searchsorted(edges, high, side="left") - 1
    same = n0 == n1
    last = np.where(same, first, last)
    keep = (last >= 0) & (first < nbin)
    first = np.clip(first[keep], 0, nbin - 1)
    last = np.clip(last[keep], 0, nbin - 1)
    slots = slots[keep]
    n = last - first + 1
    repeated = np.repeat(np.arange(first.size), n)
    offs = np.arange(repeated.size) - np.repeat(np.cumsum(n) - n, n)
    np.add.at(counts, (slots[repeated], np.repeat(first, n) + offs), 1)


def scan_range(task):
    start, stop = task
    records = np.memmap(_G['capture'], dtype=RECORD_DTYPE, mode='r',
                        offset=_G['offset'], shape=(_G['n_records'],))
    edges = _G['edges']
    shell_array = _G['shells']
    measure = np.zeros((shell_array.size, CANON_N_BINS))
    counts = np.zeros((shell_array.size, CANON_N_BINS), dtype=np.int64)
    for s0 in range(start, stop, CHUNK):
        block = np.asarray(records[s0:min(s0 + CHUNK, stop)])
        slot = np.searchsorted(shell_array, block["shell"])
        ok = (slot < shell_array.size) & \
             (shell_array[np.minimum(slot, shell_array.size - 1)] == block["shell"])
        block = block[ok]; slot = slot[ok]
        _accumulate_hist_chunk(measure, edges, block, slot)
        count_hist_chunk(counts, edges, block, slot)
    return measure, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', default='/gpfs/kjhan/lumina_runner2/scratch/'
                    'a2_02c/a2_02c_segments_g2_2P2400000.bin')
    ap.add_argument('--deck', default=os.path.join(
        ROOT, 'data/tardis_reference_toy06_19p48d'))
    ap.add_argument('--run', default='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4')
    ap.add_argument('--ref', default=os.path.join(
        ROOT, 'data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv'))
    ap.add_argument('--bin', default=os.path.join(
        ROOT, 'data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin'))
    ap.add_argument('--fixture', default=os.path.join(ROOT, 'l1bf_fixture'))
    ap.add_argument('--oracle-ledger', default=os.path.join(
        ROOT, 'validation/a2_05/L1BF_GATE_LEDGER.json'))
    ap.add_argument('--truth-npz', default=os.path.join(
        ROOT, 'validation/a2_05/oracle_truth_contrib.npz'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'validation/a2_05'))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    report = []

    def say(s=''):
        print(s, flush=True)
        report.append(s)

    say('A2-05 L-1bf CHAIN lane (fixed-seed MC capture -> production commit)')
    say(f'  capture={a.capture}')
    header, _ = capture_layout(Path(a.capture))
    n_shells_all = int(header['n_shells'])
    delta_t = float(header['delta_t_s'])
    generation = int(header['generation'])
    n_records = int(header['segment_count'])
    epoch_capture = float(header['time_explosion_s'])
    say(f'  shells={n_shells_all} generation={generation} delta_t={delta_t:.6e}s '
        f'records={n_records}')

    import csv
    with open(os.path.join(a.deck, 'geometry.csv'), newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['shell_id']))
    v_mid = np.asarray([0.5e-5 * (float(r['v_inner']) + float(r['v_outer']))
                        for r in rows])
    if v_mid.size != n_shells_all:
        raise SystemExit(f'deck shells {v_mid.size} != capture {n_shells_all}')
    sel, sel_v = [], []
    for lb, vt in zip(LABELS, TARGET_V):
        k = int(np.argmin(np.abs(v_mid - vt)))
        sel.append(k); sel_v.append(v_mid[k])
    order = np.argsort(sel)
    shells_sorted = np.asarray(sorted(set(sel)), dtype=np.int64)
    say('  shell match: ' + '  '.join(
        f'{lb}->MC{sel[i]}(v={sel_v[i]:.0f} vs {TARGET_V[i]})'
        for i, lb in enumerate(LABELS)))

    with open(a.capture, 'rb') as f:
        raw_head = f.read(HEADER_BYTES)
    shell_table_bytes = struct.unpack_from('<2Q', raw_head, 64)[1]
    offset = HEADER_BYTES + shell_table_bytes

    _G.update(capture=a.capture, offset=offset, n_records=n_records,
              edges=canonical_edges(), shells=shells_sorted)
    bounds = np.linspace(0, n_records, a.workers + 1, dtype=np.int64)
    tasks = list(zip(bounds[:-1], bounds[1:]))
    with Pool(a.workers) as pool:
        parts = pool.map(scan_range, tasks)
    measure = np.sum([p[0] for p in parts], axis=0)
    counts = np.sum([p[1] for p in parts], axis=0)
    say(f'  capture scan done ({time.time()-t0:.0f}s, {a.workers} workers); '
        f'total measure={measure.sum():.4e} total counts={int(counts.sum())}')

    volumes = np.asarray([header['volumes_cm3'][int(s)] for s in shells_sorted])
    n_sel = shells_sorted.size

    # sigma table + populations + truth (same assembly as the ORACLE lane)
    bk = read_bake_bin(a.bin)
    ref_Z, ref_i, ref_E, ref_g = [], [], [], []
    with open(os.path.join(a.ref, 'levels.csv')) as f:
        for r in csv.DictReader(f):
            ref_Z.append(int(r['atomic_number'])); ref_i.append(int(r['ion_number']))
            ref_E.append(float(r['energy_eV'])); ref_g.append(float(r['g']))
    ref_Z, ref_i = np.array(ref_Z), np.array(ref_i)
    ref_E, ref_g = np.array(ref_E), np.array(ref_g)
    ions, popcache = [], {}
    for spec in IONS:
        if spec['pop'] not in popcache:
            popcache[spec['pop']] = parse_popcob(f"{a.run}/{spec['pop']}")
        _, allions, order_i, ND = popcache[spec['pop']]
        CI = allions[spec['cmf']][0]
        NF = CI.shape[1]
        n_ion = CI.sum(axis=1)
        osc = parse_osc(os.path.join(a.run, spec['osc']))
        E_eV = np.asarray(osc.levels['E_cm'][:NF], float) * CM2EV
        gl = np.asarray(osc.levels['g'][:NF], float)
        nu_th = (osc.ionization_eV - E_eV) * EV / H
        selr = np.where((ref_Z == spec['Z']) & (ref_i == spec['ion']))[0]
        if not (selr.size >= NF and np.max(np.abs(ref_E[selr][:NF] - E_eV)) < 1e-6
                and np.array_equal(ref_g[selr][:NF], gl)):
            raise SystemExit(f'{spec["lab"]}: ref row identity failed')
        S_D = np.asarray(bk['sigma'][selr[:NF]], dtype=float)
        # CMFGEN populations at the velocity-matched depth of each gated shell
        rt = open(f'{a.run}/RVTJ').read()
        from certify_rate_machine import rvtj_block
        V = rvtj_block(rt, 'Velocity (km/s)', CI.shape[0])
        depth_of = [int(np.argmin(np.abs(V - vt))) for vt in TARGET_V]
        ions.append(dict(spec=spec, NF=NF, CI=CI, n_ion=n_ion, nu_th=nu_th,
                         S_D=S_D, depth_of=depth_of))
    all_sigma = np.concatenate([io['S_D'] for io in ions], axis=0)
    all_nuth = np.concatenate([io['nu_th'] for io in ions])
    offsets_l = np.cumsum([0] + [io['NF'] for io in ions])

    # fixture input, mode 2 (MC commit)
    NBIN_LEG = 1000
    NU_MIN_LEG, NU_MAX_LEG = 1.5e14, 3.0e16
    dln_leg = math.log(NU_MAX_LEG / NU_MIN_LEG) / NBIN_LEG
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'l1bf_input_chain.bin')
    epoch = epoch_capture
    with open(path, 'wb') as f:
        f.write(b'A205IN01')
        # fresh owner => the production commit enforces generation == 1;
        # the capture's own generation (2) is recorded in the ledger.
        f.write(struct.pack('<QQQd', 2, n_sel, 1, epoch))
        v = 1.0e8 + 1.0e7 * np.arange(n_sel + 1)
        f.write(v[:-1].astype('<f8').tobytes())
        f.write(v[1:].astype('<f8').tobytes())
        f.write(np.ascontiguousarray(measure, dtype='<f8').tobytes())
        f.write(np.ascontiguousarray(counts.astype(np.uint64)).tobytes())
        f.write(volumes.astype('<f8').tobytes())
        f.write(struct.pack('<d', delta_t))
        f.write(struct.pack('<Qdd', NBIN_LEG, NU_MIN_LEG, dln_leg))
        f.write(struct.pack('<Q', all_sigma.shape[0]))
        for k in range(all_sigma.shape[0]):
            f.write(struct.pack('<d', float(all_nuth[k])))
            f.write(np.ascontiguousarray(all_sigma[k], dtype='<f8').tobytes())
    proc = subprocess.run([a.fixture, path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'fixture rc={proc.returncode}: {proc.stderr[:400]}')
    n_lev = all_sigma.shape[0]
    gamma = np.zeros((n_lev, n_sel)); state = np.zeros((n_lev, n_sel), int)
    var = np.zeros((n_lev, n_sel))
    seen = 0
    done_marker = False
    for line in proc.stdout.splitlines():
        if line.startswith('GAMMA '):
            t = line.split()
            gamma[int(t[1]), int(t[2])] = float(t[4])
            state[int(t[1]), int(t[2])] = int(t[3])
            var[int(t[1]), int(t[2])] = float(t[7])
            seen += 1
        elif line.strip() == 'A2_05_L1BF_FIXTURE DONE':
            done_marker = True
    # Mechanism judgment (2nd re-review): completeness, marker, state
    # validity and zero STALE are asserted, not assumed.
    mech_checks = dict(
        done_marker=done_marker,
        rows_complete=(seen == n_lev * n_sel),
        states_legal=bool(np.all((state >= 1) & (state <= 5))),
        stale_zero=int((state == 5).sum()) == 0,
    )
    say(f'  fixture chain lane done ({time.time()-t0:.0f}s); mech checks: '
        + ' '.join(f'{k}={v}' for k, v in mech_checks.items()))

    slot_of = {int(s): i for i, s in enumerate(shells_sorted)}
    oracle = {}
    if os.path.exists(a.oracle_ledger):
        led = json.load(open(a.oracle_ledger))
        for row in led.get('rows', []):
            oracle[(row['ion'], row['shell'])] = row.get('gamma_view')

    truth = np.load(a.truth_npz, allow_pickle=False) \
        if os.path.exists(a.truth_npz) else None
    if truth is None:
        raise SystemExit(f'truth contributions missing: {a.truth_npz} '
                         '(run the ORACLE gate first)')
    truth_contrib = truth['truth_contrib']

    rows_out = []
    say()
    say('  --- CHAIN lane: Gamma_chain [s^-1]; qualification = CI '
        f'(rel half-width <= {CI_REL_LIMIT:.4f}) AND truth-side f_cov >= '
        '0.999; otherwise the failing reason is named ---')
    for j, io in enumerate(ions):
        lab = io['spec']['lab']
        cells = []
        for q, lb in enumerate(LABELS):
            slot = slot_of[sel[q]]
            d = io['depth_of'][q]
            p = io['CI'][d] / io['n_ion'][d]
            g_lev = gamma[offsets_l[j]:offsets_l[j+1], slot]
            st_lev = state[offsets_l[j]:offsets_l[j+1], slot]
            v_lev = var[offsets_l[j]:offsets_l[j+1], slot]
            usable = (st_lev == VALID) | (st_lev == EXACT_ZERO)
            G = float((p * np.where(usable, g_lev, 0.0)).sum())
            VarG = float((p * p * np.where(usable, v_lev, 0.0)).sum())
            ci = 1.96 * math.sqrt(VarG)
            rel = ci / G if G > 0 else float('inf')
            # truth-side coverage: active set from the ORACLE lane's
            # CMFGEN-field contribution (state-independent denominator)
            contrib = truth_contrib[offsets_l[j]:offsets_l[j+1], q]
            total = contrib.sum()
            if total > 0:
                order_c = np.argsort(contrib)[::-1]
                csum = np.cumsum(contrib[order_c])
                n_active = int(np.searchsorted(csum, 0.999 * total) + 1)
                active = order_c[:n_active]
                fcov = float(contrib[active][usable[active]].sum() /
                             contrib[active].sum())
            else:
                fcov = float('nan')
            ci_ok = rel <= CI_REL_LIMIT
            cov_ok = fcov >= 0.999
            qualified = bool(ci_ok and cov_ok)
            reason = 'OK' if qualified else \
                ('BLOCKED_INSUFFICIENT_SAMPLING' if not cov_ok
                 else 'UNDERPOWERED')
            go = oracle.get((lab, lb))
            ratio = (G / go) if (go and go > 0) else None
            rows_out.append(dict(ion=lab, shell=lb, mc_shell=int(sel[q]),
                                 gamma_chain=G, ci_half=ci, ci_rel=rel,
                                 fcov_truth=fcov, qualified=qualified,
                                 reason=reason,
                                 blocked_share=float(p[~usable].sum()),
                                 ratio_vs_oracle=ratio))
            tag = f'{lb}={G:.3e}(ci{rel:.3f} fcov{fcov:.3f}' + \
                  ('' if qualified else ' ' + reason) + \
                  (f' r={ratio:.3f})' if ratio else ')')
            cells.append(tag)
        say(f'  {lab:7s}: ' + ' '.join(cells))
    n_q = sum(1 for r in rows_out if r['qualified'])
    mech_ok = all(mech_checks.values())
    say()
    say(f'  qualified cells (CI AND coverage): {n_q}/{len(rows_out)}; '
        f'mechanism verdict: {"PASS" if mech_ok else "FAIL"} '
        f'({mech_checks}); no legacy fallback exists in the chain.')
    ledger = dict(schema='lumina-a2-05-l1bf-chain-v2', lane='CHAIN',
                  mech_checks=mech_checks, mech_verdict=bool(mech_ok),
                  capture=a.capture, capture_generation=generation,
                  shells={LABELS[i]: int(sel[i]) for i in range(len(sel))},
                  ci_rel_limit=CI_REL_LIMIT, rows=rows_out,
                  qualified_cells=n_q, total_cells=len(rows_out),
                  elapsed_s=round(time.time() - t0, 1))
    with open(os.path.join(a.out, 'CHAIN_LANE_LEDGER.json'), 'w') as f:
        json.dump(ledger, f, indent=2)
    with open(os.path.join(a.out, 'CHAIN_LANE_REPORT.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    say(f'\n[done] {time.time()-t0:.0f}s -> {a.out}/CHAIN_LANE_LEDGER.json')
    return 0 if mech_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
