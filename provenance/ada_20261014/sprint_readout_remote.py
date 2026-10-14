#!/usr/bin/env python3
"""스프린트 판독: Lumina 런의 관측 스펙트럼을 공개 CMFGEN toy06 19.48 d 와 비교하고, 회차별 T_e 를 CMFGEN T 와 비교한다.

출력(새 디렉터리): summary.json, bands.csv, iter_te.csv, spectra_resampled.csv, compare.png(가능할 때).
연산은 Slurm 계산 작업 또는 사용자가 지정한 lageunha에서만 한다.
"""
import argparse
import csv
import glob
import json
import math
import os
import socket
import sys
from pathlib import Path

import numpy as np

PUBLIC = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06')
DAY = 19.48
COMMON = (2500.0, 25500.0)
BANDS = (('2500-4000', 2500.0, 4000.0), ('4000-6000', 4000.0, 6000.0), ('6000-8000', 6000.0, 8000.0),
         ('8000-10000', 8000.0, 10000.0), ('10000-25500', 10000.0, 25500.0))
CMFGEN_T = Path('/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/fixed_te_profile.txt')


def read_lumina(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    assert a.shape[1] == 2 and np.all(np.isfinite(a)) and np.all(np.diff(a[:, 0]) > 0), path
    return a[:, 0], a[:, 1]


def read_public_spectrum(day=DAY):
    lines = (PUBLIC / 'spectra_toy06_cmfgen.txt').read_text().splitlines()
    times = [float(t) for t in [ln for ln in lines if ln.startswith('#TIMES[d]:')][0].split()[1:]]
    hit = [i for i, t in enumerate(times) if abs(t - day) <= 5e-6 * day]
    assert len(hit) == 1
    col = 1 + hit[0]
    rows = [ln.split() for ln in lines if ln.strip() and not ln.startswith('#')]
    lam = np.array([float(r[0]) for r in rows])
    L = np.array([float(r[col]) for r in rows])
    return lam, L


def read_public_lbol(day=DAY):
    for ln in (PUBLIC / 'lbol_edep_toy06_cmfgen.txt').read_text().splitlines():
        if ln.startswith('#') or not ln.strip():
            continue
        t = ln.split()
        if abs(float(t[0]) - day) <= 5e-6 * day:
            return float(t[1]), float(t[2]) if len(t) > 2 else None
    return None, None


def integ(lam, f, lo, hi):
    """구간 [lo,hi] 의 사다리꼴 적분(끝점 선형 보간)."""
    lo, hi = max(lo, lam[0]), min(hi, lam[-1])
    if not hi > lo:
        return float('nan')
    inside = (lam > lo) & (lam < hi)
    x = np.concatenate(([lo], lam[inside], [hi]))
    y = np.interp(x, lam, f)
    return float(np.trapz(y, x))


def iter_te(work):
    rows = []
    cm = {}
    if CMFGEN_T.exists():
        for ln in CMFGEN_T.read_text().split('\n'):
            t = ln.split()
            if len(t) == 2:
                cm[int(t[0])] = float(t[1])
    for f in sorted(glob.glob(str(work / 'physics_comparison' / 'physics_DET_iter*.shell.csv'))):
        it = int(Path(f).name.split('iter')[1][:4])
        with open(f, newline='') as fh:
            rr = list(csv.DictReader(fh))
        te = {int(r['shell_id']): float(r['T_e_K']) for r in rr}
        rat = [te[s] / cm[s] for s in te if s in cm and cm[s] > 0]
        heat = sum(float(r.get('sum_heating') or 0) for r in rr)
        cool = sum(float(r.get('sum_cooling') or 0) for r in rr)
        rows.append({'iter': it, 'Te_shell0': te.get(0), 'Te_shell25': te.get(25), 'Te_shell49': te.get(49),
                     'Te_over_cmfgen_median': float(np.median(rat)) if rat else '',
                     'Te_over_cmfgen_min': min(rat) if rat else '', 'Te_over_cmfgen_max': max(rat) if rat else '',
                     'sum_heating': heat, 'sum_cooling': cool})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run root containing work/')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--spectrum', default='lumina_spectrum_det_obs.csv')
    a = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID') or socket.gethostname().lower() == 'lageunha', 'run on the authorized compute host'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    work = Path(a.run) / 'work'
    res = {'label': a.label, 'run': a.run, 'day': DAY}
    lbol_pub, edep_pub = read_public_lbol()
    res['public_lbol'], res['public_edep'] = lbol_pub, edep_pub
    sp = work / a.spectrum
    rows = []
    if sp.exists():
        lamL, LL = read_lumina(sp)
        lamC, LC = read_public_spectrum()
        res['lumina_integral_all'] = integ(lamL, LL, lamL[0], lamL[-1])
        cs = (max(COMMON[0], lamC[0], lamL[0]), min(COMMON[1], lamC[-1], lamL[-1]))
        res['common_support'] = cs
        Lc, Ll = integ(lamC, LC, *cs), integ(lamL, LL, *cs)
        res['common_L_cmfgen'], res['common_L_lumina'] = Lc, Ll
        res['common_ratio'] = Ll / Lc
        res['pass_luminosity_10pct'] = bool(abs(Ll / Lc - 1.0) <= 0.10)
        for name, lo, hi in BANDS:
            c, l = integ(lamC, LC, lo, hi), integ(lamL, LL, lo, hi)
            rows.append({'band': name, 'L_cmfgen': c, 'L_lumina': l, 'ratio': l / c if c else '',
                         'norm_ratio': (l / Ll) / (c / Lc) if c and Ll else ''})
        # 정규화 형태: 공통 지원 광도로 나눈 스펙트럼을 같은 격자(로그 1000 점)에 리샘플
        grid = np.exp(np.linspace(math.log(cs[0]), math.log(cs[1]), 1000))
        nC = np.interp(grid, lamC, LC) / Lc
        nL = np.interp(grid, lamL, LL) / Ll
        sel = (grid >= 3000) & (grid <= 9000)
        res['shape_rms_dex_3000_9000'] = float(np.sqrt(np.mean(np.log10(np.maximum(nL[sel], 1e-300) /
                                                                       np.maximum(nC[sel], 1e-300)) ** 2)))
        with open(out / 'spectra_resampled.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wavelength_angstrom', 'L_cmfgen', 'L_lumina', 'norm_cmfgen', 'norm_lumina'])
            for g, c, l, a1, b1 in zip(grid, np.interp(grid, lamC, LC), np.interp(grid, lamL, LL), nC, nL):
                w.writerow([g, c, l, a1, b1])
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            ax[0].plot(lamC, LC, 'k-', lw=0.8, label='CMFGEN 19.48 d')
            ax[0].plot(lamL, LL, 'r-', lw=0.8, label=a.label)
            ax[0].set_xlim(2500, 10000)
            ax[0].set_ylabel('L_lambda [erg/s/A]')
            ax[0].legend()
            ax[1].plot(grid, nC, 'k-', lw=0.8, label='CMFGEN (norm)')
            ax[1].plot(grid, nL, 'r-', lw=0.8, label=a.label + ' (norm)')
            ax[1].set_xlim(2500, 10000)
            ax[1].set_xlabel('wavelength [A]')
            ax[1].set_ylabel('L_lambda / L_common')
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(out / 'compare.png', dpi=120)
            res['plot'] = str(out / 'compare.png')
        except Exception as e:
            res['plot'] = 'not made: %r' % e
    else:
        res['spectrum'] = 'absent: %s' % sp
    with open(out / 'bands.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['band', 'L_cmfgen', 'L_lumina', 'ratio', 'norm_ratio'])
        w.writeheader()
        w.writerows(rows)
    it = iter_te(work)
    if it:
        with open(out / 'iter_te.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(it[0].keys()))
            w.writeheader()
            w.writerows(it)
    res['iterations_dumped'] = len(it)
    res['bands'] = rows
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({k: res.get(k) for k in ('label', 'common_ratio', 'pass_luminosity_10pct',
                                              'shape_rms_dex_3000_9000', 'iterations_dumped')}, default=str))
    for r in rows:
        print('%-12s ratio %s  norm %s' % (r['band'], r['ratio'], r['norm_ratio']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
