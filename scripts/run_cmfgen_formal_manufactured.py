#!/usr/bin/env python3
"""작은 독립 제조해로 angular 수송·산란·에너지·격자 반응을 함께 판독한다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simpson_regular(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) % 2 != 1 or len(x) < 3 or not np.allclose(np.diff(x), x[1] - x[0]):
        raise ValueError('Simpson 적분에는 홀수 개의 등간격 점이 필요함')
    return float((x[1] - x[0]) / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def simpson_nonuniform(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) % 2 != 1 or len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError('홀수 개의 증가하는 적분 좌표가 필요함')
    h0, h1 = x[1:-1:2]-x[:-2:2], x[2::2]-x[1:-1:2]
    return float(np.sum((h0+h1)/6*((2-h1/h0)*y[:-2:2] + (h0+h1)**2/(h0*h1)*y[1:-1:2] + (2-h0/h1)*y[2::2])))


def absorption_moments(r, order=256):
    """Rin=1, Rout=2, chi=eta=Binner=1인 구의 연속해. native 결과를 쓰지 않는다."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = []
    for radius in r:
        mu_core = np.sqrt(1 - 1 / radius ** 2)
        value = np.zeros(4)
        for lo, hi in ((-1., 0.), (0., mu_core), (mu_core, 1.)):
            if lo == hi:
                continue
            mu = lo + (nodes + 1) * (hi - lo) / 2
            path = radius * mu + np.sqrt(4 - radius ** 2 * (1 - mu ** 2))
            intensity = -np.expm1(-path)
            intensity[mu > mu_core] = 1
            for k in range(4):
                value[k] += np.dot(weights, mu ** k * intensity) * (hi - lo) / 4
        result.append(value)
    return np.array(result)


def analyze_static(out, mode):
    a = np.loadtxt(out / 'MANUFACTURED_NODES.dat')
    r, j, h, k, nn, eta, chi, scat = a[:, 1:].T
    iterations = np.atleast_2d(np.loadtxt(out / 'MANUFACTURED_ITERATIONS.dat'))
    net = float(r[0] ** 2 * h[0] - r[-1] ** 2 * h[-1])
    exchange = simpson_nonuniform(r[::-1], (r * r * (eta - (chi - scat) * j))[::-1])
    normalization = max(abs(net), abs(exchange))
    result = dict(calls=len(iterations), j_residual=float(iterations[-1, 1]),
                  source_residual=float(iterations[-1, 2]), h_change_over_j=float(iterations[-1, 3]),
                  angular_support_min=float(iterations[:, 5].min()),
                  net_boundary_r2H=net, volume_exchange=exchange,
                  energy_defect_absolute=abs(net - exchange),
                  energy_defect_relative=abs(net - exchange) / normalization if normalization else 0.)
    if mode == 'absorb':
        exact = absorption_moments(r)
        for col, name in enumerate(('J', 'H', 'K', 'N')):
            result[name + '_max_absolute_error'] = float(np.max(abs(a[:, col + 2] - exact[:, col])))
    if mode == 'zero':
        result['zero_field_max_absolute'] = float(np.max(abs(a[:, 2:6])))
    if mode == 'thermal':
        # 외부 진공 경계에서는 LTE가 아니다. 충분히 깊은 곳의 J=B만 확인한다.
        result['deep_LTE_J_error'] = float(np.max(abs(j[r <= 1.5] - 1)))
    return result


def analyze_redshift(out, nfreq):
    a = np.loadtxt(out / 'REDSHIFT_FIELD.dat')
    s, intensity, previous, b = a[::-1].T
    dilution = simpson_regular(s, 3 * b * intensity)
    frequency = simpson_regular(s, b * (previous - intensity) * nfreq)
    boundary = float(intensity[-1] - intensity[0])
    defect = boundary + dilution - frequency
    return dict(max_absolute_error=float(np.max(abs(intensity - 1))), boundary_change=boundary,
                expansion_dilution=dilution, frequency_transport=frequency,
                energy_defect_absolute=abs(defect),
                energy_defect_relative=abs(defect) / max(abs(boundary), abs(dilution), abs(frequency)))


def run(exe, out, grids, core_rays, radial_mapping):
    if out.exists():
        raise ValueError('기존 제조해 출력 보존')
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    fixture_manifest = exe.with_suffix('.build.json')
    fixture_build = json.loads(fixture_manifest.read_text())
    source = Path(fixture_build['source'])
    if sha(exe) != fixture_build['exe_sha256'] or sha(source) != fixture_build['source_sha256']:
        raise ValueError('제조해 실행파일 또는 실제 빌드 소스가 바뀜')
    shutil.copy2(source, out / source.name)
    build_manifest = Path(fixture_build['transport_build_manifest'])
    report = dict(scope='독립 제조해 수송 검증; 최종 Si/S 동질성 또는 released-T 합격 아님',
                  exe=str(exe), exe_sha256=sha(exe), producer_sha256=sha(__file__),
                  fixture_sha256=sha(source), fixture_build=fixture_build,
                  build_manifest_sha256=sha(build_manifest), core_rays=core_rays,
                  radial_mapping=radial_mapping, cases=[],
                  final_goal='NOT_JUDGED', physical_equivalence='NOT_JUDGED')
    env = dict(os.environ, LUMINA_FORMAL_PRIMARY='1', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='')
    cases = [(mode, n, None, core_rays, 'radial') for mode in ('absorb', 'sharp', 'scatter', 'thick') for n in grids]
    cases += [(mode, grids[1], None, core_rays, 'control') for mode in ('zero', 'thermal')]
    cases += [('redshift', n, 512 * 2 ** index, core_rays, 'coupled') for index, n in enumerate(grids)]
    cases += [('absorb', grids[0], None, nc, 'angular') for nc in (32,64,128)]
    cases += [('redshift', grids[0], nf, core_rays, 'frequency') for nf in (1024,2048)]
    cases += [('redshift', n, 2048, core_rays, 'spatial') for n in grids[:2]]
    started = time.monotonic()
    for mode, n, nfreq, nc, dimension in cases:
        directory = out / (mode + '_' + str(n) + '_' + dimension + '_' + str(nfreq or nc)); directory.mkdir()
        command = [str(exe), mode, str(n)] + ([str(nfreq)] if nfreq else [str(nc)])
        if not nfreq and radial_mapping == 'cosine':
            command += ['cosine']
        before = time.monotonic()
        try:
            with (directory / 'native.stdout').open('x') as stdout, (directory / 'native.stderr').open('x') as stderr:
                p = subprocess.run(command, cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=20)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        case = dict(mode=mode, n=n, nfreq=nfreq, core_rays=nc, refinement=dimension, command=command, returncode=rc,
                    elapsed_seconds=time.monotonic() - before, output=str(directory))
        if rc == 0:
            case['metrics'] = analyze_redshift(directory, nfreq) if nfreq else analyze_static(directory, mode)
        elif (directory / 'FORMAL_PRIMARY_FAILURE.txt').exists():
            case['failure'] = (directory / 'FORMAL_PRIMARY_FAILURE.txt').read_text().splitlines()[0]
        report['cases'].append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        if time.monotonic() - started > 250:
            report['partial'] = True
            break
    report['elapsed_seconds'] = time.monotonic() - started
    report['all_cases_executed'] = len(report['cases']) == len(cases)
    report['all_native_calls_returned_zero'] = report['all_cases_executed'] and all(c['returncode'] == 0 for c in report['cases'])
    with (out / 'manufactured-report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--grids', type=int, nargs=3, default=[17, 33, 65])
    p.add_argument('--core-rays', type=int, default=129)
    p.add_argument('--radial-mapping', choices=['uniform','cosine'], default='uniform')
    a = p.parse_args()
    if any(n < 7 or n > 257 or n % 2 != 1 for n in a.grids) or any(y != 2*x-1 for x,y in zip(a.grids,a.grids[1:])):
        p.error('격자는 h/h2/h4의 홀수 7..257이어야 함')
    if a.core_rays < 1 or max(a.grids)+a.core_rays > 500:
        p.error('core ray와 shell ray 합계는 500 이하여야 함')
    run(a.exe.resolve(), a.out.resolve(), a.grids, a.core_rays, a.radial_mapping)
