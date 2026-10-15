#!/usr/bin/env python3
"""2일 미수렴 초기 반복의 Si/S를 Saha predictor 및 같은 시각의 CMFGEN과 비교.

수렴 물리해 비교기와 분리한다. 입력은 기존 C reader가 검증·추출한 CSV이며
복사장 재계산/공개값 주입/외삽/분포 재정규화/최종 PASS 판정은 하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from read_cmf_common_material import public_block, interpolate, write_csv


def material(path, prefix):
    with path.open() as handle:
        status = handle.readline().strip()
        if not status.startswith('# ' + prefix):
            raise ValueError('물질표의 상태가 요청한 미수렴 진단과 다름')
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        key = tuple(int(row[k]) for k in ('shell', 'Z', 'stage'))
        if key in indexed:
            raise ValueError('중복 물질 행')
        for k, value in row.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError('비유한/음수 물질 자료')
        indexed[key] = row
    if not indexed:
        raise ValueError('빈 물질표')
    return status, indexed


def fraction(row, mass):
    nuclei = float(row['rho_g_cm3']) * float(row['mass_fraction']) / (mass * 1.660539066e-24)
    population = float(row['n_ion_cm3'])
    if nuclei == 0:
        if population != 0:
            raise ValueError('원소 핵수0에 양의 인구')
        return None, 0.
    value = population / nuclei
    if value > 1 + 1e-10:
        raise ValueError('핵수를 초과하는 이온 분포')
    volume = 4 * math.pi / 3 * (float(row['r_outer_cm'])**3 - float(row['r_inner_cm'])**3)
    if not volume > 0:
        raise ValueError('양의 셸 부피 필요')
    return value, nuclei * volume


def compare(run, public):
    _, before = material(run / 'toy_initial_material.csv', 'BOOTSTRAP_LTE_SAHA;')
    status, after = material(run / 'iterate_material.csv', 'UNCONVERGED_INITIAL_ITERATE;')
    match = re.search(r'completed_iterations=(\d+); physical_history=0$', status)
    if not match or before.keys() != after.keys():
        raise ValueError('완료 반복 또는 두 물질표의 레이아웃 불일치')
    day = 2.0  # C 추출기는 toy06의 epoch=172800을 확인한다.
    ph, phys = public_block(public / 'phys_toy06_cmfgen.txt', day)
    details, metrics, missing, plasma = [], [], {}, {}
    identity_fields = ('r_inner_cm', 'r_outer_cm', 'rho_g_cm3', 'T_K', 'mass_fraction')
    for z, name, mass in ((14, 'si', 28.0855), (16, 's', 32.065)):
        header, reference = public_block(public / f'ionfrac_{name}_toy06_cmfgen.txt', day)
        columns = {int(k[len(name):]): i for i, k in enumerate(header) if k.startswith(name)}
        for (s, element, q), row in after.items():
            if element != z:
                continue
            old = before[(s, z, q)]
            if any(row[k] != old[k] for k in identity_fields):
                raise ValueError('동일 초기 모델/T가 아닌 두 물질표')
            initial, weight = fraction(old, mass)
            current, _ = fraction(row, mass)
            velocity = (float(row['r_inner_cm']) + float(row['r_outer_cm'])) / 2 / (day * 86400) / 1e5
            ref = interpolate(reference, velocity, columns[q]) if q in columns else None
            details.append(dict(shell=s, Z=z, stage=q, velocity_km_s=velocity,
                saha_fraction=initial, iterate_fraction=current, reference_fraction=ref,
                nuclei_shell=weight,
                saha_error=initial-ref if initial is not None and ref is not None else None,
                iterate_error=current-ref if current is not None and ref is not None else None))
            plasma[s] = dict(shell=s, velocity_km_s=velocity, T_K=float(row['T_K']),
                ne_saha=float(old['n_e_cm3']), ne_iterate=float(row['n_e_cm3']),
                T_reference=interpolate(phys, velocity, ph.index('temp[K]')),
                ne_reference=interpolate(phys, velocity, ph.index('ne[/cm^3]')))
        stages = sorted({r['stage'] for r in details if r['Z'] == z})
        missing[str(z)] = sorted(set(columns) - set(stages))
        for q in stages:
            selected = [r for r in details if r['Z'] == z and r['stage'] == q and r['iterate_error'] is not None]
            total = math.fsum(r['nuclei_shell'] for r in selected)
            entry = dict(Z=z, stage=q, comparable_shells=len(selected))
            for kind in ('saha', 'iterate'):
                entry[kind + '_mean_absolute_fraction_error'] = math.fsum(abs(r[kind+'_error']) for r in selected)/len(selected) if selected else None
                entry[kind + '_nuclei_weighted_absolute_fraction_error'] = math.fsum(abs(r[kind+'_error'])*r['nuclei_shell'] for r in selected)/total if total else None
            metrics.append(entry)
    if set(plasma) != {s for s, z, q in after}:
        raise ValueError('일부 셸의 Si/S가 누락됨')
    out = run / 'iterate-comparison'
    out.mkdir(exist_ok=False)
    write_csv(out / 'ionization.csv', details)
    write_csv(out / 'plasma.csv', [plasma[s] for s in sorted(plasma)])
    paths = [run/'toy_initial_material.csv', run/'iterate_material.csv', run/'iterate.sha256',
        public/'phys_toy06_cmfgen.txt', public/'ionfrac_si_toy06_cmfgen.txt', public/'ionfrac_s_toy06_cmfgen.txt',
        Path(__file__), Path(__file__).with_name('read_cmf_common_material.py')]
    hashes = {}
    for path in paths:
        with path.open('rb') as handle:
            hashes[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = dict(day=day, completed_iterations=int(match[1]), status='UNCONVERGED_INITIAL_ITERATE',
        goal_verdict='NOT_JUDGED', physical_history=False, radiation_recomputed=False,
        spectrum_compared=False, velocity_interpolation='linear; no extrapolation',
        fraction_denominator='rho*mass_fraction/(mass_amu*AMU); no renormalization',
        missing_stages=missing, missing_stages_are_not_zero=True, ionization=metrics,
        input_sha256=hashes)
    with (out/'summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--public', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.public), indent=2, allow_nan=False))
