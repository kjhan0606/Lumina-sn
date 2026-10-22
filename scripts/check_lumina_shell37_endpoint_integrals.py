#!/usr/bin/env python3
"""셀37 반사실험 F의 광자 대역 비를 독립 longdouble 적분으로 재현한다."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_lumina_outer_input import sha


PAIR = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_pair_20261022-jXG2f6')
GRID = Path('/gpfs/kjhan/cmfgen_runs/lumina_full_sr_verified_20261020-yb8k9y/grid/frequency.npy')
BANDS = ((100, 400), (400, 800), (800, 1600), (1600, 3000))


def photon_integral(nu, intensity):
    x = np.asarray(nu, dtype=np.longdouble)
    y = np.asarray(intensity, dtype=np.longdouble) / x
    return np.sum((y[:-1] + y[1:]) * np.diff(x) / 2, dtype=np.longdouble)


def check(transport_root):
    root = Path(transport_root)
    pair = json.loads((PAIR / 'execution.json').read_text())
    run = json.loads((root / 'execution.json').read_text())
    if pair['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or run['status'] != 'COUNTERFACTUAL_DIAGNOSTIC_COMPLETE':
        raise ValueError('완성된 F 짝과 반사실험 필요')
    paths = dict(reference=PAIR / 'Jref.npy', P0=PAIR / 'Fref.npy',
                 outer=root / 'F-outer.npy', inner=root / 'F-inner.npy')
    identity = dict(reference=pair['Jref']['sha256'], P0=pair['F_reference']['sha256'],
                    outer=run['endpoint_results']['outer']['field']['sha256'],
                    inner=run['endpoint_results']['inner']['field']['sha256'])
    arrays = {}
    try:
        for name, path in paths.items():
            if sha(path) != identity[name]:
                raise ValueError('광자 적분 입력 변경: ' + name)
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        nu = np.load(GRID, mmap_mode='r', allow_pickle=False)
        try:
            if (nu.shape != (1115020,) or
                    any(value.shape != (91, len(nu)) for value in arrays.values())):
                raise ValueError('자체 격자/셀 차원 오류')
            result = {}
            maximum = 0.
            for lo, hi in BANDS:
                key = f'{lo}-{hi}A'
                mask = (2.99792458e18 / nu >= lo) & (2.99792458e18 / nu < hi)
                freq = nu[mask]
                if len(freq) < 2:
                    raise ValueError('빈 UV 대역')
                reference = photon_integral(freq, arrays['reference'][37, mask])
                if not reference > 0:
                    raise ValueError('참조 대역 광자수0')
                result[key] = {}
                for name in ('P0', 'outer', 'inner'):
                    value = photon_integral(freq, arrays[name][37, mask])
                    signed = float((value - reference) / reference)
                    residual = (pair['reference_residual'] if name == 'P0' else
                                run['endpoint_results'][name]['reference_residual'])
                    expected = residual['selected_shells']['37'][key]['signed_over_reference']
                    difference = abs(signed - expected)
                    maximum = max(maximum, difference)
                    result[key][name] = dict(field_over_reference=float(value / reference),
                                             signed_residual=signed,
                                             recorded_signed_residual=expected,
                                             absolute_difference=difference)
            if maximum > 5e-12:
                raise ValueError('독립 광자 적분과 저장 잔차 불일치')
            return dict(schema='lumina-shell37-independent-photon-integral-v1',
                        shell=37, integration='independent longdouble trapezoid of J/nu',
                        maximum_absolute_signed_residual_difference=maximum,
                        result=result, field_sha256=identity,
                        grid_sha256=sha(GRID), physical_RT_executions=0,
                        accepted_physical_field=False, final_goal='NOT_MET')
        finally:
            nu._mmap.close()
    finally:
        for value in arrays.values():
            value._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('기존 독립 적분 결과 보존')
    report = check(args.transport_root)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(out=str(args.out),
                          maximum_difference=report['maximum_absolute_signed_residual_difference']),
                     ensure_ascii=False))
