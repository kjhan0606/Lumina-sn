#!/usr/bin/env python3
"""셀37의 부피평균 물질 계수와 CMFGEN 양 끝 native 점 계수를 대조한다.

두 native 계수의 산술 중간값은 진단용 공간 proxy일 뿐 CMFGEN의
monotone ray 보간, 실제 셀 평균 또는 formal 수송 결과가 아니다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_lumina_native_formal_coefficients import band_metrics, formal, sha


def compare(reference_field, preparation, cells_file, frequency_grid, cell=37):
    p = json.loads(preparation.read_text())
    c = json.loads(cells_file.read_text())
    if c['input_sha256'][str(preparation)] != sha(preparation):
        raise ValueError('준비된 셀/외부 입력 신원 불일치')
    material_path = next(Path(path) for path in c['input_sha256'] if Path(path).name == 'material-cells.json')
    if sha(material_path) != c['input_sha256'][str(material_path)]:
        raise ValueError('셀 물질 보고서 신원 불일치')
    material = json.loads(material_path.read_text())
    map_path = material_path.parent / 'cell_map.csv'
    if sha(map_path) != material['output_sha256']['cell_map.csv']:
        raise ValueError('셀 좌표 매핑 신원 불일치')
    with map_path.open() as stream:
        mapping = list(csv.DictReader(stream))[cell]
    inner = int(mapping['inner_native_depth_1based']) - 1
    outer = int(mapping['outer_native_depth_1based']) - 1
    if (int(mapping['cell']) != cell or inner != 89 - cell or outer != 88 - cell
            or c['cells'] != 89 or c['frequencies'] != 1115020):
        raise ValueError('셀37/native 대응 불일치')
    if p['input_sha256'][str(frequency_grid)] != sha(frequency_grid):
        raise ValueError('주파수 격자 신원 불일치')
    freq = np.load(frequency_grid, mmap_mode='r', allow_pickle=False)
    if freq.ndim != 1 or len(freq) != c['frequencies'] or not np.all(np.diff(freq) > 0):
        raise ValueError('Lumina 주파수 격자 오류')
    geometry, ref_freq, coefficients = formal(reference_field, p['reference_field_sha256'], [inner, outer])
    if not np.isclose(float(mapping['r_inner_cm']), geometry[0, inner] * 1e10, rtol=1e-14, atol=0):
        raise ValueError('셀 내부 반경 불일치')
    if not np.isclose(float(mapping['r_outer_cm']), geometry[0, outer] * 1e10, rtol=1e-14, atol=0):
        raise ValueError('셀 외부 반경 불일치')
    bands = ((100, 400), (400, 800), (800, 1600), (1600, 3000))
    result = {}
    for component, name in enumerate(('cell_chi.npy', 'cell_eta.npy')):
        path = cells_file.parent / name
        if sha(path) != c['output_sha256'][name]:
            raise ValueError('셀 계수 신원 불일치: ' + name)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != (89, len(freq)):
            raise ValueError('셀 계수 차원 오류')
        projected = np.interp(np.log(ref_freq), np.log(freq), array[cell])
        native_inner, native_outer = coefficients[component]
        midpoint = 0.5 * (native_inner + native_outer)
        result[name] = {}
        for lo, hi in bands:
            key = f'{lo}-{hi}A'
            cell_mid = band_metrics(ref_freq, midpoint, projected, lo, hi)
            half_gradient = band_metrics(ref_freq, midpoint, native_outer, lo, hi)
            result[name][key] = dict(cell_vs_arithmetic_native_midpoint=cell_mid,
                native_endpoint_signed_contrast_over_midpoint_abs=
                    2 * half_gradient['signed_integral_difference_over_reference_abs'],
                native_endpoint_unsigned_contrast_over_midpoint_abs=
                    2 * half_gradient['unsigned_integral_difference_over_reference_abs'])
    return dict(schema='lumina-shell37-formal-coefficient-proxy-v1',
                shell=cell, native_inner_index=inner, native_outer_index=outer,
                mapping='CMFGEN native radius endpoints; midpoint proxy is arithmetic coefficient mean',
                frequency_mapping='Lumina log(nu)-linear to CMFGEN native frequencies',
                unit_mapping='CMFGEN formal chi/eta divided by 1e10',
                limitation='No CMFGEN ray interpolation or cell-average reference is claimed; spatial and frequency interpolation effects are not separated.',
                input_sha256={str(reference_field): p['reference_field_sha256'],
                              str(preparation): c['input_sha256'][str(preparation)],
                              str(cells_file): sha(cells_file),
                              str(frequency_grid): p['input_sha256'][str(frequency_grid)],
                              str(material_path): c['input_sha256'][str(material_path)],
                              str(map_path): material['output_sha256']['cell_map.csv']},
                result=result, physical_RT_executions=0, final_goal='NOT_MET')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-field', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--frequency-grid', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('기존 결과를 덮어쓰지 않음')
    report = compare(args.reference_field, args.preparation, args.cells, args.frequency_grid)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(out=str(args.out), result=report['result']), ensure_ascii=False))
