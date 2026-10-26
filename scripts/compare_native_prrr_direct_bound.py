#!/usr/bin/env python3
"""고정 원래점의 완성 C bound BF와 원 PRRR의 SL별 율을 읽기 전용 대조한다.

원 PRRR에는 준위용해가 포함되므로 이 비교만으로 수용 판정을 내리지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from compare_cmfgen_bf_superlevel_rates import parse_prrr_superlevel_rates


def load_material(path):
    with Path(path).open('rb') as stream:
        tag, ns, nc, nl, epoch = struct.unpack('<8sIIId', stream.read(28))
        if tag != b'LUMNOD01' or ns != 90 or not nc or not nl or epoch <= 0:
            raise ValueError('원래점 물질 header 불일치')
        columns = np.frombuffer(stream.read(12*nc), dtype='<i4').reshape(nc, 3).copy()
        physical = np.frombuffer(stream.read(8*4*ns), dtype='<f8').reshape(ns, 4).copy()
        population = np.frombuffer(stream.read(8*ns*nc), dtype='<f8').reshape(ns, nc).copy()
        if stream.read(1):
            raise ValueError('물질 후행 데이터')
    return nl, physical, {tuple(map(int, row)):population[:, i]
                          for i, row in enumerate(columns)}


def load_full(path, nl, ns):
    with Path(path).open('rb') as stream:
        tag, nlevel, npoint = struct.unpack('<8sII', stream.read(16))
        if (tag, nlevel, npoint) != (b'LMATFL01', nl, ns):
            raise ValueError('전체 준위 인구 header 불일치')
        result = np.frombuffer(stream.read(8*nl*ns), dtype='<f8').reshape(nl, ns).copy()
        if stream.read(1) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError('전체 준위 인구 손상')
    return result


def compare(bound_result, material, full_population, levels, prrr_dir, node):
    report = json.loads(Path(bound_result).read_text())
    if (report.get('state') != 'COMPLETE_BOUND_ONLY' or
            report.get('complete_bound_frequency_integral') is not True or
            report.get('native_node') != node or report.get('material_updates') != 0 or
            report.get('input_identity', {}).get('material_record') != 70 or
            Path(report.get('full_population_check', {}).get('source', '')).resolve() !=
            Path(full_population).resolve()):
        raise ValueError('완성·동일시점 C bound 결과만 허용')
    root = Path(bound_result).parent
    unit = np.load(root/'unit-rates.npy', allow_pickle=False)
    with np.load(root/'roster.npz', allow_pickle=False) as roster:
        lower, target, Z, stage, probability = [roster[key].copy() for key in
                                                  ('lower', 'target', 'Z', 'stage', 'probability')]
    nr = len(lower)
    if (unit.shape != (nr, 6) or report.get('routes_complete') != nr or
            not np.isfinite(unit).all() or np.any(unit[:, :3] < 0)):
        raise ValueError('완성 route 배열 불일치')
    nl, physical, superlevel = load_material(material)
    rows = list(csv.DictReader(Path(levels).open()))
    if len(rows) != nl or not 0 <= node < len(physical):
        raise ValueError('준위/물질 차원 불일치')
    full = load_full(full_population, nl, len(physical))
    groups = {}
    for index, row in enumerate(rows):
        key = (int(row['atomic_number']), int(row['ion_number']), int(row['super_level']))
        groups.setdefault(key, []).append(index)
    for key, members in groups.items():
        if key not in superlevel or not np.isclose(
                full[members, node].sum(), superlevel[key][node], rtol=1e-12, atol=0):
            raise ValueError(f'전체 준위와 SL 물질 불일치: {key}')
    if np.any(lower < 0) or np.any(lower >= nl) or np.any(target >= nl):
        raise ValueError('route 준위 주소 범위 밖')
    output = {}
    for element, ion, label in ((14, 2, 'Sk2'), (16, 2, 'S2')):
        indices = np.flatnonzero((Z == element) & (stage == ion))
        sl_ids = [int(row['super_level']) for row in rows if
                  int(row['atomic_number']) == element and int(row['ion_number']) == ion-1]
        nsl = max(sl_ids)+1
        values = np.zeros((nsl, 2), dtype=float)
        for i in indices:
            low = int(lower[i]); upper = int(target[i]); p = float(probability[i])
            row = rows[low]
            if (int(row['atomic_number']), int(row['ion_number'])) != (element, ion-1):
                raise ValueError('lower ion 주소 불일치')
            sl = int(row['super_level'])
            if upper >= 0:
                target_row = rows[upper]
                if (int(target_row['atomic_number']), int(target_row['ion_number'])) != (element, ion):
                    raise ValueError('upper ion 주소 불일치')
                upper_key = (element, ion, int(target_row['super_level']))
            else:
                upper_key = (element, ion, -1)
            values[sl, 0] += p*float(unit[i, 0])*float(full[low, node])
            values[sl, 1] += p*float(unit[i, 1]+unit[i, 2])*float(superlevel[upper_key][node])
        native_photo = parse_prrr_superlevel_rates(Path(prrr_dir)/f'{label}PRRR', label, 90, nsl)
        native_reverse = parse_prrr_superlevel_rates(Path(prrr_dir)/f'{label}PRRR', label, 90, nsl,
                                                      process='recombination')
        native = np.column_stack((native_photo[:, node], native_reverse[:, node]))
        output[label] = dict(native_node=node, routes=len(indices),
            native_total=native.sum(axis=0).tolist(), C_bound_total=values.sum(axis=0).tolist(),
            native_minus_C_bound=(native-values).sum(axis=0).tolist(),
            per_SL=[dict(sl=sl, native_photo=float(native[sl, 0]),
                         C_bound_photo=float(values[sl, 0]),
                         native_reverse=float(native[sl, 1]),
                         C_bound_reverse=float(values[sl, 1]),
                         reverse_delta=float(native[sl, 1]-values[sl, 1]))
                    for sl in range(nsl)])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bound-result', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--full-population', type=Path, required=True)
    parser.add_argument('--levels', type=Path, required=True)
    parser.add_argument('--prrr-dir', type=Path, required=True)
    parser.add_argument('--node', type=int, default=49)
    args = parser.parse_args()
    print(json.dumps(compare(args.bound_result, args.material, args.full_population,
                             args.levels, args.prrr_dir, args.node), indent=2))
