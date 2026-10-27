#!/usr/bin/env python3
"""원70 동일 물질/복사장에서 Si II 역전 준위의 직접 BF 기여를 집계한다.

저장된 총 BF율은 bound와 dissolution을 포함한다. 새 복사장 수송이나
준위 간 간접 연쇄의 크기를 계산하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_material_inventory import sha


def weighted_photoionization(rows, columns, populations, fractions, roster, photo_rates,
                             levels=(123, 124), node=54):
    """실제 lower 주소를 기준으로 Si II 전 경로와 지정 FL의 n_l P Γ를 비교한다."""
    lower = np.asarray(roster['lower'], dtype='i8')
    selected = np.flatnonzero((roster['Z'] == 14) & (roster['stage'] == 2))
    if not len(selected) or any(int(rows[lower[i]]['atomic_number']) != 14 or
                                int(rows[lower[i]]['ion_number']) != 1 for i in selected):
        raise ValueError('Si II lower 원자 주소 불일치')
    lookup = {tuple(map(int, row)): i for i, row in enumerate(columns)}
    if len(lookup) != len(columns) or populations.ndim != 2 or not 0 <= node < populations.shape[1]:
        raise ValueError('SL 주소 또는 반경점 오류')
    actual_levels = [i for i, row in enumerate(rows) if int(row['atomic_number']) == 14 and
                     int(row['ion_number']) == 1 and int(row['level_number']) in levels]
    if len(actual_levels) != len(levels): raise ValueError('대상 Si II FL 주소 누락/중복')
    sl = np.array([lookup[(int(rows[i]['atomic_number']), int(rows[i]['ion_number']),
                           int(rows[i]['super_level']))] for i in lower[selected]])
    full = populations[sl]*fractions[lower[selected]]
    flow = full*roster['probability'][selected, None]*photo_rates[selected]
    if (not np.isfinite(flow).all() or np.any(flow < 0) or
            np.any(roster['probability'][selected] < 0)):
        raise ValueError('BF 인구/확률/광이온화율 오류')
    total = flow.sum(axis=0)
    if np.any(total <= 0): raise ValueError('비교할 양의 Si II 총 광이온화 없음')
    parts = []
    for level in levels:
        indices = [i for i in actual_levels if int(rows[i]['level_number']) == level]
        mask = lower[selected] == indices[0]; value = flow[mask].sum(axis=0)
        parts.append(dict(FL_1based=level+1, routes=int(mask.sum()),
            node_rate_cm3_s=float(value[node]), node_fraction=float(value[node]/total[node]),
            threshold_Hz=roster['threshold'][selected[mask]].tolist()))
    mask = np.isin(lower[selected], actual_levels); joint = flow[mask].sum(axis=0)/total
    summed = {}
    for i, route in enumerate(selected):
        key = int(lower[route]); summed[key] = summed.get(key, 0.)+float(flow[i, node])
    leaders = sorted(summed.items(), key=lambda item: -item[1])[:8]
    return dict(node=node, SiII_routes=len(selected), total_rate_cm3_s=float(total[node]),
        levels=parts, combined_node_fraction=float(joint[node]),
        combined_max_fraction=float(joint.max()), combined_max_node=int(joint.argmax()),
        combined_median_fraction=float(np.median(joint)),
        dominant_levels=[dict(FL_1based=int(rows[i]['level_number'])+1,
                              configuration=rows[i]['configuration'], fraction=float(value/total[node]))
                         for i, value in leaders])


def analyze(root, levels_path):
    root, levels_path = Path(root).resolve(), Path(levels_path).resolve()
    report_path = root/'result.json'; report = json.loads(report_path.read_text())
    if report['state'] != 'COMPLETE_FROZEN_MATERIAL_BF_90': raise ValueError('완료 원70 BF만 사용')
    names = ('material-snapshot.npz', 'route-roster.npz', 'total-rates.npy')
    for name in names:
        if sha(root/name) != report['output_sha256'][name]: raise ValueError('원70 BF 결과 변경: '+name)
    expected_levels = [digest for path, digest in report['data_sha256'].items() if path.endswith('/atomic/levels.csv')]
    if len(set(expected_levels)) != 1 or sha(levels_path) != expected_levels[0]:
        raise ValueError('원70 BF와 다른 원자 준위 주소')
    rows = list(csv.DictReader(levels_path.open()))
    with np.load(root/'material-snapshot.npz', allow_pickle=False) as archive:
        snapshot = {k: archive[k] for k in ('columns', 'superlevel', 'fraction')}
    with np.load(root/'route-roster.npz', allow_pickle=False) as archive:
        roster = {k: archive[k] for k in archive.files}
    rates = np.load(root/'total-rates.npy', mmap_mode='r', allow_pickle=False)
    try:
        result = weighted_photoionization(rows, snapshot['columns'], snapshot['superlevel'],
                                         snapshot['fraction'], roster, rates[:, :, 0])
    finally: rates._mmap.close()
    result.update(baseline='원70 물질·원70 복사장, 고정온도 미수렴; 새 자체장 아님',
        rate_scope='직접 광이온화 n_lower × 경로확률 × 총 BF photo율; 간접 BB 연쇄 미포함',
        input_sha256={str(p): sha(p) for p in (report_path, levels_path, *[root/n for n in names])},
        new_integrals=0, new_transport=0, final_goal='NOT_JUDGED')
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--levels', dest='levels_path', type=Path, required=True)
    return p.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(analyze(**vars(parse_args())), ensure_ascii=False, indent=2))
