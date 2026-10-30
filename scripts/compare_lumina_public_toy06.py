#!/usr/bin/env python3
"""완료된 물질응답과 공개 toy06의 19.48일 분포를 속도점에서 비교한다.

읽기 전용이다. 보간값은 구간 평균이나 동일 물리조건의 수렴 인증이 아니다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def public_block(path, epoch='19.480'):
    rows, matching, active = [], 0, False
    for line in path.read_text().splitlines():
        if line.startswith('#TIME:'):
            active = line.split()[1] == epoch
            matching += int(active)
        elif active and line.strip() and not line.startswith('#'):
            rows.append([float(value) for value in line.split()])
    values = np.asarray(rows, dtype=float)
    if (matching != 1 or values.shape[0] != 100 or values.ndim != 2 or
            not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0)):
        raise ValueError('공개 19.48일 블록 또는 속도 격자 오류')
    return values


def interp_columns(table, velocity, column):
    """범위 밖은 외삽하지 않고 NaN으로 남긴다."""
    v = np.asarray(velocity, dtype=float)
    return np.interp(v, table[:, 0], table[:, column], left=np.nan, right=np.nan)


def compare(result, public):
    result, public = Path(result), Path(public)
    report = json.loads(result.read_text())
    if (report.get('state') != 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE'
            or report.get('fixed_accepted_temperature') is not True):
        raise ValueError('완료된 고정온도 물질응답만 비교 가능')
    epoch = report['source_identity']['epoch_seconds']
    if epoch != 19.48 * 86400:
        raise ValueError('공개 toy06 19.48일과 응답 epoch 불일치')
    root = result.parent
    paths = {name: root / name for name in ('Si-S-comparison.csv', 'physical-profiles.npz')}
    for name, path in paths.items():
        if sha(path) != report['output_sha256'][name]:
            raise ValueError('완료 응답의 표 또는 물리 프로필 해시 불일치')
    public_paths = {z: public / f'ionfrac_{sym}_toy06_cmfgen.txt'
                    for z, sym in ((14, 'si'), (16, 's'))}
    public_paths['phys'] = public / 'phys_toy06_cmfgen.txt'
    pub = {key: public_block(path) for key, path in public_paths.items()}
    if any(pub[z].shape[1] != 6 for z in (14, 16)) or pub['phys'].shape[1] != 5:
        raise ValueError('공개 이온 단계 또는 물리량 열 형상 오류')
    with np.load(paths['physical-profiles.npz'], allow_pickle=False) as p:
        radius = p['radius_cm'].copy()
        temperature = p['new_response_temperature'].copy()
        ne = p['new_response_ne'].copy()
    if (radius.shape != (90,) or temperature.shape != (90,) or ne.shape != (90,)
            or not np.isfinite([radius, temperature, ne]).all()
            or np.any(radius <= 0) or np.any(temperature <= 0) or np.any(ne <= 0)):
        raise ValueError('물리 프로필 주소 또는 값 오류')
    velocity = radius / epoch / 1.e5
    values = {}
    with paths['Si-S-comparison.csv'].open(newline='') as f:
        for row in csv.DictReader(f):
            z, q, node = int(row['Z']), int(row['charge']), int(row['node'])
            if z not in (14, 16) or q not in (1, 2, 3, 4):
                continue
            key = (z, q, node)
            x = float(row['new_J_response_fraction'])
            if (key in values or not 0 <= node < 90 or not np.isfinite(x) or x < 0
                    or float(row['radius_cm']) != radius[node]):
                raise ValueError('이온분율 표의 점·주소·값 오류')
            values[key] = x
    if len(values) != 2 * 4 * 90:
        raise ValueError('Si/S II–V 분율 표 누락')

    stage, profiles = {}, []
    for z in (14, 16):
        for q in (1, 2, 3, 4):
            ref = interp_columns(pub[z], velocity, q + 1)
            candidate = np.array([values[(z, q, i)] for i in range(90)])
            valid = np.isfinite(ref)
            if not valid.any():
                raise ValueError('공개 속도와 겹치는 점 없음')
            index = np.flatnonzero(valid)[np.argmax(abs(candidate[valid] - ref[valid]))]
            stage[f'{z}:{q}'] = dict(compared_nodes=int(valid.sum()),
                max_abs_fraction=float(abs(candidate[index] - ref[index])),
                node_1based=int(index + 1), velocity_km_s=float(velocity[index]),
                public_fraction=float(ref[index]), lumina_fraction=float(candidate[index]))
            for i in np.flatnonzero(valid):
                profiles.append(dict(node_1based=int(i + 1), Z=z, charge=q,
                    velocity_km_s=float(velocity[i]), public_fraction=float(ref[i]),
                    lumina_fraction=float(candidate[i])))
    overlap = np.isfinite(interp_columns(pub['phys'], velocity, 1))
    return dict(state='DIAGNOSTIC_NOT_JOINT_CONVERGENCE', epoch_days=19.48,
        scope='공개 시간의존 CMFGEN과 고정-T Lumina의 이온분율; 동등 물리계약 아님',
        mapping='공개 속도점별 선형 보간; 외삽 없음; 점값이며 셀 평균 아님',
        excluded_node_1based=(np.flatnonzero(~overlap) + 1).tolist(),
        input_sha256={'result': sha(result), **{k: sha(v) for k, v in paths.items()},
                      **{str(k): sha(v) for k, v in public_paths.items()}},
        selected_response_sha256=report['selected_response']['response_sha256'],
        stages=stage, profiles=profiles,
        thermodynamic_profiles=[dict(node_1based=int(i + 1), velocity_km_s=float(velocity[i]),
            public_temperature=float(interp_columns(pub['phys'], velocity, 1)[i]),
            lumina_temperature=float(temperature[i]),
            public_ne=float(interp_columns(pub['phys'], velocity, 3)[i]),
            lumina_ne=float(ne[i])) for i in np.flatnonzero(overlap)],
        new_physical_calculations=0, final_equivalence='NOT_JUDGED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.result, args.public), ensure_ascii=False, allow_nan=False))
