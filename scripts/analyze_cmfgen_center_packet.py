#!/usr/bin/env python3
"""물질70/J71의 선택된 formal packet·광선합·기존 완료 장을 비교한다.

native 노드 검증이며 CMFGEN 셀중심 J나 자기일관 이온분포 판정은 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import compensated_moments, formal_state, geometry, rays
from prepare_cmfgen_mom_capture import sha
from run_cmfgen_formal_primary import fields


INVENTORY = Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v2/formal-build.json')


def analyze(root, out=None, comparison=None):
    field = root/'field'
    report = json.loads((root/'center-packet-run.json').read_text())
    comparison_path = comparison if comparison is not None else root/'center-field-comparison.json'
    comparison = json.loads(comparison_path.read_text())
    if (report['returncode'] != 97 or report['new_field_sha256'] != comparison['captured_sha256'] or
        report['baseline_field_sha256'] != comparison['original_sha256'] or
        comparison['records'] != 197291 or comparison['calls_mismatch_count'] != 0 or
        not all(comparison[k] for k in ('geometry_equal','frequency_equal','weights_equal')) or
        max(comparison['component_max_frequency_normalized_difference']) > 1e-10):
        raise ValueError('전체 장의 수치적 계측 무간섭 조건 미충족')
    indices=report.get('frequency_indices',[60714,60715])
    if indices not in ([60714,60715],[44360,44361]):
        raise ValueError('목표 formal packet 주파수 쌍 불일치')
    target = out if out is not None else root/'center-packet-analysis.json'
    if target.exists():
        raise FileExistsError('기존 포획 판독 보존')
    inventory = json.loads(INVENTORY.read_text())['ray_inventory']
    _, original = fields(Path(report['original_field'])/'FORMAL_PRIMARY_FIELDS.bin')
    rows = []
    for index in indices:
        stems = sorted(p.name[:-8] for p in field.iterdir()
                       if p.name.startswith(f'FORMAL_{index}_') and p.name.endswith('_out.bin')
                       and not p.name.endswith('_rays_out.bin'))
        if not stems:
            raise ValueError(f'{index} 실제 formal packet 없음')
        for stem in stems:
            paths = [field/(stem+suffix) for suffix in (
                '_in.bin', '_out.bin', '_geometry.bin', '_rays_in.bin', '_rays_out.bin')]
            if any(not p.is_file() for p in paths):
                raise ValueError('formal packet 구성 누락: '+stem)
            phase_in = formal_state(paths[0], inventory)
            phase_out = formal_state(paths[1], inventory)
            g = geometry(paths[2])
            previous = rays(paths[3])
            actual = rays(paths[4])
            if (phase_in['phase'][:2] != (71, index) or
                phase_out['phase'][:2] != (71, index) or
                any(g[k] != actual[k] or g[k] != previous[k] for k in ('NC', 'NP', 'ND'))):
                raise ValueError('계측 세대·주파수·광선 기하 불일치: '+stem)
            moments, bound = compensated_moments(g, actual)
            stored = np.array([phase_out['ray'][f'{k}NU_STORE']['values'] for k in 'JHKN'])
            error = abs(stored-moments)
            if stored.shape != moments.shape or np.any(error > bound):
                raise ValueError('실제 ray 합과 formal JHKN 불일치: '+stem)
            native = np.asarray(original['field'][index-1, 0, :])
            j = stored[0]
            max_rel = float(np.max(abs(j-native)/np.maximum(abs(native), 1e-300)))
            rows.append(dict(phase=phase_out['phase'], frequency_Hz=float(phase_out['args']['FREQ']*1e15),
                             wavelength_A=float(2.99792458e18/(phase_out['args']['FREQ']*1e15)),
                             max_ray_moment_over_roundoff_bound=float(np.max(np.divide(
                                 error, bound, out=np.zeros_like(error), where=bound>0))),
                             max_J_relative_to_original_final=max_rel,
                             native49_J_over_final=float(j[49]/native[49]),
                             native50_J_over_final=float(j[50]/native[50]),
                             native49_50_J=[float(j[49]), float(j[50])],
                             ray_points_inserted=g['RAY_POINTS_INSERTED'],
                             packet_sha256={p.name:sha(p) for p in paths}))
    output = dict(scope=__doc__, original_field_sha256=report['baseline_field_sha256'],
                  new_field_sha256=report['new_field_sha256'],
                  byte_identical=report['field_byte_identical'],
                  numerically_equal_within_1e_10=True,
                  comparison_sha256=sha(comparison_path),native_nodes_only=True,
                  native_nodes_zero_based=[49,50],
                  cases=rows, final_goal='NOT_MET')
    with target.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(target), cases=[{k:v for k,v in row.items()
        if k not in ('packet_sha256',)} for row in rows]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path,
                        help='기존 분석을 보존하고 새 경로에 정정 요약을 기록한다')
    parser.add_argument('--comparison', type=Path,
                        help='정정된 동일 native 위치의 전주파수 비교 결과')
    args = parser.parse_args()
    analyze(args.root.resolve(),args.out.resolve() if args.out is not None else None,
            args.comparison.resolve() if args.comparison is not None else None)
