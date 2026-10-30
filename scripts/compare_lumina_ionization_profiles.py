#!/usr/bin/env python3
"""봉인된 원 물질과 후보의 같은 원래점 Si/S 분율을 비교한다."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


STAGES = {'Si_II': (14, 1), 'Si_III': (14, 2), 'S_II': (16, 1), 'S_III': (16, 2)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def diagnostics(reference, candidate, threshold=1.e-8):
    old, new = np.asarray(reference), np.asarray(candidate)
    if (old.ndim != 1 or new.shape != old.shape or not len(old) or
        not np.isfinite([old, new]).all() or np.any(old < 0) or np.any(new < 0)):
        raise ValueError('분율 비교 형상/부호 오류')
    active = (old > threshold) | (new > threshold)
    supported = (old > 0) & (new > 0)
    log = np.full(old.shape, np.nan)
    log[supported] = np.log10(new[supported]) - np.log10(old[supported])
    chosen = active & supported
    values = abs(log[chosen])
    return log, dict(known_fraction_active_points=int(active.sum()),
        active_support_mismatch_count=int(np.count_nonzero(active & ~supported)),
        max_abs_fraction=float(abs(new - old).max()),
        worst_absolute_node_zero_based=int(np.argmax(abs(new - old))),
        active_log_abs_median_dex=float(np.median(values)) if len(values) else None,
        active_log_abs_p95_dex=float(np.quantile(values, .95)) if len(values) else None,
        active_log_abs_max_dex=float(values.max()) if len(values) else None,
        active_log_above_0p1_dex_count=int(np.count_nonzero(values > .1)),
        active_log_above_0p1_dex_nodes_zero_based=np.flatnonzero(chosen & (abs(log) > .1)).tolist())


def response_selection(report, result_path, rejected_evaluation=None):
    if rejected_evaluation is None:
        return result_path.parent / 'accepted-response.npz', report['accepted_output_sha256'], {}
    index = rejected_evaluation
    if type(index) is not int or not 1 <= index <= len(report.get('evaluations', [])):
        raise ValueError('완료된 실제 후보 번호 필요')
    decisions = [row for row in report.get('history', []) if row.get('evaluation') == index]
    if (len(decisions) != 1 or decisions[0].get('accepted') is not False or
        decisions[0].get('phase') not in ('line_search', 'frozen_direction')):
        raise ValueError('명시적으로 거부된 실제 방향 후보만 진단 가능')
    folder = result_path.parent / ('evaluation-' + str(index))
    evaluation_report = folder / 'result.json'
    if json.loads(evaluation_report.read_text()) != report['evaluations'][index - 1]:
        raise ValueError('완료 보고서와 실제 후보 기록 불일치')
    return folder / 'response.npz', report['evaluations'][index - 1]['output_sha256']['response.npz'], {
        'candidate_status': 'REJECTED_DIAGNOSTIC_ONLY', 'rejected_evaluation': index,
        'candidate_accepted': False, 'decision': decisions[0],
        'evaluation_report_sha256': sha(evaluation_report)}


def compare(result_path, expected_hash, out, rejected_evaluation=None):
    result_path, out = result_path.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError('기존 비교 보존')
    if sha(result_path) != expected_hash:
        raise ValueError('봉인 후보 보고서 변경')
    report = json.loads(result_path.read_text())
    selected, selected_hash, selection = response_selection(report, result_path, rejected_evaluation)
    accepted = result_path.parent / 'accepted-response.npz'
    inputs = {result_path: expected_hash, accepted: report['accepted_output_sha256'], selected: selected_hash}
    if selection:
        inputs[selected.parent / 'result.json'] = selection['evaluation_report_sha256']
    paths = {}
    for name, suffix in (('snapshot', '/material-snapshot.npz'), ('nodes', '/native_material_nodes.bin')):
        matches = [(Path(p), h) for p, h in report['input_sha256'].items() if p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('원 물질 입력 경로 누락/중복: ' + name)
        paths[name], digest = matches[0]
        inputs[paths[name]] = digest
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('물질/수용 후보 입력 변경')
    with np.load(paths['snapshot']) as f:
        initial = {k: f[k] for k in ('columns', 'superlevel', 'temperature', 'ne')}
    with np.load(selected) as f:
        candidate = {k: f[k] for k in ('columns', 'population', 'temperature', 'ne')}
    with paths['nodes'].open('rb') as stream:
        tag, ns, nc, nl, epoch = struct.unpack('<8sIIId', stream.read(28))
        columns = np.fromfile(stream, dtype='<i4', count=3 * nc).reshape(nc, 3)
        physical = np.fromfile(stream, dtype='<f8', count=4 * ns).reshape(ns, 4)
        native_population = np.fromfile(stream, dtype='<f8', count=ns * nc).reshape(ns, nc).T
        trailing = stream.read(1)
    if (tag != b'LUMNOD01' or ns != 90 or not np.isfinite(epoch) or epoch <= 0 or
        trailing or not np.array_equal(native_population, initial['superlevel']) or
        not np.array_equal(columns, initial['columns']) or not np.array_equal(columns, candidate['columns']) or
        not np.array_equal(physical[:, 1], initial['temperature']) or
        not np.array_equal(physical[:, 2], initial['ne']) or
        any(v.shape != (nc, ns) for v in (initial['superlevel'], candidate['population'])) or
        np.any(np.diff(physical[:, 0]) >= 0)):
        raise ValueError('같은 원래90점/원자 주소/물질이 아님')
    values, summaries = {}, {}
    for name, (Z, q) in STAGES.items():
        element = columns[:, 0] == Z
        selected = element & (columns[:, 1] == q)
        if not selected.any():
            raise ValueError('요청한 이온이 없음: ' + name)
        old_total = initial['superlevel'][element].sum(axis=0)
        new_total = candidate['population'][element].sum(axis=0)
        if np.any(old_total <= 0) or not np.allclose(old_total, new_total, rtol=1.e-12, atol=0):
            raise ValueError('원소 보존 실패: ' + name)
        old = initial['superlevel'][selected].sum(axis=0) / old_total
        new = candidate['population'][selected].sum(axis=0) / new_total
        log, summary = diagnostics(old, new)
        values[name] = (old, new, log)
        summaries[name] = summary
    rows = []
    for i in range(ns):
        row = dict(native_node_zero_based=i, radius_cm=float(physical[i, 0]),
            velocity_km_s=float(physical[i, 0] / epoch / 1.e5),
            T_reference_K=float(initial['temperature'][i]), T_candidate_K=float(candidate['temperature'][i]),
            ne_reference_cm3=float(initial['ne'][i]), ne_candidate_cm3=float(candidate['ne'][i]))
        label = 'Lumina' if rejected_evaluation is None else 'Lumina_REJECTED_eval' + str(rejected_evaluation)
        for name, (old, new, log) in values.items():
            row.update({name + '_reference': float(old[i]), name + '_' + label: float(new[i]),
                        name + '_log10_' + label + '_over_reference': float(log[i]) if np.isfinite(log[i]) else ''})
        rows.append(row)
    out.mkdir()
    table = out / 'profiles.csv'
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('비교 중 원본 변경')
    summary = dict(state=('COMPLETE_SAME_NATIVE_NODE_IONIZATION_COMPARISON' if rejected_evaluation is None
                          else 'COMPLETE_REJECTED_CANDIDATE_IONIZATION_DIAGNOSTIC'),
        **selection,
        baseline='봉인된 native 원 물질 입력; 물리 출처와 앵커 자격은 별도 확인 필요',
        producer_declared_baseline=report.get('comparison_baseline'),
        source_run=result_path.parent.parent.name, points=ns, summary=summaries,
        input_sha256=dict(result=expected_hash, accepted=inputs[accepted], candidate_response=selected_hash,
            snapshot=inputs[paths['snapshot']], nodes=inputs[paths['nodes']]),
        csv_sha256=sha(table), node_mapping='동일 원래90점; shell 평균/중심 보간 없음',
        weighting='동일 가중 점별 통계; 최종 v2 질량가중 상태 판정 아님',
        active_scope='fraction>1e-8인 합집합 부분집합. 율/열/불투명도 활성 항목의 완전한 합집합 판정 아님',
        zero_floor_applied=False, new_rates=0, new_transport=0, final_equivalence='NOT_JUDGED',
        final_goal='NOT_MET', source_script_sha256=sha(Path(__file__)))
    (out / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rejected-evaluation', type=int,
                        help='명시적으로 거부된 실제 후보를 수용 물질과 구분한 진단 전용 비교')
    args = parser.parse_args()
    compare(args.result, args.sha256, args.out, args.rejected_evaluation)
