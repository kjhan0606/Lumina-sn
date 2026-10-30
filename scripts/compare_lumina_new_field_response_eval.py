#!/usr/bin/env python3
"""저장된 새 복사장 물질 평가의 Si/S 분포를 read-only로 재현한다.

분율 비교를 다시 계산할 뿐, 수용·수렴 판정이나 BF/BB/SE/수송을 수행하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
from lumina_new_field_response import comparison


def input_path(inputs, basename):
    matches = [Path(path) for path in inputs if Path(path).name == basename]
    if len(matches) != 1:
        raise ValueError(f'봉인 입력 {basename}의 주소가 하나로 특정되지 않음')
    return matches[0]


def evaluation_files(result, report, index):
    evaluations = report.get('evaluations')
    if type(index) is not int or not isinstance(evaluations, list) or not 1 <= index <= len(evaluations):
        raise ValueError('완료된 평가 인덱스가 결과에 없음')
    entry = evaluations[index-1]
    directory = result.parent/f'evaluation-{index}'
    if Path(entry.get('root', '')).resolve() != directory.resolve():
        raise ValueError('평가 결과 주소 불일치')
    response = directory/'response.npz'; balance = directory/'balance.npz'
    if B.sha(response) != entry.get('response_sha256') or B.sha(balance) != entry.get('balance_sha256'):
        raise ValueError('저장 평가 응답/열장부 해시 불일치')
    return entry, response, balance


def checked_input(report, basename):
    path = input_path(report['input_sha256'], basename)
    if B.sha(path) != report['input_sha256'][str(path)]:
        raise ValueError(f'봉인 입력 변경: {basename}')
    return path


def accepted_manifest(preparation):
    manifest_path = Path(preparation['accepted_material']).resolve()
    expected_manifest = preparation.get('input_sha256', {}).get(str(manifest_path))
    if not expected_manifest or B.sha(manifest_path) != expected_manifest:
        raise ValueError('수용 물질 manifest의 봉인 해시 불일치')
    return json.loads(manifest_path.read_text())


def accepted_material_paths(preparation):
    manifest = accepted_manifest(preparation)
    material = Path(preparation['material']).resolve()
    expected_material = manifest.get('output_sha256', {}).get('native_material_nodes.bin')
    if not expected_material or B.sha(material) != expected_material:
        raise ValueError('수용 입력 물질 파일의 봉인 해시 불일치')
    return material


def original_reference_path(preparation, report):
    # Match the producer: the original is an input to the accepted-material
    # manifest, not the accepted output recorded in field preparation.
    manifest = accepted_manifest(preparation)
    inputs = manifest.get('input_sha256', {})
    reference = input_path(inputs, 'native_material_nodes.bin')
    expected = inputs[str(reference)]
    if (not expected or B.sha(reference) != expected or
            report.get('input_sha256', {}).get(str(reference)) != expected):
        raise ValueError('원 기준 물질의 manifest/실행 봉인 해시 불일치')
    return reference


def compare_result(result_path, expected_sha256, index):
    result = Path(result_path).resolve()
    if B.sha(result) != expected_sha256:
        raise ValueError('결과 보고서 SHA256 불일치')
    report = json.loads(result.read_text())
    entry, response_path, balance_path = evaluation_files(result, report, index)

    prep_path = checked_input(report, 'preparation.json')
    feedback_path = checked_input(report, 'feedback.json')
    model = checked_input(report, 'MODEL').parent
    model_spec = checked_input(report, 'MODEL_SPEC')
    links = checked_input(report, 'atomic_links.txt')
    levels = checked_input(report, 'levels.csv')
    preparation = json.loads(prep_path.read_text())
    feedback = json.loads(feedback_path.read_text())
    accepted_path = accepted_material_paths(preparation)

    reference_path = original_reference_path(preparation, report)

    rows = list(csv.DictReader(levels.open()))
    models = B.atomic.load_cmfgen_model_spec(model_spec, links)
    reference = B.read_material(reference_path, rows, models)
    accepted = B.read_material(accepted_path, rows, models)
    with np.load(response_path, allow_pickle=False) as saved:
        columns = saved['columns']; population = saved['population']
        temperature = saved['temperature']; ne = saved['ne']
    if (not np.array_equal(columns, reference['columns']) or
            not np.array_equal(columns, accepted['columns']) or
            not np.array_equal(reference['physical'][:, 0], accepted['physical'][:, 0])):
        raise ValueError('원자 주소 또는 shell 반경 불일치')
    summary, _ = comparison(columns, reference['superlevel'], accepted['superlevel'],
                            population, accepted['physical'][:, 0])
    j_sha = feedback.get('last_complete', {}).get('J', {}).get('sha256')
    if not j_sha:
        raise ValueError('완료 복사장의 SHA가 feedback 기록에 없음')
    return dict(
        diagnostic='DIAGNOSTIC_ONLY_NOT_FULL_V2_ACCEPTANCE',
        result_state=report.get('state'), final_goal=report.get('final_goal'),
        selected_response_is_this_evaluation=(report.get('selected_response') == entry),
        evaluation_index=index, generation=entry.get('generation'),
        baseline='ACCEPTED_MANIFEST_ORIGINAL_NATIVE_MATERIAL_NOT_INDEPENDENTLY_CERTIFIED',
        producer_declared_baseline=report.get('comparison_baseline'), input_J_sha256=j_sha,
        result_sha256=expected_sha256, response_sha256=entry['response_sha256'],
        balance_sha256=entry['balance_sha256'], reference_material_sha256=B.sha(reference_path),
        accepted_material_sha256=B.sha(accepted_path),
        accepted_population_equals_reference=bool(np.array_equal(
            reference['superlevel'], accepted['superlevel'])),
        max_relative_temperature_change=float(np.max(abs(temperature/accepted['temperature']-1))),
        max_relative_ne_change=float(np.max(abs(ne/accepted['ne']-1))),
        activity_basis='ION_FRACTION_ONLY; RATE/HEAT/OPACITY ACTIVITY NOT AVAILABLE',
        ion_summary=[row for row in summary if row['Z'] in (14, 16)],
        new_SE=0, new_BF_BB_integrals=0, new_transport=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--evaluation-index', type=int, required=True,
                        help='result.json 내 1기준 evaluations 순번')
    args = parser.parse_args()
    result = compare_result(args.result, args.result_sha256, args.evaluation_index)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
