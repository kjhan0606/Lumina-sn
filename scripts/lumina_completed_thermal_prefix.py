"""진행 중 실행을 완료로 꾸미지 않고 완료 기점/centered4응답만 봉인한다."""
import argparse
import hashlib
import json
from pathlib import Path

import cmfgen_model_continuum as B

SCHEMA = 'LUMINA_COMPLETED_BASE_AND_FOUR_PROBES_V1'
COEFFICIENTS = ('bound-rates.npy', 'route-roster.npz', 'jbar.npy',
                'line-status.npy', 'FF-coefficients.npz', 'line-profile-mass.npy')


def validate(value):
    if value.get('schema') != SCHEMA or value.get('parent_completion_certified') is not False:
        raise ValueError('완료 부분자료의 명시적 신원 오류')
    raw = value['parent_report_json']
    if hashlib.sha256(raw.encode()).hexdigest() != value['parent_report_sha256']:
        raise ValueError('포획한 원 보고서 변경')
    report = json.loads(raw); root = Path(value['producer_run_root'])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError('완료 부분자료의 원 실행 주소 오류')
    base = report.get('thermal_initial_response')
    probes = report.get('thermal_evaluation_order', [])[:4]
    directions = report.get('thermal_directions', [])[:1]
    if (not base or len(probes) != 4 or len(directions) != 1 or
            report.get('BB_contract') != 'SOURCE_SAMPLED_TRAPEZOID_A_TIMES_MASS_V1'):
        raise ValueError('완료 기점/네 응답/첫 방향/계수 계약 결손')
    direction = directions[0]
    if (direction.get('base_response') != base or direction.get('probes') != probes or
            direction.get('linearization_kind') != 'centered' or
            direction.get('centered_log_step') != 1.e-4 or
            direction.get('field') != report.get('field') or
            direction.get('source_identity') != report.get('source_identity') or
            direction.get('energy_policy') != report.get('thermal_energy_policy') or
            Path(direction['root']) != root/'thermal-direction-1'):
        raise ValueError('완료 부분자료의 동일 장/기점/centered 방향 불일치')
    for index, entry in enumerate([base, *probes]):
        location = Path(entry['root'])
        if (location.parent != root or not location.name.startswith('evaluation-') or
                entry not in report.get('evaluations', []) or
                (index and entry.get('fixed_field_thermal_callback') is not True)):
            raise ValueError('완료 부분자료의 응답 주소/순서 오류')
        for name, key in (('response.npz', 'response_sha256'), ('balance.npz', 'balance_sha256')):
            if B.sha(location/name) != entry[key]:
                raise ValueError('완료 부분자료의 응답/열장부 변경')
    expected_names = {*COEFFICIENTS, 'thermal-direction-1/result.json',
                      'thermal-direction-1/linearization.npz'}
    hashes = value['completed_output_sha256']
    if set(hashes) != expected_names or any(B.sha(root/name) != digest for name, digest in hashes.items()):
        raise ValueError('완료 부분자료의 계수/방향 파일 변경')
    if (json.loads((root/'thermal-direction-1/result.json').read_text()) != direction or
            hashes['thermal-direction-1/linearization.npz'] != direction['linearization_sha256']):
        raise ValueError('완료 부분자료의 저장 방향 불일치')
    return report, root, hashes


def unpack(value):
    """일반 보고서는 그대로 두고 명시적 부분자료만 검사한다."""
    return validate(value) if value.get('schema') == SCHEMA else None


def capture(parent, output):
    parent = Path(parent).resolve(); output = Path(output).resolve()
    if output.exists() or output.parent == parent.parent:
        raise ValueError('별도 새 부분자료 경로 필요; 원 실행은 변경하지 않음')
    raw = parent.read_text()
    names = (*COEFFICIENTS, 'thermal-direction-1/result.json',
             'thermal-direction-1/linearization.npz')
    value = dict(schema=SCHEMA, parent_completion_certified=False,
        parent_report_json=raw, parent_report_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        producer_run_root=str(parent.parent),
        completed_output_sha256={name: B.sha(parent.parent/name) for name in names})
    validate(value)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    return B.sha(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(capture(args.parent, args.out))
