"""진행 중 부모와 완료 부분자료를 구분하고 불변 입력만 재사용한다."""
import json
from pathlib import Path

import pytest

import lumina_completed_thermal_prefix as P
import lumina_thermal_replay as R
from test_lumina_thermal_refinement_reuse import saved


@pytest.fixture
def prefix_saved(saved):
    root, report, args, _ = saved
    report.update(state='STARTED', BB_contract='SOURCE_SAMPLED_TRAPEZOID_A_TIMES_MASS_V1',
                  evaluations=[report['thermal_initial_response'], *report['thermal_evaluation_order']])
    for name in P.COEFFICIENTS:
        (root/name).write_bytes(name.encode())
    parent = root/'result.json'; parent.write_text(json.dumps(report))
    output = root.parent/'completed-prefix.json'
    return root, report, args, parent, output


def test_live_parent_prefix_reuses_exact_completed_probes_without_certifying_completion(prefix_saved):
    root, report, args, parent, output = prefix_saved
    before = parent.read_bytes()
    digest = P.capture(parent, output)
    assert parent.read_bytes() == before
    result = R.load(output, digest, **args)
    assert len(result['replay']) == 4
    assert result['provenance']['source_report_state'] == 'STARTED'
    assert result['provenance']['source_completion_certified'] is False
    # 원 보고서의 이후 진행은 완료 배열의 재사용을 무효화하지 않는다.
    parent.write_text(json.dumps(report | {'phase': 'later evaluation'}))
    again = R.load(output, digest, **args)
    assert again['first'][2] == result['first'][2]


@pytest.mark.parametrize('fault', ['probe', 'coefficient', 'snapshot', 'direction', 'mode'])
def test_prefix_rejects_mutation_and_implicit_replay(prefix_saved, fault):
    root, report, args, parent, output = prefix_saved
    digest = P.capture(parent, output)
    if fault == 'probe':
        (Path(report['thermal_evaluation_order'][0]['root'])/'response.npz').write_bytes(b'changed')
    elif fault == 'coefficient':
        (root/'bound-rates.npy').write_bytes(b'changed')
    elif fault == 'direction':
        (root/'thermal-direction-1/result.json').write_text('{}')
    elif fault == 'mode':
        args['direction_source'] = False
    else:
        value = json.loads(output.read_text()); value['parent_report_json'] += ' '
        output.write_text(json.dumps(value)); digest = P.B.sha(output)
    with pytest.raises(ValueError):
        R.load(output, digest, **args)


def test_incomplete_probes_and_existing_output_are_not_relabelled(prefix_saved):
    root, report, args, parent, output = prefix_saved
    P.capture(parent, output)
    with pytest.raises(ValueError, match='별도 새'):
        P.capture(parent, output)
    report['thermal_evaluation_order'].pop()
    parent.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='결손'):
        P.capture(parent, output.with_name('incomplete.json'))
    assert not output.with_name('incomplete.json').exists()
