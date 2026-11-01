"""완료 방향의 읽기 전용 재사용과 실행 모드 경계 시험. 실제 물리 callback 없음."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_new_field_response as N
import lumina_thermal_replay as R
from test_lumina_thermal_refinement import fixture


@pytest.fixture
def saved(tmp_path):
    _, source = fixture()
    root = tmp_path/'run'; root.mkdir(); (tmp_path/'build').mkdir()
    libraries = [tmp_path/'build'/name for name in ('bridge.so', 'lines.so')]
    for path in libraries: path.write_bytes(path.name.encode())
    columns = np.array([[14, 1, 0], [14, 2, 0]])
    entries = []; values = [source['first'], *source['probes']]
    for i, (data, heat, metadata) in enumerate(values):
        data['columns'] = columns
        output = root/('evaluation-'+str(i+1)); output.mkdir()
        np.savez(output/'response.npz', **data); np.savez(output/'balance.npz', **heat)
        metadata.update(root=str(output), response_sha256=R.B.sha(output/'response.npz'),
                        balance_sha256=R.B.sha(output/'balance.npz'), fixed_field_thermal_callback=i>0)
        entries.append(metadata)
    basis = source['basis']; field = {'J':'pinned'}; identity = {'material':'pinned'}
    direction = N.preserve_thermal_direction(root, 1, source['first'], basis['jacobian'],
        basis['scaled_residual'], basis['log_direction'], basis['charge_scale'], basis['thermal_scale'],
        field=field, source_identity=identity, probes=entries[1:], policy='physical_stored')
    report = dict(state='COMPLETE_CURRENT_FIELD_THERMAL_DIRECTION_NOT_ACCEPTED_MATERIAL',
        field=field, source_identity=identity, thermal_energy_policy='physical_stored',
        temperature_direction_only=True, material_published=False, temperature_updated=False,
        temperature_linearization=dict(evaluations_used=4, accepted_iterations=0),
        thermal_initial_response=entries[0], thermal_evaluation_order=entries[1:],
        thermal_directions=[direction], thermal_implementation_sha256={'physics':'unchanged'},
        input_sha256={str(p):R.B.sha(p) for p in libraries},
        output_sha256={str(p.relative_to(root)):R.B.sha(p) for p in (
            root/'thermal-direction-1/result.json', root/'thermal-direction-1/linearization.npz')})
    args = dict(field=field, source_identity=identity, implementation=report['thermal_implementation_sha256'],
        state=dict(columns=columns, superlevel=np.ones((2, 2))), library=libraries[0], line_library=libraries[1],
        inputs={}, policy='physical_stored', direction_source=True)
    def read(**updates):
        path = root/'result.json'; path.write_text(json.dumps(report))
        return R.load(path, R.B.sha(path), **(args|updates))
    return root, report, args, read


def test_completed_direction_exposes_four_probes_and_exact_basis_without_acceptance(saved):
    root, report, args, read = saved
    result = read()
    assert len(result['replay']) == 4
    np.testing.assert_array_equal(result['direction_basis']['temperature'], result['first'][0]['temperature'])
    assert result['provenance']['centered_direction_source']
    assert not result['provenance']['previous_acceptance_reused']
    assert result['provenance']['new_material_evaluations'] == 0
    assert result['projection'] is None
    assert str(root/'thermal-direction-1/linearization.npz') in args['inputs']


@pytest.mark.parametrize('fault', ['mode', 'prefix', 'projection', 'policy', 'incomplete', 'count',
                                    'base', 'probe_order', 'matrix', 'record', 'publication'])
def test_only_exact_completed_direction_can_be_reused(saved, fault):
    root, report, args, read = saved
    updates = {}
    if fault == 'mode': updates['direction_source'] = False
    elif fault == 'prefix': updates['prefix'] = 4
    elif fault == 'projection': updates['project_charge'] = True
    elif fault == 'policy': updates['policy'] = 'native_stored'
    elif fault == 'incomplete': report['state'] = 'INCOMPLETE_PRESERVED'
    elif fault == 'count': report['thermal_evaluation_order'].pop()
    elif fault == 'base': report['thermal_initial_response'] = report['thermal_evaluation_order'][0]
    elif fault == 'probe_order': report['thermal_evaluation_order'].reverse()
    elif fault == 'matrix': (root/'thermal-direction-1/linearization.npz').write_bytes(b'changed')
    elif fault == 'record': (root/'thermal-direction-1/result.json').write_text('{}')
    else: report['material_published'] = True
    with pytest.raises(ValueError): read(**updates)


@pytest.mark.parametrize('fault', ['seed', 'seed_sha', 'direction', 'direction_sha', 'prefix',
                                    'projection', 'secant', 'mode', 'duration', 'budget'])
def test_refinement_requires_pinned_inputs_and_no_mixed_mode_before_output(fault):
    args = dict(seconds=42600, temperature_evaluations=12, refine_temperature_direction=True,
        resume_result='baseline', resume_result_sha='baseline-sha',
        resume_thermal_result='direction', resume_thermal_result_sha='direction-sha')
    if fault == 'seed': args['resume_result'] = None
    elif fault == 'seed_sha': args['resume_result_sha'] = None
    elif fault == 'direction': args['resume_thermal_result'] = None
    elif fault == 'direction_sha': args['resume_thermal_result_sha'] = None
    elif fault == 'prefix': args['thermal_replay_prefix'] = 4
    elif fault == 'projection': args['charge_projection'] = True
    elif fault == 'secant': args['secant_after_accept'] = True
    elif fault == 'mode': args['temperature_direction_only'] = True
    elif fault == 'duration': args['seconds'] = 42601
    else: args['temperature_evaluations'] = 13
    with pytest.raises(ValueError, match='유한 온도수정'): N.run(SimpleNamespace(**args))


@pytest.mark.parametrize('executor,hostname,seconds', [
    ('manual-lageunha','lageunha',43200), ('slurm','syntax',43200), ('slurm','grammar058',108000)])
def test_refinement_cannot_expand_executor_or_time(monkeypatch, executor, hostname, seconds):
    monkeypatch.setattr(N, 'execution_resources', lambda *a:dict(
        executor=executor, hostname=hostname, long_thermal=True, allocated_seconds=seconds))
    with pytest.raises(ValueError, match='grammar Slurm'):
        N.run(SimpleNamespace(seconds=42600, temperature_evaluations=12, refine_temperature_direction=True,
            resume_result='base', resume_result_sha='sha', resume_thermal_result='dir', resume_thermal_result_sha='sha'))
