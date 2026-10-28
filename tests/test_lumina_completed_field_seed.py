"""완료 계수 재사용의 같은 장·물질·생산자 및 실제 배열 대조 시험."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_completed_field_seed as C


@pytest.fixture
def archive(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    (tmp_path/'build').mkdir(); (tmp_path/'source/scripts').mkdir(parents=True)
    for name in ('bridge.so', 'lines.so'):
        (tmp_path/'build'/name).write_bytes(name.encode())
    for name in C.PRODUCERS:
        (tmp_path/'source/scripts'/name).write_bytes(Path(C.__file__).with_name(name).read_bytes())
    state = dict(temperature=np.array([9000., 10000.]), ne=np.array([20., 40.]),
                 columns=np.array([[14, 1, 0], [14, 2, -1]]),
                 superlevel=np.array([[3., 4.], [7., 16.]]))
    np.save(root/'bound-rates.npy', np.ones((3, 2, 6)))
    np.save(root/'jbar.npy', np.ones((4, 2)))
    np.save(root/'line-status.npy', np.array([1, 2, 1, 1], dtype='u1'))
    np.savez(root/'route-roster.npz', lower=np.arange(3))
    np.savez(root/'FF-coefficients.npz', signature='ff', temperature=state['temperature'],
             heating=np.ones((2, 2)), cooling=np.ones((2, 2)))
    np.savez(root/'physical-profiles.npz', accepted_temperature=state['temperature'], accepted_ne=state['ne'])
    evaluation = root/'evaluation-1'; evaluation.mkdir()
    np.savez(evaluation/'response.npz', temperature=state['temperature'], ne=state['ne']*1.01,
             columns=state['columns'], population=state['superlevel'], charge_residual=np.zeros(2))
    entry = dict(root=str(evaluation), response_sha256=C.B.sha(evaluation/'response.npz'))
    inputs = {str(tmp_path/'MODEL'): 'model'}
    for name in ('bridge.so', 'lines.so'):
        inputs[str(tmp_path/'build'/name)] = C.B.sha(tmp_path/'build'/name)
    report = dict(state='COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE',
        fixed_accepted_temperature=True, charge_converged=True, full_BF_integrations=1,
        full_BF_seconds=7000., field={'J': 'new-J', 'sweep': 1}, source_identity={'material': 'accepted'},
        input_sha256=inputs.copy(), selected_response=entry, evaluations=[entry],
        output_sha256={name:C.B.sha(root/name) for name in (*C.COEFFICIENTS, 'physical-profiles.npz')})
    args = dict(field=report['field'].copy(), source_identity=report['source_identity'].copy(),
                state=state, inputs=inputs.copy(), library=tmp_path/'build/bridge.so',
                line_library=tmp_path/'build/lines.so', routes=3, lines=4)
    return root, report, args


def load(archive):
    root, report, args = archive
    path = root/'result.json'; path.write_text(json.dumps(report))
    return C.load(path, C.B.sha(path), **args)


def change_npz(path, **updates):
    with np.load(path) as f:
        data = {k:f[k] for k in f.files}
    np.savez(path, **{**data, **updates})


def test_completed_coefficients_are_read_only_and_ne_is_only_a_starting_guess(archive):
    root, report, args = archive
    before = {name:(root/name).read_bytes() for name in C.COEFFICIENTS}
    seed = load(archive)
    np.testing.assert_array_equal(seed['ne'], args['state']['ne']*1.01)
    assert seed['provenance']['saved_heat_reused'] is False
    assert seed['provenance']['original_full_BF_seconds'] == 7000.
    assert 'balance.npz' not in seed['paths']
    for name, value in before.items():
        assert (root/name).read_bytes() == value
        assert str(root/name) in args['inputs']


@pytest.mark.parametrize('fault', ['incomplete', 'not_fixed', 'not_charge', 'multiple_BF',
    'field', 'material', 'input', 'library', 'line_library', 'producer', 'temperature', 'ne', 'columns'])
def test_mixed_context_is_rejected(archive, fault):
    root, report, args = archive
    if fault == 'incomplete': report['state'] = 'INCOMPLETE_PRESERVED'
    elif fault == 'not_fixed': report['fixed_accepted_temperature'] = False
    elif fault == 'not_charge': report['charge_converged'] = False
    elif fault == 'multiple_BF': report['full_BF_integrations'] = 2
    elif fault == 'field': args['field'] = {'J': 'old-J', 'sweep': 1}
    elif fault == 'material': args['source_identity'] = {'material': 'old70'}
    elif fault == 'input': args['inputs'][str(root.parent/'MODEL')] = 'changed'
    elif fault in ('library', 'line_library'): args[fault].write_bytes(b'changed')
    elif fault == 'producer':
        (root.parent/'source/scripts'/C.PRODUCERS[0]).write_text('changed')
    elif fault in ('temperature', 'ne'): args['state'][fault] *= 1.01
    else: args['state']['columns'][0, 1] = 0
    with pytest.raises(ValueError): load(archive)


@pytest.mark.parametrize('name', ['bound-rates.npy', 'jbar.npy', 'FF-coefficients.npz',
                                'route-roster.npz', 'line-status.npy', 'physical-profiles.npz'])
def test_corrupted_coefficient_is_rejected(archive, name):
    root, _, _ = archive
    (root/name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='파일 변경'): load(archive)


def test_wrong_report_pin_is_rejected(archive):
    root, _, args = archive
    load(archive)
    with pytest.raises(ValueError, match='보고서 변경'):
        C.load(root/'result.json', '0'*64, **args)


@pytest.mark.parametrize('fault', ['negative_bound', 'bad_bound_shape', 'nan_jbar',
    'bad_status', 'bad_ff_temperature', 'negative_ff', 'bad_ff_shape', 'nan_charge', 'unclosed_charge',
    'response_hash', 'response_address'])
def test_semantically_bad_sealed_arrays_are_rejected(archive, fault):
    root, report, _ = archive
    name = None
    if fault == 'negative_bound':
        name = 'bound-rates.npy'; np.save(root/name, -np.ones((3, 2, 6)))
    elif fault == 'bad_bound_shape':
        name = 'bound-rates.npy'; np.save(root/name, np.ones((3, 1, 6)))
    elif fault == 'nan_jbar':
        name = 'jbar.npy'; np.save(root/name, np.full((4, 2), np.nan))
    elif fault == 'bad_status':
        name = 'line-status.npy'; np.save(root/name, np.zeros(4, dtype='u1'))
    elif fault.startswith('bad_ff') or fault == 'negative_ff':
        name = 'FF-coefficients.npz'
        if fault == 'bad_ff_temperature': change_npz(root/name, temperature=np.ones(2))
        elif fault == 'negative_ff': change_npz(root/name, heating=-np.ones((2, 2)))
        else: change_npz(root/name, heating=np.ones((1, 2)))
    elif fault in ('nan_charge', 'unclosed_charge'):
        path = root/'evaluation-1/response.npz'
        change_npz(path, charge_residual=np.full(2, np.nan if fault == 'nan_charge' else 1.))
        report['selected_response']['response_sha256'] = C.B.sha(path)
    elif fault == 'response_hash':
        (root/'evaluation-1/response.npz').write_bytes(b'changed')
    else: report['selected_response']['root'] = str(root/'outside')
    if name: report['output_sha256'][name] = C.B.sha(root/name)
    with pytest.raises(ValueError): load(archive)
