import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_lumina_final_native_pair as pair


def fixture(tmp_path):
    response = tmp_path / 'response.npz'
    balance = tmp_path / 'balance.npz'
    response.write_bytes(b'selected response')
    balance.write_bytes(b'selected balance')
    report = dict(state='COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE',
                  source_commit='b'*40, field={'J': {'sha256': 'a'*64}},
                  thermal_energy_policy='native_instantaneous_scenario', charge_converged=True,
                  selected_response=dict(root=str(tmp_path), response_sha256=pair.C.sha(response),
                                         balance_sha256=pair.C.sha(balance)))
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(report))
    return dict(result=str(path), producer_commit='b'*40, J_sha256='a'*64), report


def test_completed_selected_identity(tmp_path):
    config, report = fixture(tmp_path)
    actual, identity = pair.final_input(config)
    assert actual == report
    assert identity['response_sha256'] == report['selected_response']['response_sha256']
    assert identity['result_sha256'] == pair.C.sha(Path(config['result']))


@pytest.mark.parametrize('key,value', [('state', 'STARTED'), ('source_commit', 'c'*40),
    ('charge_converged', False), ('thermal_energy_policy', 'physical_stored'),
    ('field', {'J': {'sha256': 'c'*64}})])
def test_wrong_producer_or_contract_refused(tmp_path, key, value):
    config, report = fixture(tmp_path)
    report[key] = value
    Path(config['result']).write_text(json.dumps(report))
    with pytest.raises(ValueError):
        pair.final_input(config)


@pytest.mark.parametrize('name', ['response.npz', 'balance.npz'])
def test_mutated_selected_artifact_refused(tmp_path, name):
    config, _ = fixture(tmp_path)
    (tmp_path/name).write_bytes(b'changed')
    with pytest.raises(ValueError):
        pair.final_input(config)


@pytest.mark.parametrize('fail_first', [False, True])
def test_whole_pair_sequential_preserves_outputs(tmp_path, monkeypatch, fail_first):
    baseline = tmp_path/'baseline'; native = baseline/'native'
    (native/'seq_logs').mkdir(parents=True)
    original = tmp_path/'original'; original.mkdir()
    build = tmp_path/'build'; build.mkdir()
    for name in ('VADAT', 'IN_ITS', 'MODEL_SPEC', 'EDDFACTOR', 'EDDFACTOR_INFO'):
        (native/name).write_text('synthetic')
    for name in ('cmfgen_fixedj_heat.exe', 'fixedj-heat-build.json'):
        (build/name).write_text('synthetic')
    (original/'MODEL').write_text('synthetic')
    (original/'material-seed.json').write_text('{}')
    (native/'seq_logs/feedback-stage.json').write_text(json.dumps(dict(
        original_physical_seed=str(original), material_source=str(original),
        restart_sha256={}, linked_inputs={})))
    config, report = fixture(tmp_path)
    np.savez(tmp_path/'balance.npz', native_instantaneous_scenario_net=np.zeros(90))
    report['selected_response']['balance_sha256'] = pair.C.sha(tmp_path/'balance.npz')
    report.update(input_sha256={str(original/'MODEL'): 'unused',
                               str(original/'native_material_nodes.bin'): 'unused'},
                  temperature_block={'fixed_field_root_converged': False})
    Path(config['result']).write_text(json.dumps(report))
    config.update(output=str(tmp_path/'output'), baseline=str(baseline), build=str(build),
                  pinned_files={}, preparation='unused', preparation_sha256='unused',
                  feedback='unused', feedback_sha256='unused')
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(pair, 'INPUTS', ('MODEL',))
    monkeypatch.setattr(pair, 'RESTART', ())
    monkeypatch.setattr(pair, 'RESTART_OVERLAY', ())
    monkeypatch.setattr(pair.shutil, 'disk_usage', lambda p: SimpleNamespace(free=64*1024**3))
    def outputs(folder, names):
        folder.mkdir()
        for name in names: (folder/name).write_text('synthetic')
    monkeypatch.setattr(pair, 'export_material', lambda *a: outputs(a[-1], ['accepted-material.json']))
    monkeypatch.setattr(pair, 'prepare_state', lambda *a: outputs(a[-1], ['NATIVE_HEAT_INPUT.bin', 'heat-state-input.json']))
    monkeypatch.setattr(pair, 'prepare_field', lambda *a: outputs(a[-1], ['EDDFACTOR', 'EDDFACTOR_INFO', 'fixedj-field-input.json']))
    monkeypatch.setattr(pair, 'refined_grid_inputs', lambda *a, **kw: dict(vadat='v', model_spec='m'))
    calls = []
    def run(root, identity):
        calls.append(root.name)
        assert identity['J_sha256'] == config['J_sha256']
        (root/'native/NATIVE_HEAT_ROW.bin').write_bytes(b'synthetic row')
        return int(fail_first)
    monkeypatch.setattr(pair.C, 'run', run)
    monkeypatch.setattr(pair, 'read_heat_row', lambda p: dict(steq_t_native=np.ones(90), frequencies=10))
    monkeypatch.setattr(pair, 'full_comparison', lambda *a: {'mocked': True})
    if fail_first:
        with pytest.raises(RuntimeError): pair.execute(path, pair.C.sha(path))
        assert calls == ['h05']
    else:
        result = pair.execute(path, pair.C.sha(path))
        assert calls == ['h05', 'h025']
        assert result['producer_root_converged'] is False
        assert result['new_transport'] == result['new_material_update'] == 0
        assert len((tmp_path/'output/native-vs-lumina-heat.csv').read_text().splitlines()) == 91
    with pytest.raises(FileExistsError): pair.execute(path, pair.C.sha(path))
