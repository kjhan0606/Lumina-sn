"""SC 완료·물질·입력장 계보만 검사한다. 합성 파일은 실제 물리 장이 아니다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_lumina_fullsr_sc as C


def record(tmp_path):
    field = tmp_path/'synthetic-J'; field.write_bytes(b'NOT_PHYSICAL_FIELD')
    j = dict(path=str(field), sha256=C.N.sha(field))
    data = dict(schema='lumina-fullsr-sc-native90-v1', complete_original_grid=True, next_frequency=-1,
        completed_frequencies=1115020, source_identity={'fixture': True},
        original_layout=dict(physical_intervals=16172, frequencies=1115020), identity=dict(input_J='ZERO_OWN_START'),
        completed_actual_F_calls=1, J=j)
    path = tmp_path/'result.json'; path.write_text(json.dumps(data))
    return path, data


@pytest.mark.parametrize('bad', ['partial', 'source', 'input', 'count', 'intervals', 'hash'])
def test_incomplete_or_mixed_fields_are_not_sent_to_rates(tmp_path, bad):
    path, data = record(tmp_path)
    if bad == 'partial': data['complete_original_grid'] = False
    elif bad == 'source': data['source_identity'] = {'wrong': True}
    elif bad == 'input': data['identity']['input_J'] = 'CMFGEN_REFERENCE'
    elif bad == 'count': data['completed_frequencies'] -= 1
    elif bad == 'intervals': data['original_layout']['physical_intervals'] -= 1
    else: data['J']['sha256'] = 'BAD'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): C.completed_entry(path, {'fixture': True}, 16172, 1)


def test_sc_report_preserves_its_actual_method_and_own_field_lineage(tmp_path):
    path, data = record(tmp_path)
    first = C.completed_entry(path, {'fixture': True}, 16172, 1)
    assert first['report']['transport'] == 'FULLSR_INVARIANT_UPWIND_SC'
    assert first['report']['accepted_physical_field'] is False
    data['identity']['input_J'] = first['J']; path.write_text(json.dumps(data))
    second = C.completed_entry(path, {'fixture': True}, 16172, 2, first['J'])
    assert second['input_iteration'] == 1 and second['iteration'] == 2


def test_adapter_records_method_and_input_identities_without_another_transport(tmp_path, monkeypatch):
    path, data = record(tmp_path)
    config = tmp_path/'config'; config.write_text('{}')
    base = tmp_path/'base.json'
    base.write_text(json.dumps(dict(config=str(config), preparation='SYNTHETIC',
        preparation_sha256='SYNTHETIC', source_identity=data['source_identity'])))
    monkeypatch.setattr(C, 'BASE', base); monkeypatch.setattr(C, 'BASE_SHA', C.N.sha(base))
    monkeypatch.setattr(C, 'verify_stage', lambda root: 'SYNTHETIC_STAGE')
    class Source:
        identity = data['source_identity']
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(C.N, 'open_sealed_native', lambda *args: Source())
    calls = []
    def rates(config, source, contract, entries, out, deadline):
        calls.append(entries); Path(out).mkdir()
        return dict(synthetic_only=True, accepted_physical_field=False)
    monkeypatch.setattr(C.R, 'compare', rates)
    out = tmp_path/'rates'; C.compare(path, out)
    evidence = json.loads((out/'transport-comparison.json').read_text())
    assert len(calls) == 1 and calls[0][0]['J'] == data['J']
    assert evidence['completed_fields'][0]['report']['transport'] == 'FULLSR_INVARIANT_UPWIND_SC'
    assert evidence['result_records'][str(path)] == C.N.sha(path)
    assert evidence['accepted_physical_field'] is False and evidence['final_goal'] == 'NOT_MET'


def test_pair_adapter_reports_j1_residual_not_j2_convergence(tmp_path, monkeypatch):
    first, data = record(tmp_path)
    for name, value in (('J1', 1.), ('J2', 1.25)):
        np.save(tmp_path/(name+'.npy'), np.full((2, 8), value))
    data['J'] = dict(path=str(tmp_path/'J1.npy'), sha256=C.N.sha(tmp_path/'J1.npy'))
    first.write_text(json.dumps(data))
    second_data = json.loads(json.dumps(data)); second_data['identity']['input_J'] = data['J']
    second_data['J'] = dict(path=str(tmp_path/'J2.npy'), sha256=C.N.sha(tmp_path/'J2.npy'))
    second = tmp_path/'second.json'; second.write_text(json.dumps(second_data))
    config = tmp_path/'config'; config.write_text('{}'); base = tmp_path/'base.json'
    base.write_text(json.dumps(dict(config=str(config), preparation='SYNTHETIC',
        preparation_sha256='SYNTHETIC', source_identity=data['source_identity'])))
    monkeypatch.setattr(C, 'BASE', base); monkeypatch.setattr(C, 'BASE_SHA', C.N.sha(base))
    monkeypatch.setattr(C, 'verify_stage', lambda root: 'SYNTHETIC_STAGE')
    class Source:
        identity = data['source_identity']
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(C.N, 'open_sealed_native', lambda *args: Source())
    def rates(config, source, contract, entries, out, deadline):
        assert len(entries) == 2; Path(out).mkdir()
        return dict(synthetic_only=True, accepted_physical_field=False)
    monkeypatch.setattr(C.R, 'compare', rates)
    out = tmp_path/'rates'; C.compare(first, out, second=second)
    evidence = json.loads((out/'transport-comparison.json').read_text())
    assert evidence['J1_residual']['relative'] == .25
    assert evidence['J1_residual']['J2_residual'] == 'NOT_MEASURED'
    assert evidence['final_goal'] == 'NOT_MET'
