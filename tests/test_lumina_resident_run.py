"""새 전체장 출력·계보 연결의 소형 CPU 시험. 실제 native90 F 실행은 없다."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_fullsr_sc_actual import InputField, completed_transport
from lumina_resident_run import run, write_completed_moments
import compare_lumina_fullsr_sc as C


def resident_record(nf=8):
    return dict(schema='lumina-resident-fullsr-native90-v1', complete_original_grid=True,
        completed_actual_F_calls=1, source_identity={'fixture': True},
        original_layout=dict(frequencies=nf, physical_intervals=16172, paths=107,
            step_observations=11160, boundary_observations=17),
        field_report=dict(complete=True, failed=False, physical_intervals=16172,
            observations=11177, next_action=16296, actions=16296), identity=dict(input_J='ZERO_OWN_START'))


def test_complete_output_preserves_signed_h_and_connects_input_lineage(tmp_path):
    values = np.ones((3, 2, 8)); values[1] *= -.2
    output = write_completed_moments(tmp_path, SimpleNamespace(complete_moments=lambda: values))
    np.testing.assert_array_equal(np.load(output['moments']['path']), values)
    source = SimpleNamespace(ns=2, nf=8, radius=np.array([2., 1.]), frequency=np.arange(8.), identity={'fixture': True})
    record = dict(resident_record(), **output)
    (tmp_path/'resident-field.json').write_text(json.dumps(record))
    field = InputField(source, output['J']['path'], output['J']['sha256'])
    try: np.testing.assert_array_equal(field.values, values[0])
    finally: field.close()
    with pytest.raises(FileExistsError): write_completed_moments(tmp_path, SimpleNamespace(complete_moments=lambda: values))
    (tmp_path/'sc-result.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='하나로 특정'): InputField(source, output['J']['path'], output['J']['sha256'])


def test_partial_field_writes_no_completed_files(tmp_path):
    def incomplete(): raise ValueError('부분/실패 장')
    with pytest.raises(ValueError, match='부분/실패'): write_completed_moments(tmp_path, SimpleNamespace(complete_moments=incomplete))
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize('bad', ['cursor', 'observations', 'intervals', 'failure', 'partial', 'count', 'frequencies'])
def test_resident_completion_rejects_incomplete_or_mixed_geometry(bad):
    record = resident_record()
    if bad == 'cursor': record['field_report']['next_action'] -= 1
    elif bad == 'observations': record['field_report']['observations'] -= 1
    elif bad == 'intervals': record['field_report']['physical_intervals'] -= 1
    elif bad == 'failure': record['field_report']['failed'] = True
    elif bad == 'partial': record['complete_original_grid'] = False
    elif bad == 'count': record['completed_actual_F_calls'] = 0
    else: record['original_layout']['frequencies'] -= 1
    assert not completed_transport(record, 8)


def test_rate_consumer_labels_resident_method_not_coarse_upwind(tmp_path):
    data = resident_record(1115020); field = tmp_path/'fixture-not-physical-J'
    field.write_bytes(b'NOT_PHYSICAL_FIELD'); data['J'] = dict(path=str(field), sha256=C.N.sha(field))
    path = tmp_path/'resident-field.json'; path.write_text(json.dumps(data))
    entry = C.completed_entry(path, data['source_identity'], 16172, 1)
    assert entry['report']['transport'] == 'FULLSR_ACTUAL_PHASE_RESIDENT_CHAIN'
    assert entry['report']['accepted_physical_field'] is False
    data['identity']['input_J'] = entry['J']; path.write_text(json.dumps(data))
    assert C.completed_entry(path, data['source_identity'], 16172, 2, entry['J'])['input_iteration'] == 1


def test_login_cli_is_rejected_before_any_output(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='Slurm 계산 노드'): run(tmp_path, cache=tmp_path/'absent')
    assert not tuple(tmp_path.iterdir())
