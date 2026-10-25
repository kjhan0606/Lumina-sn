"""SC 완료·물질·입력장 계보만 검사한다. 합성 파일은 실제 물리 장이 아니다."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_lumina_fullsr_sc as C


def record(tmp_path):
    field = tmp_path/'synthetic-J'; field.write_bytes(b'NOT_PHYSICAL_FIELD')
    j = dict(path=str(field), sha256=C.N.sha(field))
    data = dict(schema='lumina-fullsr-sc-native90-v1', complete_original_grid=True, next_frequency=-1,
        completed_frequencies=1115020, source_identity={'fixture': True},
        original_layout=dict(physical_intervals=16172), identity=dict(input_J='ZERO_OWN_START'),
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
