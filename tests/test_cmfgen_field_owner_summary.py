"""완료/phase/전깊이 coverage를 확인한 뒤에만 인구 응답을 해석한다."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_cmfgen_field_owner_pair import read_response,verify_certificate_transition


def sample(tmp_path, defect=None):
    rows = []
    ref = {}
    for Z in (14, 16):
        for s in range(2, 7):
            ref[1, Z, s] = .2
            rows.append(dict(depth_1based=1, Z=Z, stage=s, pre_fraction=.2,
                candidate_fraction=.2, committed_fraction=.2, Te_K=10000., ne_cm3=1e7,
                n_element=1e6, scaled_se_residual=1e-15, conservation_residual=0.,
                kappa_2=1e11, completed=1, candidate_valid=1))
    if defect == 'duplicate':
        rows[-1] = rows[0]
    if defect == 'phase':
        rows[0]['pre_fraction'] = .3
    if defect == 'invalid':
        rows[0]['candidate_valid'] = 0
    if defect == 'element_sum':
        rows[0]['candidate_fraction'] = .3
    path = tmp_path / 'response.csv'; log = tmp_path / 'owner.log'
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    log.write_text('CMF_OWNER_RESPONSE_COMPLETE depths=1\n' if defect != 'partial' else 'working\n')
    return path, log, ref


def test_valid_full_response(tmp_path):
    path, log, ref = sample(tmp_path)
    assert len(read_response(path, log, ref, 1)) == 10


@pytest.mark.parametrize('defect', ['duplicate', 'phase', 'invalid', 'element_sum', 'partial'])
def test_inconsistent_response_is_not_accepted(tmp_path, defect):
    path, log, ref = sample(tmp_path, defect)
    with pytest.raises(ValueError):
        read_response(path, log, ref, 1)


def transition_rows(tmp_path):
    path,log,ref=sample(tmp_path);old=read_response(path,log,ref,1)
    for row in old.values():
        row.update(dimension='3',raw_rank='1',graph_components='1',topology_pass='0',numerical_pass='0')
    new={k:dict(r) for k,r in old.items()}
    for row in new.values():
        row.update(legacy_topology_pass='0',legacy_numerical_pass='0',adoption_path='CERTIFIED_NUMERICAL',
            certificate_attempted='1',certificate_status='0',certificate_pass='1',
            certificate_probability_error_display='1e-16',certificate_relative_error_display='2e-16',
            commit_performed='1',topology_pass='1',numerical_pass='1',boundary_pass='1')
    return new,old


def test_certificate_transition_retains_original_candidate_and_legacy_flags(tmp_path):
    new,old=transition_rows(tmp_path)
    result=verify_certificate_transition(new,old)
    assert result['certified']==2 and result['legacy_flags_preserved']


@pytest.mark.parametrize('field,value',[('candidate_fraction','.3'),('scaled_se_residual','1e-16'),
    ('legacy_numerical_pass','1'),('certificate_pass','0'),('boundary_pass','0')])
def test_certificate_does_not_hide_changed_candidate_old_verdict_or_failed_physics(tmp_path,field,value):
    new,old=transition_rows(tmp_path);new[1,14,3][field]=value
    with pytest.raises(ValueError):verify_certificate_transition(new,old)
