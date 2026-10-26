import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_material_restart import common_start,sha


def fixture(tmp_path,*,bad_charge=False,changed_T=False):
    columns=np.array([[z,q,0 if q==1 else -1] for z in (14,16,20,26,27,28) for q in (1,2)])
    population=np.ones((12,3));ne=np.sum(columns[:,1,None]*population,axis=0)
    reference=dict(columns=columns,superlevel=population,temperature=np.full(3,1.e4),ne=ne)
    response=dict(columns=columns,population=population,temperature=reference['temperature']*(1.1 if changed_T else 1.),
        ne=ne,charge_residual=np.full(3,np.nan if bad_charge else 0.),ledger=np.zeros((6,3,7)),energy=np.ones((3,3)))
    directory=tmp_path/'evaluation-1';directory.mkdir()
    for p in (tmp_path/'accepted-response.npz',directory/'response.npz'):np.savez(p,**response)
    np.savez(directory/'balance.npz',native_stored_net=np.zeros(3))
    source=tmp_path/'source-input';source.write_text('unchanged')
    report=dict(state='COMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION',charge_solved=True,
        temperature_updated=False,points=3,history=[dict(evaluation=1,accepted=True)],
        evaluations=[dict(output_sha256={name:sha(directory/name) for name in ('response.npz','balance.npz')})],
        accepted_output_sha256=sha(tmp_path/'accepted-response.npz'),input_sha256={str(source):sha(source)})
    path=tmp_path/'result.json';path.write_text(json.dumps(report))
    return path,reference,source


def test_pinned_candidate_and_same_evaluation_energy_are_consumed_without_writes(tmp_path):
    path,reference,source=fixture(tmp_path);identities={}
    before={p:sha(p) for p in (path,source,path.parent/'accepted-response.npz')}
    candidate,energy,directory=common_start(path,sha(path),reference,identities)
    np.testing.assert_array_equal(candidate['population'],reference['superlevel'])
    assert 'native_stored_net' in energy and directory.name=='evaluation-1'
    assert all(sha(p)==h for p,h in before.items())
    assert str(path) in identities and str(source) in identities


@pytest.mark.parametrize('bad_charge,changed_T',[(True,False),(False,True)])
def test_semantically_invalid_candidate_is_rejected_even_with_valid_file_hashes(tmp_path,bad_charge,changed_T):
    path,reference,_=fixture(tmp_path,bad_charge=bad_charge,changed_T=changed_T)
    with pytest.raises(ValueError):common_start(path,sha(path),reference,{})


def test_tampered_old_input_or_wrong_report_hash_is_rejected(tmp_path):
    path,reference,source=fixture(tmp_path)
    with pytest.raises(ValueError,match='신원'):common_start(path,'0'*64,reference,{})
    source.write_text('changed')
    with pytest.raises(ValueError,match='원 입력'):common_start(path,sha(path),reference,{})
