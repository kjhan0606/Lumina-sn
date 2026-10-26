"""완성 전주파수 BF만 실물 C handoff로 넘기는 계약의 소형 음성 대조."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_occupation_native_test as handoff


def fixture(tmp_path):
    routes=[SimpleNamespace(lower=0,target=1,Z=14,stage=1,probability=1.,threshold_Hz=2.),
            SimpleNamespace(lower=2,target=-1,Z=16,stage=2,probability=.5,threshold_Hz=3.)]
    point=SimpleNamespace(routes=routes,node=49,saha=np.array([4.,5.]),
                          sigma_edge=np.array([6.,7.]))
    np.save(tmp_path/'unit-rates.npy',np.ones((2,6)))
    np.save(tmp_path/'routes-complete.npy',np.ones(2,dtype=bool))
    np.savez(tmp_path/'roster.npz',lower=[0,2],target=[1,-1],Z=[14,16],stage=[1,2],
             probability=[1.,.5],threshold=[2.,3.],saha=[4.,5.],sigma_edge=[6.,7.])
    sealed=dict(state='COMPLETE_BOUND_ONLY',complete_bound_frequency_integral=True,
        routes_complete=2,routes_expected=2,native_node=49,frequencies=7,input_identity={'id':'same'},
        source_unchanged=True,pseudo_included=False,BB_included=False,SE_RE_solution=False,
        material_updates=0,actual_new_transport=0,
        source_sha256={str(Path(__file__)):handoff.B.sha(Path(__file__))})
    path=tmp_path/'result.json';path.write_text(json.dumps(sealed))
    return point,path,sealed


def test_complete_full_bound_contract(tmp_path):
    point,path,_=fixture(tmp_path)
    values,seal=handoff.load_complete_full_bound(path,point,{'id':'same'},7)
    assert values.shape==(2,6) and seal['source_result_sha256']==handoff.B.sha(path)


def test_partial_and_route_mismatch_rejected(tmp_path):
    point,path,sealed=fixture(tmp_path)
    sealed['routes_complete']=1;path.write_text(json.dumps(sealed))
    with pytest.raises(ValueError,match='완성'):
        handoff.load_complete_full_bound(path,point,{'id':'same'},7)
    sealed['routes_complete']=2;path.write_text(json.dumps(sealed))
    np.savez(tmp_path/'roster.npz',lower=[0,2],target=[1,-1],Z=[14,16],stage=[1,2],
             probability=[1.,.4],threshold=[2.,3.],saha=[4.,5.],sigma_edge=[6.,7.])
    with pytest.raises(ValueError,match='route'):
        handoff.load_complete_full_bound(path,point,{'id':'same'},7)
