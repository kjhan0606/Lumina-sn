"""실제계산 실행기 제어흐름과 원래점 율 어댑터를 작은 합성 입력으로 검증한다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_native90_campaign as C
import lumina_native90_rates as R
from test_lumina_checkpoint_rates import fixture as rate_fixture


@pytest.mark.parametrize('failure', [0, 1, 2])
def test_campaign_saves_only_complete_fields_and_never_claims_convergence(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(C, 'NATIVE_SHAPE', (2, 8))
    kernel_path=tmp_path/'kernel';kernel_path.write_bytes(b'SYNTHETIC_NOT_LOADABLE')
    for name in ('planck', 'gradient'): np.save(tmp_path/(name+'.npy'),np.zeros(8))
    identity={'fixture': True}
    contract=dict(schema='lumina-native90-first-pair-v1',shape=[2,8],limits=C.LIMITS,
        library=str(kernel_path),kernel_sha256=C.N.sha(kernel_path), input_sha256={},code_sha256={},c_source_sha256={},
        boundary={name:dict(path=str(tmp_path/(file+'.npy')),sha256=C.N.sha(tmp_path/(file+'.npy')))
                  for name,file in (('inner_planck.npy','planck'),('inner_gradient.npy','gradient'))},
        config='SYNTHETIC',preparation='SYNTHETIC',preparation_sha256='SYNTHETIC',
        source_identity=identity,plan_layout={'physical_intervals':3})
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    class Source:
        ns=2;nf=8;radius=np.array([2.,1.]);frequency=np.arange(1.,9.)
        def __enter__(self): return self
        def __exit__(self,*args): pass
    Source.identity=identity
    class Plan:
        def __init__(self,*args,**kwargs): pass
    class Provider:
        def __init__(self,*args,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def bind_view(self): pass
    class Kernel:
        calls=0
        def __init__(self,*args): pass
        def execute_native90(self,*args,**kwargs):
            Kernel.calls+=1
            assert 0<kwargs['max_seconds']<=1500
            if failure and Kernel.calls==failure: raise C.DriverFailure(4,{'status':4,'completed_actual_F_calls':0})
            moments=np.full((3,2,8),float(Kernel.calls))
            return moments[0],dict(moments=moments,physical_intervals=3,completed_actual_F_calls=1)
    monkeypatch.setattr(C.N,'open_sealed_native',lambda *args:Source())
    monkeypatch.setattr(C,'open_geometry',lambda *args:None)
    monkeypatch.setattr(C,'CompiledDriverPlan',Plan);monkeypatch.setattr(C,'CompiledInput',Provider)
    monkeypatch.setattr(C,'DriverKernel',Kernel)
    calls=[]
    def rates(config,source,contract,completed,out,deadline):
        calls.append(len(completed)); assert deadline>C.time.monotonic()
        return {'synthetic_fixture':True,'accepted_physical_field':False}
    monkeypatch.setattr(C.R,'compare',rates)
    code=C.run(path,C.N.sha(path));record=json.loads((tmp_path/'campaign.json').read_text())
    assert code==int(bool(failure)) and Kernel.calls==record['F_reservations']==(1 if failure==1 else 2)
    assert len(record['completed'])==(failure-1 if failure else 2)
    assert calls==([len(record['completed'])] if record['completed'] else [])
    assert record['accepted_physical_field'] is False and record['final_goal']=='NOT_MET'
    if failure:
        assert not (tmp_path/'J-000002.npy').exists() and 'J1_residual' not in record
        if failure==1: assert not (tmp_path/'J-000001.npy').exists()
    else:
        assert record['J1_residual']['relative']==1. and record['J1_residual']['J2_residual']=='NOT_MEASURED'
        assert record['completed'][1]['input_J']==record['completed'][0]['J']
    with pytest.raises(ValueError): C.run(path,C.N.sha(path))


def test_native_rate_blocks_do_not_average_spatial_nodes_or_extrapolate():
    nu=np.array([1.,2.,4.,8.]);j=np.array([[1.,4.,9.,16.],[10.,20.,30.,40.]])
    result=R.radiation_block(nu,nu,j,nu,{'J1':j,'J2':2*j})
    np.testing.assert_array_equal(result['J1'],j)
    np.testing.assert_array_equal(result['J2'],2*j)
    sample=R.radiation_block(np.array([np.sqrt(2),np.sqrt(8)]),nu,j,nu,{'J1':j})
    np.testing.assert_allclose(sample['J1'],[[2.5,6.5],[15.,25.]])
    with pytest.raises(ValueError): R.radiation_block(np.array([.5,1.]),nu,j,nu,{'J1':j})
    with pytest.raises(ValueError): R.radiation_block(np.array([1.,2.]),nu,j,nu[1:],{'J1':j[:,1:]})


@pytest.mark.parametrize('block', [1,7,512])
def test_native_adapter_reuses_production_rate_integrator(block):
    nu,rows,state,route=rate_fixture();j=np.stack([np.ones(len(nu)),np.full(len(nu),3.)])*1e-6
    def fields(sl):return R.radiation_block(nu[sl],nu,j,nu,{'J1':j,'J2':2*j})
    [(key,result)]=list(R.ion_rates_stream(nu,fields,[route],rows,{},state,False,
        targets=((14,2),),integration_block_bins=block))
    for process,expected in result['total']['CMFGEN'].items():
        np.testing.assert_allclose(result['total']['J1'][process],expected,rtol=3e-14)
        doubled=(2*result['total']['CMFGEN']['stimulated']+result['total']['CMFGEN']['spontaneous']
                 if process=='reverse' else expected*(1 if process=='spontaneous' else 2))
        np.testing.assert_allclose(result['total']['J2'][process],doubled,rtol=3e-14)
