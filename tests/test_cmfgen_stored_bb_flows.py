"""보존 native ZNET/fine BB 블록의 주소·부호·핵수 보존 직접시험."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_cmfgen_ir_grid_impact import (add_bb_flow, compare_native_bb_flows,
                                            pairwise_native_blocks, rank_native_bb_heat)


def test_independent_response_heat_applies_occupation_only_to_absorption(monkeypatch):
    import lumina_model_response_balance as M
    from lumina_model_freefree import FreeFreeCoefficients
    obj=M.ResponseBalance.__new__(M.ResponseBalance)
    pop=np.array([[10.,12.],[2.,3.]])
    fraction=np.ones_like(pop);occupation=np.array([[1.,1.],[.5,.25]])
    t=np.array([10000.,11000.]);ne=np.array([1.e6,2.e6]);nu=1.e15;a=3.;ratio=.5
    obj.columns=np.array([[14,1,0],[14,1,1]]);obj.index={tuple(k):i for i,k in enumerate(obj.columns)}
    obj.roster={k:np.empty(0,dtype='i4') for k in ('lower','target','Z','stage')}
    obj.jbar=np.array([[1.e-3,2.e-3]]);obj.profile_mass=np.array([.9])
    obj.lines=dict(lower=np.array([0]),upper=np.array([1]),nu0=np.array([nu]),
                   A=np.array([a]),g_ratio=np.array([ratio]))
    obj.owner=SimpleNamespace(full_to_sl=np.array([0,1]),
        line_scaling=lambda *args,**kwargs:np.ones((1,2)),
        internal_energy=lambda *args:np.zeros(2))
    obj.radius=np.array([2.,1.]);obj.epoch=1.;obj.library=None;obj.tmin=0.
    obj.scaling_enabled=False;obj.scaling_limit=.5;obj.density_limit=1.e30
    obj.shock=obj.xray=obj.stored=obj.instant=np.zeros(2)
    obj.coefficients=FreeFreeCoefficients('synthetic',t,np.zeros((1,2)),np.zeros((1,2)))
    zero=dict(heating=np.zeros(2),cooling=np.zeros(2))
    obj.ff=SimpleNamespace(contract=lambda *args,**kwargs:dict(terms={
        'all_positive_ions':zero,'native_upper_ions':zero}))
    monkeypatch.setattr(M,'adiabatic',lambda *args,**kwargs:dict(signed_total=np.zeros(2)))
    bul=a*M.B.C**2/(2*M.B.H*nu**3)
    forward=pop[0]*occupation[1]*bul/ratio*obj.jbar[0]*M.B.H*nu
    reverse=pop[1]*(a*.9+bul*obj.jbar[0])*M.B.H*nu
    ledger=np.zeros((2,2,2));ledger[0,:,1]=forward;ledger[1,:,1]=-reverse
    data=dict(population=pop,temperature=t,ne=ne,energy=np.zeros((2,3)),ledger=ledger)
    trial=SimpleNamespace(fraction=fraction,state=dict(occupation=occupation),
                          radiation_generation=1,material_generation=1)
    result,_=obj.evaluate(data,trial,np.empty((0,2,6)))
    np.testing.assert_allclose(result['bb'],np.stack([forward,reverse]),rtol=3.e-15)
    np.testing.assert_allclose(result['native_bb'],result['bb'],rtol=0,atol=0)


def test_bound_bound_flow_conserves_number_and_internal_sl_is_zero():
    target=np.zeros((2,3),dtype=np.longdouble)
    flow=np.array([1.,-2.,3.])
    add_bb_flow(target,0,1,flow)
    np.testing.assert_array_equal(target.sum(axis=0),0)
    before=target.copy()
    add_bb_flow(target,1,1,flow)
    np.testing.assert_array_equal(target,before)


def test_native_znet_comparison_uses_species_and_full_level_keys(tmp_path):
    atoms=[(14,2,'Sk2'),(14,3,'SkIII'),(16,2,'S2'),(16,3,'SIII')]
    rows=[];columns=[];headers=[]
    for i,(z,stage,label) in enumerate(atoms):
        for l in range(2):
            rows.append(dict(atomic_number=str(z),ion_number=str(stage-1),level_number=str(l)))
            columns.append((z,stage-1,l))
        headers.append(f'{i+1} {label}(upper-lower) 1.000000 1 2 1 2\n'+
                       ' '.join(['0.5']*90)+'\n')
    path=tmp_path/'NETRATE';path.write_text('\n'.join(headers))
    pop=np.tile(np.array([10.,2.]*4)[:,None],(1,90))
    lines=dict(lower=np.array([0,2,4,6]),upper=np.array([1,3,5,7]),
               nu0=np.full(4,1.e15),A=np.full(4,3.),g_ratio=np.ones(4))
    sink={}
    report=compare_native_bb_flows(path,lines,np.zeros((4,90)),np.ones(4),pop,
        np.ones((8,90)),np.array(columns),rows,np.arange(8),np.zeros(4,dtype=bool),vector_sink=sink)
    assert report['matched_lines']==4
    for ion in report['by_ion'].values():
        assert ion['matched_lines']==1
        assert ion['max_absolute_row_delta']==pytest.approx(3.)
        assert ion['delta_over_BB_gross_median']==pytest.approx(.5)
        assert ion['number_conservation_relative']==0.
        assert ion['depths'][0]['native_print_L1_bound']==pytest.approx(6.e-7)
    twice={}
    native=compare_native_bb_flows(path,lines,np.zeros((4,90)),np.ones(4),pop,
        np.ones((8,90)),np.array(columns),rows,np.arange(8),np.zeros(4,dtype=bool),2*pop,twice)
    assert native['population']=='native full snapshot'
    for ion in native['by_ion'].values():
        assert ion['max_absolute_row_delta']==pytest.approx(6.)
        assert ion['delta_over_BB_gross_median']==pytest.approx(.5)
    assert pairwise_native_blocks(sink,twice)['Z14_stage2']['max_L1']==pytest.approx(6.)


@pytest.fixture
def ranked_lines(tmp_path):
    path=tmp_path/'NETRATE'
    path.write_text('1 Sk2(u-l) 1.000000 1 2 1 2\n0.5 0.5\n\n'
                    '2 Sk2(v-w) 2.000000 3 4 3 4\n0.5 0.5\n')
    pop=np.array([[10.,20.],[2.,3.],[15.,25.],[4.,5.]])
    rows=[dict(atomic_number='14',ion_number='1',level_number=str(i)) for i in range(4)]
    columns=np.array([[14,1,i] for i in range(4)])
    lines=dict(lower=np.array([0,2]),upper=np.array([1,3]),nu0=np.array([1.e15,2.e15]),
               A=np.array([3.,3.]),g_ratio=np.ones(2))
    return [path,lines,np.zeros((2,2)),np.ones(2),pop,np.ones((4,2)),columns,rows,
            np.arange(4),np.zeros(2,dtype=bool),pop.copy(),np.ones((2,2))]


def test_bb_heat_ranking_uses_common_native_pop_and_retains_largest_transitions(ranked_lines):
    import cmfgen_model_continuum as B
    ranked_lines[10]*=2  # C의 다른 full 재구성 대신 같은 native full 인구를 양쪽에 쓴다.
    result=rank_native_bb_heat(*ranked_lines,depths=(0,1),keep=1)
    expected=1.5*B.H*(1.e15*ranked_lines[10][1]+2.e15*ranked_lines[10][3])
    np.testing.assert_allclose(result['by_ion']['Z14_stage2']['delta'],expected,rtol=3.e-15)
    assert result['matched_lines']==2 and result['depths']==[1,2]
    assert all(v[0]['line_id']==2 for v in result['top_lines'].values())
    assert result['final_goal']=='NOT_MET' and result['new_physical_evaluations']==0


@pytest.mark.parametrize('fault',['missing','duplicate','frequency'])
def test_bb_heat_ranking_rejects_misaligned_native_records(ranked_lines,fault):
    path=ranked_lines[0];text=path.read_text()
    if fault=='missing':text=text.replace('3 4 3 4','3 4 3 9')
    if fault=='duplicate':text+=text
    if fault=='frequency':text=text.replace('2.000000','2.100000')
    path.write_text(text)
    with pytest.raises(ValueError):rank_native_bb_heat(*ranked_lines,depths=(0,1))


@pytest.mark.parametrize('depths,keep',[((),1),((0,0),1),((0,.5),1),((0,2),1),((0,1),1.5)])
def test_bb_heat_ranking_rejects_invalid_depth_or_heap_size(ranked_lines,depths,keep):
    with pytest.raises(ValueError):rank_native_bb_heat(*ranked_lines,depths=depths,keep=keep)
