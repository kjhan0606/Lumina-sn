"""원래점 선택과 생산 PHOT 가중오차를 작은 독립 수축으로 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_reconstruction_phot as D
from test_lumina_checkpoint_rates import fixture


def material():
    nu,rows,state,route=fixture()
    state.update(physical=np.array([[2.,7000.,1e7,1.],[1.,12000.,1e8,2.]]),
        superlevel=state['full'].copy(),columns=np.array([[14,1,0],[14,2,0]]),
        sl_index={(14,1,0):0,(14,2,0):1},epoch_seconds=1.)
    return nu,rows,state,route


def test_spatial_selection_retains_all_levels_and_terminal_topology_without_averaging():
    nu,rows,state,route=material();nodes=np.array([1,0]);selected=D.select_nodes(state,nodes)
    for key in ('temperature','ne','physical'):np.testing.assert_array_equal(selected[key],state[key][nodes])
    for key in ('full','occupation','superlevel'):np.testing.assert_array_equal(selected[key],state[key][:,nodes])
    assert selected['sl_index'] is state['sl_index'] and selected['columns'] is state['columns']
    j=np.full((2,len(nu)),1e-6)
    def rates(s,field):
        return next(D.P.ion_rates_stream(nu,lambda sl:{'J':field[:,sl]},[route],rows,{},s,False,
            targets=((14,2),),integration_block_bins=7))[1]
    full=rates(state,j);subset=rates(selected,j[nodes])
    for process,value in full['total']['J'].items():
        np.testing.assert_allclose(subset['total']['J'][process],value[nodes],rtol=3e-14)
    selected['full'][:]=0
    assert np.any(state['full']>0)


@pytest.mark.parametrize('nodes',[[0,0],[-1],[2],[],[True],[.5]])
def test_bad_node_selection_cannot_replace_the_requested_native_points(nodes):
    _nu,_rows,state,_route=material()
    with pytest.raises(ValueError):D.select_nodes(state,nodes)


def test_cancellation_does_not_hide_absolute_photon_weighted_reconstruction_error():
    nu,rows,state,route=material()
    reference=np.full((2,len(nu)),1e-6);delta=np.zeros_like(reference)
    coefficients=D.R.B.continuum_grid([route],rows,state,nu)
    weighted=coefficients['chi_abs'][0]*D.P.photon_weights(nu)
    delta[0,5]=1e-8;delta[0,13]=-delta[0,5]*weighted[5]/weighted[13]
    original=reference+delta;assert np.all(original>=0)
    fields=D.error_fields(reference,original,reference.copy())
    result=next(D.P.ion_rates_stream(nu,lambda sl:{k:v[:,sl] for k,v in fields.items()},
        [route],rows,{},state,False,targets=((14,2),),integration_block_bins=7))[1]['total']
    bias=result['original']['photo'][0]-result['reference']['photo'][0]
    error=result['abs_original_error']['photo'][0]
    assert abs(bias)<result['reference']['photo'][0]*1e-14
    assert error>result['reference']['photo'][0]*1e-5
    np.testing.assert_allclose(error,np.sum(abs(delta[0])*weighted),rtol=3e-14)
    np.testing.assert_array_equal(result['abs_candidate_error']['photo'],np.zeros(2))
    np.testing.assert_array_equal(result['original']['spontaneous'],result['reference']['spontaneous'])


def test_block_attribution_preserves_global_endpoint_weights_and_final_rates():
    nu,rows,state,route=material();j=np.full((2,len(nu)),1e-6);seen=[]
    def run(observer):
        return next(D.P.ion_rates_stream(nu,lambda sl:{'J':j[:,sl]},[route],rows,{},state,False,
            targets=((14,2),),integration_block_bins=7,block_report=observer))[1]
    baseline=run(None)
    def observer(key,sl,bound,total):
        assert key==(14,2) and bound is total
        seen.append((sl.start,sl.stop,total['J']['photo'].copy()))
    actual=run(observer)
    assert [(a,b) for a,b,_ in seen]==[(i,min(len(nu),i+7)) for i in range(0,len(nu),7)]
    for process,value in baseline['total']['J'].items():
        np.testing.assert_array_equal(actual['total']['J'][process],value)
    np.testing.assert_allclose(sum(v for _,_,v in seen),baseline['total']['J']['photo'],rtol=3e-14)
