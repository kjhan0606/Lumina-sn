"""다점 BF 배치의 직접 적분·trial 갱신·실패 원자성 시험."""
import ctypes as C
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import cmfgen_model_continuum as B
from lumina_model_material_rates import Block,DP,BP,ModelMaterialRates
from lumina_model_point_rates import ModelPointBoundRates,Rates
from lumina_model_point_reference import ReferencePointRates


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    output=tmp_path_factory.mktemp('material-rates')/'point.so'
    subprocess.run(['gcc','-O2','-std=c11','-fopenmp','-fPIC','-shared','-Wl,-z,defs',
                    '-I'+str(ROOT/'src'),str(ROOT/'src/bf_rate_jnu.c'),
                    str(ROOT/'src/radiation_field.c'),str(ROOT/'src/seed_capability.c'),'-lm',
                    '-o',str(output)],check=True)
    return output


def example(library,lte_field=False):
    rows=[dict(atomic_number=14,ion_number=1,super_level=0,g='2',energy_eV='0'),
          dict(atomic_number=14,ion_number=1,super_level=1,g='4',energy_eV='19.6'),
          dict(atomic_number=14,ion_number=2,super_level=0,g='1',energy_eV='0'),
          dict(atomic_number=14,ion_number=2,super_level=1,g='3',energy_eV='34.5')]
    models={(14,2):dict(z_screen=2.,ionization_eV=20.),
            (14,3):dict(z_screen=3.,ionization_eV=35.,terminal_g=2.)}
    routes=[]
    for lower,target,stage,edge_ev in ((0,2,2,20.),(1,2,2,.4),(2,-1,3,35.),(3,-1,3,.5)):
        edge=edge_ev*B.EV/B.H
        routes.append(B.Route(14,stage,lower,target,0 if target>=0 else -1,0,.7,
            1. if target>=0 else 2.,0.,edge,
            ((lambda nu:np.full_like(nu,1.e-18),np.array([edge]),edge,'synthetic'),),True))
    columns=np.array([[14,1,0],[14,1,1],[14,2,0],[14,2,1],[14,3,-1]])
    state=dict(columns=columns,superlevel=np.array([[1.e7,2.e7,3.e7],[1.e3,2.e3,3.e3],
        [1.e5,2.e5,3.e5],[1.e2,2.e2,3.e2],[1.e-2,2.e-2,3.e-2]]),
        temperature=np.array([8000.,12000.,18000.]),ne=np.array([1.e8,2.e8,5.e8]))
    # native 용해 지지의 문턱 바로 아래도 실제로 표본화한다.
    nu=np.unique(np.r_[np.linspace(15.,42.,61),20.*(1.-np.array([1.e-9,1.e-7])),
                       35.*(1.-np.array([1.e-9,1.e-7]))])*B.EV/B.H
    width=np.zeros_like(nu);width[1:]+=.5*np.diff(nu);width[:-1]+=.5*np.diff(nu)
    evaluation=nu.copy();evaluation[20:24]=evaluation[21]
    if lte_field:
        j=2.*B.H*nu[None,:]**3/B.C**2/np.expm1(B.H*nu[None,:]/(B.KB*state['temperature'][:,None]))
    else:
        j=np.linspace(1.e-6,3.e-6,len(nu))[None,:]*np.array([[1.],[2.],[3.]])
    provider=ModelMaterialRates(routes,rows,models,library,nu,width,j,
        radiation_generation=7,evaluation=evaluation)
    return provider,state


def test_batch_matches_existing_scalar_at_each_point_and_all_six_terms(library):
    provider,state=example(library);trial=provider.prepare(state,material_generation=11)
    bound,total=provider.integrate(trial,material_generation=11,radiation_generation=7,
                                   frequency_block=17,route_block=3)
    expected=[np.zeros_like(bound),np.zeros_like(total)]
    for node in range(provider.n_points):
        point=ModelPointBoundRates(provider.routes,provider.rows,provider.models,trial.state,node,library)
        reference=ReferencePointRates(point)
        for r in range(len(provider.routes)):
            values=reference.contract(r,provider.frequency,provider.width,provider.J[node],provider.evaluation,
                                      dissolution_enabled=True)
            for accumulator,value in zip(expected,values):
                accumulator[r,node]=value
    np.testing.assert_allclose(bound,expected[0],rtol=5.e-12,atol=0)
    np.testing.assert_allclose(total,expected[1],rtol=5.e-12,atol=0)
    split=provider.integrate(trial,material_generation=11,radiation_generation=7,frequency_block=4,route_block=1)
    np.testing.assert_allclose(split,[bound,total],rtol=5.e-12,atol=0)
    assert np.any(total[:,:,0]>bound[:,:,0])


def test_trial_owns_input_and_positive_ion_change_rebuilds_occupation(library):
    provider,state=example(library);first=provider.prepare(state,material_generation=1)
    before=provider.integrate(first,material_generation=1,radiation_generation=7)
    state['superlevel'][:,0]*=10.
    second=provider.prepare(state,material_generation=2)
    after=provider.integrate(second,material_generation=2,radiation_generation=7)
    assert first.material_signature!=second.material_signature
    assert np.any(first.state['occupation'][:,0]!=second.state['occupation'][:,0])
    np.testing.assert_array_equal(first.state['ne'],second.state['ne'])
    np.testing.assert_array_equal(first.state['temperature'],second.state['temperature'])
    np.testing.assert_array_equal(before[0][:,:,0],after[0][:,:,0])
    assert np.any(before[1][:,0]!=after[1][:,0])
    with pytest.raises(ValueError,match='stale'):
        provider.integrate(first,material_generation=2,radiation_generation=7)
    with pytest.raises(ValueError,match='stale'):
        provider.integrate(second,material_generation=2,radiation_generation=8)
    with pytest.raises(ValueError):
        provider.integrate(second,material_generation=2,radiation_generation=7,route_indices=[])
    with pytest.raises(ValueError):
        provider.integrate(second,material_generation=2,radiation_generation=7,route_indices=[.5])


def test_planck_field_preserves_route_lte_balance_for_rates_and_heat(library):
    provider,state=example(library,lte_field=True);trial=provider.prepare(state,material_generation=1)
    for values in provider.integrate(trial,material_generation=1,radiation_generation=7):
        np.testing.assert_allclose(values[:,:,0]*trial.saha,values[:,:,1]+values[:,:,2],rtol=5.e-12,atol=0)
        np.testing.assert_allclose(values[:,:,3]*trial.saha,values[:,:,4]+values[:,:,5],rtol=5.e-12,atol=0)


def test_bad_late_c_input_preserves_both_complete_output_arrays(library):
    dll=C.CDLL(str(library));fn=dll.bf_rate_material_block_contract
    fn.argtypes=[C.POINTER(Block),C.POINTER(Rates),C.POINTER(Rates)];fn.restype=C.c_int
    arrays=[np.array(x,dtype='f8') for x in ([1.e15,1.01e15],[1.e15,1.01e15],[1.e12,1.e12],
        [1.e-5,1.e-5,1.e-5,np.nan],[1.e-18,1.e-18],[1.e15],[1.e-8,1.e-8],[1.e-14,1.e-14])]
    block=Block(2,1,2,*[a.ctypes.data_as(DP) for a in arrays],0,BP(),DP(),DP(),DP(),DP(),DP())
    bound=np.full((1,2,6),-123.);total=bound.copy()
    assert fn(C.byref(block),bound.ctypes.data_as(C.POINTER(Rates)),total.ctypes.data_as(C.POINTER(Rates)))!=0
    np.testing.assert_array_equal(bound,-123.);np.testing.assert_array_equal(total,-123.)
