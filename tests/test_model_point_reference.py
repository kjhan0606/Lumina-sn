import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cmfgen_model_continuum as B
import compare_lumina_checkpoint_rates as R
from lumina_model_point_reference import ReferencePointRates,reference_line_rates
from test_model_point_dissolution import synthetic_point


def reference_point(tmp_path):
    point = synthetic_point(tmp_path)
    point.milne_prefactor = np.array([
        B.ground_lte_reference(point.rows,point.models,point.state)[0,0]/point.upper_population[0]])
    point.inverse_kT_Hz = B.H/(B.KB*point.temperature)
    return point,ReferencePointRates(point)


def test_mapped_coefficients_use_actual_frequency_for_reverse_and_signed_heat(tmp_path):
    point, producer = reference_point(tmp_path)
    edge = point.routes[0].threshold_Hz
    nu = edge*np.array([.8,.84,.89,.94,.9998,.99999,1.02,1.1,1.2])
    evaluation = edge*np.array([.8,.85,.9,.95,.9999,1.,1.,1.1,1.2])
    width = np.linspace(1e10,2e10,len(nu))
    j = np.linspace(1e-5,2e-5,len(nu))
    bound,total = producer.contract(0,nu,width,j,evaluation,dissolution_enabled=True)
    unique,index = np.unique(evaluation,return_inverse=True)
    coefficients = B.continuum_grid(point.routes,point.rows,point.state,unique,4)
    probability = point.routes[0].probability
    lower,upper = point.state['full'][0,0],point.upper_population[0]
    for output,with_pseudo in ((bound,False),(total,True)):
        if with_pseudo:
            B.add_dissolved_continuum(coefficients,point.routes,point.rows,point.models,
                                      point.state,unique,support_policy='native-SE')
        sampled = R.mapped_rate_coefficients(coefficients,nu,unique,index,[point.temperature])
        weight = 4*np.pi*width/(B.H*nu)
        terms = np.array([sampled['chi_abs'][0]*j,
                          sampled['eta_spont'][0],sampled['chi_stim'][0]*j])*weight
        expected = np.r_[terms.sum(axis=1),(terms*(B.H*(nu-edge))).sum(axis=1)]
        physical = output*np.array([lower,upper,upper,lower,upper,upper])*probability
        np.testing.assert_allclose(physical,expected,rtol=5e-14,atol=0)
    assert total[0] > bound[0]


def test_reference_block_partition_preserves_global_weights(tmp_path):
    point, producer = reference_point(tmp_path)
    edge = point.routes[0].threshold_Hz
    nu = edge*np.array([.9,.99,1.,1.05,1.1])
    width = np.ones(len(nu))*1e10
    evaluation = edge*np.array([.91,.999,1.,1.04,1.1])
    j = np.ones(len(nu))*1e-5
    whole = producer.contract(0,nu,width,j,evaluation,dissolution_enabled=True)
    a = producer.contract(0,nu[:2],width[:2],j[:2],evaluation[:2],dissolution_enabled=True)
    b = producer.contract(0,nu[2:],width[2:],j[2:],evaluation[2:],dissolution_enabled=True)
    np.testing.assert_allclose(np.array(a)+np.array(b),whole,rtol=5e-14,atol=0)
    with pytest.raises(ValueError,match='공통격자'):
        producer.contract(0,nu[::-1],width,j,evaluation,dissolution_enabled=True)


def test_native_line_mass_is_explicit_not_a_Jbar_normalization():
    def contract(*args):
        return dict(status=np.array([1,1]),profile_mass=np.array([.8,1.2]),
                    spontaneous=np.array([10.,20.]),jbar=np.array([3.,4.]))
    result = reference_line_rates(SimpleNamespace(contract=contract),[],[],[])
    np.testing.assert_array_equal(result['spontaneous_native_discrete'],[8.,24.])
    np.testing.assert_array_equal(result['spontaneous'],[10.,20.])
    np.testing.assert_array_equal(result['jbar'],[3.,4.])
