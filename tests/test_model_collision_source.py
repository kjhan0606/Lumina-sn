from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_model_collision_source import IonCollisionSource,ex_e1x,pair_blocks,table_values
from l1a_collision import _ex_e1x


def fixture():
    return IonCollisionSource(16,3,7,3,np.array([0.,1.,2.]),np.array([2.,4.,6.]),
        np.array([0,1,0]),.1,1.,np.array([1]),np.array([1.,4.]),
        np.array([[1.,16.]]),np.array([2]),np.array([.5]),{},dict(pairs=3))


def test_all_pairs_ordered_complete_including_no_radiative_line():
    blocks=list(pair_blocks(7,5))
    pairs=list(zip(np.concatenate([b[0] for b in blocks]),np.concatenate([b[1] for b in blocks])))
    assert pairs==[(i,j) for i in range(7) for j in range(i+1,7)]
    assert all(len(b[0])<=5 for b in blocks)
    assert list(pair_blocks(1,1))==[]
    with pytest.raises(ValueError):list(pair_blocks(3,0))


def test_ex_e1x_matches_native_polynomial_on_both_branches():
    x=np.array([1.e-8,.2,1.,1.00001,3.,15.,100.])
    assert np.allclose(ex_e1x(x),[_ex_e1x(float(v)) for v in x],rtol=1.e-14,atol=0.)
    with pytest.raises(ValueError):ex_e1x([0.,1.])


def test_log_log_table_and_temperature_dependency_not_frozen():
    v=table_values([1.,4.],[[1.,16.]],[5000.,10000.,20000.,40000.,80000.])
    assert np.array_equal(v[:,0],[1.,1.,4.,16.,16.])
    with pytest.raises(ValueError):table_values([1.,4.],[[1.,0.]],[20000.])


def test_blocks_preserve_table_vr_default_and_90_temperatures():
    source=fixture();temperatures=np.linspace(10000.,30000.,90)
    blocks=list(source.blocks(temperatures,maximum=2))
    assert np.array_equal(np.concatenate([b[0] for b in blocks]),[7,7,8])
    assert np.array_equal(np.concatenate([b[1] for b in blocks]),[8,9,9])
    omega=np.concatenate([b[2] for b in blocks],axis=1)
    assert omega.shape==(90,3)
    assert np.allclose(omega[:,0],(temperatures/10000.)**2,rtol=1.e-14)
    assert np.all(omega[:,2]==.1)
    assert sum(b[3]['TABULATED'] for b in blocks)==1
    assert sum(b[3]['VR'] for b in blocks)==1
    assert sum(b[3]['OMEGA_SET'] for b in blocks)==1
    assert omega[0,0]!=omega[-1,0]
    with pytest.raises(ValueError):list(source.blocks([0.]))


def test_one_omega_has_lte_collision_detailed_balance_not_two_independent_rates():
    source=fixture();temperature=20000.;ne=1.e8
    pop=source.g*np.exp(-source.energy/(8.617333262145e-5*temperature))
    for lower,upper,values,_ in source.blocks(temperature,maximum=2):
        lo,up=lower-source.offset,upper-source.offset
        base=8.63e-6*ne/np.sqrt(temperature)*values[0]
        forward=pop[lo]*base/source.g[lo]*np.exp(-(source.energy[up]-source.energy[lo])/(8.617333262145e-5*temperature))
        reverse=pop[up]*base/source.g[up]
        assert np.allclose(forward,reverse,rtol=1.e-14,atol=0.)
        assert not np.allclose(forward,1.01*reverse,rtol=1.e-6,atol=0.)
