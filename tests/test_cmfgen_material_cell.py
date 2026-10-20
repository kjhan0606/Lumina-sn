"""같은 양의 물질의 해를 source 비율 보간이 아닌 독립 formal 적분과 대조한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cmfgen_packet_cell_material import kernel,linear_material_reference

NATIVE=Path('/gpfs/kjhan/cmfgen_src/cur_cmf')


@pytest.fixture(scope='module')
def material(tmp_path_factory):
    call,_,_=kernel(tmp_path_factory.mktemp('material_cell'),NATIVE)
    return lambda i,chi,eta,b,blue,length,alpha,tol=1e-11:call('material',i,chi,eta,b,blue,length,alpha,tol)


@pytest.mark.parametrize('tau',[1e-280,1e-12,.01,.5,1.,10.,1e3,1e6,1e280])
def test_constant_source_thin_and_deep_without_intensity_floor(material,tau):
    for incoming in (0.,.3):
        value,error,count,status=material(incoming,[1,1],[2,2],[0,0],[0,0],tau,0.)
        exact=incoming*np.exp(-tau)+2*(-np.expm1(-tau))
        assert status==0 and count>0
        assert value==pytest.approx(exact,rel=2e-12,abs=0)
        assert value>=0 and error>=0


def test_weak_nonzero_emission_at_zero_downstream_endpoint_is_not_subtracted_away(material):
    value,_,_,status=material(0.,[1e200,1e200],[1e200,0],[0,0],[0,0],1.,0.)
    assert status==0 and value==pytest.approx(1e-200,rel=2e-12,abs=0)
    value,_,_,status=material(1e280,[1,1],[0,0],[0,0],[0,0],1000.,0.)
    assert status==0 and value==pytest.approx(np.exp(np.log(1e280)-1000),rel=2e-12,abs=0)


def test_positive_material_and_blue_against_independent_formal_integral(material):
    rng=np.random.default_rng(20261020)
    for case in range(24):
        chi=10.**rng.uniform(-2,3,2);eta=10.**rng.uniform(-8,0,2)
        b=10.**rng.uniform(-7,-1,2);blue=10.**rng.uniform(-5,1,2)
        alpha=(0.,4.,22187.)[case%3]
        effective=chi+(alpha+3)*b
        length=10.**rng.uniform(-3,5)/effective.mean()
        incoming=10.**rng.uniform(-8,-2)
        reference,accuracy=linear_material_reference(incoming,chi,eta,b,blue,length,alpha)
        value,error,count,status=material(incoming,chi,eta,b,blue,length,alpha)
        assert status==0,(case,status,count)
        assert value>0 and error>=0
        assert value==pytest.approx(reference,rel=5e-9,abs=0),(case,accuracy,count)


@pytest.mark.parametrize('defect',['chi','eta','blue','length','tolerance'])
def test_unsupported_material_is_rejected_not_repaired(material,defect):
    chi=[1,2];eta=[1,1];blue=[0,0];length=1.;tol=1e-11
    if defect=='chi':chi[0]=0
    if defect=='eta':eta[0]=-1
    if defect=='blue':blue[0]=np.nan
    if defect=='length':length=0
    if defect=='tolerance':tol=0
    _,_,_,status=material(0.,chi,eta,[0,0],blue,length,0.,tol)
    assert status!=0
