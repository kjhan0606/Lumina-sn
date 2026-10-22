"""봉인 Bézier SC의 Python 독립 재현에 필요한 해석적 직접시험."""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_cmfgen_local_sc_packet import bezier_source, check, local_sc, replay_ray
import pytest


def test_bezier_constant_source_integrates_exactly():
    for tau in (1e-5,.01,.5,1.,10.,100.):
        got=bezier_source(2.,2.,2.,tau,1.)
        np.testing.assert_allclose(got,-2.*math.expm1(-tau),rtol=2e-13,atol=0)


def test_local_sc_constant_source_has_analytic_both_directions():
    n=6
    geometry=dict(B_P=np.zeros(n),B_M=np.zeros(n),
                  S_P=np.arange(n,0,-1,dtype='f8'),
                  S_M=np.arange(n,dtype='f8'))
    chi=np.full(n,.2)
    eta=chi*3.
    previous=np.zeros(n)
    current=np.full(n,7.)
    expected=7.*math.exp(-.2)+3.*(-math.expm1(-.2))
    for direction in ('P','M'):
        got=local_sc(chi,eta,previous,geometry,current,3,.01,direction)
        np.testing.assert_allclose(got,expected,rtol=2e-13,atol=0)


def test_full_noncore_ray_constant_source_matches_analytic_path():
    n=6
    geometry=dict(R=np.ones(n),B_P=np.zeros(n),B_M=np.zeros(n),
                  S_P=np.arange(n,0,-1,dtype='f8'),
                  S_M=np.arange(n,dtype='f8'),MU_P=np.zeros(n))
    args=dict(dLOG_NU=.01,NC=0)
    previous=dict(I_P=np.zeros(n),I_M=np.zeros(n))
    plus,minus=replay_ray(args,geometry,previous,
                          np.full(n,.2),np.full(n,.6),1)
    depth=np.arange(n,dtype='f8')
    np.testing.assert_allclose(minus,3*(-np.expm1(-.2*depth)),rtol=2e-13,atol=0)
    np.testing.assert_allclose(plus,3*(-np.expm1(-.2*(2*(n-1)-depth))),
                               rtol=2e-13,atol=0)


def test_core_inner_boundary_override_propagates_outward():
    n=6
    geometry=dict(R=np.ones(n),B_P=np.zeros(n),B_M=np.zeros(n),
                  S_P=np.arange(n,0,-1,dtype='f8'),
                  S_M=np.arange(n,dtype='f8'),MU_P=np.zeros(n))
    previous=dict(I_P=np.zeros(n),I_M=np.zeros(n))
    plus,_=replay_ray(dict(dLOG_NU=.01,NC=1),geometry,previous,
                      np.full(n,.2),np.full(n,.6),1,inner_plus=5.)
    expected=3.+2.*np.exp(-.2*(n-1-np.arange(n)))
    np.testing.assert_allclose(plus,expected,rtol=2e-13,atol=0)


def test_blue_propagation_requires_all_calls_and_full_rays():
    with pytest.raises(ValueError,match='모든 호출'):
        check(Path('/nonexistent'),propagate_blue=True)
