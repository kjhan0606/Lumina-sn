"""같은 C 광선/계수 경로에 이산 상풍 P0 커널을 잇는 소형 직접시험."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_outer_c_backend import COuterTransport
from lumina_outer_native_spatial import NativeSpatialGeometry, build_variant as build_continuous, set_native_fixed
from lumina_upwind_outer_variant import build_variant
from test_lumina_outer_transport import kernels,problem


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build_variant(tmp_path_factory.mktemp('upwind-outer')/'src')


def test_same_native_geometry_fixed_j_runs_once(library,kernels):
    p=problem(kernels)
    j=np.full((p.n+2,p.nf),1e-4)
    plan=NativeSpatialGeometry(p.g,subdivisions=2,extra_native_indices=(10,11))
    nodes=[np.ascontiguousarray(a[list(p.g.reference_indices)]) for a in
           (p.node_chi,p.node_es,p.node_eta)]
    with COuterTransport(library,plan,p.nu,p.cell_chi,p.cell_es,p.cell_eta,
            *nodes,p.planck,p.gradient,64*1024**2,weighted_workers=2,
            moment_count=1,approved_cpu_limit=2) as operator:
        chi=np.full((p.n+1,p.nf),1e-14)
        eta=np.full((p.n+1,p.nf),1e-18)
        set_native_fixed(operator,chi,eta,j)
        first,info=operator.apply_j(j)
        second,_=operator.apply_j(j)
    assert info['actual_ray_threads']<=2
    assert first.shape==j.shape and np.isfinite(first).all() and np.all(first>=0)
    np.testing.assert_array_equal(first,second)


def test_existing_build_is_preserved(library):
    with pytest.raises(FileExistsError):
        build_variant(library.parent)


def test_zero_drift_reduces_to_same_continuous_ray_operator(library,kernels,tmp_path):
    p=problem(kernels)
    static=replace(p.g,velocity=np.zeros_like(p.g.velocity),
                   sigma=np.zeros_like(p.g.sigma))
    plan=NativeSpatialGeometry(static,subdivisions=2,extra_native_indices=(10,11))
    assert all(step.drift==0 for _,_,negative,positive in plan.paths
               for step in (*negative,*positive))
    nodes=[np.ascontiguousarray(a[list(p.g.reference_indices)]) for a in
           (p.node_chi,p.node_es,p.node_eta)]
    j=np.full((p.n+2,p.nf),1e-4)
    chi=np.full((p.n+1,p.nf),1e-14)
    eta=np.full((p.n+1,p.nf),1e-18)
    continuous=build_continuous(tmp_path/'continuous')
    values=[]
    for source in (continuous,library):
        with COuterTransport(source,plan,p.nu,p.cell_chi,p.cell_es,p.cell_eta,
                *nodes,p.planck,p.gradient,64*1024**2,weighted_workers=2,
                moment_count=1,approved_cpu_limit=2) as operator:
            set_native_fixed(operator,chi,eta,j)
            result,_=operator.apply_j(j)
            values.append(result)
    np.testing.assert_allclose(values[1],values[0],rtol=5e-13,atol=0.)
