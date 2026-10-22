"""고정 J native 셀 보간의 격리 직접시험; 실제 CMFGEN 수용은 아니다."""
import ctypes as ct
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_outer_c_backend import COuterTransport, Report, pointer
from lumina_outer_native_spatial import NativeSpatialGeometry, build_variant, set_native_fixed
from lumina_outer_transport import CoupledGeometry
from test_lumina_outer_transport import kernels, problem


@pytest.fixture(scope='module')
def native_library(tmp_path_factory):
    return build_variant(tmp_path_factory.mktemp('native-spatial')/'src')


def operator(library, plan, p):
    nodes = [np.ascontiguousarray(a[list(p.g.reference_indices)]) for a in
             (p.node_chi,p.node_es,p.node_eta)]
    return COuterTransport(library,plan,p.nu,p.cell_chi,p.cell_es,p.cell_eta,
        *nodes,p.planck,p.gradient,64*1024**2,weighted_workers=2,
        moment_count=1,approved_cpu_limit=2)


def test_native_plan_preserves_unsplit_p0_geometry(kernels):
    p = problem(kernels)
    baseline = CoupledGeometry(p.g)
    native = NativeSpatialGeometry(p.g)
    assert len(native.paths) == len(baseline.paths)
    assert native.half_segments == baseline.half_segments
    assert np.array_equal(native.sample_radii,baseline.sample_radii)
    for old,new in zip(baseline.paths,native.paths):
        assert old[:2] == new[:2]
        for original,reconstructed in zip(old[2:],new[2:]):
            assert len(original) == len(reconstructed)
            for a,b in zip(original,reconstructed):
                assert (a.length,a.drift,a.exterior,a.index,a.observations,a.z) == (
                    b.length,b.drift,b.exterior,b.index,b.observations,b.z)
                assert b.weight == a.weight if a.exterior else 0 <= b.weight <= 1


def test_native_observation_radii_keep_fixed_source_indices(native_library,kernels):
    p = problem(kernels)
    reference = NativeSpatialGeometry(p.g,subdivisions=2)
    same = NativeSpatialGeometry(p.g,subdivisions=2,
                                 extra_native_indices=tuple(p.g.reference_indices))
    assert np.array_equal(same.sample_radii,reference.sample_radii)
    assert same.half_segments == reference.half_segments
    j = np.full((p.n+2,p.nf),1e-4)
    with operator(native_library,same,p) as c:
        original,_ = c.apply_j(j)
    with operator(native_library,reference,p) as c:
        repeated,_ = c.apply_j(j)
    np.testing.assert_array_equal(original,repeated)

    target = NativeSpatialGeometry(p.g,subdivisions=2,extra_native_indices=(10,11))
    assert np.array_equal(target.sample_radii[:-2],reference.sample_radii[:-2])
    assert np.array_equal(target.sample_radii[-2:],p.g.native_radius[[10,11]])
    assert set(reference.impacts).issubset(set(target.impacts))
    with operator(native_library,target,p) as c:
        chi = np.full((p.n+1,p.nf),1e-14)
        eta = np.full((p.n+1,p.nf),1e-18)
        set_native_fixed(c,chi,eta,j)
        result,_ = c.apply_j(j)
    assert result.shape == j.shape and np.isfinite(result).all()
    assert np.all(result >= 0)
    with pytest.raises(ValueError,match='native 관측'):
        NativeSpatialGeometry(p.g,extra_native_indices=(10,10))


def test_uniform_native_source_matches_p0_and_rejects_new_j(native_library,kernels):
    p = problem(kernels)
    p.cell_chi[:] = 1e-14
    p.cell_es[:] = 2e-15
    p.cell_eta[:] = 1e-18
    j = np.full((p.n+2,p.nf),1e-4)
    plan = NativeSpatialGeometry(p.g,subdivisions=2)
    with operator(native_library,plan,p) as c:
        baseline,_ = c.apply_j(j)
        native_chi = np.full((p.n+1,p.nf),1e-14)
        native_eta = np.full((p.n+1,p.nf),1e-18+2e-15*1e-4)
        set_native_fixed(c,native_chi,native_eta,j)
        reconstructed,_ = c.apply_j(j)
        np.testing.assert_allclose(reconstructed,baseline,rtol=3e-13,atol=0)
        with pytest.raises(ValueError,match='native 고정'):
            set_native_fixed(c,native_chi,native_eta,j)
        wrong = j.copy()
        with pytest.raises(ValueError,match='status=1'):
            c.apply_j(wrong)
        report = Report()
        assert c.dll.cmf_outer_solve(c.context,pointer(j),2,1e-8,ct.byref(report)) == 1


def test_native_gradient_changes_actual_rays(native_library,kernels):
    p = problem(kernels)
    j = np.full((p.n+2,p.nf),1e-4)
    plan = NativeSpatialGeometry(p.g,subdivisions=2)
    with operator(native_library,plan,p) as c:
        baseline,_ = c.apply_j(j)
        # 안쪽/바깥쪽의 방향성을 모두 사용한다. 임의 점 선택으로 참조를 맞추지 않는다.
        chi = np.linspace(1.2e-14,0.8e-14,p.n+1)[:,None]*np.ones((1,p.nf))
        eta = np.linspace(0.8e-18,1.2e-18,p.n+1)[:,None]*np.ones((1,p.nf))
        set_native_fixed(c,np.ascontiguousarray(chi),np.ascontiguousarray(eta),j)
        native,_ = c.apply_j(j)
        assert np.isfinite(native).all() and np.all(native >= 0)
        assert np.max(np.abs(native-baseline)) > 1e-12*np.max(baseline)


def test_new_source_and_output_cannot_overwrite(native_library):
    with pytest.raises(FileExistsError,match='보존'):
        build_variant(native_library.parent)
