"""상주 광선과 관측자 적분을 정적 회색 해석해로 직접 비교한다."""
import math
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lumina_fullsr_chain_gpu import DeviceInput
from lumina_observer_spectrum import ObserverDriverPlan, disk_weights
from lumina_phase_geometry import PhaseGeometry
from lumina_resident_actual import manufactured_source
from lumina_resident_field import ResidentField
from lumina_resident_observer import ResidentObserver
import run_lumina_resident_observer as runner
from stage_lumina_fullsr_sc import stage
from test_lumina_resident_field import Runtime, Library, driver_library, frozen


def test_observer_staging_includes_actual_runner_and_dependencies(tmp_path):
    project = Path(__file__).resolve().parents[1]
    stage(project, tmp_path/'stage', observer=True)
    manifest = json.loads((tmp_path/'stage/stage.json').read_text())['files']
    for name in ('scripts/run_lumina_resident_observer.py', 'scripts/lumina_resident_observer.py',
                 'scripts/lumina_observer_spectrum.py', 'scripts/lumina_resident_run.py',
                 'src/cmf_fullsr_chain.cu'):
        assert name in manifest and (tmp_path/'stage/code'/name).is_file()


def test_resident_observer_matches_static_ray_solution(driver_library, frozen, tmp_path):
    geometry, source, provider, _, chi, eta = manufactured_source(tmp_path)
    plan = ObserverDriverPlan(geometry, source, disk_subdivisions=1)
    runtime = Runtime(); gpu = SimpleNamespace(lib=Library(runtime, frozen))
    boundary = .4+.02*np.arange(source.nf); gradient = np.zeros(source.nf)
    with source, PhaseGeometry(driver_library, geometry, source) as factory, DeviceInput(runtime, provider) as device:
        with ResidentField(gpu, runtime, device, factory, plan, boundary, gradient,
                           kernel_identity='SYNTHETIC_CPU_ONLY') as field:
            observer = ResidentObserver(field)
            with pytest.raises(ValueError, match='미완료'):
                observer.finish()
            deadline = time.monotonic()+120
            for _ in plan.paths:
                observer.advance_path(deadline=deadline)
            actual = observer.finish()
            assert not actual['accepted_physical_spectrum']
            assert not actual['scattering_convergence_certified']
            assert actual['physical_intervals'] == len(plan.arrays['steps'])
    assert not runtime.data
    radius = plan.outer_radius; impact = plan.observer_impact; core = plan.observer_core
    outer_z = np.sqrt((radius-impact)*(radius+impact))
    lengths = 2*outer_z
    inner = plan.boundaries[0]
    lengths[core] = outer_z[core]-np.sqrt((inner-impact[core])*(inner+impact[core]))
    tau = lengths[:, None]*chi[None, :]
    factor = np.ones_like(tau)
    np.divide(-np.expm1(-tau), tau, out=factor, where=tau != 0)
    rays = np.where(core[:, None], boundary[None, :]*np.exp(-tau), 0.) + eta[None, :]*lengths[:, None]*factor
    weights = disk_weights(impact, core, radius)
    expected = 4*math.pi**2*radius**2*np.sum(weights[:, None]*rays, axis=0)
    np.testing.assert_allclose(actual['luminosity_erg_s_Hz'], expected, rtol=2e-12, atol=0.)


def test_observer_checkpoint_resumes_without_repeating_rays(driver_library, frozen, tmp_path):
    geometry, source, provider, _, _, _ = manufactured_source(tmp_path)
    plan = ObserverDriverPlan(geometry, source); runtime = Runtime()
    gpu = SimpleNamespace(lib=Library(runtime, frozen)); boundary = np.full(source.nf, .4)
    kwargs = dict(kernel_identity='SYNTHETIC_CPU_ONLY')
    with source, PhaseGeometry(driver_library, geometry, source) as factory, DeviceInput(runtime, provider) as device:
        with ResidentField(gpu, runtime, device, factory, plan, boundary, np.zeros(source.nf), **kwargs) as field:
            original = ResidentObserver(field)
            for _ in range(len(plan.paths)//2): original.advance_path(deadline=time.monotonic()+120)
            checkpoint = original.checkpoint(tmp_path/'observer-checkpoint')
            for _ in range(len(plan.paths)-original.spectrum.next_path):
                original.advance_path(deadline=time.monotonic()+120)
            expected = original.finish()['luminosity_erg_s_Hz']
        with ResidentField(gpu, runtime, device, factory, plan, boundary, np.zeros(source.nf), **kwargs) as field:
            resumed = ResidentObserver(field); resumed.restore(tmp_path/'observer-checkpoint')
            assert resumed.spectrum.next_path == checkpoint['completed_paths']
            for _ in range(len(plan.paths)-resumed.spectrum.next_path):
                resumed.advance_path(deadline=time.monotonic()+120)
            np.testing.assert_array_equal(resumed.finish()['luminosity_erg_s_Hz'], expected)
        metadata = tmp_path/'observer-checkpoint/observer.json'
        altered = json.loads(metadata.read_text()); altered['completed_paths'] += 1
        metadata.write_text(json.dumps(altered))
        with ResidentField(gpu, runtime, device, factory, plan, boundary, np.zeros(source.nf), **kwargs) as field:
            resumed = ResidentObserver(field)
            with pytest.raises(ValueError, match='완료 위치'):
                resumed.restore(tmp_path/'observer-checkpoint')
            with pytest.raises(ValueError, match='완료 위치'):
                resumed.advance_path(deadline=time.monotonic()+120)
    assert not runtime.data


def test_observer_runner_publishes_completed_synthetic_spectrum(driver_library, frozen, tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    geometry, source, provider, _, _, _ = manufactured_source(tmp_path)
    plan = ObserverDriverPlan(geometry, source)
    root = tmp_path/'run'; root.mkdir()
    (root/'chain_cuda.so').write_text('synthetic')
    (root/'geometry.so').symlink_to(driver_library)
    class TestRuntime(Runtime):
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def report(self): return {'synthetic': True}
    runtime = TestRuntime(); gpu = SimpleNamespace(lib=Library(runtime, frozen), device_uuid=lambda:'gpu-test')
    class Provider:
        def __init__(self, *args, **kwargs): self.__dict__.update(provider.__dict__)
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def bind_view(self): pass
        def prepare_dense(self, **kwargs): return {'synthetic': True}
        def borrow_dense_view(self): return provider.borrow_dense_view()
    class Field:
        def __init__(self, *args): pass
        def close(self): pass
    def inputs(stack, *args):
        stack.enter_context(source)
        return ({'cpu_library': str(driver_library)}, source, geometry,
                (np.full(source.nf, .4), np.zeros(source.nf)), 'prep-sha')
    monkeypatch.setattr(runner, 'verify_stage', lambda root: 'stage-sha')
    monkeypatch.setattr(runner, 'open_field_inputs', inputs)
    monkeypatch.setattr(runner, 'InputField', Field)
    monkeypatch.setattr(runner, 'CompiledInput', Provider)
    monkeypatch.setattr(runner, 'ChainGPU', lambda path: gpu)
    monkeypatch.setattr(runner, 'Runtime', lambda *args, **kwargs: runtime)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='gpu-test, synthetic, 10000\n'))
    status = runner.run(root, accepted_input=tmp_path/'prep', accepted_input_sha256='prep-sha',
                        input_j=tmp_path/'j', input_j_sha256='j-sha', seconds=600)
    report = json.loads((root/'observer-result.json').read_text())
    assert status == 0 and report['status'] == 'COMPLETE_FROZEN_OBSERVER_NOT_CONVERGENCE'
    assert report['completed_paths'] == len(plan.paths) and not report['accepted_physical_spectrum']
    with np.load(root/'observer-spectrum.npz', allow_pickle=False) as data:
        assert np.isfinite(data['luminosity_erg_s_Hz']).all()
    assert not runtime.data
