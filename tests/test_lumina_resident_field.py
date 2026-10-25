"""상주 host 흐름·checkpoint를 CPU 대역에서 검사한다. 실제 GPU 증거와 구분한다."""
import ctypes as ct
import json
from types import SimpleNamespace
import time

import numpy as np
import pytest

from test_lumina_fullsr_frozen import frozen
from test_lumina_nodal_driver import driver_library
from lumina_fullsr_sc import SCALED
from lumina_fullsr_chain_gpu import DeviceInput
from lumina_phase_geometry import PhaseGeometry
from lumina_phase_cache import CachedPhaseGeometry, build_cache
from lumina_resident_field import ResidentField, actions
from lumina_resident_actual import manufactured_source, analytic_moments
from run_lumina_fullsr_chain_gpu import positive


class Buffer:
    def __init__(self, owner, address, size): self.owner, self.address, self.size = owner, address, size
    def close(self): self.owner.data.pop(self.address)


class Runtime:
    status = 0
    def __init__(self): self.data = {}; self.next = 256; self.error = 0
    def allocate(self, size):
        address = self.next; self.next += size+256; self.data[address] = np.zeros(size, dtype='u1')
        return Buffer(self, address, size)
    def array(self, address, shape, dtype='f8'):
        if not isinstance(address, int): address = ct.cast(address, ct.c_void_p).value
        for first, data in self.data.items():
            if first <= address < first+len(data):
                return np.frombuffer(data, offset=address-first, count=int(np.prod(shape)), dtype=dtype).reshape(shape)
        raise ValueError('모의 장치 주소 범위 밖')
    def upload(self, target, array): self.data[target.address][:array.nbytes] = np.frombuffer(array.tobytes(), dtype='u1')
    def upload_pointer(self, target, pointer, size):
        self.data[target.address][:size] = np.frombuffer(ct.string_at(pointer, size), dtype='u1')
    def download(self, source, shape, dtype='f8'): self.sync(); return self.array(source.address, shape, dtype).copy()
    def zero(self, target): self.data[target.address][:] = 0
    def sync(self):
        if self.error: raise RuntimeError('모의 GPU 상태 '+str(self.error))


class Library:
    def __init__(self, runtime, cpu): self.rt, self.cpu = runtime, cpu
    def lumina_fullsr_chain_stream(self, view, steps, first, count, lengths, deltas, locations, phases, h, incoming, output, status):
        if self.rt.error: return 0
        v = view._obj; ns, nf = v.ns, v.nf
        initial = self.rt.array(incoming, nf, SCALED)
        if np.any(initial['mantissa'] < 0): self.rt.error = 175; return 0
        canonical = self.rt.array(v.canonical, (3, ns, nf)); j = self.rt.array(v.j, (ns, nf))
        chi = np.tile(canonical[0, 0], (steps+1, 1))
        eta = np.tile(canonical[2, 0]+canonical[1, 0]*j[0], (steps+1, 1))
        code, result, _, report = self.cpu.chain(self.rt.array(lengths, steps), self.rt.array(deltas, steps), chi, eta, initial)
        assert code == 0, report
        self.rt.array(output, nf, SCALED)[:] = result; return 0
    def lumina_history_cuda_boundary(self, nf, planck, gradient, mu, ray, status):
        if not self.rt.error: self.rt.array(ray, nf)[:] = self.rt.array(planck, nf)+mu*self.rt.array(gradient, nf)
        return 0
    def lumina_fullsr_chain_encode(self, nf, incoming, output, status):
        if not self.rt.error: self.rt.array(output, nf, SCALED)[:] = positive(self.rt.array(incoming, nf))
        return 0
    def lumina_fullsr_chain_decode(self, nf, incoming, output, counts, status):
        if self.rt.error: return 0
        raw = self.rt.array(incoming, nf, SCALED); values = np.ldexp(raw['mantissa'], raw['exponent'].astype('i4'))
        self.rt.array(output, nf)[:] = values
        self.rt.array(counts, 3, 'u8')[0] += np.count_nonzero(raw['mantissa'] == 0)
        return 0
    def lumina_history_cuda_accumulate(self, ns, nf, n, nodes, weights, ray, moments, status):
        if self.rt.error: return 0
        nn = self.rt.array(nodes, n, 'i4'); ww = self.rt.array(weights, (n, 3)); ii = self.rt.array(ray, nf)
        values = self.rt.array(moments, (3, ns, nf))
        for node, weight in zip(nn, ww): values[:, node] += weight[:, None]*ii[None, :]
        return 0


def test_resident_control_preserves_core_events_all_moments_and_checkpoint(driver_library, frozen, tmp_path):
    g, source, provider, plan, chi, eta = manufactured_source(tmp_path)
    runtime = Runtime(); gpu = SimpleNamespace(lib=Library(runtime, frozen))
    boundary = .4+.02*np.arange(8); gradient = .02-.002*np.arange(8)
    expected = analytic_moments(plan, chi, eta, boundary, gradient)
    with source, PhaseGeometry(driver_library, g, source) as factory, DeviceInput(runtime, provider) as device:
        kwargs = dict(kernel_identity='CPU_CONTROL_NOT_GPU_EVIDENCE')
        with ResidentField(gpu, runtime, device, factory, plan, boundary, gradient, **kwargs) as field:
            with pytest.raises(ValueError, match='부분/실패'): field.complete_moments()
            field.advance(deadline=time.monotonic()+120); original = field.complete_moments()
            assert field.intervals == len(plan.arrays['steps'])
            assert field.observations == len(plan.arrays['event_node'])+int(plan.arrays['paths'][:, 4].sum())
            assert abs(original-expected).max()/expected[0].max() < 1e-12
        build_cache(tmp_path/'geometry-cache', factory, plan)
        with CachedPhaseGeometry(tmp_path/'geometry-cache', factory, plan) as cached, ResidentField(
                gpu, runtime, device, cached, plan, boundary, gradient, **kwargs) as field:
            field.advance(deadline=time.monotonic()+120, max_actions=len(field.sequence)//2)
            field.checkpoint(tmp_path/'checkpoint')
        with CachedPhaseGeometry(tmp_path/'geometry-cache', factory, plan) as cached, ResidentField(
                gpu, runtime, device, cached, plan, boundary, gradient, **kwargs) as field:
            field.restore(tmp_path/'checkpoint'); field.advance(deadline=time.monotonic()+120)
            np.testing.assert_array_equal(field.complete_moments(), original)
        metadata = json.loads((tmp_path/'checkpoint/checkpoint.json').read_text()); metadata['intervals'] += 1
        (tmp_path/'checkpoint/checkpoint.json').write_text(json.dumps(metadata))
        with CachedPhaseGeometry(tmp_path/'geometry-cache', factory, plan) as cached, ResidentField(
                gpu, runtime, device, cached, plan, boundary, gradient, **kwargs) as field:
            with pytest.raises(ValueError, match='완료 위치'): field.restore(tmp_path/'checkpoint')
    assert not runtime.data


def test_resident_failure_never_publishes_partial_field_or_checkpoint(driver_library, frozen, tmp_path):
    g, source, provider, plan, _, _ = manufactured_source(tmp_path)
    runtime = Runtime(); gpu = SimpleNamespace(lib=Library(runtime, frozen))
    with source, PhaseGeometry(driver_library, g, source) as factory, DeviceInput(runtime, provider) as device:
      with ResidentField(gpu, runtime, device, factory, plan, np.full(8, .4), np.zeros(8),
                         kernel_identity='CPU_CONTROL_NOT_GPU_EVIDENCE') as field:
        field.advance(deadline=time.monotonic()+60, max_actions=1)
        invalid = np.zeros(8, dtype=SCALED); invalid[0]['mantissa'] = -.5; runtime.upload(field.ray, invalid)
        with pytest.raises(RuntimeError, match='175'): field.advance(deadline=time.monotonic()+60, max_actions=1)
        with pytest.raises(ValueError, match='부분/실패'): field.complete_moments()
        with pytest.raises(ValueError, match='실패 후'): field.checkpoint(tmp_path/'invalid')
        assert field.cursor == 1 and not (tmp_path/'invalid').exists()
    assert not runtime.data
