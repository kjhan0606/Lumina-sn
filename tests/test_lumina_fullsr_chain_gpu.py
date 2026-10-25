"""장치 없이 host 연결/축약 순서·자원 반환을 확인한다. GPU 정확도는 실제 실행에서 본다."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
from lumina_fullsr_chain_gpu import ChainGPU
from lumina_fullsr_frozen import AFFINE
from lumina_fullsr_sc import SCALED
from run_lumina_fullsr_chain_gpu import positive, scalar, relative_scaled, selected_columns, manufactured
from stage_lumina_fullsr_sc import stage
from test_lumina_fullsr_frozen import frozen


class FakeBuffer:
    def __init__(self, runtime, address, size): self.runtime, self.address, self.size = runtime, address, size
    def close(self): self.runtime.data.pop(self.address); self.runtime.closed.append(self.address)


class FakeRuntime:
    status = 0
    def __init__(self): self.data = {}; self.next = 1; self.closed = []; self.fail = False
    def allocate(self, size):
        address = self.next; self.next += 1; self.data[address] = None; return FakeBuffer(self, address, size)
    def upload(self, target, array): self.data[target.address] = array.copy()
    def sync(self):
        if self.fail: raise RuntimeError('모의 장치 실패')
    def download(self, source, shape, dtype='f8'): return self.data[source.address].copy()


class FakeLibrary:
    def __init__(self, runtime, cpu): self.rt, self.cpu = runtime, cpu
    def lumina_fullsr_chain_build(self, steps, rays, lengths, deltas, chi, eta, maps, status):
        d = self.rt.data
        code, _, built, report = self.cpu.chain(d[lengths], d[deltas], d[chi], d[eta], np.zeros(rays, dtype=SCALED))
        assert code == 0, report
        d[maps] = built.copy(); return 0
    def lumina_fullsr_chain_reduce(self, rows, rays, incoming, outgoing, status):
        a = self.rt.data[incoming]; result = np.zeros(((rows+1)//2, rays), dtype=AFFINE)
        for k in range(len(result)):
            for j in range(rays):
                left = a[2*k, j]
                if 2*k+1 == rows: result[k, j] = left; continue
                right = a[2*k+1, j]
                t = scalar(right['transmission'])*scalar(left['transmission'])
                s = scalar(right['transmission'])*scalar(left['source'])+scalar(right['source'])
                result[k, j] = ((t.m, t.e), (s.m, s.e))
        self.rt.data[outgoing] = result; return 0
    def lumina_fullsr_chain_apply(self, rays, maps, initial, output, status):
        result = np.zeros(rays, dtype=SCALED)
        for j in range(rays):
            m = self.rt.data[maps][0, j]
            p = scalar(m['transmission'])*scalar(self.rt.data[initial][j])+scalar(m['source'])
            result[j] = (p.m, p.e)
        self.rt.data[output] = result; return 0


def fake_gpu(cpu):
    runtime = FakeRuntime(); gpu = ChainGPU.__new__(ChainGPU); gpu.lib = FakeLibrary(runtime, cpu)
    return gpu, runtime


@pytest.mark.parametrize('steps', [1, 2, 5, 8])
def test_wrapper_reduction_preserves_noncommuting_order_and_releases_buffers(frozen, steps):
    gpu, runtime = fake_gpu(frozen)
    lengths = np.linspace(.1, .7, steps); deltas = .01*lengths
    chi = np.array([np.linspace(.1, 3., steps+1), np.linspace(-.4, .2, steps+1)]).T.copy()
    eta = np.array([np.linspace(.7, .1, steps+1), np.linspace(.2, .8, steps+1)]).T.copy()
    initial = positive([.4, .2])
    code, expected, _, report = frozen.chain(lengths, deltas, chi, eta, initial); assert code == 0, report
    result, report = gpu.execute(runtime, lengths, deltas, chi, eta, initial)
    assert relative_scaled(result, expected).max() < 2e-14
    assert not runtime.data and len(runtime.closed) == 8
    assert report['endpoint_maps'] == steps*2


def test_wrapper_does_not_return_candidate_when_device_status_fails(frozen):
    gpu, runtime = fake_gpu(frozen); runtime.fail = True
    with pytest.raises(RuntimeError, match='모의 장치'):
        gpu.execute(runtime, [1.], [.1], [[.4], [.7]], [[.3], [.2]], positive([.4]))
    assert not runtime.data and len(runtime.closed) == 8


def test_manufactured_independent_integral_and_scaled_recovery(frozen):
    gpu, runtime = fake_gpu(frozen)
    report = manufactured(frozen, gpu, runtime)
    assert report['passed'] and max(report['independent_relative']) < 1e-12
    assert report['underflow_exponent'] < -1074


def test_selected_columns_seal_global_uv_top_and_pair_budget():
    source = SimpleNamespace(nf=1115020, frequency=np.array([1e12]), dlognu=1.1267110173509078e-5)
    columns = selected_columns(source, [624712, 624968, 625223])
    assert len(columns)*3*2*2 <= 512 and np.all(np.diff(columns) > 0)
    assert {0, source.nf-2, source.nf-1, 624712, 624968, 625223} <= set(columns)


def test_scaled_comparison_does_not_discard_underflow():
    a = np.zeros(2, dtype=SCALED); a[0] = (.5, -2400); a[1] = (.5, 2400)
    b = a.copy(); b['mantissa'] *= 1+1e-8
    np.testing.assert_allclose(relative_scaled(a, b), 1e-8/(1+1e-8), rtol=1e-7)


def test_gpu_staging_captures_import_and_shared_source_closure(tmp_path):
    root = tmp_path/'run'; stage(ROOT, root, chain_gpu=True)
    for name in ('cmf_fullsr_chain.cu', 'cmf_fullsr_linear_map.hpp', 'cmf_positive_moments.h',
                 'cmf_linear_provider.hpp', 'cmf_frequency_history.h', 'cmf_gpu_runtime.cu', 'cmf_fullsr_chain.h'):
        assert (root/'code/src'/name).is_file()
    assert (root/'code/scripts/run_lumina_fullsr_chain_gpu.py').is_file()
    assert (root/'code/scripts/lumina_gpu_driver.py').is_file()
