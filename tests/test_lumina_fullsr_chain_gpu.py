"""장치 없이 host 연결/축약 순서·자원 반환을 확인한다. GPU 정확도는 실제 실행에서 본다."""
from pathlib import Path
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
from lumina_fullsr_chain_gpu import ChainGPU, DeviceInput, ProviderTiles, LOCATION_DTYPE
from lumina_fullsr_frozen import AFFINE
from lumina_fullsr_sc import SCALED
from lumina_linear_native import DenseView
from run_lumina_fullsr_chain_gpu import (positive, scalar, relative_scaled, selected_columns, manufactured,
                                       array_relative, scaled_counts, grid_comparison)
from run_lumina_fullsr_chain_gpu import run_full
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
    def download(self, source, shape, dtype='f8'):
        return self.data[source.address].reshape(-1)[:int(np.prod(shape))].astype(dtype, copy=True).reshape(shape)


class FakeLibrary:
    def __init__(self, runtime, cpu): self.rt, self.cpu = runtime, cpu
    def lumina_fullsr_chain_build(self, steps, rays, lengths, deltas, chi, eta, maps, status):
        d = self.rt.data
        code, _, built, report = self.cpu.chain(d[lengths], d[deltas], d[chi], d[eta], np.zeros(rays, dtype=SCALED))
        assert code == 0, report
        d[maps] = built.copy(); return 0
    def lumina_fullsr_chain_provider_build(self, view, steps, rays, lengths, deltas, locations, phases, columns, maps, status):
        d = self.rt.data; phase = d[phases]; cols = d[columns]
        chi = .2+.03*cols[None, :]+.1*phase[:, None]
        eta = .3+.02*cols[None, :]+.05*phase[:, None]
        code, _, built, report = self.cpu.chain(d[lengths], d[deltas], chi, eta, np.zeros(rays, dtype=SCALED))
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


def test_reusable_tiles_keep_geometry_and_handle_short_last_tile(frozen):
    gpu, runtime = fake_gpu(frozen); device_input = SimpleNamespace(runtime=runtime, view=DenseView())
    lengths = np.linspace(.1, .7, 5); deltas = lengths*.01; phases = np.linspace(.2, 0., 6)
    locations = np.zeros(6, dtype=LOCATION_DTYPE)
    with ProviderTiles(gpu, runtime, device_input, lengths, deltas, locations, phases, width=4) as work:
        allocations = runtime.next
        for columns in (np.arange(4, dtype='i4'), np.arange(4, 6, dtype='i4')):
            initial = positive(np.full(len(columns), .4))
            chi = .2+.03*columns[None, :]+.1*phases[:, None]
            eta = .3+.02*columns[None, :]+.05*phases[:, None]
            code, expected, _, report = frozen.chain(lengths, deltas, chi, eta, initial); assert code == 0, report
            result, timing, response = work.execute(columns, initial, transmission=True)
            assert relative_scaled(result, expected).max() < 2e-14
            assert response.shape == (len(columns),) and timing['outputs'] == len(columns)
            assert runtime.next == allocations
    assert not runtime.data and len(runtime.closed) == 9


def test_reference_j_is_rejected_before_gpu_upload():
    provider = SimpleNamespace(identity={'field': {'role': 'REFERENCE_NATIVE90_J'}})
    with pytest.raises(ValueError, match='참조 J'): DeviceInput(None, provider)


def test_vector_scaled_difference_matches_scalar_without_floor():
    rng = np.random.default_rng(418)
    first = np.zeros(1000, dtype=SCALED); second = first.copy()
    for data in (first, second):
        data['mantissa'] = rng.uniform(.5, 1., 1000); data['exponent'] = rng.integers(-2400, 2400, 1000)
    first['mantissa'][:10] = 0.; second['mantissa'][5:15] = 0.
    np.testing.assert_allclose(array_relative(first, second), relative_scaled(first, second), rtol=2e-15, atol=0)
    source = SimpleNamespace(frequency=np.arange(1000, dtype=float)+1)
    report = grid_comparison(first, second, source)
    assert report['both_true_zero'] == 5 and report['one_true_zero'] == 10


def test_scaled_classification_distinguishes_zero_underflow_and_gain_range():
    data = np.zeros(5, dtype=SCALED)
    for i, e in enumerate((-2400, -1074, -1073, 2400), start=1): data[i] = (.5, e)
    assert scaled_counts(data) == dict(true_zero=1, binary64_underflow_nonzero=2, binary64_overflow=1)


def test_failed_full_interval_records_terminal_state_without_starting_gpu(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    assert run_full(root, parent_seconds=60) == 1
    record = json.loads((root/'full-interval.json').read_text())
    assert record['status'] == 'INCOMPLETE' and record['phase'] == 'FINISHED'
    assert record['complete_original_frequencies'] is False and not record['cases']
    assert not (root/'I-phase1.npy').exists() and not (root/'complete-tiles.npy').exists()
