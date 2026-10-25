"""동일 구간 드라이버의 계산 순서·캐시·실패 원자성을 장치 없이 확인한다."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lumina_gpu_driver import Arena, Work, Location
from lumina_gpu_linear import LinearDriver, Piece
from lumina_gpu_hybrid import SlabLocalHybridDriver


class Buffer:
    def __init__(self, values, offset=0, size=None):
        self.values = values
        self.offset = offset
        self.size = values.nbytes-offset if size is None else size
        self.address = self

    def part(self, offset, size):
        assert offset >= 0 and offset+size <= self.size
        return Buffer(self.values, self.offset+offset, size)


class Runtime:
    status = None

    def download(self, buffer, shape, dtype='f8'):
        return np.frombuffer(buffer.values, dtype=dtype, count=int(np.prod(shape)),
                             offset=buffer.offset).copy().reshape(shape)

    def sync(self):
        pass


def driver(local, nf=35, width=4, fail_slab=None):
    d = LinearDriver.__new__(LinearDriver)
    d.slab_local_refinement = local
    d.nf, d.width, d.rt = nf, width, Runtime()
    d.bound = lambda: None
    source = object()
    d.provider = SimpleNamespace(_source=source)
    d.layout = {'slots':4096, 'workspace':100000}
    # 한 묶음의 완결 mesh는 들어가되 여러 묶음의 mesh는 함께 못 담는 캐시.
    arena_bytes = max((3*nf*d.map_bytes+255)//256*256,
                      128*((3*width*d.map_bytes+255)//256*256))
    d.arena = Arena(Buffer(np.zeros(arena_bytes, dtype='u1')))
    d.ray = Buffer(np.linspace(.1, .2, nf))
    d.characteristic = Buffer(np.zeros(nf))
    d.candidate = Buffer(np.zeros(nf))
    d.errors = Buffer(np.zeros((nf+width-1)//width))
    d.scores = Buffer(np.zeros(4096))
    d.limited = Buffer(np.zeros(1, dtype='u8'))
    d.lib = SimpleNamespace(lumina_history_cuda_reconstruct=lambda *args:0)
    d.geometry = lambda plan, index, leaves:(leaves,None,None,0.)
    d.final_mesh = {}
    maps = {}

    def group(geometry, row, begin, count, target, work):
        if begin//width == fail_slab and count != nf:
            raise TimeoutError('선택한 주파수 묶음의 실패')
        leaf = geometry[0][row]
        h = leaf.b-leaf.a
        rate = 4.+.3*(np.arange(begin, begin+count)%7)
        scale = np.power(10., (np.arange(begin, begin+count)%3-1)*200.)
        coarse = h*np.exp(rate*(leaf.a+.5*h))*scale
        fine = .5*h*(np.exp(rate*(leaf.a+.25*h))+np.exp(rate*(leaf.a+.75*h)))*scale
        maps[target.offset] = (coarse, fine)
        work.take('map_queries', 3*count)

    def replay(leaves, begin, count, work):
        assert all(a.b == b.a for a,b in zip(leaves, leaves[1:]))
        fine = np.zeros(count)
        error = np.zeros(count)
        components = []
        for leaf in leaves:
            coarse, value = maps[leaf.cache[1].offset]
            fine += value
            component = np.abs(coarse-value)
            error += component
            components.append(component)
        output = d.ray.values[begin:begin+count]+fine
        tolerance = 5e-4*output
        ratios = error/tolerance
        d.candidate.values[begin:begin+count] = output
        for at in range(0,count,width):
            d.errors.values[(begin+at)//width] = ratios[at:at+width].max()
        for row, component in enumerate(components):
            d.scores.values[row] = (component/tolerance).max()
        if count != nf:
            d.final_mesh[begin//width] = [(leaf.a,leaf.b) for leaf in leaves]
        work.take('replay_leaf_columns', len(leaves)*count)

    d.group, d.replay = group, replay
    plan = SimpleNamespace(source=source, panels=1, arrays={'steps':[None]})
    return d, plan


def solve(local, nf=35, width=4):
    d, plan = driver(local, nf, width)
    work = Work(10000000, time.monotonic()+30)
    report = d.interval(plan, 0, work, max_seconds=30)
    assert d.arena.used == 0 and not d.arena.live
    return d, work, report


@pytest.mark.parametrize('nf,width', [(35,4),(37,8),(8,8)])
def test_finishing_slab_preserves_values_errors_and_refinement_mesh(nf,width):
    old, ow, prior = solve(False,nf,width)
    new, nw, actual = solve(True,nf,width)
    np.testing.assert_array_equal(new.ray.values, old.ray.values)
    np.testing.assert_array_equal(new.errors.values, old.errors.values)
    assert new.final_mesh == old.final_mesh
    assert actual['global_error'] <= 1. and prior['global_error'] <= 1.
    assert actual['cache_evictions'] == 0
    if nf > width:
        assert prior['cache_evictions'] > 0
        assert nw.counts['map_queries'] < ow.counts['map_queries']
    assert actual['refinement_order'] == 'FINISH_ACTIVE_SLAB'


def test_partial_completed_slabs_do_not_replace_input_on_failure():
    d, plan = driver(True, fail_slab=1)
    original = d.ray
    values = original.values.copy()
    work = Work(10000000, time.monotonic()+30)
    deadline = work.deadline
    with pytest.raises(TimeoutError):
        d.interval(plan,0,work,max_seconds=30)
    assert d.ray is original
    np.testing.assert_array_equal(d.ray.values,values)
    assert d.arena.used == 0 and not d.arena.live
    assert work.deadline == deadline
    assert not d._interval_complete
    with pytest.raises(ValueError, match='완성 구간'):
        d.weighted(np.ones(d.nf))


def test_new_candidate_is_explicit_and_keeps_original_default():
    assert SlabLocalHybridDriver.slab_local_refinement
    assert not LinearDriver.slab_local_refinement


def test_reporting_failure_after_completed_mesh_keeps_previous_input():
    d, plan = driver(True,nf=8,width=8)
    original = d.ray
    values = original.values.copy()
    download = d.rt.download
    def fail_last(buffer, *args):
        if buffer is d.limited:
            raise RuntimeError('완료 보고의 장치 조회 실패')
        return download(buffer,*args)
    d.rt.download = fail_last
    work = Work(10000000,time.monotonic()+30)
    with pytest.raises(RuntimeError,match='장치 조회 실패'):
        d.interval(plan,0,work,max_seconds=30)
    assert d.ray is original and not d._interval_complete
    np.testing.assert_array_equal(d.ray.values,values)
    assert not d.arena.live and d.arena.used == 0


@pytest.mark.parametrize('expire_after', [None,2])
def test_bounded_query_keeps_global_piece_and_checks_deadline_between_batches(expire_after):
    d = SlabLocalHybridDriver.__new__(SlabLocalHybridDriver)
    calls = []
    class Query:
        def __call__(self, *args):
            p = args[0]._obj
            calls.append((p.begin,p.count,args[4],args[5],args[6].offset))
            return 0
    def plan(*args):
        p = args[-1]._obj
        p.begin,p.count,p.nf = args[6],args[7],4097
        p.bins,p.leaves,p.source_leaves,p.offset,p.span = 2052,2048,128,0.,1.
        return 0
    d.lib = SimpleNamespace(lumina_linear_cuda_plan=plan,
        lumina_linear_cuda_build=lambda *args:0,lumina_linear_cuda_query_range=Query())
    d.view = Piece().view
    d.provider = SimpleNamespace(_source=SimpleNamespace(dlognu=.001))
    d.tree = SimpleNamespace(size=10*1024**2,address=1)
    work = Work(10000000,time.monotonic()+30)
    syncs = []
    def sync():
        syncs.append(1)
        if len(syncs) == expire_after: work.deadline = -1.
    d.rt = SimpleNamespace(status=None,sync=sync)
    d.piece = lambda *args:([Location(1,0,1,.5)]*3,1.,.002,.001)
    target = Buffer(np.zeros(3*2050*d.map_bytes,dtype='u1'))
    if expire_after:
        with pytest.raises(TimeoutError):d.group((None,None,None,None),0,7,2050,target,work)
        assert len(calls) == expire_after
    else:
        d.group((None,None,None,None),0,7,2050,target,work)
        assert len(calls) == len(syncs) == 9
        assert [row[2:4] for row in calls] == [(0,1024),(1024,1024),(2048,2)]*3
        assert all(row[:2] == (7,2050) for row in calls)
        assert [row[4] for row in calls] == [k*2050*d.map_bytes for k in range(3) for _ in range(3)]
        assert work.counts['map_queries'] == 3*2050


if __name__ == '__main__':
    import json
    rows = []
    for nf,width in ((35,4),(37,8),(8,8)):
        _,old_work,old_report = solve(False,nf,width)
        _,new_work,new_report = solve(True,nf,width)
        rows.append(dict(nf=nf,width=width,
            old_queries=old_work.counts['map_queries'],new_queries=new_work.counts['map_queries'],
            old_evictions=old_report['cache_evictions'],new_evictions=new_report['cache_evictions']))
    print(json.dumps(rows))
