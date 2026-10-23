"""공간선형 B GPU 외곽. C의 수송 맵/전ν 최대값 수용은 사용하지 않는다."""
import ctypes as ct
import math
from pathlib import Path
import time

import numpy as np

from lumina_linear_native import DenseView
from lumina_gpu_driver import (Runtime as BaseRuntime, GpuDriver, Arena, Leaf, Location,
    P, I, D, U, S, check, CACHE_LIMIT)
from lumina_linear_native import NativeMap, Positive

MAP_BYTES = ct.sizeof(NativeMap)
NODE_BYTES = 272
LOCAL_BYTES = 64
POSITIVE_DTYPE = np.dtype([('m', '<f8'), ('e', '<i8')])


def endpoint_prefix(piece,chi_block,source_block):
    if piece.offset>=piece.nf-1:return 0
    last=math.ceil(min(piece.nf-1,piece.offset+piece.count-1+piece.span))
    if piece.bins:
        end=max(((piece.bins+block-1)//block)*block for block in (chi_block,source_block))
        last=max(last,min(piece.nf-1,end))
    return last+1


class Piece(ct.Structure):
    _fields_ = [('view', DenseView), ('locations', Location*2)]+[
        (k, D) for k in ('length', 'h', 'da', 'db', 'offset', 'span')]+[
        (k, I) for k in ('begin', 'count', 'base', 'nf', 'block', 'leaves', 'bins', 'source_leaves')]


class Runtime(BaseRuntime):
    def __init__(self, library, baseline_free, *, _node_bytes=NODE_BYTES, _map_bytes=MAP_BYTES):
        functions = {
            'lumina_linear_cuda_node_bytes': ([], S), 'lumina_linear_cuda_map_bytes': ([], S),
            'lumina_linear_cuda_local_bytes': ([], S),
            'lumina_linear_cuda_plan': ([P, P]+[D]*4+[I]*3+[P], I),
            'lumina_linear_cuda_build': ([P, P, P], I),
            'lumina_linear_cuda_query': ([P, P, D, I, P, P], I),
            'lumina_linear_cuda_replay': ([I]*4+[D]+[P]*8, I),
            'lumina_linear_cuda_weighted': ([I]+[P]*5, I),
            'lumina_history_cuda_reconstruct': ([I, D]+[P]*4, I),
            'lumina_history_cuda_boundary': ([I, P, P, D, P, P], I),
            'lumina_history_cuda_accumulate': ([I]*3+[P]*5, I)}
        abis = (('lumina_linear_cuda_node_bytes', _node_bytes), ('lumina_linear_cuda_map_bytes', _map_bytes),
                ('lumina_linear_cuda_local_bytes', LOCAL_BYTES))
        super().__init__(library, baseline_free, _functions=functions, _abis=abis)


def memory_layout(nf, width, *, node_bytes=NODE_BYTES, extra_bytes=0):
    if type(nf) is not int or type(width) is not int or not 8 <= nf <= 2000000 or not 1 <= width <= 65536:
        raise ValueError('B GPU 격자/폭 오류')
    if type(node_bytes) is not int or node_bytes<NODE_BYTES or type(extra_bytes) is not int or extra_bytes<0:
        raise ValueError('모멘트 작업공간 크기 오류')
    arena = max(640*1024**2, 3*nf*MAP_BYTES)
    arena = (arena+255)//256*256
    blocks = (nf-2)//32+1; leaves = 1 << (blocks-1).bit_length(); tree = 2*leaves*node_bytes
    results = max(96*1024**2, nf*LOCAL_BYTES); slots = min(4096, results//(width*LOCAL_BYTES))
    slabs = (nf+width-1)//width
    weighted = nf*8+((nf+4095)//4096)*32
    aux = 3*nf*8+nf*16+4096*8*2+slabs*8+8+weighted
    workspace = arena+tree+results+aux+extra_bytes
    if slots < 1 or workspace > CACHE_LIMIT: raise MemoryError('B 활성 적분 작업공간1GiB 초과')
    return dict(arena=arena, tree=tree, results=results, slots=slots, workspace=workspace)


class LinearDriver(GpuDriver):
    node_bytes=NODE_BYTES
    map_bytes=MAP_BYTES
    chi_block=32
    source_block=None
    interval_limit_seconds=60.
    projection_bytes=0
    endpoint_cache=False
    source_integrator=0
    cooperative_shared_bytes=0
    engine='SPATIAL_LINEAR_B'
    def layout_for(self,nf,width):
        return memory_layout(nf,width,node_bytes=self.node_bytes,extra_bytes=self.projection_bytes)

    def tree_bytes(self,piece):
        return 2*piece.leaves*self.node_bytes

    def __init__(self, runtime, provider, cpu_library, *, block_width=256):
        layout = self.layout_for(provider.nf,block_width)
        if runtime.lib.lumina_linear_cuda_node_bytes()!=self.node_bytes:
            raise ValueError('GPU driver와 바이너리의 모멘트 저장 구조가 다르다')
        if runtime.lib.lumina_linear_cuda_map_bytes()!=self.map_bytes:
            raise ValueError('GPU driver와 바이너리의 맵 저장 구조가 다르다')
        endpoint=getattr(runtime.lib,'lumina_linear_cuda_endpoint_bytes',None)
        if endpoint is not None:
            endpoint.argtypes=[I];endpoint.restype=S
        actual=0 if endpoint is None else endpoint(provider.nf)
        if actual!=(4*provider.nf*8 if self.endpoint_cache else 0):
            raise ValueError('GPU driver와 바이너리의 끝점 캐시 비용이 다르다')
        integrator=getattr(runtime.lib,'lumina_linear_cuda_source_integrator',None)
        if integrator is not None:
            integrator.argtypes=[];integrator.restype=I
        if (0 if integrator is None else integrator())!=self.source_integrator:
            raise ValueError('GPU driver와 바이너리의 원천 적분기가 다르다')
        cooperative=getattr(runtime.lib,'lumina_linear_cuda_cooperative_bytes',None)
        if cooperative is not None:cooperative.argtypes=[];cooperative.restype=S
        if (0 if cooperative is None else cooperative())!=self.cooperative_shared_bytes:
            raise ValueError('GPU driver와 바이너리의 협력 상태 비용이 다르다')
        if self.cooperative_shared_bytes:
            compatible=runtime.lib.lumina_linear_cuda_cooperative_device
            compatible.argtypes=[];compatible.restype=I;check(compatible())
        if Path(cpu_library).resolve() != provider._binary or provider.identity.get('field', {}).get('role', '').startswith('REFERENCE'):
            raise ValueError('같은 native context와 자체 J 입력 필요')
        self.rt, self.provider, self.nf, self.ns, self.width = runtime, provider, provider.nf, provider.ns, block_width
        self.lib = runtime.lib; self.blocks = []; self.view = self.generation = None; self.layout = layout
        self.cpu = ct.CDLL(str(Path(cpu_library).resolve())); self.geom = self.cpu.cmf_nodal_driver_geometry
        self.geom.argtypes = [P, I]+[P]*5+[D, I]+[P]*6; self.geom.restype = I
        self.arena = Arena(self.alloc(layout['arena'])); self.tree = self.alloc(layout['tree'])
        self.results = self.alloc(layout['results']); self.pointers = self.alloc(4096*8); self.scores = self.alloc(4096*8)
        self.ray, self.characteristic, self.candidate = [self.alloc(self.nf*8) for _ in range(3)]
        self.frequency_errors = self.alloc(self.nf*16); self.limited = self.alloc(8)
        self.errors = self.alloc(((self.nf+self.width-1)//self.width)*8)
        self.weight_buffer = self.alloc(self.nf*8); self.weight_pairs = self.alloc(((self.nf+4095)//4096)*32)
        self.canonical = self.outer = self.j = self.radii = self.external = None
        self.moments = self.planck = self.gradient = self.event_nodes = self.event_weights = None; self.event_plan = None
        self.upload_generation()

    def group(self, geometry, row, begin, width, target, work):
        lengths, deltas, locations, _ = geometry
        for k in range(3):
            loc, length, da, db = self.piece(lengths, deltas, locations, row, k)
            endpoints = (Location*2)(loc[0], loc[2]); p = Piece()
            check(self.lib.lumina_linear_cuda_plan(ct.byref(self.view), endpoints, length,
                self.provider._source.dlognu, da, db, begin, width, self.chi_block, ct.byref(p)))
            if self.tree_bytes(p) > self.tree.size: raise MemoryError('B 모멘트 tree 작업공간')
            if self.endpoint_cache:
                prefix=endpoint_prefix(p,self.chi_block,self.source_block or self.chi_block)
                work.counts.setdefault('endpoint_reconstructed_columns',0)
                work.take('endpoint_reconstructed_columns',2*(prefix+(prefix<p.nf)))
            blocks=(p.bins+self.chi_block-1)//self.chi_block
            built = min(p.nf-1, blocks*self.chi_block)
            work.take('bin_maps', built); work.take('tree_compositions', p.leaves-1)
            work.take('padding_leaves', p.leaves-blocks)
            if self.projection_bytes:
                source_block=self.source_block or self.chi_block
                blocks=(p.bins+source_block-1)//source_block
                source_built=min(p.nf-1,blocks*source_block)
                # 과거 바이너리는 이 위치를 padding으로 썼다. 새 분리 변형에서만 읽는다.
                source_leaves=p.source_leaves if self.source_block is not None else p.leaves
                # 고수준 구조 작업 단위다. scalar FLOP/CPU formal 수와 혼용하지 않는다.
                for kind,count in (('source_projected_vertices',source_built+blocks),('source_projection_nodes',blocks),
                    ('source_tree_compositions',source_leaves-1),('source_padding_nodes',source_leaves-blocks)):
                    work.counts.setdefault(kind,0);work.take(kind,count)
            check(self.lib.lumina_linear_cuda_build(ct.byref(p), self.tree.address, self.rt.status))
            work.take('map_queries', width)
            check(self.lib.lumina_linear_cuda_query(ct.byref(p), self.tree.address, 1e-8, 100000,
                target.part(k*width*self.map_bytes, width*self.map_bytes).address, self.rt.status))

    def replay(self, current, begin, width, work, *, incident_errors=None):
        if len(current)*width*LOCAL_BYTES > self.results.size or len(current)*width > 2097152:
            raise MemoryError('B 순서 replay 작업공간 한도')
        pointers = np.array([x.cache[1].address for x in current], dtype='u8'); self.rt.upload(self.pointers, pointers)
        work.take('replay_leaf_columns', len(current)*width)
        arguments=(len(current), width, width, self.width, 1e-5,
            self.pointers.address, self.characteristic.part(begin*8, width*8).address,
            self.candidate.part(begin*8, width*8).address, self.results.address, self.scores.address,
            self.errors.part((begin//self.width)*8, ((width+self.width-1)//self.width)*8).address,
            self.frequency_errors.part(begin*16, width*16).address)
        if incident_errors is None:
            check(self.lib.lumina_linear_cuda_replay(*arguments,self.rt.status))
        else:
            uncertain=getattr(self.lib,'lumina_linear_cuda_replay_uncertain',None)
            if uncertain is None or incident_errors.owner is not self.rt:
                raise ValueError('동일 장치의 입사 절대오차 버퍼와 지원 replay가 필요하다')
            uncertain.argtypes=[I]*4+[D]+[P]*9;uncertain.restype=I
            check(uncertain(*arguments,incident_errors.part(begin*16,width*16).address,self.rt.status))

    def interval(self, plan, index, work, *, max_seconds=60.):
        # 새 호출이 초기 실패해도 이전 구간의 진행 상태를 재사용하지 않는다.
        self.last_completed_batch = None
        self.bound()
        if not 0 < max_seconds <= self.interval_limit_seconds: raise ValueError('B 구간 시간 한도')
        if plan.source is not self.provider._source or plan.panels != 1 or type(index) is not int or not 0 <= index < len(plan.arrays['steps']):
            raise ValueError('동일 native 계획의 물리 구간 필요')
        started = time.monotonic(); previous_deadline = work.deadline
        work.deadline = min(previous_deadline, started+max_seconds)
        slabs = (self.nf+self.width-1)//self.width; leaves = [[Leaf(0., 1.)] for _ in range(slabs)]
        cached = []; evictions = 0; rounds = 1; maximum_leaves = 1
        try:
            work.check(); geometry = self.geometry(plan, index, [Leaf(0., 1.)])
            work.take('reconstructed_columns', self.nf)
            check(self.lib.lumina_history_cuda_reconstruct(self.nf, geometry[3], self.ray.address,
                self.characteristic.address, self.limited.address, self.rt.status))
            allocation = self.arena.take(3*self.nf*self.map_bytes)
            if allocation is None: raise MemoryError('B 초기 전ν 맵 작업공간')
            offset, maps = allocation
            try:
                self.group(geometry, 0, 0, self.nf, maps, work)
                initial = Leaf(0., 1.); initial.cache = offset, maps
                self.replay([initial], 0, self.nf, work)
                errors = self.rt.download(self.errors, slabs)
            finally:
                self.arena.release(offset)
            for i in range(slabs): leaves[i][0].score = float(errors[i])
            while True:
                work.check(); active = int(errors.argmax()); worst = float(errors[active])
                if math.isnan(worst) or worst < 0: raise RuntimeError('B 상대오차 결과 비정상')
                self.last_completed_batch = dict(interval_index=index, rounds=rounds, maximum_error_ratio=worst,
                    active_slab=active, active_leaves=len(leaves[active]), completed_candidate=True)
                if worst <= 1.: break
                current = leaves[active]; peak = max(x.score for x in current); updated = []
                for leaf in current:
                    if leaf.score < .5*peak: updated.append(leaf); continue
                    middle = .5*(leaf.a+leaf.b)
                    if leaf.depth >= 32 or middle <= leaf.a or middle >= leaf.b: raise RuntimeError('B 공간 깊이32/표현 한도')
                    if leaf.cache is not None: self.arena.release(leaf.cache[0]); leaf.cache = None
                    updated.extend((Leaf(leaf.a, middle, leaf.depth+1), Leaf(middle, leaf.b, leaf.depth+1)))
                if len(updated) > min(4096, self.layout['slots']): raise RuntimeError('B 공간 leaf/활성 슬롯 한도')
                leaves[active] = current = updated
                begin = active*self.width; width = min(self.width, self.nf-begin)
                missing = [x for x in current if x.cache is None]
                geometry = self.geometry(plan, index, missing)
                for row, leaf in enumerate(missing):
                    allocation = self.arena.take(3*width*self.map_bytes)
                    while allocation is None and cached:
                        other = cached.pop(0)
                        if other == active: continue
                        for x in leaves[other]:
                            if x.cache is not None: self.arena.release(x.cache[0]); x.cache = None
                        evictions += 1; allocation = self.arena.take(3*width*self.map_bytes)
                    if allocation is None: raise MemoryError('B 활성 맵 cache 한도')
                    offset, maps = allocation
                    try: self.group(geometry, row, begin, width, maps, work)
                    except Exception: self.arena.release(offset); raise
                    leaf.cache = offset, maps
                if active in cached: cached.remove(active)
                cached.append(active); self.replay(current, begin, width, work)
                scores = self.rt.download(self.scores, len(current)); value = self.rt.download(
                    self.errors.part(active*8, 8), 1)
                for leaf, score in zip(current, scores): leaf.score = float(score)
                errors[active] = value[0]; rounds += 1; maximum_leaves = max(maximum_leaves, len(current))
            self.rt.sync(); work.check(); self.ray, self.candidate = self.candidate, self.ray
            return dict(elapsed_seconds=time.monotonic()-started, global_error=float(errors.max()), global_tolerance=1.,
                error_normalization='maximum of per-frequency relative ratios; no intensity floor or global bright norm',
                error_certificate=False, max_mesh_leaves=maximum_leaves, rounds=rounds, cache_evictions=evictions,
                cache_peak_bytes=self.arena.peak, active_workspace_bytes=self.layout['workspace'],
                limited_polynomials=int(self.rt.download(self.limited, 1, 'u8')[0]), engine=self.engine)
        finally:
            work.deadline = previous_deadline
            for slab in leaves:
                for leaf in slab:
                    if leaf.cache is not None: self.arena.release(leaf.cache[0]); leaf.cache = None

    def weighted(self, weights):
        """기계적 양의 가중 I 요약. 실제 생산율이라는 이름으로 사용하지 않는다."""
        self.bound(); values = np.ascontiguousarray(weights, dtype='f8')
        if values.shape != (self.nf,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('동일ν의 비음수 가중치 필요')
        self.rt.upload(self.weight_buffer, values)
        check(self.lib.lumina_linear_cuda_weighted(self.nf, self.weight_buffer.address, self.ray.address,
            self.frequency_errors.address, self.weight_pairs.address, self.rt.status))
        parts = self.rt.download(self.weight_pairs, ((self.nf+4095)//4096, 2), POSITIVE_DTYPE)
        total = error = Positive()
        for row in parts:
            total = total+Positive(float(row[0]['m']), int(row[0]['e']))
            error = error+Positive(float(row[1]['m']), int(row[1]['e']))
        error = error+total*(64.*len(parts)*np.finfo('f8').eps)
        return dict(value=total.value(), error=error.value(), role='WEIGHTED_LOCAL_I_NOT_PHOTO_RATE')
