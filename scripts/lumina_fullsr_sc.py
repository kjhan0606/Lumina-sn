"""fullSR 불변량 상풍·SC. 원 계수/기하/각도 가중치를 읽기 전용으로 재사용한다.

전체 원격자 완료는 수송 완결성일 뿐 수치 수렴·물리 동질성 합격이 아니다.
"""
import ctypes as ct
from pathlib import Path

import numpy as np

from lumina_gpu_driver import GpuDriver, Location, Leaf

P, I, D = ct.c_void_p, ct.c_int, ct.c_double
SCALED = np.dtype([('mantissa', 'f8'), ('exponent', 'i8')], align=True)


class Stats(ct.Structure):
    _fields_ = [(k, ct.c_uint64) for k in ('completed_frequencies', 'space_updates',
        'observation_updates', 'heap_bytes')]+[('elapsed_seconds', D)]+[(k, I) for k in
        ('status', 'next_frequency', 'failed_path', 'failed_frequency', 'failed_vertex')]


class SCPlan:
    """기존 광선의 끝점 관측 사건을 보존하고 각 물리구간만 균일 세분한다."""
    def __init__(self, plan, provider, cpu_library, *, spatial_subdivisions=4):
        if (type(spatial_subdivisions) is not int or not 1 <= spatial_subdivisions <= 128 or
                plan.source is not provider._source or
                not np.array_equal(plan.arrays['radius'], provider.g.radius) or
                not np.array_equal(plan.arrays['velocity'], provider.g.velocity)):
            raise ValueError('동일 계수·기하 및 공간 세분 한도 필요')
        self.source, self.provider = plan.source, provider
        helper = GpuDriver.__new__(GpuDriver)
        helper.provider = provider
        helper.cpu = ct.CDLL(str(Path(cpu_library).resolve()))
        helper.geom = helper.cpu.cmf_nodal_driver_geometry
        helper.geom.argtypes = [P, I]+[P]*5+[D, I]+[P]*6
        helper.geom.restype = I
        leaves = [Leaf(k/spatial_subdivisions, (k+1)/spatial_subdivisions)
                  for k in range(spatial_subdivisions)]
        a = plan.arrays
        path_ptr, kind, locations, length, drift, mu = [0], [], [], [], [], []
        event_ptr, event_node, event_weights = [0], [], []

        def vertex(mode, loc, dl=0., delta=0., cosine=0., nodes=(), weights=()):
            kind.append(mode); locations.append(Location(loc.mode, loc.first, loc.last, loc.weight))
            length.append(dl); drift.append(delta); mu.append(cosine)
            event_node.extend(nodes); event_weights.extend(weights); event_ptr.append(len(event_node))

        for path, (mb, mn, pb, pn, core) in enumerate(a['paths']):
            for leg, (begin, count) in enumerate(((mb, mn), (pb, pn))):
                for number, index in enumerate(range(begin, begin+count)):
                    lengths, deltas, loc, _ = helper.geometry(plan, index, leaves)
                    if leg == 0 and number == 0:
                        vertex(1, loc[0])
                    if leg == 1 and number == 0 and core:
                        vertex(2, loc[0], cosine=float(a['core_mu'][path]),
                               nodes=[self.source.ns-1], weights=[a['core_weights'][path]])
                    for row in range(spatial_subdivisions):
                        start, end = map(int, a['event_ptr'][index:index+2])
                        nodes = a['event_node'][start:end] if row == spatial_subdivisions-1 else ()
                        weights = a['event_weights'][start:end] if row == spatial_subdivisions-1 else ()
                        vertex(0, loc[5*row+4], float(lengths[row, 0]),
                               float(deltas[row, 0]-deltas[row, 2]), nodes=nodes, weights=weights)
            path_ptr.append(len(kind))
        self.locations = (Location*len(locations))(*locations)
        self.arrays = dict(path_ptr=np.array(path_ptr, dtype='i4'), kind=np.array(kind, dtype='i4'),
            length=np.array(length, dtype='f8'), drift=np.array(drift, dtype='f8'),
            core_mu=np.array(mu, dtype='f8'), event_ptr=np.array(event_ptr, dtype='i4'),
            event_node=np.array(event_node, dtype='i4'),
            event_weights=np.array(event_weights, dtype='f8').reshape(-1, 3))
        self.spatial_subdivisions = spatial_subdivisions
        self.npath, self.nvertex, self.nevent = len(path_ptr)-1, len(kind), len(event_node)
        if (self.nevent != plan.layout()['step_observations']+plan.layout()['boundary_observations'] or
                np.count_nonzero(self.arrays['kind'] == 2) != a['paths'][:, 4].sum()):
            raise ValueError('광선·core·관측 사건 직렬화 누락')
        for value in self.arrays.values(): value.flags.writeable = False

    def layout(self):
        return dict(paths=self.npath, spatial_vertices=self.nvertex, observations=self.nevent,
            spatial_subdivisions=self.spatial_subdivisions, scaled_blue_bytes=self.nvertex*SCALED.itemsize,
            serialized_bytes=sum(v.nbytes for v in self.arrays.values())+ct.sizeof(self.locations))


class Kernel:
    def __init__(self, library):
        self.library = ct.CDLL(str(Path(library).resolve()))
        self.advance_function = self.library.cmf_fullsr_sc_advance
        self.advance_function.argtypes = [P, I, I]+[P]*6+[I]+[P]*5+[D]+[I]*5+[D]+[P]*4
        self.advance_function.restype = I

    def advance(self, plan, provider, planck, gradient, blue, moments, *, escape=None,
                first=None, last=0, tile_width=128, threads=1, max_seconds=120., frequency_subdivisions=1):
        if plan.provider is not provider: raise ValueError('계획의 동일 제공자 필요')
        if 'REFERENCE' in getattr(getattr(provider, '_field', None), 'identity', {}).get('role', ''):
            raise ValueError('참조 J는 자체장 수송 source로 사용하지 않는다')
        view = provider.borrow_dense_view()
        nf, ns = view.nf, view.ns
        if first is None: first = nf-1
        integers = first, last, tile_width, threads, frequency_subdivisions
        if (any(type(x) is not int for x in integers) or not 0 <= last <= first < nf or
                not 1 <= tile_width <= 4096 or not 1 <= threads <= 16 or
                not 1 <= frequency_subdivisions <= 16 or not np.isfinite(max_seconds) or
                not 0 < max_seconds <= 2700.):
            raise ValueError('주파수·스레드·시간 한도 오류')
        for values, shape, dtype, writable in ((blue, (plan.nvertex,), SCALED, True),
                (moments, (3, ns, nf), np.dtype('f8'), True),
                (escape, (plan.npath, nf), np.dtype('f8'), True),
                (planck, (nf,), np.dtype('f8'), False), (gradient, (nf,), np.dtype('f8'), False)):
            if values is None:
                if shape in ((3, ns, nf), (plan.npath, nf)): continue
                raise ValueError('경계/청색 상태 누락')
            if (not isinstance(values, np.ndarray) or values.shape != shape or values.dtype != dtype or
                    not values.flags.c_contiguous or (writable and not values.flags.writeable)):
                raise ValueError('SC 배열 크기·dtype·연속/쓰기 상태 오류')
        if moments is None and escape is None: raise ValueError('출력 하나 이상 필요')
        a, stats = plan.arrays, Stats()
        address = lambda x: None if x is None else x.ctypes.data
        code = self.advance_function(ct.byref(view), plan.npath, plan.nvertex,
            *[address(a[k]) for k in ('path_ptr', 'kind')], ct.addressof(plan.locations),
            *[address(a[k]) for k in ('length', 'drift', 'core_mu')], plan.nevent,
            *[address(a[k]) for k in ('event_ptr', 'event_node', 'event_weights')],
            address(planck), address(gradient), provider._source.dlognu, frequency_subdivisions,
            first, last, tile_width, threads, max_seconds, address(blue), address(moments), address(escape), ct.byref(stats))
        report = {key: getattr(stats, key) for key, _ in Stats._fields_}
        report.update(complete_requested_range=code == 0, reached_red_edge=code == 0 and last == 0,
                      accepted_physical_field=False, final_goal='NOT_MET')
        return code, report
