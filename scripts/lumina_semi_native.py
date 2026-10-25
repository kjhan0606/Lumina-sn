"""원래점 제공자와 전역ν 스트림을 연결하는 순차 광선 후보.

생성은 기하/메모리 계획만 만든다. 실제 계수로 apply하는 권한은 별도이며
참조 J 연결 객체를 자체 장으로 받아들이지 않는다. Python 경로의 생산 속도는 미검증이다.
"""
from collections import OrderedDict
import time

import numpy as np

import lumina_outer_atmosphere as A
from lumina_native_columns import NodalColumnProvider
from lumina_nodal_transport import NodalGeometry
from lumina_semi_lagrangian import interval_geometry
import lumina_semi_stream as S


class NativeStreamingPlan(NodalGeometry):
    def __init__(self, geometry, source, *, panels=1, block_width=32):
        if (type(panels) is not int or not 1 <= panels <= 16 or
                type(block_width) is not int or not 1 <= block_width <= 32 or
                not 8 <= source.nf <= 2000000 or not np.array_equal(geometry.native_radius, source.radius)):
            raise ValueError('원래점/전역ν 기하 연결 입력 오류')
        self.source = source; self.panels = panels; self.block_width = block_width
        self._endpoints = {}; self._geometry = OrderedDict()
        super().__init__(geometry, subdivisions=1)
        self._weights = self._angular_weights()

    def _path(self, impact, coordinates, radii, subdivisions):
        steps = super()._path(impact, coordinates, radii, subdivisions)
        for i, step in enumerate(steps):
            self._endpoints[id(step)] = float(coordinates[i]), float(coordinates[i+1]), float(radii[i]), float(radii[i+1])
        return steps

    def _mu(self, impact, z, index):
        radius = self.sample_radii[index]; beta = float(A.velocity_at(self.g, radius)/A.C)
        mu = z/radius
        return float((mu-beta)/(1-beta*mu))

    def _angular_weights(self):
        by_node = [[] for _ in self.sample_radii]
        core = float(self.boundaries[0])
        for path, (impact, is_core, minus, plus) in enumerate(self.paths):
            for leg, steps in enumerate((minus, plus)):
                for number, step in enumerate(steps):
                    for node in step.observations:
                        by_node[node].append((self._mu(impact, step.z, node), (path, leg, number, node)))
                if leg == 0 and is_core:
                    z = float(np.sqrt((core-impact)*(core+impact)))
                    by_node[-1].append((self._mu(impact, z, len(by_node)-1), (path, 2, 0, len(by_node)-1)))
        weights = {}
        for entries in by_node:
            ordered = sorted(entries, key=lambda x: x[0])
            if len(ordered) < 2 or ordered[0][0] != -1 or ordered[-1][0] != 1:
                raise ValueError('원래점 각도 범위/경계 관측 누락')
            for (x, left), (y, right) in zip(ordered[:-1], ordered[1:]):
                width = y-x
                weights.setdefault(left, np.zeros(3))[:] += [width/4, width*(2*x+y)/12, width*(3*x*x+2*x*y+y*y)/24]
                weights.setdefault(right, np.zeros(3))[:] += [width/4, width*(x+2*y)/12, width*(x*x+2*x*y+3*y*y)/24]
        return weights

    def _interval(self, step, impact):
        key = id(step)
        if key in self._geometry:
            self._geometry.move_to_end(key); return self._geometry[key]
        data = interval_geometry(self.g, impact, *self._endpoints[key], self.panels)
        self._geometry[key] = data
        if len(self._geometry) > 128: self._geometry.popitem(last=False)
        return data

    def layout(self):
        nf, ns = self.source.nf, self.source.ns; blocks = (nf+self.block_width-1)//self.block_width
        largest_drift_bins = max(step.drift for _, _, a, b in self.paths for step in a+b)/self.source.dlognu
        return dict(native_nodes=ns, frequencies=nf, physical_intervals=self.half_segments,
            paths=len(self.paths), angular_events=len(self._weights),
            panels=self.panels, frequency_blocks=blocks,
            largest_physical_interval_drift_bins=largest_drift_bins,
            largest_interval_average_bins_per_panel=largest_drift_bins/self.panels,
            minimum_python_block_dispatches=self.half_segments*blocks,
            scalar_formal_steps=self.half_segments*self.panels*nf,
            kernel_entries=self.half_segments*self.panels*blocks,
            source_column_callback_calls=self.half_segments*blocks*(self.panels+1)*2,
            moment_output_bytes=3*ns*nf*8, two_ray_buffers_bytes=2*nf*8,
            moment_addition_temporary_bytes=3*nf*8, two_boundary_arrays_bytes=2*nf*8,
            own_J_input_bytes=ns*nf*8, geometry_cache_limit=128,
            memory_scope='주요 장/광선 배열 명세; 제공자/경계/기하/Python/RSS는 별도',
            actual_model_F_calls=0, physical_equivalence='NOT_JUDGED')

    def apply(self, field, segment, planck, gradient, *, max_seconds=120.):
        if getattr(field, 'identity', {}).get('role') == 'REFERENCE_INPUT_CONNECTION_ONLY_NOT_SELF_SOLUTION':
            raise ValueError('참조 J 연결 객체는 자체 수송해 입력이 아니다')
        if not callable(segment) or not np.isfinite(max_seconds) or not 0 < max_seconds <= 120:
            raise ValueError('원래점 직접 실행 입력/상한 오류')
        nf, ns = self.source.nf, self.source.ns
        planck, gradient = (np.asarray(v) for v in (planck, gradient))
        if (planck.shape != (nf,) or gradient.shape != (nf,) or
                not np.isfinite(planck).all() or np.any(planck < 0) or not np.isfinite(gradient).all()):
            raise ValueError('같은 격자의 내부 경계 배열 오류')
        provider = NodalColumnProvider(self.g, self.source, field)
        output = np.zeros((3, ns, nf)); started = time.monotonic(); reconstructions = 0
        def accumulate(key, intensity):
            output[:, key[-1], :] += self._weights[key][:, None]*intensity[None, :]
        for path, (impact, is_core, minus, plus) in enumerate(self.paths):
            intensity = np.zeros(nf)
            for leg, steps in enumerate((minus, plus)):
                if leg == 1 and is_core:
                    core = float(self.boundaries[0]); z = float(np.sqrt((core-impact)*(core+impact)))
                    intensity = planck+self._mu(impact, z, ns-1)*gradient
                    if not np.isfinite(intensity).all() or np.any(intensity < 0): raise ValueError('내부 확산 경계의 음수 강도')
                    accumulate((path, 2, 0, ns-1), intensity)
                for number, step in enumerate(steps):
                    remaining = max_seconds-(time.monotonic()-started)
                    if remaining <= 0: raise TimeoutError('원래점 소형 직접시험 시간 상한')
                    data = self._interval(step, impact)
                    incident = intensity
                    intensity, report = S.propagate(segment, self.source.dlognu, nf, lambda i: incident[i],
                        lambda a, b: data['geometry'][a, b],
                        lambda x, i: provider.at(data['radius'][x], i), lambda x: data['delta'][x],
                        panels=self.panels, block_width=self.block_width, max_seconds=min(120., remaining))
                    reconstructions += report['physical_interval_reconstructions']
                    for node in step.observations: accumulate((path, leg, number, node), intensity)
        return output[0], dict(moments=output, moment_layout='(3,native_nodes,global_frequency)',
            reconstructions=reconstructions, physical_intervals=self.half_segments,
            geometry_cache_entries=len(self._geometry), retained_angular_intensity_cube=False,
            profile_cache_peak_entries=provider.profile_cache_peak_entries,
            elapsed_seconds=time.monotonic()-started, accepted_physical_field=False, final_goal='NOT_JUDGED')
