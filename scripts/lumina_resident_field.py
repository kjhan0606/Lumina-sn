"""기존 광선/경계/관측 계획과 실제 r–nu GPU chain을 연결한다. 자체 J 입력은 불변이다."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lumina_fullsr_sc import SCALED
from lumina_gpu_driver import check, check_host
from lumina_native_columns import sha


def array_identity(arrays):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        array = np.ascontiguousarray(array)
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def actions(plan):
    result = []
    for path, row in enumerate(plan.arrays['paths']):
        result.append(('zero', path))
        for leg in (0, 1):
            if leg == 1 and row[4]: result.append(('core', path))
            begin, count = map(int, row[2*leg:2*leg+2])
            result.extend(('interval', i) for i in range(begin, begin+count))
    if sum(kind == 'interval' for kind, _ in result) != len(plan.arrays['steps']):
        raise ValueError('물리구간 계획의 누락')
    return result


class ResidentField:
    def __init__(self, gpu, runtime, device_input, factory, plan, planck, gradient, *, subdivisions=4, kernel_identity):
        self.rt, self.gpu, self.input, self.factory, self.plan = runtime, gpu, device_input, factory, plan
        self.nf, self.ns = plan.source.nf, plan.source.ns; self.subdivisions = subdivisions
        if (device_input.source is not plan.source or factory.provider._source is not plan.source or
                not np.array_equal(factory.radius, plan.arrays['radius']) or
                not np.array_equal(factory.velocity, plan.arrays['velocity']) or
                type(subdivisions) is not int or not 1 <= subdivisions <= 16):
            raise ValueError('같은 원 자료/기하·phase 세분 필요')
        required = ('lumina_fullsr_chain_stream', 'lumina_fullsr_chain_encode', 'lumina_fullsr_chain_decode',
                    'lumina_history_cuda_boundary', 'lumina_history_cuda_accumulate')
        if any(not hasattr(gpu.lib, name) for name in required): raise ValueError('상주 chain/aux API가 없는 바이너리')
        planck, gradient = [np.ascontiguousarray(a, dtype='f8') for a in (planck, gradient)]
        if (planck.shape != (self.nf,) or gradient.shape != planck.shape or np.any(planck < 0) or
                not np.isfinite(planck).all() or not np.isfinite(gradient).all()): raise ValueError('같은 native 경계 필요')
        self.identity = dict(schema='lumina-resident-field-v1', kernel=kernel_identity,
            geometry_factory=factory.identity,
            input=device_input.identity, plan=array_identity(plan.arrays),
            boundary=array_identity(dict(planck=planck, gradient=gradient)), subdivisions=subdivisions)
        self.sequence = actions(plan); self.cursor = 0; self.failed = False; self.blocks = []
        self.intervals = 0; self.observations = 0; self.started = time.monotonic()
        try:
            self.ray, self.shadow = [self.allocate(self.nf*16) for _ in range(2)]
            self.double_ray = self.allocate(self.nf*8); self.conversions = self.allocate(24)
            self.moments = self.allocate(3*self.ns*self.nf*8)
            self.boundaries = [self.upload(a) for a in (planck, gradient)]
            self.nodes = self.upload(np.ascontiguousarray(plan.arrays['event_node'], dtype='i4'))
            self.weights = self.upload(np.ascontiguousarray(plan.arrays['event_weights'], dtype='f8'))
            self.core_node = self.upload(np.array([self.ns-1], dtype='i4'))
            self.core_weights = self.upload(np.ascontiguousarray(plan.arrays['core_weights'], dtype='f8'))
            self.geometry = [self.allocate(size) for size in (65536*8, 65536*8, 65537*24, 65537*8)]
            self.rt.zero(self.ray); self.rt.zero(self.shadow); self.rt.zero(self.moments); self.rt.zero(self.conversions)
            self.rt.sync()
        except Exception:
            self.close(); raise

    def allocate(self, size):
        value = self.rt.allocate(int(size)); self.blocks.append(value); return value

    def upload(self, array):
        value = self.allocate(array.nbytes); self.rt.upload(value, array); return value

    def advance(self, *, deadline, max_actions=None):
        if self.failed: raise ValueError('실패 세대는 재개/발행하지 않는다. 원 checkpoint에서 새 문맥으로 복원한다')
        if max_actions is not None and (type(max_actions) is not int or max_actions < 0): raise ValueError('action 수 오류')
        stop = len(self.sequence) if max_actions is None else min(len(self.sequence), self.cursor+max_actions)
        try:
            while self.cursor < stop:
                if time.monotonic() >= deadline: raise TimeoutError('전체장 마감; 마지막 유효 checkpoint를 보존')
                kind, index = self.sequence[self.cursor]
                if kind == 'zero':
                    self.rt.zero(self.ray)
                elif kind == 'core':
                    check(self.gpu.lib.lumina_history_cuda_boundary(self.nf, *[b.address for b in self.boundaries],
                        float(self.plan.arrays['core_mu'][index]), self.double_ray.address, self.rt.status))
                    check(self.gpu.lib.lumina_fullsr_chain_encode(self.nf, self.double_ray.address, self.ray.address, self.rt.status))
                    check(self.gpu.lib.lumina_history_cuda_accumulate(self.ns, self.nf, 1, self.core_node.address,
                        self.core_weights.address+index*24, self.double_ray.address, self.moments.address, self.rt.status))
                    self.rt.sync(); self.observations += 1
                else:
                    row = self.factory.grid(self.plan.arrays['steps'][index], self.subdivisions)
                    for buffer, array in zip(self.geometry, (row['length'], row['drift'], row['locations'], row['phase'])):
                        self.rt.upload(buffer, array)
                    check(self.gpu.lib.lumina_fullsr_chain_stream(ct.byref(self.input.view), len(row['length']), 0, self.nf,
                        *[b.address for b in self.geometry], self.plan.source.dlognu,
                        self.ray.address, self.shadow.address, self.rt.status))
                    first, last = map(int, self.plan.arrays['event_ptr'][index:index+2])
                    if last > first:
                        check(self.gpu.lib.lumina_fullsr_chain_decode(self.nf, self.shadow.address, self.double_ray.address,
                                                                   self.conversions.address, self.rt.status))
                        check(self.gpu.lib.lumina_history_cuda_accumulate(self.ns, self.nf, last-first,
                            self.nodes.address+first*4, self.weights.address+first*24,
                            self.double_ray.address, self.moments.address, self.rt.status))
                    # 구간 수송과 관측 누적이 모두 끝난 뒤에만 광선 상태를 교환한다.
                    self.rt.sync(); self.ray, self.shadow = self.shadow, self.ray
                    self.intervals += 1; self.observations += last-first
                self.cursor += 1
        except Exception:
            self.failed = True; raise
        return self.report()

    def report(self):
        return dict(next_action=self.cursor, actions=len(self.sequence), physical_intervals=self.intervals,
                    observations=self.observations, complete=self.cursor == len(self.sequence) and not self.failed,
                    failed=self.failed, elapsed_seconds=time.monotonic()-self.started,
                    accepted_physical_field=False, final_goal='NOT_MET')

    def complete_moments(self):
        if self.failed or self.cursor != len(self.sequence): raise ValueError('부분/실패 장은 완성장으로 발행하지 않는다')
        values = self.rt.download(self.moments, (3, self.ns, self.nf))
        if not np.isfinite(values).all() or np.any(values[[0, 2]] < 0): raise ValueError('완성 J/K의 음수 또는 비유한 모멘트')
        return values

    def checkpoint(self, root):
        if self.failed: raise ValueError('실패 후 부분 모멘트를 checkpoint로 승격하지 않는다')
        root = Path(root).resolve(); root.mkdir(exist_ok=False); self.rt.sync()
        check_host((3*self.ns*self.nf+self.nf*2)*8)
        data = {'ray.npy': self.rt.download(self.ray, self.nf, dtype=SCALED),
                'moments.npy': self.rt.download(self.moments, (3, self.ns, self.nf)),
                'conversions.npy': self.rt.download(self.conversions, 3, dtype='u8')}
        hashes = {}
        for name, array in data.items():
            with (root/name).open('xb') as stream: np.save(stream, array, allow_pickle=False)
            hashes[name] = sha(root/name)
        record = dict(identity=self.identity, next_action=self.cursor, intervals=self.intervals,
                      observations=self.observations, files=hashes, complete=self.cursor == len(self.sequence))
        with (root/'checkpoint.json').open('x') as stream: json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return record

    def restore(self, root):
        if self.failed or self.cursor: raise ValueError('새 상주 세대에서만 복원한다')
        root = Path(root).resolve(); record = json.loads((root/'checkpoint.json').read_text())
        if record['identity'] != self.identity or not 0 <= record['next_action'] <= len(self.sequence):
            raise ValueError('checkpoint의 입력/기하/소스/바이너리 정체성 불일치')
        prefix = self.sequence[:record['next_action']]
        expected_intervals = sum(kind == 'interval' for kind, _ in prefix)
        expected_observations = sum(1 if kind == 'core' else int(self.plan.arrays['event_ptr'][index+1]-
            self.plan.arrays['event_ptr'][index]) if kind == 'interval' else 0 for kind, index in prefix)
        if record['intervals'] != expected_intervals or record['observations'] != expected_observations:
            raise ValueError('checkpoint의 완료 위치/구간/관측 수 불일치')
        for name, target, shape, dtype in (('ray.npy', self.ray, (self.nf,), SCALED),
            ('moments.npy', self.moments, (3, self.ns, self.nf), np.dtype('f8')),
            ('conversions.npy', self.conversions, (3,), np.dtype('u8'))):
            if sha(root/name) != record['files'][name]: raise ValueError('checkpoint 원본 변경')
            array = np.load(root/name, mmap_mode='r', allow_pickle=False)
            if array.shape != shape or array.dtype != dtype: raise ValueError('checkpoint 형상/dtype 불일치')
            self.rt.upload(target, array); array._mmap.close()
        self.rt.sync(); self.cursor = record['next_action']; self.intervals = record['intervals']; self.observations = record['observations']

    def close(self):
        for value in reversed(self.blocks): value.close()
        self.blocks = []

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
