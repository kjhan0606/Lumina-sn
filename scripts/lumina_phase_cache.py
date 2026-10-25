"""물질/J와 독립인 기존 C phase 기하를 한 번 만들고 읽기 전용으로 재사용한다."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from lumina_fullsr_chain_gpu import LOCATION_DTYPE
from lumina_fullsr_sc_actual import BASE, BASE_SHA, publish
from lumina_nodal_driver import CompiledDriverPlan
from lumina_phase_geometry import PhaseGeometry
from lumina_resident_field import array_identity
import lumina_local_micro as M
from lumina_native_columns import sha

SCHEMA = 'lumina-actual-phase-cache-v1'
NODE_ARRAYS = ('x', 'radius', 'phase', 'locations')
CELL_ARRAYS = ('length', 'drift')


def geometry_identity(factory, plan, subdivisions):
    """온도/준위인구/J를 키에 넣지 않는다. 반경·속도·적분 격자만 기하를 정한다."""
    if (factory.provider._source is not plan.source or not np.array_equal(factory.radius, plan.arrays['radius']) or
            not np.array_equal(factory.velocity, plan.arrays['velocity']) or
            type(subdivisions) is not int or not 1 <= subdivisions <= 16):
        raise ValueError('같은 원 기하/phase 세분 필요')
    return dict(schema=SCHEMA, factory=factory.identity, dlognu=float(plan.source.dlognu),
        subdivisions=subdivisions, geometry=array_identity({name: plan.arrays[name] for name in
            ('radius', 'velocity', 'gauss_x', 'gauss_w', 'steps')}),
        native_radius=array_identity(dict(radius=plan.source.radius)))


def build_cache(root, factory, plan, *, subdivisions=4, seconds=240):
    if not np.isfinite(seconds) or not 0 < seconds <= 300: raise ValueError('기하 준비는 내부300초 이내')
    identity = geometry_identity(factory, plan, subdivisions)
    root = Path(root).resolve(); root.mkdir(exist_ok=False)
    start = time.monotonic(); deadline = start+seconds; arrays = {}
    record = dict(identity=identity, complete=False, completed_intervals=0, actual_F_calls=0, files={})
    publish(root/'geometry.json', record)
    def check_time():
        if time.monotonic() >= deadline: raise TimeoutError('기하 준비 마감; 부분 cache는 사용하지 않는다')
    try:
        steps = plan.arrays['steps']; sizes = []
        for step in steps:
            check_time(); sizes.append(factory.count(step, subdivisions))
        cell_ptr = np.r_[0, np.cumsum(sizes, dtype='i8')]
        node_ptr = cell_ptr+np.arange(len(cell_ptr), dtype='i8')
        for name, values in (('steps', steps), ('cell_ptr', cell_ptr), ('node_ptr', node_ptr)):
            with (root/(name+'.npy')).open('xb') as stream: np.save(stream, values, allow_pickle=False)
        for name in NODE_ARRAYS+CELL_ARRAYS:
            size = int(node_ptr[-1] if name in NODE_ARRAYS else cell_ptr[-1])
            arrays[name] = np.lib.format.open_memmap(root/(name+'.npy'), mode='w+',
                dtype=LOCATION_DTYPE if name == 'locations' else 'f8', shape=(size,))
        maximum_roundoff = 0.
        for index, step in enumerate(steps):
            check_time(); grid = factory.grid(step, subdivisions)
            if len(grid['length']) != sizes[index]: raise ValueError('같은 C 기하 크기 변경')
            for name, values in arrays.items():
                ptr = node_ptr if name in NODE_ARRAYS else cell_ptr
                values[int(ptr[index]):int(ptr[index+1])] = grid[name]
            maximum_roundoff = max(maximum_roundoff, grid['max_phase_roundoff'])
            record['completed_intervals'] = index+1
        for values in arrays.values(): values.flush()
        for name in (*NODE_ARRAYS, *CELL_ARRAYS, 'steps', 'cell_ptr', 'node_ptr'):
            check_time(); record['files'][name+'.npy'] = sha(root/(name+'.npy'))
        record.update(complete=True, cells=int(cell_ptr[-1]), nodes=int(node_ptr[-1]),
            maximum_phase_roundoff=maximum_roundoff, elapsed_seconds=time.monotonic()-start)
        check_time(); publish(root/'geometry.json', record)
        return record
    except Exception as exc:
        record.update(complete=False, failure=str(exc), elapsed_seconds=time.monotonic()-start)
        publish(root/'geometry.json', record); raise
    finally:
        for values in arrays.values(): values._mmap.close()


class CachedPhaseGeometry:
    def __init__(self, root, factory, plan, *, subdivisions=4):
        root = Path(root).resolve(); record = json.loads((root/'geometry.json').read_text()); self.arrays = {}
        expected = geometry_identity(factory, plan, subdivisions)
        if record.get('complete') is not True or record.get('identity') != expected:
            raise ValueError('완성된 같은 원 phase cache 필요')
        names = set(NODE_ARRAYS+CELL_ARRAYS+('steps', 'cell_ptr', 'node_ptr'))
        if set(record.get('files', {})) != {name+'.npy' for name in names}: raise ValueError('기하 cache 파일 명세 오류')
        try:
            for name in names:
                path = root/(name+'.npy')
                if sha(path) != record['files'][name+'.npy']: raise ValueError('기하 cache 원본 변경')
                self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            n = len(plan.arrays['steps']); a = self.arrays
            if (not np.array_equal(a['steps'], plan.arrays['steps']) or a['steps'].dtype != np.dtype('f8') or
                    any(a[name].shape != (n+1,) or a[name].dtype != np.dtype('i8') for name in ('cell_ptr', 'node_ptr')) or
                    a['cell_ptr'][0] != 0 or np.any(np.diff(a['cell_ptr']) < 1) or
                    np.any(np.diff(a['cell_ptr']) > 65536) or
                    not np.array_equal(a['node_ptr'], a['cell_ptr']+np.arange(n+1)) or
                    record.get('completed_intervals') != n or record.get('cells') != int(a['cell_ptr'][-1]) or
                    record.get('nodes') != int(a['node_ptr'][-1])):
                raise ValueError('기하 cache 구간/절점 명세 오류')
            for name in NODE_ARRAYS+CELL_ARRAYS:
                size = int(a['node_ptr'][-1] if name in NODE_ARRAYS else a['cell_ptr'][-1])
                dtype = LOCATION_DTYPE if name == 'locations' else np.dtype('f8')
                if a[name].shape != (size,) or a[name].dtype != dtype: raise ValueError('기하 cache 배열 형상 오류')
            self.identity = sha(root/'geometry.json'); self.subdivisions = subdivisions
            self.provider = SimpleNamespace(_source=plan.source)
            self.radius, self.velocity = factory.radius, factory.velocity
            self.maximum_roundoff = record['maximum_phase_roundoff']
            self.lookup = {step.tobytes(): i for i, step in enumerate(a['steps'])}
        except Exception:
            self.close(); raise

    def grid(self, step, subdivisions=4):
        step = np.ascontiguousarray(step, dtype='f8')
        if step.shape != (5,) or type(subdivisions) is not int or subdivisions != self.subdivisions:
            raise ValueError('같은 cache 기하/세분 필요')
        index = self.lookup.get(step.tobytes())
        if index is None: raise ValueError('cache 밖의 물리구간')
        row = {}
        for name in NODE_ARRAYS+CELL_ARRAYS:
            ptr = self.arrays['node_ptr' if name in NODE_ARRAYS else 'cell_ptr']
            row[name] = self.arrays[name][int(ptr[index]):int(ptr[index+1])]
        return dict(row, max_phase_roundoff=self.maximum_roundoff, elapsed_seconds=0.)

    def close(self):
        for values in self.arrays.values(): values._mmap.close()
        self.arrays = {}
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--library', required=True)
    parser.add_argument('--seconds', type=int, default=240)
    args = parser.parse_args()
    if sha(BASE) != BASE_SHA: raise ValueError('원 봉인 계획 변경')
    config = json.loads(BASE.read_text())
    with M.N.open_sealed_native(config['config'], config['preparation'], config['preparation_sha256']) as source:
        if source.identity != config['source_identity']: raise ValueError('원 물질 신원 변경')
        geometry = M.open_geometry(config['preparation']); plan = CompiledDriverPlan(geometry, source, panels=1)
        with PhaseGeometry(args.library, geometry, source) as factory:
            report = build_cache(args.root, factory, plan, seconds=args.seconds)
    print(json.dumps(dict(complete=report['complete'], intervals=report['completed_intervals'],
        cells=report['cells'], elapsed_seconds=report['elapsed_seconds'], actual_F_calls=0)))


if __name__ == '__main__': main()
