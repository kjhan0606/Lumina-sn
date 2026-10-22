#!/usr/bin/env python3
"""원13003 일부 출력의 균일 공간격자 수렴 조사. 전체장이나 종료 mesh 재현은 아니다."""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import time

import numpy as np

import lumina_native_cost_diag as D
from lumina_gpu_driver import Leaf, Location, check
from lumina_gpu_linear import LinearDriver, Piece
from lumina_linear_native import Map, Positive


def advance(value, error, transfer):
    output, local_error = transfer.apply(value)
    return output, transfer.upper_t * error + local_error


def ratio(numerator, denominator):
    if not denominator.m:
        return 0. if not numerator.m else None
    try:
        value = math.ldexp(numerator.m / denominator.m,
                           numerator.e - denominator.e)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def scaled(value):
    return dict(m=value.m, e=value.e)


def run(root, parent, parent_sha, library, library_sha):
    root, parent, library = [Path(p).resolve() for p in (root, parent, library)]
    target = root / 'cpu-spatial-curve.json'
    if target.exists() or target.with_suffix('.partial').exists():
        raise FileExistsError('기존 공간격자 조사 보존')
    if D.sha(parent) != parent_sha or D.sha(library) != library_sha:
        raise ValueError('부모 결과 또는 CPU 바이너리 신원 변경')
    failed = json.loads(parent.read_text())
    if failed['status'] != 'GPU_LOCAL_NOT_MET' or not failed['error'].startswith('TimeoutError:'):
        raise ValueError('완료된 시간초과 결과 필요')
    plan = json.loads(parent.with_name('gpu-local-plan.json').read_text())
    if D.sha(parent.with_name('gpu-local-plan.json')) != failed['plan_sha256']:
        raise ValueError('부모 계획 신원 변경')
    start = time.monotonic()
    record = dict(status='STARTED', parent=str(parent), parent_sha256=parent_sha,
                  library=str(library), library_sha256=library_sha,
                  source_sha256=D.sha(__file__), rows=[], GPU_seconds=0,
                  actual_model_F_calls=0, SE_RE_updates=0, final_goal='NOT_MET',
                  scope='원13003의 별도 균일 mesh; 종료 시 저장되지 않은 GPU mesh와 다르다.',
                  spatial_reference='같은 적분기의 mesh 변화이며 독립 물리 참조가 아니다.')
    D.publish(target, record)
    try:
        with D.inputs(root, record) as (provider, old, _piece, _ends):
            if old['source_identity'] != plan['source_identity']:
                raise ValueError('원 물질·격자·입력 정체성 불일치')
            import lumina_local_micro as M
            from lumina_nodal_driver import CompiledDriverPlan
            geometry = M.open_geometry(old['preparation'])
            driver_plan = CompiledDriverPlan(geometry, provider._source, panels=1, block_width=256)
            if driver_plan.arrays['steps'][13003].tolist() != plan['intervals'][0]['step']:
                raise ValueError('원13003 물리 구간 변경')
            helper = LinearDriver.__new__(LinearDriver)
            helper.provider = provider
            helper.geom = ct.CDLL(old['cpu_library']).cmf_nodal_driver_geometry
            helper.geom.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p]*5 + [ct.c_double, ct.c_int] + [ct.c_void_p]*6
            helper.geom.restype = ct.c_int
            planner = ct.CDLL(old['gpu_library']).lumina_linear_cuda_plan
            planner.argtypes = [ct.c_void_p]*2 + [ct.c_double]*4 + [ct.c_int]*3 + [ct.c_void_p]
            planner.restype = ct.c_int
            probe = ct.CDLL(str(library)).native_cost_probe
            probe.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p]*3
            probe.restype = ct.c_int
            view = provider.borrow_dense_view()
            slab = failed['last_completed_batch']['active_slab']
            begin = slab * 256
            columns = [begin, begin+127, begin+255, 624968]
            if len(set(columns)) != 4 or any(not 0 <= c < provider.nf for c in columns):
                raise ValueError('네 출력열의 범위 오류')
            record['columns'] = columns
            previous = {}
            calls = 0
            for panels in (1, 2, 4, 8, 16, 32, 64, 128):
                pieces = helper.geometry(driver_plan, 13003,
                    [Leaf(i/panels, (i+1)/panels) for i in range(panels)])
                values = [Positive() for _ in columns]
                errors = [Positive() for _ in columns]
                totals = dict(build_seconds=0., query_seconds=0., source_panels=0)
                for i in range(panels):
                    loc, length, da, db = helper.piece(*pieces[:3], i, 0)
                    endpoints = (Location*2)(loc[0], loc[2])
                    piece = Piece()
                    check(planner(ct.byref(view), endpoints, length, provider._source.dlognu,
                                  da, db, 0, provider.nf, 2, ct.byref(piece)))
                    for j, column in enumerate(columns):
                        if time.monotonic()-start >= 170.:
                            raise TimeoutError('CPU 공간격자 조사 내부170초')
                        raw = D.NativeMap()
                        stats = np.zeros(4, dtype='u8')
                        times = np.full(2, -1.)
                        code = probe(ct.byref(piece), column, ct.byref(raw), stats.ctypes.data, times.ctypes.data)
                        calls += 1
                        if code:
                            record['failure'] = dict(panels=panels, leaf=i, column=column, code=code)
                            raise RuntimeError('고정 적분기 실패; 다른 값으로 대체하지 않는다')
                        transfer = Map(raw.transmission.positive(), raw.source.positive(),
                                       raw.source_error.positive(), raw.tau_error)
                        values[j], errors[j] = advance(values[j], errors[j], transfer)
                        totals['build_seconds'] += float(times[0])
                        totals['query_seconds'] += float(times[1])
                        totals['source_panels'] += int(raw.panels)
                rows = []
                for column, value, error in zip(columns, values, errors):
                    gap = value.difference(previous[column]) if column in previous else None
                    rows.append(dict(column=column, value=scaled(value), numerical_error=scaled(error),
                        numerical_error_ratio=ratio(error, value),
                        successive_mesh_relative_difference=None if gap is None else ratio(gap, value)))
                    previous[column] = value
                record['rows'].append(dict(spatial_panels=panels, outputs=rows, **totals))
                record['map_calls'] = calls
                record['elapsed_seconds'] = time.monotonic()-start
                D.publish(target, record)
            record['status'] = 'CPU_SPATIAL_CURVE_FINISHED_NOT_PHYSICAL_ACCEPTANCE'
            return 0
    except Exception as error:
        record.update(status='CPU_SPATIAL_CURVE_STOPPED', error=type(error).__name__+': '+str(error))
        return 1
    finally:
        record['elapsed_seconds'] = time.monotonic()-start
        D.publish(target, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    for key in ('parent', 'parent-sha256', 'library', 'library-sha256'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.parent, args.parent_sha256, args.library, args.library_sha256))
