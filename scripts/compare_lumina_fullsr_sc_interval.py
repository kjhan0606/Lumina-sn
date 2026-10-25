#!/usr/bin/env python3
"""원13003의 같은 영입사·동결 source를 보존된 독립 미세3열과 대조한다."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_fullsr_sc import Kernel, Location, SCALED
from lumina_fullsr_sc_actual import BASE, BASE_SHA, publish
from lumina_gpu_driver import GpuDriver, Leaf
import lumina_local_micro as M
from lumina_nodal_driver import CompiledDriverPlan


def frozen_interval(kernel, provider, lengths, deltas, loc, subdivisions, columns, deadline, reconstruction):
    from lumina_linear_native import Positive
    ray = np.zeros(provider.nf, dtype=SCALED)
    report = dict(cell_calls=0, primitive_maps=0, compositions=0, elapsed_seconds=0.,
                  bin_subdivisions=subdivisions, ray_grid_subdivisions=1,
                  requested_red_column=min(columns), red_halo_cells=len(lengths)-1,
                  reconstruction='POSITIVE_CUBIC_U' if reconstruction else 'LINEAR_U')
    code = 0
    for k in range(len(lengths)):
        remaining = deadline-time.monotonic()
        if remaining <= 1: raise TimeoutError('중점 동결 비교 전체 시간 한도')
        # cubic U의 왼쪽 이웃을 위해 남은 공간 조각 수만큼 red halo를 둔다.
        lower = max(0, min(columns)-(len(lengths)-1-k))
        code, updated, part = kernel.cell(provider, loc[5*k+2], float(lengths[k, 0]),
            float(deltas[k, 0]-deltas[k, 2]), ray, last=lower,
            bin_subdivisions=subdivisions, reconstruction=reconstruction, max_seconds=remaining-1)
        for key in ('primitive_maps', 'compositions', 'elapsed_seconds'): report[key] += part[key]
        report['cell_calls'] += 1
        if code:
            report['failed_cell'] = k; report['failure'] = part
            return code, None, report
        ray = updated
    values = np.array([Positive(float(x['mantissa']), int(x['exponent'])).value() for x in ray[columns]])
    return code, values, report


def compare(root, library, seconds=160, cases=None, engine='UPWIND_SC', reconstruction=1, bin_rule='MEAN_CHI'):
    cases = ((4, 1), (16, 1), (64, 1), (64, 4)) if cases is None else cases
    if (type(seconds) is not int or not 1 <= seconds <= 160 or not 1 <= len(cases) <= 4 or
            any(len(case) != 2 or any(type(x) is not int for x in case) or
                not 1 <= case[0] <= 4096 or not 1 <= case[1] <= 16 for case in cases)):
        raise ValueError('최대4조합·공간4096·주파수16·내부160초 한도')
    if (engine not in ('UPWIND_SC', 'FROZEN_MIDPOINT') or reconstruction not in (0, 1) or
            bin_rule not in ('MEAN_CHI', 'LINEAR_CHI')):
        raise ValueError('명시적 수송/재구성 선택 필요')
    root = Path(root).resolve(); output = root/'interval-comparison.json'
    if output.exists(): raise FileExistsError('기존 독립 대조 보존')
    if M.N.sha(BASE) != BASE_SHA: raise ValueError('원 봉인 변경')
    start = time.monotonic(); c = json.loads(BASE.read_text())
    if M.N.sha(c['cpu_library']) != c['sealed_sha256'][c['cpu_library']]:
        raise ValueError('기존 제공자·기하 바이너리 변경')
    record = dict(status='STARTED', cases=[], final_goal='NOT_MET', accepted_physical_field=False,
        interval=13003, engine=engine, bin_rule=bin_rule, input_J_I='EXPLICIT_ZERO', reference_columns=c['fine_columns'],
        reference_values=c['fine_values'], input_plan_sha256=BASE_SHA,
        kernel_sha256=M.N.sha(library), runner_sha256=M.N.sha(__file__),
        scope='독립 미세 구간3열과 SC 공간·주파수 세분 대조; 전체장 오차 보증 아님')
    publish(output, record)
    try:
        with M.N.open_sealed_native(c['config'], c['preparation'], c['preparation_sha256']) as source:
            if source.identity != c['source_identity']: raise ValueError('원 계수 변경')
            geometry = M.open_geometry(c['preparation'])
            original = CompiledDriverPlan(geometry, source, panels=1, block_width=128)
            with CompiledInput(c['cpu_library'], geometry, source, M.ZeroTrial(source),
                               slab_width=128, cache_rows=128, profiles=1024) as provider:
                provider.bind_zero_trial(); record['dense_preparation'] = provider.prepare_dense()
                helper = GpuDriver.__new__(GpuDriver); helper.provider = provider
                helper.geom = ct.CDLL(c['cpu_library']).cmf_nodal_driver_geometry
                helper.geom.argtypes = [ct.c_void_p, ct.c_int]+[ct.c_void_p]*5+[
                    ct.c_double, ct.c_int]+[ct.c_void_p]*6
                helper.geom.restype = ct.c_int
                if engine == 'FROZEN_MIDPOINT':
                    from lumina_fullsr_frozen import FrozenKernel
                    kernel = FrozenKernel(library, linear_chi=bin_rule=='LINEAR_CHI')
                else: kernel = Kernel(library)
                boundary = np.zeros(source.nf)
                for space, frequency in cases:
                    leaves = [Leaf(k/space, (k+1)/space) for k in range(space)]
                    lengths, deltas, loc, _ = helper.geometry(original, 13003, leaves)
                    if engine == 'FROZEN_MIDPOINT':
                        code, actual, report = frozen_interval(kernel, provider, lengths, deltas, loc,
                            frequency, c['fine_columns'], start+seconds, reconstruction)
                        row = dict(spatial_subdivisions=space, bin_subdivisions=frequency, report=report)
                        if code == 0:
                            reference = np.array(c['fine_values'])
                            row.update(values=actual.tolist(), signed_relative=(actual/reference-1).tolist())
                        record['cases'].append(row); record['elapsed_seconds'] = time.monotonic()-start
                        publish(output, record)
                        if code: raise RuntimeError('중점 동결 비교 미완: '+str(code))
                        continue
                    locations = [loc[0]]+[loc[5*k+4] for k in range(space)]
                    n = space+1
                    plan = SimpleNamespace(provider=provider, npath=1, nvertex=n, nevent=0,
                        locations=(Location*n)(*locations), arrays=dict(
                            path_ptr=np.array([0, n], dtype='i4'), kind=np.array([1]+[0]*space, dtype='i4'),
                            length=np.r_[0., lengths[:, 0]], drift=np.r_[0., deltas[:, 0]-deltas[:, 2]],
                            core_mu=np.zeros(n), event_ptr=np.zeros(n+1, dtype='i4'),
                            event_node=np.empty(0, dtype='i4'), event_weights=np.empty((0, 3))))
                    blue = np.zeros(n, dtype=SCALED); escape = np.zeros((1, source.nf))
                    remaining = seconds-(time.monotonic()-start)
                    if remaining <= 2: raise TimeoutError('독립 대조 시간 한도')
                    code, report = kernel.advance(plan, provider, boundary, boundary, blue, None,
                        escape=escape, last=min(c['fine_columns']), tile_width=128, threads=1,
                        frequency_subdivisions=frequency, max_seconds=remaining-1)
                    row = dict(spatial_subdivisions=space, frequency_subdivisions=frequency, report=report)
                    if code == 0:
                        actual = escape[0, c['fine_columns']]; reference = np.array(c['fine_values'])
                        row.update(values=actual.tolist(), signed_relative=(actual/reference-1).tolist())
                    record['cases'].append(row); record['elapsed_seconds'] = time.monotonic()-start
                    publish(output, record)
                    if code: raise RuntimeError('SC 비교 미완: '+str(code))
        record['status'] = 'INDEPENDENT_INTERVAL_COMPARISON_COMPLETE_NOT_FIELD'
    except Exception as exc:
        record.update(status='INCOMPLETE', error=type(exc).__name__+': '+str(exc))
    record['elapsed_seconds'] = time.monotonic()-start; publish(output, record)
    return 0 if record['status'].startswith('INDEPENDENT') else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('library', type=Path)
    p.add_argument('--case', dest='cases', nargs=2, type=int, action='append')
    p.add_argument('--engine', choices=('UPWIND_SC', 'FROZEN_MIDPOINT'), default='UPWIND_SC')
    p.add_argument('--reconstruction', type=int, choices=(0, 1), default=1)
    p.add_argument('--bin-rule', choices=('MEAN_CHI', 'LINEAR_CHI'), default='MEAN_CHI')
    args = p.parse_args(); raise SystemExit(compare(**vars(args)))
