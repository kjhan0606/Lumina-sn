#!/usr/bin/env python3
"""봉인 물질의 ≤512 구간–출력 시험. 전체장/물질/스펙트럼 갱신은 하지 않는다."""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from lumina_compiled_input import CompiledInput
from lumina_fullsr_chain_gpu import ChainGPU, DeviceInput, LOCATION_DTYPE
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import SCALED, Location
from lumina_fullsr_sc_actual import BASE, BASE_SHA, InputField, publish, verify_stage
from lumina_gpu_driver import Runtime
from lumina_linear_native import Positive
from lumina_nodal_driver import CompiledDriverPlan
import lumina_local_micro as M

OLD_GPU = Path('/gpfs/kjhan/cmfgen_runs/lumina_slab_local_20261025-nJ6bSE/history_cuda.so')
OLD_GPU_SHA = '72e0bea5e4114cfad2a895756a591ce7c8aa2b3e408af35b18764f86a6e78476'
OWN_J = Path('/gpfs/kjhan/cmfgen_runs/lumina_fullsr_sc_20261025-LcW8Dq/run/J.npy')
OWN_J_SHA = '21dd68ee432d75eb746509571a320b25e647c2be65cf248a1d7ee1f566336ca9'


def positive(values):
    out = np.zeros(np.shape(values), dtype=SCALED)
    out['mantissa'], out['exponent'] = np.frexp(values); return out


def scalar(raw): return Positive(float(raw['mantissa']), int(raw['exponent']))


def relative_scaled(first, second):
    result = []
    for raw_a, raw_b in zip(first, second):
        a, b = scalar(raw_a), scalar(raw_b); denominator = b if a <= b else a
        difference = a.difference(b)
        if not difference.m: result.append(0.); continue
        result.append(math.ldexp(difference.m/denominator.m, difference.e-denominator.e))
    return np.array(result)


def selected_columns(source, fine):
    # 사전 고정된 전 주파수 표본과 Si/S를 포함하는 UV 대역, 최고nu 양옆이다.
    uv = [370., 500., 600., 750., 912., 1500.]
    columns = np.r_[np.linspace(0, source.nf-1, 24, dtype='i8'), source.nf-2, fine,
                    [round(math.log((M.A.C*1e8/w)/source.frequency[0])/source.dlognu) for w in uv]]
    return np.unique(np.clip(columns.astype('i8'), 0, source.nf-1))


def profile(provider, interval, columns, check):
    h = provider._source.dlognu; span = interval.point(0.)[1]/h
    ticks = np.arange(math.floor(span*4), -1, -1, dtype='i8'); ticks = ticks[ticks/4 < span]
    phases = np.r_[span, ticks/4]
    if not 2 <= len(phases) <= 65537: raise ValueError('phase 조각 수 상한/영 길이 구간')
    x = np.zeros(len(phases)); delta = np.empty_like(x)
    chi = np.empty((len(x), len(columns))); eta = np.empty_like(chi); started = time.monotonic()
    locations = (Location*len(x))(); prepare = provider._lib.cmf_nodal_input_prepare
    prepare.argtypes = [ct.c_void_p, ct.c_double, ct.c_void_p]; prepare.restype = ct.c_int
    for i, phase in enumerate(phases):
        if i%128 == 0: check()
        if i == 0: point = 0.
        elif phase == 0: point = 1.
        else:
            point = brentq(lambda q: interval.point(float(q))[1]-phase*h, float(x[i-1]), 1.,
                           xtol=4*np.finfo(float).eps, rtol=4*np.finfo(float).eps)
        x[i] = point; radius, delta[i], _, _ = interval.point(float(point))
        if prepare(provider._ctx, radius, ct.byref(locations[i])): raise ValueError('실제 반경 위치 준비 미완')
        a, b = provider.at([radius], columns+delta[i]/h); chi[i], eta[i] = a[0], b[0]
    return x, delta, chi, eta, ticks, np.frombuffer(locations, dtype=LOCATION_DTYPE), time.monotonic()-started


def manufactured(cpu, gpu, runtime):
    lengths = np.array([.2, .7, .1]); deltas = np.array([.002, .03, .001])
    chi = np.array([[.2, -.4, .7], [3., .7, .7], [.1, -1., .7], [.7, .4, .7]])
    eta = np.array([[.3, .7, .3], [.5, .1, .3], [.8, .6, .3], [.4, .3, .3]])
    incoming = positive(np.array([.4, .2, .4])); expected = np.array([.4, .2, .4])
    # CPU 구현과 별개인 선형 계수의 formal 적분이다. 입사와 원천이 모두 비영이다.
    for k, (length, drift) in enumerate(zip(lengths, deltas)):
        for j in range(3):
            q0 = chi[k, j]*length+3*drift; q1 = chi[k+1, j]*length+3*drift
            s = length*quad(lambda u: (eta[k, j]*(1-u)+eta[k+1, j]*u)*
                math.exp(-(1-u)*(q1+.5*(q0-q1)*(1-u))), 0., 1., epsabs=1e-13, epsrel=1e-13)[0]
            expected[j] = expected[j]*math.exp(-.5*(q0+q1))+s
    code, baseline, _, report = cpu.chain(lengths, deltas, chi, eta, incoming)
    if code: raise RuntimeError('제조 CPU chain 실패: '+str(report))
    actual, timing = gpu.execute(runtime, lengths, deltas, chi, eta, incoming)
    errors = relative_scaled(actual, positive(expected)); paired = relative_scaled(actual, baseline)
    # 실제0으로 소실시키지 않은 중간 underflow를 다음 gain에서 되살린다.
    tiny, _ = gpu.execute(runtime, [1.], [0.], [[1200.], [1200.]], [[0.], [0.]], positive([1.]))
    recovered, _ = gpu.execute(runtime, [1.], [0.], [[-1200.], [-1200.]], [[0.], [0.]], tiny)
    return dict(independent_relative=errors.tolist(), CPU_GPU_relative=paired.tolist(), timing=timing,
                underflow_exponent=int(tiny[0]['exponent']), recovered_relative=float(relative_scaled(recovered, positive([1.]))[0]),
                passed=bool(errors.max() <= 1e-10 and paired.max() <= 1e-8 and tiny[0]['mantissa'] > 0 and
                            tiny[0]['exponent'] < -1074 and relative_scaled(recovered, positive([1.]))[0] <= 1e-10))


def run(root):
    root = Path(root).resolve(); output = root/'chain-gpu.json'
    if output.exists(): raise FileExistsError('기존 GPU 결과 보존')
    started = time.monotonic(); deadline = started+420
    record = dict(status='STARTED', phase='OPEN_INPUTS', final_goal='NOT_MET', accepted_physical_field=False,
        actual_model_F_calls=0, SE_RE_updates=0, cases=[], parent_seconds=600, internal_seconds=420,
        automatic_retries=0, total_pair_evaluations=0,
        scope='≤512 실제 구간–출력 쌍의 수치/비용; 전체 J·이온분포·observer 갱신 아님')
    def save():
        record.update(elapsed_seconds=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(output, record)
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('GPU 표본 묶음 내부420초')
    save(); own_field = None
    try:
        record['stage_sha256'] = verify_stage(root)
        if M.N.sha(BASE) != BASE_SHA or M.N.sha(OLD_GPU) != OLD_GPU_SHA: raise ValueError('기존 봉인/장치 runtime 변경')
        c = json.loads(BASE.read_text())
        if M.N.sha(c['cpu_library']) != c['sealed_sha256'][c['cpu_library']]: raise ValueError('계수 제공자 변경')
        record.update(input_plan_sha256=BASE_SHA, old_runtime_sha256=OLD_GPU_SHA,
                      kernel_sha256=M.N.sha(root/'chain_cuda.so'), cpu_sha256=M.N.sha(root/'frozen.so'))
        # CUDA0은 Slurm이 배정한 장치다. NVML의 물리0과 혼동하지 않는다.
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True, timeout=10)
        gpu = ChainGPU(root/'chain_cuda.so'); device_uuid = gpu.device_uuid()
        matches = [line.split(',') for line in query.stdout.splitlines() if line.split(',')[0].strip().lower() == device_uuid.lower()]
        if len(matches) != 1: raise ValueError('배정 CUDA0과 NVML GPU 신원 일치 실패')
        _, name, free = matches[0]; record.update(device_name=name.strip(), device_uuid=device_uuid)
        local = gpu.workspace()
        with Runtime(OLD_GPU, int(free.strip())*1024**2, _functions={}, _abis=()) as runtime:
            memory = runtime.report(); record.update(memory_begin=memory, chain_kernel_local_bytes=local)
            if local > memory['kernel_local']: raise MemoryError('새 kernel local이 기존 보수 예약보다 큼')
            cpu = FrozenKernel(root/'frozen.so', linear_chi=True)
            record['phase'] = 'MANUFACTURED'; save(); record['manufactured'] = manufactured(cpu, gpu, runtime); save()
            if not record['manufactured']['passed']: raise ValueError('독립 제조/CPU-GPU/gain 복원 미달; 실제 표본 실행하지 않음')
            with M.N.open_sealed_native(c['config'], c['preparation'], c['preparation_sha256']) as source:
                if source.identity != c['source_identity']: raise ValueError('원 물질 계수 변경')
                g = M.open_geometry(c['preparation']); plan = CompiledDriverPlan(g, source, panels=1, block_width=128)
                columns = selected_columns(source, c['fine_columns']); indices = [0, 13003, len(plan.arrays['steps'])-1]
                record.update(columns=columns.tolist(), interval_indices=indices,
                    planned_pair_evaluations=len(columns)*len(indices)*2*2,
                    input_J=dict(path=str(OWN_J), sha256=OWN_J_SHA), source_identity=source.identity)
                if record['planned_pair_evaluations'] > 512: raise ValueError('전체 표본 쌍 예산 초과')
                # source 세대마다 별도 context. J1을 각도별 입사강도로 오인하지 않는다.
                for source_mode in ('J0_I0', 'OWN_J1_SYNTHETIC_ENTRY_U'):
                    check()
                    field = M.ZeroTrial(source) if source_mode == 'J0_I0' else InputField(source, OWN_J, OWN_J_SHA)
                    if isinstance(field, InputField): own_field = field
                    with CompiledInput(c['cpu_library'], g, source, field, slab_width=128, cache_rows=128, profiles=1024) as provider:
                        if source_mode == 'J0_I0': provider.bind_zero_trial()
                        else: provider.bind_view()
                        dense = provider.prepare_dense()
                        with DeviceInput(runtime, provider) as device_input:
                          for index in indices:
                            check(); record['phase'] = source_mode+'_'+str(index); save()
                            interval = M.Interval(g, plan.arrays['steps'][index])
                            x, delta, chi, eta, ticks, locations, coefficient_seconds = profile(provider, interval, columns, check)
                            initial = np.zeros(len(columns), dtype=SCALED); entry_seconds = 0.
                            if source_mode != 'J0_I0':
                                entry_start = time.monotonic()
                                # 같은 양의 native U 재구성/최고nu 상수 연장을 사용한다.
                                # 이 입사는 제조 U이며 물리 광선/CMFGEN I를 주입한 것이 아니다.
                                native = positive(1e-8*np.exp(3*source.dlognu*(np.arange(source.nf)-c['fine_columns'][0])))
                                code, shifted, report = cpu.begin(provider, delta[0], native, max_seconds=120.)
                                if code: raise RuntimeError('입구 U 재구성 미완: '+str(report))
                                initial = shifted[columns].copy()
                                for j in range(len(initial)):
                                    value = scalar(initial[j])*math.exp(3*delta[0]); initial[j] = (value.m, value.e)
                                expected_entry = positive(1e-8*np.exp(3*source.dlognu*(
                                    np.minimum(source.nf-1, columns+delta[0]/source.dlognu)-c['fine_columns'][0])))
                                if relative_scaled(initial, expected_entry).max() > 1e-10: raise ValueError('입구 U 독립 대조 미달')
                                entry_seconds = time.monotonic()-entry_start
                            row = dict(source_mode=source_mode, interval=index, phase_span_bins=float(delta[0]/source.dlognu),
                                dense_preparation=dense, coefficient_seconds=coefficient_seconds,
                                device_source_upload_seconds=device_input.upload_seconds,
                                entry_reconstruction_seconds=entry_seconds, entry_kind='ZERO' if source_mode=='J0_I0' else 'SYNTHETIC_U_NATIVE_POSITIVE_RECONSTRUCTION',
                                samples=[]); results = []
                            record['cases'].append(row)
                            for subdivision in (2, 4):
                                check(); selected = np.r_[0, 1+np.flatnonzero(ticks%(4//subdivision) == 0)]
                                xx = x[selected]; dd = delta[selected]
                                lengths = np.array([interval.geometry(float(a), float(b)) for a, b in zip(xx[:-1], xx[1:])])
                                cc = chi[selected]; ee = eta[selected]; cpu_start = time.monotonic()
                                code, baseline, _, report = cpu.chain(lengths, dd[:-1]-dd[1:], cc, ee, initial,
                                                                    max_seconds=min(120., deadline-time.monotonic()))
                                if code: raise RuntimeError('실제 CPU chain 미완: '+str(report))
                                cpu_seconds = time.monotonic()-cpu_start
                                result, timing = gpu.execute(runtime, lengths, dd[:-1]-dd[1:], cc, ee, initial,
                                    device_input=device_input, locations=locations[selected], phases=dd/source.dlognu, columns=columns)
                                paired = relative_scaled(result, baseline)
                                sample = dict(phase_subdivisions=subdivision, CPU_seconds=cpu_seconds,
                                    maximum_CPU_GPU_relative=float(paired.max()), timing=timing)
                                row['samples'].append(sample)
                                if paired.max() > 1e-8:
                                    sample['CPU_GPU_relative'] = paired.tolist(); save()
                                    raise ValueError('같은 이산화 CPU/GPU 차이 미달')
                                if source_mode == 'J0_I0' and index == 13003:
                                    fine = result[np.searchsorted(columns, c['fine_columns'])]
                                    actual = np.array([scalar(v).value() for v in fine])
                                    sample.update(fine_values=actual.tolist(), fine_signed_relative=(actual/np.array(c['fine_values'])-1).tolist())
                                    if np.max(abs(actual/np.array(c['fine_values'])-1)) > 1e-4:
                                        save(); raise ValueError('독립 원3열 기준 미달')
                                results.append(result)
                                record['total_pair_evaluations'] += len(columns)
                            difference = relative_scaled(*results)
                            row.update(phase2_phase4_maximum_relative=float(difference.max()),
                                       phase2_phase4_relative=difference.tolist(), phase_converged=bool(difference.max() <= 1e-4))
                            save()
                    if isinstance(field, InputField): field.close(); own_field = None
                record['memory_end'] = runtime.report()
                spans = np.array([M.Interval(g, step).point(0.)[1]/source.dlognu for step in plan.arrays['steps']])
                total_cells = int(np.sum(np.floor(4*spans).astype('i8')+1))
                seconds_per_map = [s['timing']['build_sync_seconds']/s['timing']['endpoint_maps']
                                   for case in record['cases'] for s in case['samples']]
                maps = total_cells*source.nf
                record['cost_estimate'] = dict(phase=4, physical_intervals=len(spans),
                    span_quantiles=np.quantile(spans, [0., .5, .9, .99, 1.]).tolist(),
                    total_phase_cells=total_cells, full_field_endpoint_maps=maps,
                    measured_build_seconds_per_map_range=[min(seconds_per_map), max(seconds_per_map)],
                    build_only_hours_range=[maps*min(seconds_per_map)/3600, maps*max(seconds_per_map)/3600],
                    caveat='선택 표본의 관측 범위 외삽일 뿐. 작은 출력 수·GPU 활용률·branch 분포·축약/기하/입구/관측 비용이 달라 생산 예측 아님')
        record['status'] = ('SAMPLES_COMPLETE_NOT_FIELD' if all(c['phase_converged'] for c in record['cases'])
                            else 'SAMPLES_COMPLETE_PHASE_REWORK_NOT_FIELD')
    except Exception as exc:
        record.update(status='INCOMPLETE_PRESERVED', error=type(exc).__name__+': '+str(exc))
    finally:
        if own_field is not None: own_field.close()
        record['phase'] = 'FINISHED'; save()
    return 0 if record['status'] == 'SAMPLES_COMPLETE_NOT_FIELD' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); a = p.parse_args(); raise SystemExit(run(**vars(a)))
