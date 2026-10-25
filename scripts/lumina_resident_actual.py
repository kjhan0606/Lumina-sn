#!/usr/bin/env python3
"""GPU 상주 경로의 단회 통합시험. 실제 native90 전체 F와 물질 갱신은 실행하지 않는다."""
import argparse
import ctypes as ct
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import time

import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_fullsr_chain_gpu import ChainGPU, DeviceInput
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import SCALED
from lumina_fullsr_sc_actual import BASE, BASE_SHA, InputField, publish, verify_stage
from lumina_gpu_driver import Runtime, check
from lumina_linear_native import DenseView
from lumina_nodal_driver import CompiledDriverPlan
from lumina_phase_geometry import PhaseGeometry
from lumina_resident_field import ResidentField, array_identity
from run_lumina_fullsr_chain_gpu import (OWN_J, OWN_J_SHA, PROFILE, PROFILE_SHA, positive,
    scalar, array_relative, scaled_counts, selected_columns)
import lumina_local_micro as M
import lumina_native_columns as N
import lumina_outer_atmosphere as A

REFERENCE = Path('/gpfs/kjhan/cmfgen_runs/lumina_chain_full_20261026-0U6yoO/run')
REFERENCE_SHA = 'ec6f7f43ce2e90d474c5ffd8d0535c09c94d52dfa43e55e3811bb53515fcd511'


class ConstantProvider:
    """전 영역의 일정한 제조 chi/eta와 비영 J를 명시한 dense 입력이다."""
    def __init__(self, source, geometry, chi, eta):
        self._source, self.g = source, geometry; ns, nf, ne = source.ns, source.nf, geometry.added+1
        self.identity = dict(source=source.identity, field=dict(role='SYNTHETIC_CONSTANT_J', value=.2))
        canonical = np.zeros((3, ns, nf)); canonical[0] = chi; canonical[1] = .1/1e15
        canonical[2] = eta-canonical[1]*.2; j = np.full((ns, nf), .2)
        outer = np.zeros((nf, 2*ne+1)); outer[:, :ne] = np.log(np.where(chi != 0, abs(chi), 1.))[:, None]
        outer[:, ne:2*ne] = np.log(eta)[:, None]; outer[:, 2*ne] = np.sign(chi)
        self.arrays = [np.ascontiguousarray(a, dtype='f8') for a in (canonical, outer, j, source.radius, geometry.radius[:ne])]
        self.identity['dense_sha256'] = array_identity(dict(canonical=canonical, outer=outer, j=j))
        self.view = DenseView(ns, nf, ne, 1, 1, *[a.ctypes.data_as(ct.POINTER(ct.c_double)) for a in self.arrays])
    def borrow_dense_view(self): return self.view


def manufactured_source(root):
    radius = np.geomspace(1e15, 2e14, 24); temperature = np.full(24, 14000.); ne = np.full(24, 1e8)
    base = A.geometry(radius, .06*A.C*radius/radius[0], np.zeros(24))
    geometry = replace(base, velocity=np.zeros_like(base.velocity))
    frequency = 1e14*np.exp(.05*np.arange(8)); chi = np.array([.7, -.4, 0., .2, 1., .3, -.1, .8])/1e15
    eta = np.linspace(.3, .6, 8)/1e15; parts = np.zeros((7, 24, 8))
    parts[0] = chi-.1/1e15; parts[1] = eta-.02/1e15; parts[6] = .1/1e15
    path = Path(root)/'manufactured-source.bin'
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sII', b'LMPHYS01', 24, 8))
        for array in (frequency, temperature, ne, parts): stream.write(np.asarray(array, dtype='<f8').tobytes())
    source = N.NativeSourceColumns(path, frequency, radius, temperature, ne)
    source.identity = dict(role='MANUFACTURED_STATIC_CONSTANT_COEFFICIENTS', sha256=N.sha(path))
    plan = CompiledDriverPlan(geometry, source, panels=1)
    return geometry, source, ConstantProvider(source, geometry, chi, eta), plan, chi, eta


def analytic_moments(plan, chi, eta, planck, gradient):
    """정적 일정 계수 광선의 해석해에 원래 각도 가중치를 적용한다."""
    result = np.zeros((3, plan.source.ns, plan.source.nf)); a = plan.arrays
    for path, row in enumerate(a['paths']):
        z_start = a['steps'][int(row[0]), 1] if row[1] else 0.; incoming = np.zeros(plan.source.nf)
        for leg in (0, 1):
            begin, count = map(int, row[2*leg:2*leg+2])
            if leg == 1 and row[4]:
                z_start = a['steps'][begin, 1]; incoming = planck+float(a['core_mu'][path])*gradient
                result[:, -1] += a['core_weights'][path, :, None]*incoming[None, :]
            for index in range(begin, begin+count):
                length = a['steps'][index, 2]-z_start; tau = chi*length
                integral = np.ones_like(tau); np.divide(-np.expm1(-tau), tau, out=integral, where=tau != 0)
                intensity = incoming*np.exp(-tau)+eta*length*integral
                first, last = map(int, a['event_ptr'][index:index+2])
                for event in range(first, last):
                    result[:, a['event_node'][event]] += a['event_weights'][event, :, None]*intensity[None, :]
    return result


def manufactured_field(gpu, runtime, root, cpu_library, kernel_sha, deadline):
    g, source, provider, plan, chi, eta = manufactured_source(root)
    boundary = .4+.02*np.arange(8); gradient = .02-.002*np.arange(8)
    expected = analytic_moments(plan, chi, eta, boundary, gradient)
    with source, PhaseGeometry(cpu_library, g, source) as factory, DeviceInput(runtime, provider) as device:
        kwargs = dict(subdivisions=4, kernel_identity=kernel_sha)
        with ResidentField(gpu, runtime, device, factory, plan, boundary, gradient, **kwargs) as field:
            field.advance(deadline=deadline); actual = field.complete_moments(); report = field.report()
        denominator = float(expected[0].max()); error = [float(abs(actual[k]-expected[k]).max()/denominator) for k in range(3)]
        if max(error) > 1e-10: raise ValueError('제조 전체 J/H/K 해석 대조 미달: '+str(error))
        with ResidentField(gpu, runtime, device, factory, plan, boundary, gradient, **kwargs) as field:
            field.advance(deadline=deadline, max_actions=len(field.sequence)//2)
            checkpoint = field.checkpoint(Path(root)/'manufactured-checkpoint')
        with ResidentField(gpu, runtime, device, factory, plan, boundary, gradient, **kwargs) as resumed:
            resumed.restore(Path(root)/'manufactured-checkpoint'); resumed.advance(deadline=deadline)
            difference = float(abs(resumed.complete_moments()-actual).max()/denominator)
            if difference > 1e-10: raise ValueError('checkpoint 복원/관측 재누적 불일치')
        return dict(analytic_JHK_error_normalized_by_J=error, checkpoint_relative=difference,
                    checkpoint_next_action=checkpoint['next_action'], report=report, source_sha256=source.identity['sha256'])


def failure_case(gpu, root, cpu_library, kernel_sha, baseline_free, deadline):
    target = Path(root)/'failure-case'; target.mkdir(exist_ok=False)
    g, source, provider, plan, _, _ = manufactured_source(target)
    with source, PhaseGeometry(cpu_library, g, source) as factory, Runtime(root/'chain_cuda.so', baseline_free, _functions={}, _abis=()) as runtime:
      with DeviceInput(runtime, provider) as device, ResidentField(gpu, runtime, device, factory, plan,
          np.full(8, .4), np.zeros(8), kernel_identity=kernel_sha) as field:
        field.advance(deadline=deadline, max_actions=1)
        invalid = np.zeros(8, dtype=SCALED); invalid[0]['mantissa'] = -.5; runtime.upload(field.ray, invalid)
        try: field.advance(deadline=deadline, max_actions=1)
        except RuntimeError as exc:
            if '175' not in str(exc): raise
        else: raise ValueError('주입한 잘못된 입사 상태를 거부하지 않음')
        try: field.complete_moments()
        except ValueError: pass
        else: raise ValueError('실패 부분장을 발행함')
        try: field.checkpoint(target/'invalid-checkpoint')
        except ValueError: pass
        else: raise ValueError('실패 모멘트를 checkpoint로 승격함')
        if (target/'invalid-checkpoint').exists(): raise ValueError('실패 checkpoint 디렉터리가 생성됨')
        return dict(expected_gpu_status=175, failed_generation_blocked=True, partial_field_not_published=True)


def run(root, *, parent_seconds=480, seconds=430):
    if not 60 <= parent_seconds <= 480 or not 1 <= seconds <= 450: raise ValueError('상주 통합시험 시간 한도 오류')
    root = Path(root).resolve(); output = root/'resident-integration.json'
    if output.exists(): raise FileExistsError('기존 상주 통합 결과 보존')
    started = time.monotonic(); deadline = started+seconds
    record = dict(status='STARTED', phase='OPEN_INPUTS', final_goal='NOT_MET', accepted_physical_field=False,
        actual_native90_F_calls=0, material_updates=0, parent_seconds=parent_seconds, internal_seconds=seconds,
        automatic_retries=0, actual_cases=[])
    def save():
        record.update(elapsed_seconds=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(output, record)
    def check_time():
        if time.monotonic() >= deadline: raise TimeoutError('상주 통합시험 내부 마감')
    save(); own = None
    try:
        record['stage_sha256'] = verify_stage(root)
        if N.sha(BASE) != BASE_SHA or N.sha(PROFILE) != PROFILE_SHA or N.sha(REFERENCE/'full-interval.json') != REFERENCE_SHA:
            raise ValueError('봉인 원 입력/전NF 참조 변경')
        config = json.loads(BASE.read_text()); reference = json.loads((REFERENCE/'full-interval.json').read_text())
        if N.sha(config['cpu_library']) != config['sealed_sha256'][config['cpu_library']]: raise ValueError('봉인 계수 제공자 변경')
        for row in reference['cases']:
            if N.sha(REFERENCE/row['output']) != row['output_sha256']: raise ValueError('완료 참조 출력 변경')
        kernel_sha = N.sha(root/'chain_cuda.so'); record.update(kernel_sha256=kernel_sha, geometry_sha256=N.sha(root/'geometry.so'))
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True, timeout=10)
        gpu = ChainGPU(root/'chain_cuda.so'); device_uuid = gpu.device_uuid()
        matches = [line.split(',') for line in query.stdout.splitlines() if line.split(',')[0].strip().lower() == device_uuid.lower()]
        if len(matches) != 1: raise ValueError('CUDA0과 NVML GPU 신원 불일치')
        _, name, free = matches[0]; record.update(device_name=name.strip(), device_uuid=device_uuid)
        with Runtime(root/'chain_cuda.so', int(free)*1024**2, _functions={}, _abis=()) as runtime:
            record['memory_begin'] = runtime.report(); record['phase'] = 'MANUFACTURED_FIELD'; save()
            record['manufactured'] = manufactured_field(gpu, runtime, root, root/'geometry.so', kernel_sha, deadline); save()
            with N.open_sealed_native(config['config'], config['preparation'], config['preparation_sha256']) as source:
                if source.identity != config['source_identity']: raise ValueError('같은 물질70 원자료 필요')
                g = M.open_geometry(config['preparation']); plan = CompiledDriverPlan(g, source, panels=1)
                if not np.array_equal(plan.arrays['steps'][13003], config['intervals'][0]['step']): raise ValueError('원13003 기하 변경')
                factory = PhaseGeometry(root/'geometry.so', g, source)
                try:
                    grids = {n: factory.grid(config['intervals'][0]['step'], n) for n in (1, 2, 4)}
                    with np.load(PROFILE, allow_pickle=False) as old:
                        errors = {}
                        original_interval = M.Interval(g, config['intervals'][0]['step'])
                        for n, grid in grids.items():
                            selected = np.r_[0, 1+np.flatnonzero(np.rint(old['phases'][1:]*4).astype('i8')%(4//n) == 0)]
                            old_x = old['x'][selected]
                            old_lengths = np.array([original_interval.geometry(float(a), float(b)) for a, b in zip(old_x[:-1], old_x[1:])])
                            errors[str(n)] = dict(maximum_x=float(abs(grid['x']-old['x'][selected]).max()),
                                maximum_radius_relative=float(np.max(abs(grid['radius']/old['radius'][selected]-1))),
                                maximum_proper_length_relative=float(np.max(abs(grid['length']/old_lengths-1))),
                                maximum_phase_roundoff=grid['max_phase_roundoff'], seconds=grid['elapsed_seconds'])
                            if (errors[str(n)]['maximum_x'] > 2e-12 or errors[str(n)]['maximum_radius_relative'] > 2e-13 or
                                    errors[str(n)]['maximum_proper_length_relative'] > 2e-8):
                                raise ValueError('기존 실제 경로/C phase 기하 불일치')
                    record['geometry_comparison'] = errors; save()
                    for source_mode in ('J0_I0_ZERO', 'OWN_J1_NONZERO_ENTRY'):
                        check_time(); record['phase'] = source_mode; save()
                        field = M.ZeroTrial(source) if source_mode == 'J0_I0_ZERO' else InputField(source, OWN_J, OWN_J_SHA)
                        if isinstance(field, InputField): own = field
                        with CompiledInput(config['cpu_library'], g, source, field, slab_width=128, cache_rows=128, profiles=1024) as provider:
                            if source_mode == 'J0_I0_ZERO': provider.bind_zero_trial()
                            else: provider.bind_view()
                            prepared = provider.prepare_dense()
                            with DeviceInput(runtime, provider) as device:
                                blocks = []
                                def alloc(size):
                                    value = runtime.allocate(int(size)); blocks.append(value); return value
                                try:
                                    current, shadow = [alloc(source.nf*16) for _ in range(2)]
                                    geo = [alloc(size) for size in (65536*8, 65536*8, 65537*24, 65537*8)]
                                    columns = selected_columns(source, config['fine_columns'])
                                    input_native = np.zeros(source.nf, dtype=SCALED)
                                    if source_mode != 'J0_I0_ZERO':
                                        input_native = positive(1e-8*np.exp(3*source.dlognu*(np.arange(source.nf)-config['fine_columns'][0])))
                                    runtime.upload(current, input_native); results = []
                                    jobs = [(13003, n, grids[n]) for n in (2, 4)] if source_mode == 'J0_I0_ZERO' else [
                                        (13003, 4, grids[4]), (13004, 4, factory.grid(plan.arrays['steps'][13004], 4))]
                                    if source_mode != 'J0_I0_ZERO' and not np.array_equal(plan.arrays['steps'][13003, [0, 2, 4]], plan.arrays['steps'][13004, [0, 1, 3]]):
                                        raise ValueError('실제 연속 구간 정체성 불일치')
                                    for index, subdivision, grid in jobs:
                                        check_time()
                                        for buffer, array in zip(geo, (grid['length'], grid['drift'], grid['locations'], grid['phase'])): runtime.upload(buffer, array)
                                        tick = time.monotonic()
                                        check(gpu.lib.lumina_fullsr_chain_stream(ct.byref(device.view), len(grid['length']), 0, source.nf,
                                            *[b.address for b in geo], source.dlognu, current.address, shadow.address, runtime.status))
                                        runtime.sync(); kernel_seconds = time.monotonic()-tick
                                        result = runtime.download(shadow, source.nf, dtype=SCALED)
                                        row = dict(source=source_mode, interval=index, subdivisions=subdivision,
                                            steps=len(grid['length']), kernel_sync_seconds=kernel_seconds, dense_preparation=prepared,
                                            scaled_counts=scaled_counts(result))
                                        record['actual_cases'].append(row)
                                        if source_mode == 'J0_I0_ZERO':
                                            original = np.load(REFERENCE/('I-phase'+str(subdivision)+'.npy'), mmap_mode='r', allow_pickle=False)
                                            error = array_relative(result, original); original._mmap.close()
                                            row.update(maximum_previous_fullNF_relative=float(error.max()), worst_column=int(np.argmax(error)))
                                            if error.max() > 1e-8: save(); raise ValueError('같은 이산화의 새 순차/기존 fold 전NF 불일치')
                                            actual = np.array([scalar(v).value() for v in result[config['fine_columns']]])
                                            row['fine_signed_relative'] = (actual/np.array(config['fine_values'])-1).tolist()
                                            if np.max(abs(actual/np.array(config['fine_values'])-1)) > 1e-4: save(); raise ValueError('독립 원3열 미달')
                                            results.append(result)
                                        else:
                                            cpu = FrozenKernel(root/'frozen.so', linear_chi=True)
                                            code, q, report = cpu.begin(provider, grid['phase'][0]*source.dlognu, input_native)
                                            if code: raise ValueError('CPU 입구 재구성 미완: '+str(report))
                                            incident = q[columns].copy()
                                            for j in range(len(incident)):
                                                p = scalar(incident[j])*math.exp(3*grid['phase'][0]*source.dlognu); incident[j] = (p.m, p.e)
                                            chi = []; eta = []
                                            for radius, phase in zip(grid['radius'], grid['phase']):
                                                a, b = provider.at([radius], columns+phase); chi.append(a[0]); eta.append(b[0])
                                            code, expected, _, report = cpu.chain(grid['length'], grid['drift'], chi, eta, incident)
                                            if code: raise ValueError('비영 source/입구 CPU 대조 미완: '+str(report))
                                            error = array_relative(result[columns], expected); row['maximum_CPU_relative'] = float(error.max())
                                            if error.max() > 1e-8: save(); raise ValueError('비영 source/입구·연속 구간 대조 미달')
                                            current, shadow = shadow, current; input_native = result
                                        save()
                                    if source_mode == 'J0_I0_ZERO':
                                        error = array_relative(*results); record['phase2_phase4_maximum_relative'] = float(error.max())
                                        if error.max() > 1e-4: raise ValueError('새 순차 전NF phase2/4 미달')
                                        with (root/'stream-phase4.npy').open('xb') as stream: np.save(stream, results[-1], allow_pickle=False)
                                        record['stream_phase4_sha256'] = N.sha(root/'stream-phase4.npy')
                                finally:
                                    for buffer in reversed(blocks): buffer.close()
                                if source_mode != 'J0_I0_ZERO':
                                    prepared = json.loads(Path(config['preparation']).read_text()); boundary = []
                                    for name in ('inner_planck.npy', 'inner_gradient.npy'):
                                        path = Path(config['preparation']).parent/name
                                        if N.sha(path) != prepared['output_sha256'][name]: raise ValueError('봉인 경계 변경')
                                        boundary.append(np.load(path, mmap_mode='r', allow_pickle=False))
                                    with ResidentField(gpu, runtime, device, factory, plan, *boundary, kernel_identity=kernel_sha) as layout:
                                        record['native90_layout_memory'] = runtime.report()
                                        record['native90_layout'] = layout.report()
                                    for value in boundary: value._mmap.close()
                        if isinstance(field, InputField): field.close(); own = None
                finally: factory.close()
            record['memory_end'] = runtime.report()
        record['failure_case'] = failure_case(gpu, root, root/'geometry.so', kernel_sha, int(free)*1024**2, deadline)
        record['status'] = 'RESIDENT_INTEGRATION_COMPLETE_NOT_PHYSICAL_ACCEPTANCE'
    except Exception as exc:
        record.update(status='INCOMPLETE_PRESERVED', error=type(exc).__name__+': '+str(exc))
    finally:
        if own is not None: own.close()
        record['phase'] = 'FINISHED'; save()
    return 0 if record['status'].startswith('RESIDENT_INTEGRATION_COMPLETE') else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--parent-seconds', type=int, default=480)
    p.add_argument('--seconds', type=int, default=430); a = p.parse_args(); raise SystemExit(run(**vars(a)))
