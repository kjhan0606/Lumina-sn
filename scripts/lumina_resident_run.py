#!/usr/bin/env python3
"""원모델의 GPU 상주 전체 F 단회 실행. 제출·횟수·GPUh 승인은 운전자가 별도 확인한다."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_fullsr_chain_gpu import ChainGPU, DeviceInput
from lumina_fullsr_sc_actual import BASE, BASE_SHA, InputField, publish, verify_stage
from lumina_gpu_driver import Runtime
from lumina_nodal_driver import CompiledDriverPlan
from lumina_phase_cache import CachedPhaseGeometry
from lumina_phase_geometry import PhaseGeometry
from lumina_resident_field import ResidentField
import lumina_local_micro as M
from lumina_native_columns import sha
from lumina_material_trial_field import open_transport_trial, validate_initial_arguments


def open_field_inputs(stack, accepted_input=None, accepted_input_sha256=None):
    """기존70과 새 수용 상태를 구분하며 같은 실제 수송 코어에 전달한다."""
    if (accepted_input is None) != (accepted_input_sha256 is None):
        raise ValueError('새 물질 준비 경로와 SHA가 함께 필요하다')
    if accepted_input is not None:
        from lumina_accepted_field_input import open_prepared
        p, source, geometry, boundary = open_prepared(accepted_input, accepted_input_sha256)
        stack.enter_context(source)
        return dict(cpu_library=p['cpu_library']), source, geometry, boundary, accepted_input_sha256
    if sha(BASE) != BASE_SHA:
        raise ValueError('원 봉인 계획 변경')
    config = json.loads(BASE.read_text())
    if sha(config['cpu_library']) != config['sealed_sha256'][config['cpu_library']]:
        raise ValueError('원 계수 제공자 변경')
    source = stack.enter_context(M.N.open_sealed_native(config['config'], config['preparation'], config['preparation_sha256']))
    if source.identity != config['source_identity'] or (source.ns, source.nf) != (90, 1115020):
        raise ValueError('원 물질70/native90 신원 변경')
    geometry = M.open_geometry(config['preparation'])
    preparation = json.loads(Path(config['preparation']).read_text()); boundary = []
    for name in ('inner_planck.npy', 'inner_gradient.npy'):
        path = Path(config['preparation']).parent/name
        if sha(path) != preparation['output_sha256'][name]:
            raise ValueError('원 내부 경계 변경')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        stack.callback(values._mmap.close); boundary.append(values)
    return config, source, geometry, boundary, BASE_SHA


def write_completed_moments(root, field):
    """완성된 field만 저장한다. 부분/기존 파일을 승격하거나 덮어쓰지 않는다."""
    root = Path(root).resolve()
    targets = [root/name for name in ('moments.npy', 'J.npy')]
    partials = [path.with_suffix('.partial.npy') for path in targets]
    if any(path.exists() for path in targets+partials): raise FileExistsError('기존 전체장 출력 보존')
    values = field.complete_moments()
    with partials[0].open('xb') as stream: np.save(stream, values, allow_pickle=False)
    with partials[1].open('xb') as stream: np.save(stream, values[0], allow_pickle=False)
    hashes = [sha(path) for path in partials]
    for first, last in zip(partials, targets): first.replace(last)
    return dict(moments=dict(path=str(targets[0]), sha256=hashes[0]),
                J=dict(path=str(targets[1]), sha256=hashes[1]))


def run(root, *, cache, seconds=42600, input_j=None, input_j_sha256=None, resume_checkpoint=None,
        accepted_input=None, accepted_input_sha256=None,
        initial_preparation=None, initial_preparation_sha256=None):
    validate_initial_arguments(input_j, input_j_sha256, initial_preparation, initial_preparation_sha256)
    if (type(seconds) is not int or not 1 <= seconds <= 85800 or
            (input_j is None) != (input_j_sha256 is None) or
            (accepted_input is None) != (accepted_input_sha256 is None)):
        raise ValueError('단회 내부 시간/자체장 해시 오류; 외부 자원 승인을 대신하지 않는다')
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('GPU 전체장은 Slurm 계산 노드에서만 실행한다')
    root = Path(root).resolve(); output = root/'resident-field.json'
    if output.exists(): raise FileExistsError('기존 실행 결과 보존; 복원도 새 실행 root에서만 한다')
    if shutil.disk_usage(root).free < 16*1024**3: raise OSError('전체장·유효 checkpoint 보존 여유16GiB 미달')
    started = time.monotonic(); deadline = started+seconds
    record = dict(schema='lumina-resident-fullsr-native90-v1', status='STARTED', phase='OPEN_INPUTS',
        final_goal='NOT_MET', accepted_physical_field=False, complete_original_grid=False,
        completed_actual_F_calls=0, started_actual_F_calls=0, material_updates=0, observer_calls=0,
        automatic_retries=0, checkpoint=None, checkpoints_written=0,
        limits=dict(GPUs=1, CPUs=1, host_device_GiB_each=16, internal_seconds=seconds),
        resume_checkpoint=None if resume_checkpoint is None else str(Path(resume_checkpoint).resolve()))
    def save():
        record.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(output, record)
    save()
    try:
        stage_sha = verify_stage(root)
        kernel_sha = sha(root/'chain_cuda.so'); geometry_sha = sha(root/'geometry.so')
        with ExitStack() as stack:
            config, source, geometry, boundary, plan_sha = open_field_inputs(stack, accepted_input, accepted_input_sha256)
            record['identity'] = dict(input_plan_sha256=plan_sha, stage_sha256=stage_sha,
                kernel_sha256=kernel_sha, geometry_sha256=geometry_sha,
                geometry_cache_sha256=sha(Path(cache)/'geometry.json'),
                input_J='ZERO_OWN_START' if input_j is None else dict(path=str(Path(input_j).resolve()), sha256=input_j_sha256),
                subdivisions=4)
            if initial_preparation is not None:
                record['identity']['initial_J_preparation'] = dict(
                    path=str(Path(initial_preparation).resolve()), sha256=initial_preparation_sha256)
            record['material_input_role'] = 'ORIGINAL70' if accepted_input is None else 'EXPLICIT_ACCEPTED_MATERIAL'
            query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'],
                check=True, capture_output=True, text=True, timeout=10)
            gpu = ChainGPU(root/'chain_cuda.so'); uuid = gpu.device_uuid()
            matches = [line.split(',') for line in query.stdout.splitlines() if line.split(',')[0].strip().lower() == uuid.lower()]
            if len(matches) != 1: raise ValueError('CUDA0/NVML GPU 신원 불일치')
            _, name, free = matches[0]; record.update(device_name=name.strip(), device_uuid=uuid)
            plan = CompiledDriverPlan(geometry, source, panels=1)
            record.update(source_identity=source.identity, original_layout=plan.layout())
            field = M.ZeroTrial(source) if input_j is None else open_transport_trial(
                source, input_j, input_j_sha256, initial_preparation=initial_preparation,
                initial_preparation_sha256=initial_preparation_sha256)
            if isinstance(field, InputField): stack.callback(field.close)
            record['input_field_role'] = field.identity
            factory = stack.enter_context(PhaseGeometry(root/'geometry.so', geometry, source))
            cached = stack.enter_context(CachedPhaseGeometry(cache, factory, plan))
            provider = stack.enter_context(CompiledInput(config['cpu_library'], geometry, source, field,
                slab_width=128, cache_rows=128, profiles=1024))
            if input_j is None: provider.bind_zero_trial()
            else: provider.bind_view()
            record['phase'] = 'PREPARE_DENSE'; save()
            remaining = deadline-time.monotonic()
            if remaining <= 120: raise TimeoutError('입력 준비 전 시간 여유 미달')
            record['dense_preparation'] = provider.prepare_dense(max_seconds=min(300., remaining-120.))
            runtime = stack.enter_context(Runtime(root/'chain_cuda.so', int(free)*1024**2, _functions={}, _abis=()))
            device = stack.enter_context(DeviceInput(runtime, provider))
            current = stack.enter_context(ResidentField(gpu, runtime, device, cached, plan, *boundary,
                kernel_identity=kernel_sha))
            if resume_checkpoint is not None: current.restore(resume_checkpoint)
            record.update(phase='RESIDENT_FULL_FIELD', started_actual_F_calls=1, field_identity=current.identity)
            save(); quarter = max(1, (len(current.sequence)+3)//4)
            while current.cursor < len(current.sequence):
                if deadline-time.monotonic() <= 120:
                    path = root/('checkpoint-'+str(current.cursor)); current.checkpoint(path)
                    record.update(checkpoint=str(path), checkpoints_written=record['checkpoints_written']+1,
                        status='TIME_LIMIT_VALID_CHECKPOINT_NOT_COMPLETE')
                    break
                before = current.cursor//quarter
                current.advance(deadline=deadline-60., max_actions=32)
                if (current.cursor < len(current.sequence) and current.cursor//quarter > before):
                    path = root/('checkpoint-'+str(current.cursor)); current.checkpoint(path)
                    record.update(checkpoint=str(path), checkpoints_written=record['checkpoints_written']+1)
                record.update(field_report=current.report(), memory=runtime.report()); save()
                print(json.dumps(dict(status='RESIDENT_PROGRESS', next_action=current.cursor,
                    actions=len(current.sequence), elapsed_seconds=record['elapsed_seconds'])), flush=True)
            record.update(field_report=current.report(), memory=runtime.report(),
                terminal_conversions=runtime.download(current.conversions, 3, dtype='u8').tolist())
            if current.cursor == len(current.sequence):
                record.update(write_completed_moments(root, current))
                record.update(status='COMPLETE_TRANSPORT_NOT_PHYSICAL_ACCEPTANCE',
                    complete_original_grid=True, completed_actual_F_calls=1)
            record['source_report'] = provider.report()
    except Exception as exc:
        record.update(status='ERROR_PARTIAL_PRESERVED', error=type(exc).__name__+': '+str(exc))
    finally:
        record['phase'] = 'FINISHED'; save()
    return 0 if record['complete_original_grid'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--cache', type=Path, required=True); parser.add_argument('--seconds', type=int, default=42600)
    parser.add_argument('--input-j', type=Path); parser.add_argument('--input-j-sha256')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--accepted-input', type=Path); parser.add_argument('--accepted-input-sha256')
    parser.add_argument('--initial-preparation', type=Path); parser.add_argument('--initial-preparation-sha256')
    raise SystemExit(run(**vars(parser.parse_args())))
