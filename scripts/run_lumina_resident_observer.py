#!/usr/bin/env python3
"""같은 물질과 완성 자체장의 관측자 스펙트럼을 별도 수송으로 계산한다."""
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
from lumina_fullsr_sc_actual import InputField, publish, verify_stage
from lumina_gpu_driver import Runtime
from lumina_native_columns import sha
from lumina_observer_spectrum import ObserverDriverPlan
from lumina_phase_geometry import PhaseGeometry
from lumina_resident_field import ResidentField
from lumina_resident_observer import ResidentObserver
from lumina_resident_run import open_field_inputs


def run(root, *, accepted_input, accepted_input_sha256, input_j, input_j_sha256,
        seconds, disk_subdivisions=1, resume_checkpoint=None):
    if (not os.environ.get('SLURM_JOB_ID') or type(seconds) is not int or
            not 600 <= seconds <= 85800):
        raise ValueError('명시된 Slurm GPU 할당과 내부 시간 필요')
    root = Path(root).resolve(); output = root/'observer-result.json'
    if output.exists(): raise FileExistsError('기존 스펙트럼 결과 보존')
    if shutil.disk_usage(root).free < 16*1024**3: raise OSError('관측 저장 공간 부족')
    started = time.monotonic(); deadline = started+seconds
    record = dict(status='STARTED', final_goal='NOT_MET', accepted_physical_spectrum=False,
                  scattering_convergence_certified=False, material_updates=0, actual_observer_calls=0,
                  scope='완성 입력 J와 고정 수용 물질의 관측자 수송; 공동수렴 아님')
    def save():
        record.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(output, record)
    save()
    try:
        stage = verify_stage(root); kernel_sha = sha(root/'chain_cuda.so')
        with ExitStack() as stack:
            config, source, geometry, boundary, prep_sha = open_field_inputs(
                stack, accepted_input, accepted_input_sha256)
            input_field = InputField(source, input_j, input_j_sha256)
            stack.callback(input_field.close)
            query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free',
                '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
            gpu = ChainGPU(root/'chain_cuda.so'); uuid = gpu.device_uuid()
            matches = [line.split(',') for line in query.stdout.splitlines()
                       if line.split(',')[0].strip().lower() == uuid.lower()]
            if len(matches) != 1: raise ValueError('CUDA와 장치 신원 불일치')
            _, device_name, free = matches[0]
            plan = ObserverDriverPlan(geometry, source, disk_subdivisions=disk_subdivisions)
            factory = stack.enter_context(PhaseGeometry(root/'geometry.so', geometry, source))
            provider = stack.enter_context(CompiledInput(config['cpu_library'], geometry, source, input_field,
                slab_width=128, cache_rows=128, profiles=1024))
            provider.bind_view()
            remaining = deadline-time.monotonic()
            if remaining <= 180: raise TimeoutError('입력 준비 시간 부족')
            dense = provider.prepare_dense(max_seconds=min(300., remaining-180.))
            runtime = stack.enter_context(Runtime(root/'chain_cuda.so', int(free)*1024**2, _functions={}, _abis=()))
            device = stack.enter_context(DeviceInput(runtime, provider))
            field = stack.enter_context(ResidentField(gpu, runtime, device, factory, plan, *boundary,
                kernel_identity=kernel_sha))
            observer = ResidentObserver(field)
            if resume_checkpoint is not None: observer.restore(resume_checkpoint)
            record.update(stage_sha256=stage, preparation_sha256=prep_sha, source_identity=source.identity,
                input_J=dict(path=str(Path(input_j).resolve()), sha256=input_j_sha256),
                device_name=device_name.strip(), device_uuid=uuid, layout=plan.layout(),
                dense_preparation=dense, status='RUNNING')
            save(); quarter = max(1, (len(plan.paths)+3)//4)
            while observer.spectrum.next_path < len(plan.paths):
                if deadline-time.monotonic() <= 180:
                    path = root/('checkpoint-'+str(observer.spectrum.next_path))
                    observer.checkpoint(path)
                    record.update(status='TIME_LIMIT_CHECKPOINT_SAVED', checkpoint=str(path)); break
                previous = observer.spectrum.next_path
                observer.advance_path(deadline=deadline-60.)
                count = observer.spectrum.next_path
                if count < len(plan.paths) and count//quarter > previous//quarter:
                    path = root/('checkpoint-'+str(count)); observer.checkpoint(path)
                    record['checkpoint'] = str(path)
                record.update(completed_paths=count, field_report=field.report(), memory=runtime.report())
                save()
            if observer.spectrum.next_path == len(plan.paths):
                result = observer.finish()
                array_names = ('frequency_Hz', 'wavelength_A', 'luminosity_erg_s_Hz', 'luminosity_erg_s_A')
                target = root/'observer-spectrum.npz'
                with target.open('xb') as stream: np.savez(stream, **{n: result[n] for n in array_names})
                record.update(status='COMPLETE_FROZEN_OBSERVER_NOT_CONVERGENCE', actual_observer_calls=1,
                    spectrum=dict(path=str(target), sha256=sha(target)),
                    spectrum_metadata={k: v for k, v in result.items() if k not in array_names})
    except Exception as error:
        record.update(status='INCOMPLETE_PRESERVED', error=type(error).__name__+': '+str(error))
    finally:
        save()
    return 0 if record['actual_observer_calls'] == 1 else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    for name in ('accepted-input', 'input-j'):
        p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--disk-subdivisions', type=int, default=1)
    p.add_argument('--resume-checkpoint', type=Path)
    raise SystemExit(run(**vars(p.parse_args())))
