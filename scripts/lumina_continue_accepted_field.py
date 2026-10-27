#!/usr/bin/env python3
"""기존 작업 종료 후 미완성 수송만 보존된 체크포인트에서 이어간다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lumina_fullsr_sc_actual import completed_transport, publish
from lumina_native_columns import sha


def read_complete(path):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record if completed_transport(record, 1115020) else None


def run(parent, staged, preparation, prep_sha, cache, seconds):
    parent, staged = Path(parent).resolve(), Path(staged).resolve()
    preparation, cache = Path(preparation).resolve(), Path(cache).resolve()
    if not os.environ.get('SLURM_JOB_ID') or sha(preparation) != prep_sha:
        raise ValueError('Slurm 또는 수용 물질 신원 오류')
    previous = staged/'feedback'
    old = json.loads((previous/'feedback.json').read_text())
    if old['accepted_preparation_sha256'] != prep_sha:
        raise ValueError('기존 작업의 수용 물질이 다름')
    result = parent/'continuation.json'
    if result.exists():
        raise FileExistsError('기존 이어계산 결과 보존')
    state = dict(state='STARTED', original_feedback=str(previous/'feedback.json'),
                 final_goal='NOT_MET', scattering_convergence_certified=False)
    publish(result, state)
    start = time.monotonic()
    try:
        second = read_complete(previous/'F-2/resident-field.json')
        if second is not None:
            state.update(state='ORIGINAL_F2_ALREADY_COMPLETE', original_result=str(previous/'F-2/resident-field.json'))
            return
        first = read_complete(previous/'F-1/resident-field.json')

        def execute(label, old_child, prior):
            folder = parent/label
            folder.mkdir()
            for name in ('code', 'stage.json', 'chain_cuda.so', 'geometry.so'):
                (folder/name).symlink_to(staged/name, target_is_directory=(staged/name).is_dir())
            remaining = min(85800, int(seconds-(time.monotonic()-start)-300))
            if remaining < 240:
                raise TimeoutError('이어계산 시간 부족')
            command = [sys.executable, str(folder/'code/scripts/lumina_resident_run.py'), str(folder),
                       '--cache', str(cache), '--accepted-input', str(preparation),
                       '--accepted-input-sha256', prep_sha, '--seconds', str(remaining)]
            if prior is not None:
                command += ['--input-j', prior['J']['path'], '--input-j-sha256', prior['J']['sha256']]
            prior_report = old_child/'resident-field.json'
            if prior_report.exists():
                prior_record = json.loads(prior_report.read_text())
                checkpoint = prior_record.get('checkpoint')
                if checkpoint:
                    command += ['--resume-checkpoint', checkpoint]
            with (folder/'stdout.log').open('x') as out, (folder/'stderr.log').open('x') as err:
                code = subprocess.run(command, stdout=out, stderr=err).returncode
            record = read_complete(folder/'resident-field.json')
            if code != 0 or record is None:
                raise RuntimeError(label+' 미완료: 부분 결과 보존')
            return record

        if first is None:
            first = execute('recovered-F1', previous/'F-1', None)
        feedback = parent/'feedback'
        feedback.mkdir()
        # 새 목록의 첫 항목이 실제 물리 순서의 두 번째 수송이다.
        second = execute('feedback/F-1', previous/'F-2', first)
        child = feedback/'F-1'
        entry = dict(directory=str(child), state='COMPLETE_F_NOT_CONVERGENCE',
                     actual_physical_sweep=2, input_J=first['J'],
                     result_sha256=sha(child/'resident-field.json'))
        info = dict(schema='lumina-resident-fixed-material-feedback-v1',
                    status='FINITE_FEEDBACK_RESPONSE_NOT_CONVERGENCE',
                    source_identity=second['source_identity'], accepted_preparation_sha256=prep_sha,
                    sweeps=[entry], last_complete=dict(J=second['J'], moments=second['moments'], sweep=1),
                    original_first_J=first['J'], scattering_convergence_certified=False, final_goal='NOT_MET')
        publish(feedback/'feedback.json', info)
        state.update(state='SECOND_SWEEP_COMPLETE', feedback=str(feedback/'feedback.json'),
                     feedback_sha256=sha(feedback/'feedback.json'), first_J=first['J'], second_J=second['J'])
    except Exception as error:
        state.update(state='INCOMPLETE_PRESERVED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        publish(result, state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('parent', 'staged', 'preparation', 'cache'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--prep-sha', required=True)
    p.add_argument('--seconds', type=int, default=85800)
    run(**vars(p.parse_args()))
