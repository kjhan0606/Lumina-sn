#!/usr/bin/env python3
"""완료 증거에 따라 다음 클러스터 작업을 딱 한 번 제출한다. 대기 루프는 없다."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def dispatch(config, digest, step):
    config = Path(config).resolve()
    if sha(config) != digest: raise ValueError('묶음 제출 구성 변경')
    c = json.loads(config.read_text()); control = config.parent
    if c['schema'] != 'lumina-accepted-bundle-dispatch-v1' or step not in ('field', 'response'):
        raise ValueError('묶음 제출 범위 오류')
    for name, expected in c['control_sha256'].items():
        if sha(control/name) != expected: raise ValueError('봉인 제출 코드 변경')
    result_path = Path(c['coefficient_result']); coefficient = json.loads(result_path.read_text())
    if (coefficient['state'] != 'COMPLETE_NEW_ACCEPTED_COEFFICIENTS_AND_BOUNDARY' or
            coefficient['source_commit'] != c['coefficient_commit']):
        raise ValueError('새 계수 단회가 완성되지 않음')
    p = coefficient['preparation']; prep = Path(p['path'])
    if prep != result_path.parent/'boundary/preparation.json' or sha(prep) != p['sha256']:
        raise ValueError('새 계수 경계 기록 변경')
    gpu = Path(c['gpu_root']); prepared = gpu/'prepared.json'
    if sha(prepared) != c['gpu_prepared_sha256']: raise ValueError('준비 GPU 실행 신원 변경')
    marker = control/(step+'-submission.json')
    if marker.exists(): raise FileExistsError('제출 시도 이력 있음; 중복 제출/자동 재시도 금지')
    common = ['sbatch', '--parsable', '--no-requeue']
    if step == 'field':
        maximum_F = c.get('limits', {}).get('maximum_F', 2)
        if type(maximum_F) is not int or not 1 <= maximum_F <= 2:
            raise ValueError('묶음 F 횟수는1~2회만 허용')
        options = ['--max-sweeps', str(maximum_F)]
        initial = c.get('initial_field')
        if initial is not None:
            for key, flag in (('J', '--input-j'), ('preparation', '--initial-preparation')):
                entry = initial.get(key, {})
                if (not isinstance(entry.get('path'), str) or not entry['path'] or
                        not isinstance(entry.get('sha256'), str) or
                        not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])):
                    raise ValueError('이전 물질 초기장/준비 기록의 명시 신원 필요')
                options += [flag, entry['path'], flag+'-sha256', entry['sha256']]
        command = common+['--output='+str(gpu/'slurm-%j.out'), '--error='+str(gpu/'slurm-%j.err'),
            str(control/'submit_lumina_accepted_feedback.slurm'), str(gpu), str(prep), p['sha256'], str(config), digest,
            *options]
        host = 'syntax'
    else:
        feedback = gpu/'feedback/feedback.json'; f = json.loads(feedback.read_text())
        last = f.get('last_complete')
        if not last or f['accepted_preparation_sha256'] != p['sha256']:
            raise ValueError('새 완성 자체장 없음; 물질응답 제출 안 함')
        entry = f['sweeps'][last['sweep']-1]
        if entry['state'] != 'COMPLETE_F_NOT_CONVERGENCE': raise ValueError('부분장 소비 금지')
        parent = Path(c['response_parent'])
        parent.mkdir(exist_ok=False)
        command = common+['--output='+str(parent/'slurm-%j.out'), '--error='+str(parent/'slurm-%j.err'),
            str(control/'submit_lumina_new_field_response.slurm'), str(parent), c['response_commit'],
            '--preparation', str(prep), '--preparation-sha', p['sha256'],
            '--feedback', str(feedback), '--feedback-sha', sha(feedback)]
        host = 'grammar'
    record = dict(state='SUBMISSION_ATTEMPT_STARTED', step=step, host=host, command=command,
                  automatic_retries=0, config_sha256=digest)
    with marker.open('x') as stream: json.dump(record, stream, indent=2)
    try:
        completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            shlex.join(command)], capture_output=True, text=True, timeout=45)
        record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?\s*', completed.stdout):
            raise RuntimeError('제출 응답 확인 실패; 자동 재시도하지 않음')
        record.update(state='SUBMITTED_ONCE', job_id=int(completed.stdout.strip().split(';')[0]))
    except Exception as error:
        record.update(state='SUBMISSION_UNCONFIRMED_NO_RETRY', error=type(error).__name__+': '+str(error))
        raise
    finally:
        marker.write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path); p.add_argument('digest'); p.add_argument('step', choices=('field', 'response'))
    print(json.dumps(dispatch(**vars(p.parse_args())), indent=2))
