#!/usr/bin/env python3
"""이어계산이 끝난 후 미사용 물질응답 단회만 제출한다."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def may_submit(marker):
    if not marker.exists():
        return True
    old = json.loads(marker.read_text())
    # 이름 해석에 실패한 경우 SSH 연결 자체가 없었다. 그 외 불확실성은 중복을 피한다.
    return (old.get('state') == 'SUBMISSION_UNCONFIRMED_NO_RETRY' and
            old.get('returncode') == 255 and 'Could not resolve hostname' in old.get('stderr', ''))


def dispatch(parent, control, address, response_commit=None):
    parent, control = Path(parent).resolve(), Path(control).resolve()
    marker = parent/'continued-response-submission.json'
    if marker.exists():
        raise FileExistsError('제출 시도 이력 보존')
    record = {'state': 'CHECKING', 'automatic_retries': 0}
    with marker.open('x') as stream:
        json.dump(record, stream)
    try:
        old_marker = control/'response-submission.json'
        if not may_submit(old_marker):
            record['state'] = 'ORIGINAL_RESPONSE_SUBMITTED_OR_UNCERTAIN_NO_DUPLICATE'
            return record
        config_path = control/'config.json'; config = json.loads(config_path.read_text())
        expected = '53783a45a1a28cb9ba5694a211e6437897de8e0c500d0a06b79223011eb73fda'
        if sha(config_path) != expected:
            raise ValueError('기존 제출 구성 변경')
        script = control/'submit_lumina_new_field_response.slurm'
        if sha(script) != config['control_sha256'][script.name]:
            raise ValueError('물질응답 제출 파일 변경')
        continuation = json.loads((parent/'continuation.json').read_text())
        if continuation['state'] == 'SECOND_SWEEP_COMPLETE':
            feedback = Path(continuation['feedback'])
            if sha(feedback) != continuation['feedback_sha256']:
                raise ValueError('이어계산 완료장 기록 변경')
        elif continuation['state'] == 'ORIGINAL_F2_ALREADY_COMPLETE':
            feedback = Path(config['gpu_root'])/'feedback/feedback.json'
        else:
            raise ValueError('완성된 두 번째 장 없음')
        f = json.loads(feedback.read_text())
        last = f.get('last_complete')
        if not last or f['sweeps'][last['sweep']-1]['state'] != 'COMPLETE_F_NOT_CONVERGENCE':
            raise ValueError('부분장 사용 금지')
        prep = Path(config['coefficient_result']).parent/'boundary/preparation.json'
        prep_sha = '4707762a819fee7f013fb8655ad5d6a738c9f64cfd1ddc897e5138caa9478887'
        if sha(prep) != prep_sha or f['accepted_preparation_sha256'] != prep_sha:
            raise ValueError('물질·복사장 신원 불일치')
        selected_commit = config['response_commit'] if response_commit is None else response_commit
        if response_commit is not None and not re.fullmatch(r'[0-9a-f]{40}', response_commit):
            raise ValueError('응답 코드 커밋 신원 오류')
        output = parent/'response'
        output.mkdir()
        command = ['sbatch', '--parsable', '--no-requeue', '--output='+str(output/'slurm-%j.out'),
                   '--error='+str(output/'slurm-%j.err'), str(script), str(output), selected_commit,
                   '--preparation', str(prep), '--preparation-sha', prep_sha,
                   '--feedback', str(feedback), '--feedback-sha', sha(feedback)]
        record.update(state='SUBMISSION_ATTEMPT_STARTED', feedback=str(feedback), command=command)
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                               '-o', 'HostKeyAlias=grammar', address, shlex.join(command)],
                              capture_output=True, text=True, timeout=45)
        record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?\s*', proc.stdout):
            raise RuntimeError('제출 결과 불확실; 자동 재시도하지 않음')
        record.update(state='SUBMITTED_ONCE', job_id=int(proc.stdout.strip().split(';')[0]))
    except Exception as error:
        record.update(state='INCOMPLETE_OR_UNCONFIRMED_NO_RETRY', error=type(error).__name__+': '+str(error))
        raise
    finally:
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--address', required=True)
    p.add_argument('--response-commit')
    print(json.dumps(dispatch(**vars(p.parse_args())), ensure_ascii=False))
