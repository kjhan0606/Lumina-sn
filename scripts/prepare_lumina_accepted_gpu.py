#!/usr/bin/env python3
"""새 GPU 실행 소스와 기존 검증 바이너리만 봉인한다. GPU 할당/실행은 없다."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from lumina_native_columns import sha
from stage_lumina_fullsr_sc import stage


def prepare(project, out, commit):
    project, out = Path(project).resolve(), Path(out).resolve()
    previous = Path('/gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run')
    cache = Path('/gpfs/kjhan/cmfgen_runs/lumina_phase_prepare_20261026-8fMV0m/run/cache')
    expected = {'chain_cuda.so': '531ded6df7c3f72ec2cd17688d6245d5fd291ddfcdfb1705fdd6a435664d4956',
                'geometry.so': '3cddc57f07232c92af77d7647f137d6f0f63904024959b34345e487da9bec2d0'}
    cache_sha = 'a7dc3241e60d7ea3a22e86d9ee66cc7acfb562092201894d64be80451f9ae5cc'
    if any(sha(previous/k) != h for k, h in expected.items()) or sha(cache/'geometry.json') != cache_sha:
        raise ValueError('재사용 바이너리/기하 cache 기록 변경')
    stage(project, out, resident=True)
    new = json.loads((out/'stage.json').read_text())
    old = json.loads((previous/'stage.json').read_text())
    for name, digest in new['files'].items():
        content = subprocess.run(['git', '-C', str(project), 'show', commit+':'+name],
            check=True, capture_output=True, timeout=10).stdout
        import hashlib
        if hashlib.sha256(content).hexdigest() != digest: raise ValueError('봉인 commit과 실행 소스 불일치: '+name)
        if name.startswith('src/') and name in old['files'] and digest != old['files'][name]:
            raise ValueError('재사용 바이너리의 기존 원천 소스 변경: '+name)
    for name in expected: shutil.copy2(previous/name, out/name)
    record = dict(state='PREPARED_NOT_SUBMITTED', source_commit=commit,
        stage_sha256=sha(out/'stage.json'), binary_sha256=expected,
        cache=str(cache), cache_manifest_sha256=cache_sha,
        resources='GPU1/CPU1/host16GiB/12h/max2F/retry0',
        new_actual_F_calls=0, requires_complete_new_coefficient_preparation=True)
    (out/'prepared.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
