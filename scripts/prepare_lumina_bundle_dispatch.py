#!/usr/bin/env python3
"""이미 승인된 묶음의 단회 제출 코드와 인수를 저장한다. 작업 제출은 하지 않는다."""
import argparse
import json
from pathlib import Path
import subprocess

from lumina_bundle_dispatch import sha


def prepare(project, out, commit, gpu_root, coefficient_result, response_parent):
    project, out, gpu_root = map(lambda p: Path(p).resolve(), (project, out, gpu_root))
    if Path(response_parent).exists(): raise FileExistsError('기존 후속 물질응답 보존')
    out.mkdir(exist_ok=False)
    names = ('lumina_bundle_dispatch.py', 'submit_lumina_accepted_feedback.slurm', 'submit_lumina_new_field_response.slurm')
    for name in names:
        content = subprocess.run(['git', '-C', str(project), 'show', commit+':scripts/'+name],
            check=True, capture_output=True, timeout=10).stdout
        (out/name).write_bytes(content)
    c = dict(schema='lumina-accepted-bundle-dispatch-v1', response_commit=commit,
        coefficient_commit='9fe7074bf228b50759c3c95c0c605df972632406',
        coefficient_result=str(Path(coefficient_result).resolve()), gpu_root=str(gpu_root),
        gpu_prepared_sha256=sha(gpu_root/'prepared.json'), response_parent=str(Path(response_parent).resolve()),
        control_sha256={name: sha(out/name) for name in names},
        limits=dict(GPU_allocations=1, GPU_hours=12, maximum_F=2,
                    response_allocations=1, response_CPU=16, response_GiB=32, response_hours=4,
                    automatic_retries=0))
    config = out/'config.json'; config.write_text(json.dumps(c, indent=2)+'\n')
    return dict(config=str(config), sha256=sha(config), jobs_submitted=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out', 'gpu-root', 'coefficient-result', 'response-parent'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
