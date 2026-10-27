#!/usr/bin/env python3
"""선택 실행기의 정확한 로컬 import/include 폐쇄만 새 실행 root에 고정한다."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


def stage(project, root, include_comparison=False, chain_gpu=False, resident=False):
    project, root = Path(project).resolve(), Path(root).resolve()
    if root.exists(): raise FileExistsError('새 계산 root만 허용')
    if resident: chain_gpu = True
    pending = [project/'scripts/lumina_fullsr_sc_actual.py', project/'scripts/lumina_fullsr_sc.py']
    if resident:
        pending += [project/'scripts/lumina_resident_actual.py', project/'scripts/lumina_phase_cache.py',
                    project/'scripts/lumina_resident_run.py', project/'scripts/lumina_resident_feedback.py']
    if chain_gpu: pending += [project/'scripts/run_lumina_fullsr_chain_gpu.py']
    if include_comparison: pending.append(project/'scripts/compare_lumina_fullsr_sc.py')
    paths = set()
    while pending:
        path = pending.pop()
        if path in paths: continue
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                name = module.split('.')[0]
                candidate = project/'scripts'/(name+'.py')
                if candidate.is_file() and candidate not in paths: pending.append(candidate)
    for name in ('cmf_fullsr_sc.cpp', 'cmf_fullsr_sc.h', 'cmf_nodal_input.h', 'cmf_linear_native.h',
                 'cmf_linear_affine.hpp', 'cmf_ordered_maps.hpp', 'cmf_linear_provider.hpp'):
        paths.add(project/'src'/name)
    paths.add(project/'scripts/run_lumina_fullsr_sc.sh')
    if chain_gpu:
        paths.add(project/'scripts/run_lumina_fullsr_chain_gpu.sh')
        pending = [project/'src'/name for name in ('cmf_fullsr_chain.cu', 'cmf_fullsr_frozen.cpp', 'cmf_gpu_runtime.cu', 'cmf_linear_cuda_aux.cu')]
        if resident:
            paths.add(project/'scripts/run_lumina_resident_field.sh')
            pending += [project/'src'/name for name in ('cmf_frequency_history.c', 'cmf_nodal_input.c', 'cmf_nodal_history.c', 'cmf_nodal_driver.c')]
        while pending:
            path = pending.pop()
            if path in paths: continue
            paths.add(path)
            for name in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), flags=re.M):
                dependency = (path.parent/name).resolve()
                if not dependency.is_relative_to(project/'src'): raise ValueError('원천 의존 경로 범위 밖')
                if dependency not in paths: pending.append(dependency)
    if include_comparison:
        paths.add(project/'scripts/run_lumina_fullsr_sc_compare.sh')
        paths.add(project/'scripts/run_lumina_fullsr_sc_pair.sh')
    root.mkdir(); manifest = {}
    for path in sorted(paths):
        relative = path.relative_to(project); target = root/'code'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    with (root/'stage.json').open('x') as stream:
        json.dump(dict(schema='lumina-fullsr-sc-stage-v1', files=manifest), stream, indent=2)
    print(json.dumps(dict(root=str(root), files=len(paths),
                         stage_sha256=hashlib.sha256((root/'stage.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('project'); p.add_argument('root')
    p.add_argument('--include-comparison', action='store_true')
    p.add_argument('--chain-gpu', action='store_true')
    p.add_argument('--resident', action='store_true')
    a = p.parse_args(); stage(**vars(a))
