#!/usr/bin/env python3
"""같은 F(Jref)에서 셀37만 native 안/바깥 계수로 바꾸는 반사실험.

이는 native 공간보간이나 수용 가능한 물리 모델이 아니라 원인 민감도 시험이다.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time

import numpy as np

from diagnose_lumina_outer_fixed_point import (PROJECT, CAMPAIGN, INPUT, LIBRARY,
                                                field_residual)
from lumina_outer_c_backend import COuterTransport
from lumina_outer_prepared import PreparedInputs
from prepare_lumina_outer_input import read_auxiliary, sha


PAIR = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_pair_20261022-jXG2f6')
SOURCE = Path('/gpfs/kjhan/cmfgen_runs/lumina_bridge_field0_last_20261020-Hklgin/run/boundary_sources')
SHELL = 37
REFERENCE_NODES = (("outer", 51), ("inner", 52))


def save_record(path, record):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_array(path, values):
    temporary = path.with_suffix('.npy.partial')
    with temporary.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def digest_row(row):
    return hashlib.sha256(np.ascontiguousarray(row).view('u1')).hexdigest()


def run(root):
    root = root.resolve()
    if (socket.gethostname().lower() != 'lageunha' or Path.cwd().resolve() != PROJECT or
            not root.is_dir() or list(root.iterdir())):
        raise ValueError('Lageunha의 확인된 프로젝트와 빈 새 root 필요')
    started = time.monotonic()
    pair = json.loads((PAIR / 'execution.json').read_text())
    source_report = json.loads((SOURCE / 'sources.json').read_text())
    preparation = json.loads((INPUT / 'boundary/preparation.json').read_text())
    cell_report = json.loads((INPUT / 'cells/cells.json').read_text())
    origin = json.loads((CAMPAIGN / 'execution.json').read_text())
    wrappers = ('lumina_outer_c_backend.py', 'lumina_outer_prepared.py',
                'lumina_outer_atmosphere.py', 'lumina_outer_transport.py',
                'run_cmfgen_formal_primary.py')
    for name in wrappers:
        expected = [value for path, value in origin['source_sha256'].items()
                    if Path(path).name == name]
        if len(expected) != 1 or sha(Path(__file__).with_name(name)) != expected[0]:
            raise ValueError('기존 F(Jref)와 Python 수송 코드가 다름: ' + name)
    if (pair['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or pair['completed_F_calls'] != 2 or
            pair['material_record'] != 70 or pair['reference_record'] != 71 or
            sha(LIBRARY) != pair['input_sha256'][str(LIBRARY)] or
            sha(PAIR / 'Jref.npy') != pair['Jref']['sha256'] or
            sha(PAIR / 'Fref.npy') != pair['F_reference']['sha256'] or
            sha(INPUT / 'boundary/preparation.json') != pair['input_sha256'][str(INPUT / 'boundary/preparation.json')] or
            sha(INPUT / 'cells/cells.json') != pair['input_sha256'][str(INPUT / 'cells/cells.json')] or
            source_report['output_sha256']['sources.bin'] != preparation['reused_source_binary_sha256'] or
            sha(SOURCE / 'sources.bin') != preparation['reused_source_binary_sha256'] or
            cell_report['input_sha256'][str(INPUT / 'boundary/preparation.json')] !=
                    sha(INPUT / 'boundary/preparation.json')):
        raise ValueError('봉인 Jref/Fref/계수/바이너리 신원 불일치')
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise ValueError('두 F 출력 및 여유 공간 부족')
    record = dict(schema='lumina-shell37-endpoint-sensitivity-v1',
                  status='PREPARING', endpoint_results={}, F_calls_completed=0,
                  physical_equivalence='NOT_JUDGED', final_goal='NOT_MET',
                  resources=dict(CPUs=16, GPU=0, memory_GiB=64,
                                 wall_seconds_max=3600, F_calls_max=2, automatic_retry=0),
                  baseline_pair_execution_sha256=sha(PAIR / 'execution.json'),
                  input_sha256={str(path): sha(path) for path in
                                (LIBRARY, INPUT / 'boundary/preparation.json',
                                 INPUT / 'cells/cells.json', SOURCE / 'sources.json',
                                 PAIR / 'Jref.npy', PAIR / 'Fref.npy')},
                  source_sha256={str(path): sha(path) for path in
                                 (Path(__file__), Path(__file__).with_name('diagnose_lumina_outer_fixed_point.py'),
                                  Path(__file__).with_name('prepare_lumina_outer_input.py'))})
    save_record(root / 'execution.json', record)
    snapshot = root / 'snapshot'
    snapshot.mkdir()
    source_files = ([Path(__file__), Path(__file__).with_name('diagnose_lumina_outer_fixed_point.py'),
                     Path(__file__).with_name('prepare_lumina_outer_input.py')] +
                    [Path(__file__).with_name(name) for name in wrappers])
    for path in source_files:
        shutil.copy2(path, snapshot / path.name)
    with PreparedInputs(LIBRARY, INPUT / 'boundary', INPUT / 'cells') as loaded:
        if (loaded.plan.n != 89 or len(loaded.nu) != 1115020 or
                tuple(loaded.plan.g.reference_indices) != (0, 14)):
            raise ValueError('기존 F(Jref) 기하·ν 불일치')
        nu = loaded.nu
        chi, es, eta = read_auxiliary(SOURCE / 'sources.bin', source_report, nu, [51, 52])
        jref = np.load(PAIR / 'Jref.npy', mmap_mode='r', allow_pickle=False)
        baseline = np.load(PAIR / 'Fref.npy', mmap_mode='r', allow_pickle=False)
        if jref.shape != (91, len(nu)) or baseline.shape != jref.shape:
            raise ValueError('같은 참조 장의 배열 차원 불일치')
        for at, (label, node) in enumerate(REFERENCE_NODES):
            if time.monotonic() - started >= 3000:
                raise TimeoutError('다음 F를 위한 실행 잔여 시간 부족')
            maps = []
            try:
                for path in (INPUT / 'cells/cell_chi.npy', INPUT / 'cells/cell_eta.npy'):
                    maps.append(np.load(path, mmap_mode='c', allow_pickle=False))
                view = cell_report['es_view']
                maps.append(np.memmap(view['path'], mode='c', dtype='<f8',
                    offset=view['offset_bytes'], shape=(89, len(nu))))
                if any(array.shape != (89, len(nu)) or not array.flags.c_contiguous for array in maps):
                    raise ValueError('private 셀 계수 배열 범위 오류')
                for array, replacement in zip(maps, (chi[at], eta[at], es[at])):
                    array[SHELL] = replacement
                record['status'] = f'F_{label.upper()}_PENDING'
                record.setdefault('changed_rows', {})[label] = dict(native_index=node,
                    shell=SHELL, chi_sha256=digest_row(chi[at]),
                    eta_sha256=digest_row(eta[at]), es_sha256=digest_row(es[at]),
                    unchanged_shells=88, original_inputs_copy_on_write=True)
                save_record(root / 'execution.json', record)
                with COuterTransport(LIBRARY, loaded.plan, nu, maps[0], maps[2], maps[1],
                    *loaded.nodes, *loaded.boundary, work_budget_bytes=16 * 1024**3,
                    weighted_workers=16, moment_count=1, approved_cpu_limit=16) as operator:
                    result, info = operator.apply_j(jref)
                saved = save_array(root / f'F-{label}.npy', result)
                record['F_calls_completed'] += 1
                record['endpoint_results'][label] = dict(field=saved,
                    actual_ray_threads=info['actual_ray_threads'],
                    reference_residual=field_residual(jref, result, jref, nu),
                    difference_from_baseline=field_residual(baseline, result, jref, nu))
                record['status'] = f'F_{label.upper()}_SAVED'
                save_record(root / 'execution.json', record)
            finally:
                for array in maps:
                    array._mmap.close()
        for name in ('cell_chi.npy', 'cell_eta.npy'):
            if sha(INPUT / 'cells' / name) != cell_report['output_sha256'][name]:
                raise ValueError('원본 셀 계수 변경')
        if sha(SOURCE / 'sources.bin') != source_report['output_sha256']['sources.bin']:
            raise ValueError('원본 native 계수 변경')
    record['status'] = 'COUNTERFACTUAL_DIAGNOSTIC_COMPLETE'
    record['elapsed_seconds'] = time.monotonic() - started
    record['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_record(root / 'execution.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out_root)
    print(json.dumps(dict(root=str(args.out_root), status=result['status'],
                          F_calls_completed=result['F_calls_completed'],
                          elapsed_seconds=result['elapsed_seconds'],
                          peak_rss_KiB=result['peak_rss_KiB']),
                     ensure_ascii=False, allow_nan=False), flush=True)
