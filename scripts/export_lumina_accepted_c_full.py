#!/usr/bin/env python3
"""봉인된 전하·SE 후보를 새 계수용 물질과 실제 C full 인구로 전달한다.

수송·율 적분·물질 갱신은 실행하지 않는다. 원본 입력과 기존 출력은 보존한다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

import cmfgen_model_continuum as B
from prepare_lumina_accepted_material import prepare


def run(args):
    if os.environ.get('SLURM_CPUS_PER_TASK') != '1' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('grammar-debug CPU1 Slurm 실행 필요')
    root = args.out.resolve()
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError('기존 결과 보존을 위해 빈 출력 root 필요')
    started = time.monotonic()
    paths = {name: getattr(args, name).resolve() for name in
             ('result', 'seed', 'model_run', 'projection', 'parameters', 'topion', 'ionref')}
    expected = dict(result=args.result_sha, projection=args.projection_sha,
                    parameters=args.parameters_sha, topion=args.topion_sha, ionref=args.ionref_sha)
    if any(B.sha(paths[name]) != digest for name, digest in expected.items()):
        raise ValueError('봉인된 결과/C importer/원자 자료 SHA 불일치')
    inputs = {str(paths[name]): digest for name, digest in expected.items()}
    inputs[str(Path(__file__).resolve())] = B.sha(Path(__file__))
    material = root/'material'
    export = prepare(paths['result'], args.result_sha, paths['seed'], paths['model_run'], material)
    if export.get('thermal_energy_policy') != 'native_instantaneous_scenario':
        raise ValueError('계승한 열정책 불일치')

    for name, key in (('topion', 'LUMINA_TOPION_LEVELS_FILE'),
                      ('ionref', 'LUMINA_IONIZATION_REFERENCE_FILE')):
        os.environ[key] = str(paths[name])
    os.environ.update(LUMINA_ARTIS_PARITY='0', LUMINA_NLTE_BF_COLLISION='1')
    full = material/'C-imported-full.bin'
    command = [str(paths['projection']), str(material/'atomic'), '90',
               str(material/'native_material_nodes.bin'), str(full),
               str(material/'occupation.bin'), str(paths['parameters'])]
    with (root/'C-export.stdout').open('x') as stdout, (root/'C-export.stderr').open('x') as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120, check=False)
    (root/'C-export.exit-code').write_text(str(completed.returncode)+'\n')
    if completed.returncode:
        raise RuntimeError('C full 인구 내보내기 실패; 부분 결과 보존')

    report = json.loads(paths['result'].read_text())
    selected = report['selected_response']
    response = Path(selected['root'])/'response.npz'
    if B.sha(response) != selected['response_sha256']:
        raise ValueError('C full 비교 직전 선택 물질 응답 변경')
    inputs[str(response)] = selected['response_sha256']
    with np.load(response, allow_pickle=False) as values:
        columns = values['columns']
        population = values['population']
        fraction = values['fraction']
    index = {tuple(map(int, row)): i for i, row in enumerate(columns)}
    with (material/'atomic/levels.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    mapping = np.array([index[int(row['atomic_number']), int(row['ion_number']),
                              int(row['super_level'])] for row in rows])
    comparison = B.consume_c_full_population(full,
                    dict(full=population[mapping]*fraction))
    if any(B.sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('실행 중 원 입력 변경')
    result = dict(state='EXPORTED_NATIVE_ACCEPTED_MATERIAL_AND_ACTUAL_C_FULL_NOT_TRANSPORT',
        accepted_manifest=dict(path=str(material/'accepted-material.json'),
                               sha256=B.sha(material/'accepted-material.json')),
        C_full_population=dict(path=str(full), sha256=B.sha(full)),
        command=command, returncode=completed.returncode, C_comparison=comparison,
        checks=export['checks'], thermal_energy_policy=export['thermal_energy_policy'],
        input_sha256=inputs, new_SE_solves=0, new_integrals=0, new_transport=0,
        final_goal='NOT_MET', wall_seconds=time.monotonic()-started)
    with (root/'result.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'seed', 'model-run', 'projection', 'parameters',
                 'topion', 'ionref', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('result-sha', 'projection-sha', 'parameters-sha', 'topion-sha',
                 'ionref-sha'):
        parser.add_argument('--'+name, required=True)
    record = run(parser.parse_args())
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
