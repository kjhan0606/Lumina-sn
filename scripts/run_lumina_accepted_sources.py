#!/usr/bin/env python3
"""수용 물질의 실제 새 전주파수 계수·경계만 생산한다. RT/SE 실행은 없다."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
import lumina_accepted_field_input as A

RUNS = Path('/gpfs/kjhan/cmfgen_runs')


def run(out, source_commit):
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '16' or
            os.environ.get('SLURM_MEM_PER_NODE') != '32768' or
            os.environ.get('OMP_NUM_THREADS') != '16'):
        raise ValueError('grammar CPU16/32GiB/OpenMP16 단회 전용')
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError('기존 계수 보존')
    start = time.monotonic()
    material = RUNS/'lumina_accepted_input_20261027-79u5Sp/material'
    paths = dict(
        accepted_material=material/'accepted-material.json',
        full_population=material/'C-imported-full.bin',
        grid=RUNS/'lumina_full_sr_verified_20261020-yb8k9y/grid/frequency.npy',
        geometry_preparation=RUNS/'lumina_outer_reuse_20261021-uTVbnT/boundary/preparation.json',
        cpu_library=RUNS/'lumina_extended_4h_20261022-rJGMVR/driver.so',
        boundary_library=RUNS/'lumina_outer_weighted_20261021-B0Fepu/pytest-threadcap-work/outer-c0/outer.so',
        original_seed=RUNS/'lumina_native70_seed_20261020-DphYnQ/seed/material-seed.json',
        model=RUNS/'toy06_19p48d_field_feedback_429038/material/MODEL')
    expected = dict(
        accepted_material='61e3090a8550bffb6e6f030a84be811407634299eefba93036d1fad30d492c1b',
        full_population='2ce239bd5d39e994e2cca0a46b283d8ccb68fc17a60bbaabcc58c9291de22ea6',
        grid='c03b9fb565a215fd11ec70ab2574c70af7f09eab74669214638177c2bbfc0a6d',
        geometry_preparation='1fad50444de5db7a583befd8d59ae7970c5ebf63c3cd423273c341ddb21582f2',
        cpu_library='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4',
        boundary_library='f88362d72782415cadd2066ff6a14d512dce5d62ec61db4c9f620c800a7fd396')
    if any(B.sha(paths[k]) != h for k, h in expected.items()):
        raise ValueError('수용 물질/격자/기하/바이너리 변경')
    manifest = json.loads(paths['accepted_material'].read_text())
    for name, digest in manifest['output_sha256'].items():
        if B.sha(material/name) != digest: raise ValueError('새 물질 출력 변경: '+name)
    for key in ('original_seed', 'model'):
        if B.sha(paths[key]) != manifest['input_sha256'][str(paths[key])]:
            raise ValueError('원 기준 입력 변경: '+key)
    phot = RUNS/'lumina_model_spec_428222/cmfgen_phot_smoothing.so'
    kernel = RUNS/'lumina_full_source_tests_20261020-0sY1jL/kernels/model_sources.so'
    out.mkdir()
    inputs = {str(p): B.sha(p) for p in (*paths.values(), phot, kernel)}
    (out/'inputs.json').write_text(json.dumps(dict(source_commit=source_commit,
        input_sha256=inputs, resources='CPU16/32GiB/2h/GPU0/retry0',
        continuum_evaluation='direct at every original frequency; no CONT_FREQ substitution',
        physical_RT_executions=0, new_SE_solves=0), indent=2)+'\n')
    print(json.dumps(dict(phase='NEW_ACCEPTED_COEFFICIENTS_BEGIN', source_commit=source_commit)), flush=True)
    S.prepare(paths['model'].parent, material/'native_material_nodes.bin', material/'atomic',
        phot, kernel, paths['full_population'], np.load(paths['grid'], allow_pickle=False),
        out/'sources', frequency_grid=paths['grid'])
    if any(B.sha(Path(p)) != h for p, h in inputs.items()): raise ValueError('실행 중 입력 변경')
    paths['sources'] = out/'sources/sources.json'
    config = out/'field-config.json'
    config.write_text(json.dumps(dict(schema='lumina-accepted-field-config-v1',
        **{k: dict(path=str(p), sha256=B.sha(p)) for k, p in paths.items()}), indent=2)+'\n')
    A.prepare(config, B.sha(config), out/'boundary')
    result = dict(state='COMPLETE_NEW_ACCEPTED_COEFFICIENTS_AND_BOUNDARY',
        source_commit=source_commit, wall_seconds=time.monotonic()-start,
        input_sha256=inputs, preparation=dict(path=str(out/'boundary/preparation.json'),
        sha256=B.sha(out/'boundary/preparation.json')), new_transport=0, new_SE_solves=0,
        final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    run(**vars(parser.parse_args()))
