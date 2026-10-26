#!/usr/bin/env python3
"""봉인된 원49점 bound 후보에 검증된 준위용해율만 더해 private C 해를 대조한다."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import socket
import struct
import subprocess
import time

import numpy as np

import cmfgen_model_continuum as B
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables

BASE_SHA = 'f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab'
HEADER = struct.Struct('<8sIIII2d')
RECORD = np.dtype([('ids', '<i4', (5,)), ('values', '<f8', (10,))])
FIXED = re.compile(r'^\[MODEL-SE\]\[FIXED-FIELD\] Z=(\d+) native_node=(\d+) '
                   r'variables=(\d+) state=PRIVATE_PARTIAL_CHANNEL_CANDIDATE '
                   r'residual=([^\s]+) physical_equivalence=NOT_PROVEN$', re.M)
ROW = re.compile(r'^\[MODEL-SE\]\[FIXED-FIELD-ROW\] Z=(\d+) stage0=(\d+) '
                 r'super0=(-?\d+) original=([^\s]+) candidate=([^\s]+)$', re.M)


def parse_candidate(output, original, node):
    fixed = FIXED.findall(output)
    if (len(fixed) != 2 or {int(r[0]) for r in fixed} != {14, 16} or
        any(int(r[1]) != node or float(r[3]) > 1e-12 for r in fixed)):
        raise ValueError('private C 후보 잔차/원소 증거 누락')
    rows = ROW.findall(output)
    observed = {}
    for z, stage, super0, old, candidate in rows:
        key = (int(z), int(stage), int(super0))
        if key in observed or key not in original:
            raise ValueError('private C 후보 원자 주소 중복/추가')
        old, candidate = float(old), float(candidate)
        if (not np.isfinite(old) or not np.isfinite(candidate) or old < 0 or candidate < 0 or
            not np.isclose(old, original[key], rtol=1e-12, atol=0)):
            raise ValueError('private C 후보의 원 물질70 불일치')
        observed[key] = candidate
    if len(rows) != sum(int(r[2]) for r in fixed) or set(observed) != set(original):
        raise ValueError('private C 후보518개 주소 누락')
    return observed, max(float(r[3]) for r in fixed)


def merge_bound_pseudo(blob, bound, pseudo, roster, node):
    if len(blob) < HEADER.size:
        raise ValueError('원 BF C 기여 파일 절단')
    magic, levels, nroute, ns, saved_node, temp, ne = HEADER.unpack_from(blob)
    if (magic != b'LMBFFUL1' or nroute != 24667 or ns != 90 or saved_node != node or
        len(blob) != HEADER.size + nroute * RECORD.itemsize or
        not np.isfinite([temp, ne]).all() or temp <= 0 or ne <= 0 or
        bound.shape != (nroute, 6) or pseudo.shape != (nroute, 6) or
        not np.isfinite(bound).all() or not np.isfinite(pseudo).all() or
        np.any(pseudo[:, :3] < 0)):
        raise ValueError('BF/pseudo C 기여 차원·물질 오류')
    records = np.frombuffer(blob, dtype=RECORD, offset=HEADER.size).copy()
    for key, column in (('lower', 0), ('target', 1)):
        if not np.array_equal(records['ids'][:, column], roster[key]):
            raise ValueError('BF C route 주소 불일치')
    for key, column in (('probability', 0), ('threshold', 1), ('sigma_edge', 2), ('saha', 3)):
        if not np.array_equal(records['values'][:, column], roster[key]):
            raise ValueError('BF C route 정규화 불일치')
    if not np.array_equal(records['values'][:, 4:], bound):
        raise ValueError('원 BF C 기여와 봉인율 배열 불일치')
    z = roster['Z']
    if np.any(pseudo[~np.isin(z, (14, 16))] != 0):
        raise ValueError('Si/S 이외 pseudo 변경')
    records['values'][:, 4:] += pseudo
    if not np.isfinite(records['values']).all() or np.any(records['values'][:, 4:7] < 0):
        raise ValueError('BF+pseudo 수치 오류')
    return blob[:HEADER.size] + records.tobytes()


def run(args):
    start = time.monotonic()
    script_sha = B.sha(Path(__file__))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('기존 후보 결과 보존')
    host = socket.gethostname().lower()
    if (not host.startswith('grammar') or not os.environ.get('SLURM_JOB_ID') or
        os.environ.get('SLURM_CPUS_PER_TASK') != '1' or
        os.environ.get('SLURM_MEM_PER_NODE') != '4096' or
        os.environ.get('OMP_NUM_THREADS') != '1' or
        os.environ.get('OPENBLAS_NUM_THREADS') != '1'):
        raise ValueError('grammar CPU1·4GiB Slurm 전용')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    paths = {key: Path(getattr(args, key)) for key in
             ('base', 'baseline', 'pseudo_result', 'bound_result', 'collision_result')}
    inputs = {key: B.sha(path) for key, path in paths.items()}
    if inputs['base'] != BASE_SHA:
        raise ValueError('원 물질70 계획 신원 변경')
    base = json.loads(paths['base'].read_text())
    config = json.loads(Path(base['config']).read_text())
    plan = json.loads(Path(config['plan']).read_text())
    baseline = json.loads(paths['baseline'].read_text())
    pseudo = json.loads(paths['pseudo_result'].read_text())
    bound = json.loads(paths['bound_result'].read_text())
    collision = json.loads(paths['collision_result'].read_text())
    node = args.node
    old = paths['baseline'].parent
    bf = baseline.get('full_bound_test') or {}
    if (baseline.get('state') != 'PASS' or baseline.get('input_identity') != base['source_identity'] or
        bf.get('native_node') != node or bf.get('source_result_sha256') != inputs['bound_result'] or
        bf.get('private_partial_channel_SE_candidate') is not True or
        pseudo.get('state') != 'COMPLETE_PRIVATE_RATES' or pseudo.get('native_node') != node or
        pseudo.get('material_record') != 70 or pseudo.get('routes') != 24667 or
        pseudo.get('source_unchanged') is not True or
        pseudo.get('radiation_J_sha256') != baseline.get('fixed_radiation_J_sha256') or
        pseudo.get('input_sha256', {}).get(str(paths['bound_result'])) != inputs['bound_result'] or
        bound.get('input_identity') != baseline['input_identity'] or
        baseline.get('point_collision_test', {}).get('source_result_sha256') != inputs['collision_result'] or
        collision.get('native_node') != node):
        raise ValueError('원 BF/BB/C 후보와 pseudo가 다른 입력')
    files = {
        'bf': old / 'bound-full-frequency-contributions.bin',
        'bb': old / 'point-lines-contributions.bin',
        'occupation': old / 'model-occupation-input.bin',
        'parameters': old / 'model-occupation-parameters.bin',
        'projection': old / 'model_projection',
        'collision': paths['collision_result'].with_suffix(''),
        'bound_array': paths['bound_result'].with_name('unit-rates.npy'),
        'roster': paths['bound_result'].with_name('roster.npz'),
        'pseudo_array': paths['pseudo_result'].with_name('pseudo-unit-rates.npy'),
        'old_imported': old / 'model-full-bound-imported-full.bin',
    }
    hashes = {key: B.sha(path) for key, path in files.items()}
    if (hashes['bf'] != bf['contribution_sha256'] or
        hashes['bb'] != baseline['point_bb_test']['contributions_sha256'] or
        hashes['occupation'] != baseline['inputs_sha256']['model-occupation-input.bin'] or
        hashes['parameters'] != baseline['inputs_sha256']['model-occupation-parameters.bin'] or
        hashes['projection'] != baseline['executable_sha256'] or
        hashes['collision'] != collision['output_sha256'] or
        hashes['collision'] != baseline['point_collision_test']['binary_sha256'] or
        hashes['bound_array'] != bf['artifact_sha256']['unit-rates.npy'] or
        hashes['roster'] != bf['artifact_sha256']['roster.npz'] or
        hashes['pseudo_array'] != pseudo['artifact_sha256']):
        raise ValueError('후보 원 C/율 출력 byte 변경')
    with np.load(files['roster'], allow_pickle=False) as r:
        roster = {key: r[key] for key in r.files}
    bound_array = np.load(files['bound_array'], allow_pickle=False)
    pseudo_array = np.load(files['pseudo_array'], allow_pickle=False)
    overlay = merge_bound_pseudo(files['bf'].read_bytes(), bound_array, pseudo_array, roster, node)
    original = {(r['Z'], r['stage0'], r['super0']): r['original']
                for r in bf['fixed_field_superlevel_profile']}
    previous = {(r['Z'], r['stage0'], r['super0']): r['candidate']
                for r in bf['fixed_field_superlevel_profile']}
    if len(original) != 518 or len(previous) != 518:
        raise ValueError('원 후보 SL518개 누락')
    native = Path(plan['same_material_reference_field']).parent / 'material'
    nd, nt = read_dimensions(native)
    variables = read_variables(native, nt - 2)
    reader = ScrtempReader(native / 'SCRTEMP', nd, nt)
    if nd != 90 or not reader.has_iteration(70) or not reader.has_iteration(71):
        raise ValueError('원70/71 native 상태 누락')
    source70 = reader.state(70)
    source71 = reader.state(71)
    labels = {'Si': 14, 'S': 16}
    native71 = {}
    for variable in variables:
        if variable.species not in labels:
            continue
        key = (labels[variable.species], variable.charge,
               -1 if variable.terminal else variable.superlevel - 1)
        if key in native71:
            raise ValueError('원71 SL 주소 중복')
        native71[key] = float(source71[variable.index - 1, node])
        if key not in original or not np.isclose(source70[variable.index - 1, node], original[key],
                                                rtol=1e-12, atol=0):
            raise ValueError('원70 C 후보 원 인구 주소 불일치')
    if set(native71) != set(original):
        raise ValueError('원71/C SL 주소518개 불일치')
    out.mkdir(parents=True)
    overlay_path = out / 'bound-plus-pseudo-contributions.bin'
    with overlay_path.open('xb') as stream:
        stream.write(overlay)
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ,
               LUMINA_TOPION_LEVELS_FILE=str(project / 'data/atomic/topion_levels.csv'),
               LUMINA_IONIZATION_REFERENCE_FILE=str(project / 'data/atomic/ionization_reference.csv'))
    common = [str(files['projection']), str(Path(config['seed']) / 'atomic'), str(nd),
              str(Path(config['seed']) / 'native_material_nodes.bin')]
    results = {}
    for name, bf_path in (('control', files['bf']), ('pseudo', overlay_path)):
        exported = out / (name + '-imported-full.bin')
        command = common + [str(exported), str(files['occupation']), str(files['parameters']),
                            str(bf_path), str(files['bb']), str(files['collision'])]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=420)
        (out / (name + '.stdout')).write_text(completed.stdout)
        (out / (name + '.stderr')).write_text(completed.stderr)
        if completed.returncode:
            raise ValueError(f'{name} private C 투영 실패: {completed.returncode}')
        if B.sha(exported) != hashes['old_imported']:
            raise ValueError(f'{name} C 원 물질 byte 불일치')
        values, residual = parse_candidate(completed.stdout, original, node)
        if name == 'control' and any(not np.isclose(values[k], previous[k], rtol=1e-12, atol=0)
                                     for k in previous):
            raise ValueError('동결 BF 제어 재생이 원 후보와 다름')
        results[name] = dict(values=values, residual=residual)
    stages = []
    for z in (14, 16):
        for charge in (1, 2, 3):
            keys = [k for k in original if k[:2] == (z, charge)]
            reference = sum(native71[k] for k in keys)
            prior = sum(previous[k] for k in keys)
            new = sum(results['pseudo']['values'][k] for k in keys)
            stages.append(dict(Z=z, charge=charge, native71=reference,
                               bound_candidate=prior, pseudo_candidate=new,
                               bound_relative=(prior / reference - 1),
                               pseudo_relative=(new / reference - 1)))
    unchanged = (all(B.sha(path) == inputs[key] for key, path in paths.items()) and
                 all(B.sha(path) == hashes[key] for key, path in files.items()) and
                 B.sha(Path(__file__)) == script_sha)
    report = dict(schema='lumina-private-direct-pseudo-candidate-v1',
                  state='COMPLETE_PRIVATE_OVERLAY' if unchanged else 'INPUT_CHANGED',
                  native_node=node, material_record=70,
                  control_max_residual=results['control']['residual'],
                  pseudo_max_residual=results['pseudo']['residual'],
                  stages=stages, overlay_sha256=B.sha(overlay_path),
                  input_sha256=inputs, artifact_sha256=hashes,
                  script_sha256=script_sha, source_unchanged=unchanged,
                  wall_seconds=time.monotonic() - start,
                  public_material_updates=0, new_transport=0,
                  SE_charge_RE_solved=False, final_goal='NOT_MET')
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('state', 'control_max_residual',
                                          'pseudo_max_residual', 'stages', 'wall_seconds')}), flush=True)
    return 0 if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'baseline', 'pseudo-result', 'bound-result', 'collision-result', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--node', type=int, default=49)
    raise SystemExit(run(parser.parse_args()))
