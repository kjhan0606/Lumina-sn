#!/usr/bin/env python3
"""물질59를 고정하고 완성된 J59/J60의 생산 C owner 입력을 짝으로 만든다.

기존 PRRR/SCRTEMP 매칭 adapter의 계약을 바꾸지 않는다. 이 입력은 새 PRRR이나
수렴 참조가 아니라, 같은 물질에서 두 외부장의 조건부 인구 응답을 비교한다.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np

import compare_cmfgen_bf_superlevel_rates as bf
from compare_cmfgen_field_response import read_complete_pair
from cmfgen_phot_smoothing import read_smoothing_controls
from cmfgen_recombination_policy import build_policy
from prepare_cmfgen_owner_response import write_input
from ptf_gated_metrics import read_rvtj_block


def material_inputs(seed, loaded):
    nd, nt = bf.read_dimensions(seed)
    reader = bf.ScrtempReader(seed / 'SCRTEMP', nd, nt)
    if reader.last_it != 59:
        raise ValueError('이번 짝비교에는 새 물질59가 필요함')
    state = reader.state(59)
    variables = bf.read_variables(seed, nt - 2)
    selected = [v for v in variables if v.species in ('Si', 'S')]
    columns = [(14 if v.species == 'Si' else 16, v.charge, v.superlevel - 1)
               for v in selected]
    if len(set(columns)) != len(columns) or any(c < 1 or c > 5 for _, c, _ in columns):
        raise ValueError('인구 키 중복/범위 오류')
    for Z in (14, 16):
        for charge in range(1, 5):
            actual = {sl for z, c, sl in columns if (z, c) == (Z, charge)}
            if actual != set(range(loaded['model'][(Z, charge + 1)]['n_super'])):
                raise ValueError('활성 SL 인구 누락')
        if [(z, c, sl) for z, c, sl in columns if z == Z and c == 5] != [(Z, 5, 0)]:
            raise ValueError('terminal VI 인구 누락/중복')
    rho = read_rvtj_block(seed / 'RVTJ', 'Mass Density (gm/cm^3)', nd)
    velocity = read_rvtj_block(seed / 'RVTJ', 'Velocity (km/s)', nd)
    if rho is None or velocity is None or np.any(velocity <= 0) or np.any(np.diff(velocity) >= 0):
        raise ValueError('실제 물질 밀도/속도 격자 오류')
    epochs = np.asarray(reader.radius) * 1e10 / (velocity * 1e5)
    epoch = float(np.median(epochs))
    if np.max(np.abs(epochs / epoch - 1)) > 1e-5:
        raise ValueError('상동팽창 시각 불일치')
    v = velocity * 1e5
    mid = .5 * (v[:-1] + v[1:])
    vin = np.r_[mid, v[-1] - (mid[-1] - v[-1])]
    vout = np.r_[v[0] + v[0] - mid[0], mid]
    profiles = np.column_stack((rho, state[nt - 1] * 1e4, state[nt - 2], vin, vout))
    populations = state[[v.index - 1 for v in selected]].T
    return columns, profiles, populations, epoch


def verify_view(seed, deck):
    # 파생 view의 provenance를 새 원덱처럼 위장하지 않는다.
    stamp = json.loads((deck / 'DECK_PROVENANCE.json').read_text())
    original = Path(stamp['deck'])
    source_run = Path(stamp['cmfgen_model_spec']).parent
    _, _, loaded = bf.verify_sealed_deck(source_run, original)
    for name, key in (('MODEL_SPEC', 'cmfgen_model_spec_sha256'),
                      ('LEVEL_SL_STEQ_LINKS', 'level_sl_steq_links_sha256')):
        if bf.sha256_file(seed / name) != stamp[key]:
            raise ValueError('물질59의 원자모형 신원 불일치: ' + name)
    # 격리 실행은 두 로컬 PHOT 경로를 복사본으로 바꾼다. 링크 텍스트의 byte가
    # 아니라 모든 이온/역할/PHOT 분기별 원자료 byte가 같은지 직접 확인한다.
    actual_links = bf.atomic.load_cmfgen_links(seed / 'atomic_links.txt')
    loaded['seed_atomic_sources'] = same_atomic_sources(loaded['links'], actual_links)
    if bf.sha256_file(deck / 'levels.csv') != bf.sha256_file(original / 'levels.csv'):
        raise ValueError('원자 준위 신원 변경')
    if not re.search(r'^\s*F\s+\[TRT_NON_TE\]', (seed / 'MODEL').read_text(), re.M):
        raise ValueError('비열적 전자 활성은 이 frozen owner 범위 밖임')
    smoother = loaded['phot_smoother']
    if smoother is None or read_smoothing_controls(seed / 'MODEL') != smoother.controls:
        raise ValueError('물질59의 PHOT 전처리 선택 불일치')
    manifest = json.loads((deck / 'cmfgen_bf_channels_manifest.json').read_text())
    if manifest.get('schema') != 'lumina-cmfgen-bf-channels-v5':
        raise ValueError('검증된 선택물리 v5가 필요함')
    for name, key in (('cmfgen_bf_channels.bin', 'channel_bin_sha256'),
                      ('cmfgen_sigma_bf.bin', 'sigma_bin_sha256')):
        if bf.sha256_file(deck / name) != manifest[key]:
            raise ValueError('봉인 BF 배열 변경: ' + name)
    bf.parse_cmfgen_bf_channels(deck / 'cmfgen_bf_channels.bin', manifest['n_levels'],
                               manifest['n_freq_bins'], manifest)
    expected = build_policy(seed / 'MODEL', loaded['model'], loaded['links'], manifest['ions'],
                            [(Z, s) for Z in (14, 16) for s in range(2, 6)])
    if expected['ions'] != manifest['recombination_policy']['ions']:
        raise ValueError('물질59와 생산 덱의 재결합 선택물리 불일치')
    return loaded


def same_atomic_sources(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('원자료 이온 집합 불일치')
    result = []
    for key in sorted(expected):
        if set(expected[key]) != set(actual[key]):
            raise ValueError('원자료 역할 집합 불일치')
        for role in sorted(expected[key]):
            paths = [expected[key][role], actual[key][role]]
            routes = [bf.atomic._phot_route_paths(p) if role == 'phot' else [p] for p in paths]
            hashes = [[bf.sha256_file(p) for p in group] for group in routes]
            if hashes[0] != hashes[1]:
                raise ValueError('원자료 byte 불일치: ' + str((key, role)))
            result.append(dict(ion=key, role=role, original_paths=list(map(str, routes[0])),
                               material59_paths=list(map(str, routes[1])), sha256=hashes[0]))
    return result


def prepare(seed, run, deck, out):
    if out.exists():
        raise ValueError('기존 짝비교 입력 보존')
    loaded = verify_view(seed, deck)
    old, new, nu, nd = read_complete_pair(seed, run)
    columns, profiles, populations, epoch = material_inputs(seed, loaded)
    if len(profiles) != nd:
        raise ValueError('장/물질 깊이 불일치')
    out.mkdir()
    for label, field in (('J59', old), ('J60', new)):
        write_input(out / (label + '_owner_input.bin'), nu, profiles, columns,
                    populations, field, epoch)
    with (out / 'reference59.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['depth_1based', 'Z', 'stage', 'population_cm3', 'fraction'])
        for depth in range(nd):
            for Z in (14, 16):
                total = sum(populations[depth, i] for i, col in enumerate(columns) if col[0] == Z)
                for charge in range(1, 6):
                    density = sum(populations[depth, i] for i, col in enumerate(columns) if col[:2] == (Z, charge))
                    writer.writerow([depth + 1, Z, charge + 1, density, density / total])
    source = Path(__file__).resolve()
    shutil.copy2(source, out / source.name)
    files = [seed / name for name in ('SCRTEMP', 'RVTJ', 'MODEL', 'MODEL_SPEC',
              'atomic_links.txt', 'LEVEL_SL_STEQ_LINKS', 'EDDFACTOR')]
    files += [run / 'EDDFACTOR', run / 'seq_logs/whole-runtime.json', run / 'seq_logs/whole-inputs.json']
    # 명시된 작은 원자 view 한 곳만 열거한다. 원자료/원래 view는 쓰지 않는다.
    files += [p for p in deck.iterdir() if p.is_file()]
    files += [source, Path(write_input.__code__.co_filename).resolve(),
              Path(read_complete_pair.__code__.co_filename).resolve()]
    report = dict(schema='cmfgen-same59-field-owner-pair-v1', seed=str(seed), run=str(run),
        deck=str(deck), depths=nd, native_frequencies=len(nu), population_columns=len(columns),
        material_record=59, epoch_seconds=epoch, PRRR_phase_match='NOT_CLAIMED',
        same_T_ne_initial_population=True, CMFGEN_converged=False,
        interpretation='고정 물질59에서 외부 J59/J60에 대한 조건부 생산 owner 응답; 새 자기일관 분포가 아님',
        field_phase_warning='J59는 물질58에서, J60는 물질59에서 생성됐다. 수리 단독 효과가 아님.',
        atomic_source_equivalence=loaded['seed_atomic_sources'],
        input_sha256={str(p): bf.sha256_file(p) for p in files},
        output_sha256={p.name: bf.sha256_file(p) for p in out.iterdir() if p.is_file()},
        final_goal='NOT_JUDGED')
    with (out / 'pair-inputs.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(out=str(out), depths=nd, record=59, input_pair_complete=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seed', 'run', 'deck', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(*(getattr(args, name).resolve() for name in ('seed', 'run', 'deck', 'out')))
