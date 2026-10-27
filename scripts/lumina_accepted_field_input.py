#!/usr/bin/env python3
"""새 수용 점물질의 완성 계수와 확산 경계를 기존 GPU 수송 입력에 연결한다.

기존 계수 생산자·경계 함수를 재사용한다. BF/BB 적분·SE·F sweep은 없다.
실물 전주파수 계수의 해시 검사는 grammar 계산 노드에서 수행한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lumina_local_micro import open_geometry
from lumina_native_columns import NativeSourceColumns, sha
from lumina_outer_c_backend import diffusion_boundary
from prepare_lumina_outer_input import read_native


def pinned(config, key, identities):
    entry = config[key]; path = Path(entry['path']).resolve(); expected = entry['sha256']
    if sha(path) != expected:
        raise ValueError('새 자체장 입력 변경: '+key)
    identities[str(path)] = expected
    return path


def prepare(config_path, config_sha, out):
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('기존 자체장 입력 보존')
    if sha(config_path) != config_sha:
        raise ValueError('새 자체장 입력 구성 신원 불일치')
    c = json.loads(config_path.read_text())
    if c['schema'] != 'lumina-accepted-field-config-v1':
        raise ValueError('명시한 새 수용 물질 구성 필요')
    identities = {str(config_path): config_sha}
    paths = {k: pinned(c, k, identities) for k in
             ('accepted_material', 'sources', 'full_population', 'grid', 'geometry_preparation',
              'cpu_library', 'boundary_library', 'original_seed', 'model')}
    m = json.loads(paths['accepted_material'].read_text())
    original = json.loads(paths['original_seed'].read_text())
    sources = json.loads(paths['sources'].read_text())
    previous = json.loads(paths['geometry_preparation'].read_text())
    if m['schema'] != 'lumina-accepted-nodal-material-v1':
        raise ValueError('별도 수용 물질 manifest 필요; 원70으로 가장하지 않음')
    material = paths['accepted_material'].parent/'native_material_nodes.bin'
    identities[str(material)] = m['output_sha256']['native_material_nodes.bin']
    if (sha(material) != identities[str(material)] or
            m['input_sha256'][str(paths['original_seed'])] != identities[str(paths['original_seed'])] or
            m['input_sha256'][str(paths['model'])] != identities[str(paths['model'])] or
            sources['schema'] != 'lumina-model-local-sources-v1' or
            sources['input_sha256'][str(material)] != identities[str(material)] or
            sources['input_sha256'][str(paths['full_population'])] != identities[str(paths['full_population'])] or
            sources['input_sha256'][str(paths['grid'])] != identities[str(paths['grid'])] or
            sources['input_sha256'][str(paths['model'])] != identities[str(paths['model'])]):
        raise ValueError('새 완성 계수의 물질/full 인구/격자/모형 계보 불일치')
    geometry_path = paths['geometry_preparation'].parent/'geometry.npz'
    identities[str(geometry_path)] = previous['output_sha256']['geometry.npz']
    if sha(geometry_path) != identities[str(geometry_path)]:
        raise ValueError('원 순수 기하 변경')
    physical, epoch = read_native(material); g = open_geometry(paths['geometry_preparation'])
    if (m['points'] != len(physical) or previous['epoch_seconds'] != epoch or
            not np.array_equal(g.native_radius, physical[:, 0]) or
            not np.array_equal(g.velocity[g.added:], physical[:, 0]/epoch) or
            original['epoch_seconds'] != epoch or original['boundary_method'] != 'DIFFUSION' or
            original['inner_radius_cm'] != physical[-1, 0]):
        raise ValueError('새 물질의 원래점 반경/epoch/속도/확산 경계 불일치')
    # 계수 본문은 준비 단계에서 확인한다. GPU 열 조회에서는 이 고정 기록을 계승한다.
    binary = paths['sources'].parent/'sources.bin'
    identities[str(binary)] = sources['output_sha256']['sources.bin']
    if sha(binary) != identities[str(binary)]:
        raise ValueError('새 완성 계수 본문 변경')
    nu = np.load(paths['grid'], allow_pickle=False)
    comparison = sources['C_full_population_comparison']['max_relative_error']
    if (nu.ndim != 1 or nu.dtype != np.dtype('f8') or
            (sources['cells'], sources['frequencies'], sources['full_levels']) !=
            (len(physical), len(nu), m['full_levels']) or
            not np.isfinite(comparison) or not 0 <= comparison <= 1.e-12):
        raise ValueError('새 계수의 전모형 차원 또는 실제 C 인구 대조 불일치')
    with NativeSourceColumns(binary, nu, physical[:, 0], physical[:, 1], physical[:, 2]) as source:
        chi = np.empty((1, source.nf))
        for first in range(0, source.nf, 65536):
            last = min(first+65536, source.nf)
            chi[:, first:last] = source.read(np.array([source.ns-1]), np.arange(first, last))[0]
    planck, gradient, boundary = diffusion_boundary(paths['boundary_library'], nu, float(physical[-1, 1]),
        original['inner_luminosity_erg_s'], float(physical[-1, 0]), chi[0])
    if any(sha(Path(p)) != h for p, h in identities.items()):
        raise ValueError('새 자체장 입력 준비 중 원자료 변경')
    out.mkdir(); (out/'geometry.npz').symlink_to(geometry_path)
    for name, value in (('inner_planck.npy', planck), ('inner_gradient.npy', gradient)):
        with (out/name).open('xb') as f:
            np.save(f, value, allow_pickle=False)
    report = dict(schema='lumina-accepted-field-preparation-v1',
        status='PREPARED_NEW_COEFFICIENT_BOUNDARY_NOT_RUN',
        accepted_material=str(paths['accepted_material']), sources=str(paths['sources']),
        full_population=str(paths['full_population']), material=str(material), grid=str(paths['grid']),
        cpu_library=str(paths['cpu_library']), boundary_library=str(paths['boundary_library']),
        native_nodes=len(physical), frequencies=len(nu), epoch_seconds=epoch,
        accepted_response_sha256=m['accepted_response_sha256'], input_sha256=identities,
        source_binary=str(binary), source_binary_sha256=identities[str(binary)],
        boundary=boundary, geometry_reused_but_old_boundary_not_reused=True,
        output_sha256={n: sha(out/n) for n in ('geometry.npz', 'inner_planck.npy', 'inner_gradient.npy')},
        new_BF_BB_integrals=0, new_SE_solves=0, actual_F_calls=0, final_goal='NOT_MET')
    with (out/'preparation.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return report


def open_prepared(path, expected_sha):
    """동일한 새 물질/계수 준비만 읽는다. 전체7평면을 GPU 시작 때 다시 해시하지 않는다."""
    path = Path(path).resolve()
    if sha(path) != expected_sha:
        raise ValueError('새 자체장 준비 기록 변경')
    p = json.loads(path.read_text())
    if p['schema'] != 'lumina-accepted-field-preparation-v1':
        raise ValueError('새 수용 물질 전용 준비 기록 필요')
    for name, expected in p['input_sha256'].items():
        if name != p['source_binary'] and sha(Path(name)) != expected:
            raise ValueError('새 자체장 실행 입력 변경: '+name)
    physical, epoch = read_native(Path(p['material']))
    g = open_geometry(path)
    if (epoch != p['epoch_seconds'] or not np.array_equal(g.native_radius, physical[:, 0]) or
            not np.array_equal(g.velocity[g.added:], physical[:, 0]/epoch)):
        raise ValueError('새 자체장 실행 기하 불일치')
    boundary = []
    for name in ('inner_planck.npy', 'inner_gradient.npy'):
        value = path.parent/name
        if sha(value) != p['output_sha256'][name]:
            raise ValueError('새 내부 경계 변경')
        a = np.load(value, allow_pickle=False)
        if a.shape != (p['frequencies'],) or not np.isfinite(a).all():
            raise ValueError('새 내부 경계 형상/유한성 오류')
        boundary.append(a)
    nu = np.load(p['grid'], allow_pickle=False)
    source = NativeSourceColumns(p['source_binary'], nu, physical[:, 0], physical[:, 1], physical[:, 2])
    if (source.ns, source.nf) != (p['native_nodes'], p['frequencies']):
        source.close(); raise ValueError('새 전체장 차원 변경')
    source.identity = dict(role='ACCEPTED_NODAL_MATERIAL_NEW_COEFFICIENTS',
        accepted_response_sha256=p['accepted_response_sha256'], preparation_sha256=expected_sha,
        source_sha256=p['source_binary_sha256'], grid_sha256=p['input_sha256'][p['grid']],
        epoch_seconds=epoch, full_source_hash_verification='INHERITED_PREPARATION_NOT_REHASHED')
    return p, source, g, boundary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(prepare(a.config, a.sha256, a.out), ensure_ascii=False, indent=2, allow_nan=False))
