#!/usr/bin/env python3
"""보존 Jbar·인구로 IR 격자 제외 선의 열 기여만 재합산한다. SE/수송 없음."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_cmfgen_fixedj_full import sha
from expand_atomic_data_cmfgen import load_cmfgen_model_spec
from lumina_model_energy import ModelThermalEnergy


def summarize(sources, response_root):
    record = json.loads((response_root / 'result.json').read_text())
    source_record = json.loads((sources / 'sources.json').read_text())
    if record['BB_contract'] != 'SOURCE_SAMPLED_TRAPEZOID_A_TIMES_MASS_V1':
        raise ValueError('보존 BB 계약은 A×profile mass 경로여야 함')
    if source_record['output_sha256']['sources.bin'] != record['source_identity']['source_sha256']:
        raise ValueError('선 catalogue와 응답의 source 신원이 다름')
    expected = {}
    def bind(path, digest):
        if sha(path) != digest:
            raise ValueError(path.name + ' 봉인 hash 불일치')
        expected[path] = digest
        return path
    def original(name):
        entries = [(Path(p), h) for p, h in record['input_sha256'].items() if Path(p).name == name]
        if len(entries) != 1:
            raise ValueError(name + ' 입력 신원 누락/중복')
        return bind(*entries[0])
    model = original('MODEL')
    spec = original('MODEL_SPEC')
    links = original('atomic_links.txt')
    levels = original('levels.csv')
    bind(sources / 'line_catalog.npz', source_record['output_sha256']['line_catalog.npz'])
    for name in ('jbar.npy', 'line-profile-mass.npy'):
        bind(response_root / name, record['output_sha256'][name])
    selected = record['selected_response']
    evaluation = response_root / Path(selected['root']).name
    bind(evaluation / 'response.npz', selected['response_sha256'])
    bind(evaluation / 'balance.npz', selected['balance_sha256'])
    rows = list(csv.DictReader(levels.open()))
    models = load_cmfgen_model_spec(spec, links)
    with np.load(evaluation / 'response.npz') as z:
        pop, fraction, columns = (z[k] for k in ('population', 'fraction', 'columns'))
    atomic_links = B.atomic.load_cmfgen_links(links)
    ions = {}
    for key, atom in models.items():
        path = Path(atomic_links[key]['osc'])
        bind(path, record['input_sha256'][str(path)])
        full = B.parse_osc(path, include_transitions=False).levels[:atom['n_full']]
        ionization = re.findall(r'^\s*(\S+)\s+!Ionization energy\b', path.read_text(), re.M)
        if len(ionization) != 1:
            raise ValueError('OSC ionization energy 누락/중복')
        edge = float(ionization[0].replace('D', 'E'))
        ions[key] = dict(edge_Hz=(edge-full['E_cm'])*B.C,
                         excitation_Hz=full['E_cm']*B.C, g=full['g'])
    # 이온 내부 BB 차이에는 중성 기준 상수가 소거된다. 물리 내부에너지는 평가하지 않는다.
    owner = ModelThermalEnergy(rows, columns, ions, {z: 0. for z, _ in models})
    with np.load(sources / 'line_catalog.npz') as z:
        lines = {k: z[k] for k in z.files}
    jbar = np.load(response_root / 'jbar.npy', mmap_mode='r')
    mass = np.load(response_root / 'line-profile-mass.npy', mmap_mode='r')
    count, nd = jbar.shape
    if count != len(lines['nu0']) or mass.shape != (count,) or fraction.shape[1] != nd:
        raise ValueError('catalogue/Jbar/인구 차원 불일치')
    domain = S.native_line_domain(model.parent)
    for path, digest in domain['source_sha256'].items():
        bind(Path(path), digest)
    excluded = ~S.native_grid_support_selection(lines['nu0'], domain)
    total = np.zeros((2, nd), dtype=np.longdouble)
    dropped = np.zeros_like(total)
    by_ion = {}
    nuclei = pop.sum(axis=0)
    group = owner.full_to_sl
    enabled = S.control(model, 'SCL_LN') == 'T'
    limit = S.number(model, 'SCL_LN_FAC')
    density_limit = S.number(model, 'SCL_DEN_LIM', '1e30')
    for first in range(0, count, 2048):
        last = min(count, first + 2048)
        lo, up = (lines[k][first:last] for k in ('lower', 'upper'))
        nu, a, ratio = (lines[k][first:last] for k in ('nu0', 'A', 'g_ratio'))
        lower = np.asarray(pop[group[lo]] * fraction[lo], dtype=np.longdouble)
        upper = np.asarray(pop[group[up]] * fraction[up], dtype=np.longdouble)
        bul = a * B.C**2 / (2 * B.H * nu**3)
        forward = lower * (bul / ratio)[:, None] * jbar[first:last] * (B.H * nu)[:, None]
        reverse = upper * (a * mass[first:last])[:, None] * (B.H * nu)[:, None]
        reverse += upper * bul[:, None] * jbar[first:last] * (B.H * nu)[:, None]
        factor = owner.line_scaling(lo, up, nu, nuclei, enabled=enabled,
                                   ratio_limit=limit, density_limit=density_limit)
        forward *= factor
        reverse *= factor
        total += np.stack([forward.sum(axis=0), reverse.sum(axis=0)])
        for offset in np.flatnonzero(excluded[first:last]):
            key = (int(rows[lo[offset]]['atomic_number']), int(rows[lo[offset]]['ion_number']) + 1)
            entry = by_ion.setdefault(key, [0, np.zeros((2, nd), dtype=np.longdouble)])
            entry[0] += 1
            values = np.stack([forward[offset], reverse[offset]])
            entry[1] += values
            dropped += values
    with np.load(evaluation / 'balance.npz') as z:
        stored = z['native_bb']
        old_net = z['native_stored_net']
    error = float(np.max(np.abs(total - stored)))
    if not np.allclose(total, stored, rtol=1.e-11, atol=1.e-13):
        raise ValueError('전체 BB 합이 보존 열장부를 재현하지 못함')
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(path.name + ' 분석 중 변경')
    delta = np.asarray(-(dropped[0] - dropped[1]), dtype='f8')
    return dict(scope='고정-state 열 재합산; 새 SE/RE/수송/물질수용 없음',
        sign='양수=가열; delta=제외 후 열잔차−기존 열잔차',
        retained_lines=count-int(excluded.sum()), excluded_lines=int(excluded.sum()),
        grid_cutoff_Hz=domain['minimum_line_Hz_grid_guard_inclusive'],
        saved_native_bb_max_abs_reproduction_error=error,
        maximum_abs_heat_delta=float(np.max(abs(delta))),
        maximum_abs_heat_delta_depth=int(np.argmax(abs(delta)))+1,
        by_ion={f'Z{z}_stage{stage}':dict(lines=n,
            maximum_abs_net=float(np.max(abs(v[0]-v[1]))),
            net_at_depth36=float(v[0,35]-v[1,35]),
            net_at_depth85=float(v[0,84]-v[1,84])) for (z,stage),(n,v) in sorted(by_ion.items())},
        depths=[dict(depth=i+1,old_heat_residual=float(old_net[i]),heat_delta=float(delta[i]),
            frozen_state_new_heat_residual=float(old_net[i]+delta[i])) for i in (0,35,51,84,85,88,89)],
        input_sha256=[dict(name=path.name, sha256=digest) for path,digest in expected.items()],
        final_goal='NOT_MET')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--response-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.sources, args.response_root), ensure_ascii=False, indent=2))
