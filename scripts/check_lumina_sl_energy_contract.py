#!/usr/bin/env python3
"""저장 후보의 SL 에너지 계약을 재현한다. grammar-debug CPU1/4GiB/180초용.

같은 Jbar의 fallback 선만 재생한다. 새 BF/BB 적분·SE·수송·설정 변경은 없다.
결과는 전자 열방정식이나 최종 스펙트럼 수용 판정이 아니다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_lumina_line_thermal_ledgers import digest
from lumina_model_energy import ModelThermalEnergy
from run_lumina_full_material_response import BASE


def cycle_work(frequency, factor, orientation):
    photon = B.H * np.asarray(frequency, dtype=np.longdouble) / B.EV
    sign = np.asarray(orientation)
    return dict(physical_cycle_defect_eV=float(np.sum(sign * photon)),
        effective_cycle_defect_eV=float(np.sum(sign * factor * photon)),
        relative_effective_cycle_defect=float(abs(np.sum(sign * factor * photon)) / photon.sum()))


def atomic_proxy(ledger):
    v = np.asarray(ledger, dtype=np.longdouble)
    # 광자 가열에서 전자 초과에너지를 빼고 원자로 들어가는 충돌 냉각을 더한다.
    # 이 값은 진단용이다. 이 항들을 photon RE에 다시 더하지 않는다.
    return (v[:, :, 0] + v[:, :, 1] - v[:, :, 2] + v[:, :, 3] + v[:, :, 4]).sum(axis=0)


def check(result, expected_sha, bb):
    result = result.resolve(); bb = bb.resolve()
    if digest(result) != expected_sha:
        raise ValueError('봉인 수용 보고서 변경')
    report = json.loads(result.read_text()); accepted = result.parent / 'accepted-response.npz'
    if not any(row.get('accepted') is True for row in report['history']):
        raise ValueError('실제 수용 후보 필요')
    if digest(accepted) != report['accepted_output_sha256']:
        raise ValueError('수용 물질 변경')
    with np.load(accepted) as f:
        state = {k: f[k] for k in f.files}
    evaluations = [row for row in report['evaluations']
                   if row['output_sha256']['response.npz'] == report['accepted_output_sha256']]
    if len(evaluations) != 1:
        raise ValueError('수용 응답 주소 모호')
    balance = Path(report['accepted_source_directory']) / 'balance.npz'
    if digest(balance) != evaluations[0]['output_sha256']['balance.npz']:
        raise ValueError('같은 수용 후보의 열장부가 아님')
    with np.load(balance) as f:
        energy = {k: f[k] for k in f.files}
    config = json.loads(Path(json.loads(BASE.read_text())['config']).read_text())
    plan = json.loads(Path(config['plan']).read_text()); material = Path(plan['initial_material'])
    levels = Path(config['cells']) / 'atomic/levels.csv'
    rows = list(csv.DictReader(levels.open()))
    models = B.atomic.load_cmfgen_model_spec(material / 'MODEL_SPEC', material / 'atomic_links.txt')
    owner = ModelThermalEnergy.from_model(material, rows, models, state['columns'],
        result.parent.parent / 'source/data/atomic/ionization_reference.csv')
    br = json.loads((bb / 'result.json').read_text())
    before = {str(result): expected_sha, str(accepted): report['accepted_output_sha256'],
        str(balance): evaluations[0]['output_sha256']['balance.npz'],
        str(levels): report['input_sha256'][str(levels)], **owner.input_sha256,
        **{str(bb / name): br['output_sha256'][name] for name in ('line-roster.npz', 'jbar.npy')}}
    if any(digest(Path(p)) != h for p, h in before.items()):
        raise ValueError('계약 재생 입력 신원 변경')
    with np.load(bb / 'line-roster.npz') as f:
        lines = {k: f[k] for k in f.files}
    lo, up, nu = (lines[k] for k in ('lower', 'upper', 'nu0')); mapping = owner.full_to_sl
    density = state['population'].sum(axis=0)
    limit = S.number(material / 'MODEL', 'SCL_LN_FAC')
    density_limit = S.number(material / 'MODEL', 'SCL_DEN_LIM', '1e30')
    if (len(lo) != 790493 or len(rows) != 24542 or len(density) != 90 or
        S.control(material / 'MODEL', 'SCL_LN') != 'T' or
        S.control(material / 'MODEL', 'SCL_SL_OPAC', 'F') != 'F' or
        np.any(density >= density_limit)):
        raise ValueError('이번에 봉인한 모형/가중/밀도 범위를 벗어남')
    ratio = (owner.mean_edge[mapping[lo]] - owner.mean_edge[mapping[up]]) / nu
    effective = owner.line_scaling(lo, up, nu, density[:1], enabled=True,
                                  ratio_limit=limit, density_limit=density_limit)[:, 0]
    # 실제 원 graph에서 찾은 대표 반례. 가장 강한 순환이나 실제 순환 유량이라는 뜻은 아니다.
    vertices = [16547, 16551, 16555]; ids = np.array([556455, 557280, 556459])
    if (list(zip(lo[ids].tolist(), up[ids].tolist())) !=
        [(vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[0], vertices[2])] or
        np.any(lines['A'][ids] <= 0) or mapping[vertices[0]] != mapping[vertices[1]]):
        raise ValueError('기록한 원자 graph 반례 변경')
    witness = dict(vertices=[dict(global_full=i, Z=int(rows[i]['atomic_number']),
        charge=int(rows[i]['ion_number']), superlevel=int(rows[i]['super_level']),
        energy_eV=float(rows[i]['energy_eV'])) for i in vertices],
        line_roster_indices=ids.tolist(), orientation=[1, 1, -1],
        frequency_Hz=nu[ids].tolist(), A_s=lines['A'][ids].tolist(),
        raw_SL_ratio=ratio[ids].tolist(), effective_factor=effective[ids].tolist(),
        **cycle_work(nu[ids], effective[ids], [1, 1, -1]))
    selected = np.flatnonzero(abs(ratio - 1) > limit)
    restoration = np.zeros(90, dtype=np.longdouble); same = np.zeros_like(restoration)
    J = np.load(bb / 'jbar.npy', mmap_mode='r')
    try:
        for start in range(0, len(selected), 2048):
            idx = selected[start:start + 2048]; l = lo[idx]; u = up[idx]; v = nu[idx]
            A = lines['A'][idx]; bul = A * B.C**2 / (2 * B.H * v**3)
            lower = state['population'][mapping[l]] * state['fraction'][l]
            upper = state['population'][mapping[u]] * state['fraction'][u]; j = J[idx]
            forward = np.asarray(lower * (bul / lines['g_ratio'][idx])[:, None] * j, dtype=np.longdouble)
            reverse = np.asarray(upper * (A[:, None] + bul[:, None] * j), dtype=np.longdouble)
            q = (forward - reverse) * (B.H * v)[:, None]
            restoration += ((1 - ratio[idx, None]) * q).sum(axis=0)
            same += q[mapping[l] == mapping[u]].sum(axis=0)
    finally:
        J._mmap.close()
    proxy = atomic_proxy(state['ledger'])
    gap = energy['native_bb'][0] - energy['native_bb'][1] - energy['bb'][0] + energy['bb'][1]
    ideal = gap - restoration
    if any(digest(Path(p)) != h for p, h in before.items()):
        raise ValueError('계약 재생 중 입력 변경')
    return dict(scope=__doc__, input_sha256=before, source_script_sha256=digest(Path(__file__)),
        witness=witness, points=90, fallback_lines=len(selected),
        same_SL_lines=int(np.count_nonzero(mapping[lo] == mapping[up])),
        arrays={name:np.asarray(value, dtype='f8').tolist() for name,value in
            (('physical_atomic_proxy', proxy), ('actual_BB_gap', gap),
             ('fallback_restoration', restoration), ('same_SL_photon', same),
             ('ideal_SL_BB_minus_photon', ideal), ('actual_native_replaced_proxy', proxy + gap),
             ('ideal_native_replaced_proxy', proxy + ideal))},
        not_actual_native_EHB=True, source_model_changed=False, new_BF_BB_integrals=0,
        new_SE_solves=0, new_transport=0, final_goal='NOT_MET')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--bb', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(check(a.result, a.sha256, a.bb), ensure_ascii=False, indent=2, allow_nan=False))
