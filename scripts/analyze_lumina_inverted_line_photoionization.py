#!/usr/bin/env python3
"""원70 동일 물질/복사장에서 Si II 역전 준위의 직접 BF 기여를 집계한다.

저장된 총 BF율은 bound와 dissolution을 포함한다. 새 복사장 수송이나
준위 간 간접 연쇄의 크기를 계산하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_material_inventory import sha
from analyze_cmfgen_signed_sources import read_full
import cmfgen_model_continuum as B
from lumina_model_collision_source import ModelCollisionSource, ex_e1x
import build_cmfgen_fullpair_omega as legacy
from build_cmfgen_fullpair_omega import same_n_key


def weighted_photoionization(rows, columns, populations, fractions, roster, photo_rates,
                             levels=(123, 124), node=54):
    """실제 lower 주소를 기준으로 Si II 전 경로와 지정 FL의 n_l P Γ를 비교한다."""
    lower = np.asarray(roster['lower'], dtype='i8')
    selected = np.flatnonzero((roster['Z'] == 14) & (roster['stage'] == 2))
    if not len(selected) or any(int(rows[lower[i]]['atomic_number']) != 14 or
                                int(rows[lower[i]]['ion_number']) != 1 for i in selected):
        raise ValueError('Si II lower 원자 주소 불일치')
    lookup = {tuple(map(int, row)): i for i, row in enumerate(columns)}
    if len(lookup) != len(columns) or populations.ndim != 2 or not 0 <= node < populations.shape[1]:
        raise ValueError('SL 주소 또는 반경점 오류')
    actual_levels = [i for i, row in enumerate(rows) if int(row['atomic_number']) == 14 and
                     int(row['ion_number']) == 1 and int(row['level_number']) in levels]
    if len(actual_levels) != len(levels): raise ValueError('대상 Si II FL 주소 누락/중복')
    sl = np.array([lookup[(int(rows[i]['atomic_number']), int(rows[i]['ion_number']),
                           int(rows[i]['super_level']))] for i in lower[selected]])
    full = populations[sl]*fractions[lower[selected]]
    flow = full*roster['probability'][selected, None]*photo_rates[selected]
    if (not np.isfinite(flow).all() or np.any(flow < 0) or
            np.any(roster['probability'][selected] < 0)):
        raise ValueError('BF 인구/확률/광이온화율 오류')
    total = flow.sum(axis=0)
    if np.any(total <= 0): raise ValueError('비교할 양의 Si II 총 광이온화 없음')
    parts = []
    for level in levels:
        indices = [i for i in actual_levels if int(rows[i]['level_number']) == level]
        mask = lower[selected] == indices[0]; value = flow[mask].sum(axis=0)
        parts.append(dict(FL_1based=level+1, routes=int(mask.sum()),
            node_rate_cm3_s=float(value[node]), node_fraction=float(value[node]/total[node]),
            threshold_Hz=roster['threshold'][selected[mask]].tolist()))
    mask = np.isin(lower[selected], actual_levels); joint = flow[mask].sum(axis=0)/total
    summed = {}
    for i, route in enumerate(selected):
        key = int(lower[route]); summed[key] = summed.get(key, 0.)+float(flow[i, node])
    leaders = sorted(summed.items(), key=lambda item: -item[1])[:8]
    return dict(node=node, SiII_routes=len(selected), total_rate_cm3_s=float(total[node]),
        levels=parts, combined_node_fraction=float(joint[node]),
        combined_max_fraction=float(joint.max()), combined_max_node=int(joint.argmax()),
        combined_median_fraction=float(np.median(joint)),
        dominant_levels=[dict(FL_1based=int(rows[i]['level_number'])+1,
                              configuration=rows[i]['configuration'], fraction=float(value/total[node]))
                         for i, value in leaders])


def radiative_line_flow(lower, upper, jbar, frequency, einstein_A, g_ratio):
    """흡수, 유도방출, 자발방출, 상향 순 전이수. 값은 같은 점의 cm^-3 s^-1."""
    lower, upper, jbar = (np.asarray(v, dtype='f8') for v in (lower, upper, jbar))
    if (lower.shape != upper.shape or lower.shape != jbar.shape or lower.ndim != 1 or
            not all(np.isfinite(v).all() and np.all(v >= 0) for v in (lower, upper, jbar)) or
            not np.isfinite([frequency, einstein_A, g_ratio]).all() or
            frequency <= 0 or einstein_A < 0 or g_ratio <= 0):
        raise ValueError('선 인구/복사장/Einstein 계수 오류')
    bul = einstein_A*B.C**2/(2*B.H*frequency**3)
    absorption = lower*bul/g_ratio*jbar
    stimulated = upper*bul*jbar
    spontaneous = upper*einstein_A
    return dict(absorption=absorption, stimulated=stimulated, spontaneous=spontaneous,
                net_up=absorption-stimulated-spontaneous)


def collision_pair_flow(lower, upper, temperature, electron_density, omega, g_lower, g_upper,
                        gap_eV, w_lower, w_upper, dissolution):
    """기존 전모형 C collision_pair와 같은 점유율·상세균형을 사용한다."""
    values = (lower, upper, temperature, electron_density, omega, g_lower, g_upper, gap_eV, w_lower, w_upper)
    if (not np.isfinite(values).all() or lower < 0 or upper < 0 or temperature <= 0 or
            electron_density <= 0 or omega <= 0 or g_lower <= 0 or g_upper <= 0 or gap_eV < 0 or
            not 0 < w_upper <= w_lower <= 1 or type(dissolution) is not bool):
        raise ValueError('충돌 쌍 물질/원자/점유확률 오류')
    base = 8.63e-6*electron_density/np.sqrt(temperature)*omega
    excitation = base*np.exp(-gap_eV/(8.617333262145e-5*temperature))/g_lower
    up = lower*excitation*(w_upper/w_lower); down = upper*base/g_upper
    lost = lower*excitation*(1.-w_upper/w_lower) if dissolution else 0.
    return dict(up_cm3_s=float(up), down_cm3_s=float(down), net_up_cm3_s=float(up-down),
                dissolution_forward_cm3_s=float(lost))


def compare_same_n_branch(lower_name, upper_name, gap_eV, temperature, omega):
    """선택된 SAME_N 분기와 가상의 다른 분기를 비교한다. 생산 물리는 바꾸지 않는다."""
    if not np.isfinite([gap_eV, temperature, omega]).all() or gap_eV <= 0 or temperature <= 0 or omega <= 0:
        raise ValueError('SAME_N 비교 온도/에너지/충돌강도 오류')
    lower_key, upper_key = same_n_key(lower_name), same_n_key(upper_name)
    same = lower_key is not None and lower_key == upper_key
    x = gap_eV/(8.617333262145e-5*temperature)
    g2 = .276*float(ex_e1x(np.array([x]))[0])
    current = max(.7 if same else .2, g2); alternative = max(.2 if same else .7, g2)
    return dict(lower_name=lower_name, upper_name=upper_name, lower_key=lower_key, upper_key=upper_key,
        same_n=same, gap_eV=float(gap_eV), x=float(x), g2=g2, selected_gbar=current,
        alternate_branch_gbar=alternative, alternate_to_selected_ratio=alternative/current,
        alternate_branch_omega=float(omega*alternative/current),
        scope='한 쌍의 가상 SAME_N 분기만 변경한 대수 비교; 전체 SE/물리 타당성 아님')


def superlevel_line_network(rows, full, lines, jbar, status, groups, target_line, node):
    """각 SL 경계를 넘는 실제 방사선의 유입/유출. 같은 SL 내부선은 제외한다."""
    output = []
    for group in groups:
        members = np.array([i for i, row in enumerate(rows) if int(row['atomic_number']) == 14 and
                            int(row['ion_number']) == 1 and int(row['super_level']) == group])
        if not len(members): raise ValueError('Si II 슈퍼준위 누락')
        lo_in = np.isin(lines['lower'], members); up_in = np.isin(lines['upper'], members)
        chosen = np.flatnonzero(lo_in ^ up_in)
        if not len(chosen) or not np.isin(status[chosen], [1, 2]).all():
            raise ValueError('슈퍼준위 선 복사장 범위 결손')
        target = np.flatnonzero(chosen == target_line)
        if len(target) != 1: raise ValueError('역전선이 두 슈퍼준위를 연결하지 않음')
        lower, upper = lines['lower'][chosen], lines['upper'][chosen]
        nu, A, ratio = (lines[name][chosen] for name in ('nu0', 'A', 'g_ratio'))
        J = np.asarray(jbar[chosen, node]); bul = A*B.C**2/(2*B.H*nu**3)
        up = full[lower, node]*bul/ratio*J
        down = full[upper, node]*(A+bul*J)
        if not np.isfinite([up, down]).all() or np.any(up < 0) or np.any(down < 0):
            raise ValueError('슈퍼준위 선 전이 비유한/음수')
        incoming = np.where(lo_in[chosen], down, up); outgoing = np.where(lo_in[chosen], up, down)
        net = incoming-outgoing; k = int(target[0]); total_in, total_out = float(incoming.sum()), float(outgoing.sum())
        ranking = np.argsort(-abs(net))[:5]
        output.append(dict(superlevel_zero_based=group, full_levels=len(members), cross_lines=len(chosen),
            incoming_cm3_s=total_in, outgoing_cm3_s=total_out, net_cm3_s=float(net.sum()),
            inverted_line_net_cm3_s=float(net[k]),
            inverted_line_incoming_fraction=float(incoming[k]/total_in) if total_in else None,
            inverted_line_outgoing_fraction=float(outgoing[k]/total_out) if total_out else None,
            largest_net_lines=[dict(lower_FL_1based=int(rows[int(lower[i])]['level_number'])+1,
                upper_FL_1based=int(rows[int(upper[i])]['level_number'])+1,
                wavelength_A=float(B.C/nu[i]*1.e8), net_into_superlevel_cm3_s=float(net[i]))
                for i in ranking]))
    return output


def analyze(root, levels_path, BB_root=None, original_full=None, collision_run=None, collision_reference=None,
            compare_native_precision=False):
    root, levels_path = Path(root).resolve(), Path(levels_path).resolve()
    report_path = root/'result.json'; report = json.loads(report_path.read_text())
    if report['state'] != 'COMPLETE_FROZEN_MATERIAL_BF_90': raise ValueError('완료 원70 BF만 사용')
    names = ('material-snapshot.npz', 'route-roster.npz', 'total-rates.npy')
    identities = {str(report_path): sha(report_path), str(levels_path): sha(levels_path)}
    for name in names:
        digest = sha(root/name)
        if digest != report['output_sha256'][name]: raise ValueError('원70 BF 결과 변경: '+name)
        identities[str(root/name)] = digest
    expected_levels = [digest for path, digest in report['data_sha256'].items() if path.endswith('/atomic/levels.csv')]
    if len(set(expected_levels)) != 1 or identities[str(levels_path)] != expected_levels[0]:
        raise ValueError('원70 BF와 다른 원자 준위 주소')
    rows = list(csv.DictReader(levels_path.open()))
    with np.load(root/'material-snapshot.npz', allow_pickle=False) as archive:
        snapshot = {k: archive[k] for k in ('columns', 'superlevel', 'fraction', 'temperature', 'ne', 'occupation')}
    with np.load(root/'route-roster.npz', allow_pickle=False) as archive:
        roster = {k: archive[k] for k in archive.files}
    rates = np.load(root/'total-rates.npy', mmap_mode='r', allow_pickle=False)
    try:
        result = weighted_photoionization(rows, snapshot['columns'], snapshot['superlevel'],
                                         snapshot['fraction'], roster, rates[:, :, 0])
    finally: rates._mmap.close()
    if (BB_root is None) != (original_full is None):
        raise ValueError('원70 BB 자료와 실제 C 복원 인구를 함께 지정해야 함')
    if BB_root is not None:
        bb = Path(BB_root).resolve(); original_full = Path(original_full).resolve()
        bb_report_path = bb/'result.json'; bb_report = json.loads(bb_report_path.read_text())
        if bb_report['state'] != 'COMPLETE_LINEAR_FIELD_BB90': raise ValueError('완료 원70 BB 필요')
        for name in ('line-roster.npz', 'jbar.npy', 'status.npy'):
            digest = sha(bb/name)
            if digest != bb_report['output_sha256'][name]: raise ValueError('원70 BB 출력 변경')
            identities[str(bb/name)] = digest
        field_key = [k for k in report['data_sha256'] if k.endswith('/FORMAL_PRIMARY_FIELDS.bin')]
        if len(field_key) != 1 or bb_report['input_sha256'].get(field_key[0]) != report['data_sha256'][field_key[0]]:
            raise ValueError('BF와 BB의 원70 복사장 불일치')
        full = read_full(original_full)
        identities[str(bb_report_path)] = sha(bb_report_path)
        identities[str(original_full)] = sha(original_full)
        if full.shape != snapshot['fraction'].shape: raise ValueError('원70 C full 인구 형상 오류')
        target = [i for i, row in enumerate(rows) if int(row['atomic_number']) == 14 and
                  int(row['ion_number']) == 1 and int(row['level_number']) in (123, 124)]
        if len(target) != 2: raise ValueError('Si II 역전 선 FL 주소 오류')
        target.sort(key=lambda i: int(rows[i]['level_number']))
        lookup = {tuple(map(int, row)): i for i, row in enumerate(snapshot['columns'])}
        for level in target:
            row = rows[level]; col = lookup[(int(row['atomic_number']), int(row['ion_number']), int(row['super_level']))]
            reconstructed = snapshot['superlevel'][col]*snapshot['fraction'][level]
            scale = np.maximum(abs(reconstructed), abs(full[level]))
            difference = np.divide(abs(reconstructed-full[level]), scale, out=np.zeros_like(scale), where=scale>0)
            if np.max(difference) > 1.e-12: raise ValueError('BF 물질과 원70 C full 인구 불일치')
        with np.load(bb/'line-roster.npz', allow_pickle=False) as data:
            lines = {k: data[k] for k in data.files}
        matches = np.flatnonzero((lines['lower'] == target[0]) & (lines['upper'] == target[1]))
        if len(matches) != 1: raise ValueError('Si II 역전 선 주소 누락/중복')
        index = int(matches[0]); status = np.load(bb/'status.npy', allow_pickle=False)
        if status[index] not in (1, 2): raise ValueError('역전 선의 원70 복사장 범위 결손')
        jbar = np.load(bb/'jbar.npy', mmap_mode='r', allow_pickle=False)
        try:
            J = np.array(jbar[index], copy=True)
            groups = [int(rows[level]['super_level']) for level in target]
            if len(set(groups)) != 2: raise ValueError('역전선은 서로 다른 두 SL을 연결해야 함')
            # 각 경계선의 실제 하위/상위 C 인구가 BF snapshot과 같은지 확인한다.
            membership = np.array([int(row['atomic_number']) == 14 and int(row['ion_number']) == 1 and
                                   int(row['super_level']) in groups for row in rows])
            cross = membership[lines['lower']] | membership[lines['upper']]
            involved = np.unique(np.r_[lines['lower'][cross], lines['upper'][cross]])
            for level in involved:
                row = rows[level]; col = lookup[(int(row['atomic_number']), int(row['ion_number']), int(row['super_level']))]
                actual = snapshot['superlevel'][col, result['node']]*snapshot['fraction'][level, result['node']]
                expected = full[level, result['node']]; scale = max(abs(actual), abs(expected))
                if scale > 0 and abs(actual-expected) > 1.e-12*scale:
                    raise ValueError('슈퍼준위 선망의 C/BF 인구 불일치')
            network = superlevel_line_network(rows, full, lines, jbar, status, groups, index, result['node'])
        finally: jbar._mmap.close()
        nu, A, ratio = (float(lines[name][index]) for name in ('nu0', 'A', 'g_ratio'))
        flow = radiative_line_flow(full[target[0]], full[target[1]], J, nu, A, ratio)
        node = result['node']
        selected_direct = sum(v['node_rate_cm3_s'] for v in result['levels'])
        result['inverted_line'] = dict(FL_1based=[124, 125], line_index=index, frequency_Hz=nu,
            oscillator_strength=float(lines['f'][index]), einstein_A_s=A, g_lower_over_upper=ratio,
            Jbar_node=float(J[node]), rates_cm3_s={k: float(v[node]) for k, v in flow.items()},
            net_down_to_selected_direct_BF_ratio=(float(-flow['net_up'][node]/selected_direct)
                                                   if selected_direct > 0 else None),
            negative_net_up_nodes=int(np.count_nonzero(flow['net_up'] < 0)),
            interpretation='같은 이온 내부의 순 BB 전이; 전하를 직접 바꾸는 이온화율이 아님')
        result['superlevel_radiative_network'] = network
        if (collision_run is None) != (collision_reference is None):
            raise ValueError('충돌 원자료 경로와 이전 응답 기록을 함께 지정해야 함')
        if collision_run is not None:
            collision_run, collision_reference = Path(collision_run).resolve(), Path(collision_reference).resolve()
            collision_report = json.loads(collision_reference.read_text())
            if (collision_report['state'] != 'COMPLETE_PRIVATE_FIXED_FIELD_ALL_MODEL_RESPONSE' or
                    collision_report['input_sha256'].get(str(report_path)) != identities[str(report_path)] or
                    collision_report['input_sha256'].get(str(bb_report_path)) != identities[str(bb_report_path)]):
                raise ValueError('충돌 참조의 원70 BF/BB 계보 불일치')
            source = ModelCollisionSource(collision_run, rows); ion = source.load(14, 2)
            if any(collision_report['collision_input_sha256'].get(k) != v for k, v in ion.identity.items()):
                raise ValueError('원70 응답과 다른 Si II 충돌 자료')
            wanted = ((10, 125), (11, 125), (115, 124), (124, 125))
            row_number = {int(row['level_number'])+1: i for i, row in enumerate(rows)
                          if int(row['atomic_number']) == 14 and int(row['ion_number']) == 1}
            pairs = {(row_number[a], row_number[b]): (a, b) for a, b in wanted}
            node = result['node']; temperature = float(snapshot['temperature'][node]); ne = float(snapshot['ne'][node])
            dissolution = B.dissolution_flag(collision_run/'MODEL'); found = {}
            for lower, upper, omega, _ in ion.blocks([temperature]):
                for (lo, up), label in pairs.items():
                    match = np.flatnonzero((lower == lo) & (upper == up))
                    if len(match) > 1 or (len(match) and label in found): raise ValueError('충돌쌍 중복')
                    if not len(match): continue
                    strength = float(omega[0, match[0]])
                    key = (lo-ion.offset)*ion.n+(up-ion.offset)
                    method = 'TABULATED' if key in ion.table_keys else 'VR' if key in ion.vr_keys else 'OMEGA_SET'
                    flow = collision_pair_flow(float(full[lo, node]), float(full[up, node]), temperature, ne,
                        strength, float(rows[lo]['g']), float(rows[up]['g']),
                        float(rows[up]['energy_eV'])-float(rows[lo]['energy_eV']),
                        float(snapshot['occupation'][lo, node]), float(snapshot['occupation'][up, node]), dissolution)
                    found[label] = dict(FL_1based=list(label), method=method, omega=strength, **flow)
            ion.verify_identity()
            if set(found) != set(wanted): raise ValueError('선택한 Si II 충돌쌍 누락')
            result['collisional_pairs'] = [found[label] for label in wanted]
            lo, up = (row_number[124], row_number[125])
            result['same_n_branch_comparison'] = compare_same_n_branch(
                rows[lo]['configuration'], rows[up]['configuration'],
                float(rows[up]['energy_eV'])-float(rows[lo]['energy_eV']),
                temperature, found[124, 125]['omega'])
            if compare_native_precision:
                ids = source.groups[14, 2]
                csv_values = legacy.ion_strengths(14, 1, 'Si II', None, ids, rows, source.models,
                                                  source.links, temperature, native_precision=False)
                edge_values = legacy.ion_strengths(14, 1, 'Si II', None, ids, rows, source.models,
                                                   source.links, temperature, native_precision=True)
                if not all(np.array_equal(csv_values[i], edge_values[i]) for i in (0, 1)):
                    raise ValueError('CSV/FEDGE 충돌쌍 순서 불일치')
                comparison = []
                for a, b in wanted:
                    loc = np.flatnonzero((csv_values[0] == a-1) & (csv_values[1] == b-1))
                    if len(loc) != 1: raise ValueError('CSV/FEDGE 대상 충돌쌍 누락')
                    current, edge = (float(value[2][loc[0]]) for value in (csv_values, edge_values))
                    if abs(current-found[a, b]['omega']) > 5.e-12*current:
                        raise ValueError('동적 생산기와 기존 CSV 생산기 불일치')
                    comparison.append(dict(FL_1based=[a, b], csv_omega=current, native_edge_omega=edge,
                                           relative_difference=edge/current-1.))
                result['native_edge_precision_comparison'] = dict(pairs=comparison,
                    csv_branches=csv_values[3]['branches'], native_branches=edge_values[3]['branches'],
                    interpretation='원 OSC FEDGE 연산의 별도 Python 예측; native 전체 실행 아님')
            result['collision_scope'] = '원70 고정 물질에서 선택한 네 쌍만; 전체 충돌망/이온화 민감도 아님'
            result['collision_temperature_K'] = temperature; result['collision_ne_cm3'] = ne
            identities.update(ion.identity); identities[str(collision_reference)] = sha(collision_reference)
    elif collision_run is not None or collision_reference is not None:
        raise ValueError('충돌 비교에는 같은 원70 BB 자료와 C 인구가 필요')
    if compare_native_precision and 'native_edge_precision_comparison' not in result:
        raise ValueError('원자료 정밀도 비교에는 같은 원70 충돌 자료가 필요')
    result.update(baseline='원70 물질·원70 복사장, 고정온도 미수렴; 새 자체장 아님',
        rate_scope='직접 광이온화 n_lower × 경로확률 × 총 BF photo율; 간접 BB 연쇄 미포함',
        input_sha256=identities,
        new_integrals=0, new_transport=0, final_goal='NOT_JUDGED')
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--levels', dest='levels_path', type=Path, required=True)
    p.add_argument('--BB-root', type=Path); p.add_argument('--original-full', type=Path)
    p.add_argument('--collision-run', type=Path); p.add_argument('--collision-reference', type=Path)
    p.add_argument('--compare-native-precision', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(analyze(**vars(parse_args())), ensure_ascii=False, indent=2))
