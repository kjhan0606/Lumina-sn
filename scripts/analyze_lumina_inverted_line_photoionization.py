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


def analyze(root, levels_path, BB_root=None, original_full=None):
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
        snapshot = {k: archive[k] for k in ('columns', 'superlevel', 'fraction')}
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
    result.update(baseline='원70 물질·원70 복사장, 고정온도 미수렴; 새 자체장 아님',
        rate_scope='직접 광이온화 n_lower × 경로확률 × 총 BF photo율; 간접 BB 연쇄 미포함',
        input_sha256=identities,
        new_integrals=0, new_transport=0, final_goal='NOT_JUDGED')
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--levels', dest='levels_path', type=Path, required=True)
    p.add_argument('--BB-root', type=Path); p.add_argument('--original-full', type=Path)
    return p.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(analyze(**vars(parse_args())), ensure_ascii=False, indent=2))
