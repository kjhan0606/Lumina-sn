#!/usr/bin/env python3
"""보존 Jbar·인구의 IR 열 기여와 선택 native BB 순율을 대조한다. SE/수송 없음."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_cmfgen_fixedj_full import sha, read_matrix
from expand_atomic_data_cmfgen import load_cmfgen_model_spec
from lumina_model_energy import ModelThermalEnergy
from cmfgen_ion_identity import parse_cmfgen_ion_id
from extract_cmfgen_line_net_fixture import netrate_header, vectors


def add_bb_flow(target, lower, upper, downward):
    """이온 내부 전이의 유입−유출. 같은 SL 안의 전이는 풀 변수에 작용하지 않는다."""
    if lower != upper:
        target[lower] += downward
        target[upper] -= downward


def compare_native_bb_flows(path, lines, jbar, mass, pop, fraction, columns, rows, group, excluded,
                            full_population=None):
    targets = {(14, 2), (14, 3), (16, 2), (16, 3)}
    z = np.array([int(r['atomic_number']) for r in rows])
    stage = np.array([int(r['ion_number'])+1 for r in rows])
    full = np.array([int(r['level_number'])+1 for r in rows])
    keys = {}
    for i, (lo, up) in enumerate(zip(lines['lower'], lines['upper'])):
        if (z[lo], stage[lo]) in targets and not excluded[i]:
            key = (int(z[lo]), int(stage[lo]), int(full[lo]), int(full[up]))
            if key in keys:
                raise ValueError('중복 native BB 주소')
            keys[key] = i
    shape = pop.shape
    native, fine, gross, rounding = (np.zeros(shape, dtype=np.longdouble) for _ in range(4))
    seen = set()
    same_sl = 0
    def selected_header(line):
        header = netrate_header(line)
        if header is not None:
            raw = header.transition.split('(', 1)[0]
            if raw in ('Sk2', 'SkIII', 'S2', 'SIII', 'Si2', 'SiIII'):
                return header
        return None
    for header, values in vectors(path, pop.shape[1], selected_header):
        atomic, ion, _ = parse_cmfgen_ion_id(header.transition.split('(', 1)[0])
        key = (atomic, ion, header.lower_full, header.upper_full)
        if key not in keys or key in seen:
            raise ValueError('NETRATE/선 catalogue 주소 누락·중복')
        seen.add(key)
        i = keys[key]
        lo, up = lines['lower'][i], lines['upper'][i]
        lc, uc = int(group[lo]), int(group[up])
        nu, a, ratio = (lines[k][i] for k in ('nu0', 'A', 'g_ratio'))
        if abs(nu-header.frequency_field*1.e15) > 5.01e8:
            raise ValueError('native 출력 반올림보다 큰 전이 주파수 차이')
        lower = np.asarray(pop[lc]*fraction[lo] if full_population is None else full_population[lo], dtype=np.longdouble)
        upper = np.asarray(pop[uc]*fraction[up] if full_population is None else full_population[up], dtype=np.longdouble)
        bul = a*B.C**2/(2*B.H*nu**3)
        upward = lower*(bul/ratio)*jbar[i]
        downward = upper*(a*mass[i]+bul*jbar[i])
        znet = np.asarray(values, dtype=np.longdouble)
        if not np.isfinite(znet).all():
            raise ValueError('비유한 native ZNET')
        native_down = upper*a*znet
        add_bb_flow(native, lc, uc, native_down)
        add_bb_flow(fine, lc, uc, downward-upward)
        if lc != uc:
            gross[lc] += downward+upward
            gross[uc] += downward+upward
            nonzero = znet != 0
            quantum = np.zeros_like(znet)
            quantum[nonzero] = .5*10.**(np.floor(np.log10(abs(znet[nonzero])))-6)
            uncertainty = upper*a*quantum
            rounding[lc] += uncertainty
            rounding[uc] += uncertainty
        else:
            same_sl += 1
    if seen != set(keys):
        raise ValueError('일부 Si/S native 전이를 대조하지 못함')
    result = {}
    for atomic, ion in sorted(targets):
        ids = np.flatnonzero((columns[:,0] == atomic) & (columns[:,1] == ion-1))
        if not len(ids):
            raise ValueError('대상 이온 SL 누락')
        delta = native[ids]-fine[ids]
        l1 = np.sum(abs(delta), axis=0)
        fine_l1 = np.sum(abs(fine[ids]), axis=0)
        gross_sum = gross[ids].sum(axis=0)
        relative = np.divide(l1, gross_sum, out=np.zeros_like(l1), where=gross_sum>0)
        worst = np.unravel_index(np.argmax(abs(delta)), delta.shape)
        conserved = np.abs(delta.sum(axis=0))
        result[f'Z{atomic}_stage{ion}'] = dict(
            matched_lines=sum(k[:2] == (atomic, ion) for k in keys),
            delta_over_BB_gross_median=float(np.median(relative)),
            delta_over_BB_gross_max=float(relative.max()),
            max_absolute_row_delta=float(abs(delta[worst])),
            max_absolute_row_depth=int(worst[1])+1,
            max_absolute_row_superlevel=int(columns[ids[worst[0]],2])+1,
            number_conservation_relative=float(np.max(np.divide(conserved,gross_sum,
                out=np.zeros_like(conserved),where=gross_sum>0))),
            depths=[dict(depth=d+1,L1_delta=float(l1[d]),fine_net_L1=float(fine_l1[d]),
                native_print_L1_bound=float(rounding[ids,d].sum()),
                delta_over_BB_gross=float(relative[d])) for d in (0,35,51,84,89)])
    return dict(scope='공통 A·인구에 native 출력 ZNET와 fine Jbar를 대입한 BB 블록 차이; native 전체 SE 잔차 아님',
                population='native full snapshot' if full_population is not None else 'Lumina response',
                current_units='cm^-3 s^-1',matched_lines=len(seen),same_superlevel_lines=same_sl,
                by_ion=result,ionization_solved=False)


def summarize(sources, response_root, *, netrate=None, expected_netrate_sha256=None,
              native_comparison=None):
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
    native_flows = None
    if netrate is not None:
        if expected_netrate_sha256 is None:
            raise ValueError('NETRATE 기대 hash가 필요함')
        bind(netrate, expected_netrate_sha256)
        if native_comparison is None:
            raise ValueError('동일-state native comparison 신원이 필요함')
        bind(native_comparison, sha(native_comparison))
        evidence = json.loads(native_comparison.read_text())['input_sha256']
        if evidence['accepted_Lumina_response']['sha256'] != selected['response_sha256']:
            raise ValueError('native snapshot과 선택 응답 신원이 다름')
        native_full = read_matrix(bind(netrate.parent/'NATIVE_FULL_POP.bin',
                                      evidence['native_full_population']['sha256']))
        index_path = bind(netrate.parent/'NATIVE_FULL_INDEX.csv',evidence['native_index']['sha256'])
        lookup = {(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
                  for i,r in enumerate(rows)}
        aligned = np.empty_like(fraction)
        mapped = set()
        native_rows = list(csv.DictReader(index_path.open()))
        if len(native_rows) != len(rows) or native_full.shape != fraction.shape:
            raise ValueError('native full snapshot 형상 불일치')
        for i, row in enumerate(native_rows):
            atomic, ion, _ = parse_cmfgen_ion_id(row['ion_label'])
            if atomic != int(row['Z']):
                raise ValueError('native index의 원소 라벨/번호 불일치')
            target = lookup[atomic,ion,int(row['full_level'])]
            if target in mapped or int(row['superlevel']) != columns[group[target],2]+1:
                raise ValueError('native full-level 대응 중복/SL 불일치')
            aligned[target] = native_full[i]
            mapped.add(target)
        native_flows = compare_native_bb_flows(netrate, lines, jbar, mass, pop, fraction,
                                              columns, rows, group, excluded, aligned)
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
        native_BB_flow_comparison=native_flows,
        final_goal='NOT_MET')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--response-root', type=Path, required=True)
    parser.add_argument('--netrate', type=Path)
    parser.add_argument('--expected-netrate-sha256')
    parser.add_argument('--native-comparison', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.sources, args.response_root,netrate=args.netrate,
        expected_netrate_sha256=args.expected_netrate_sha256,
        native_comparison=args.native_comparison), ensure_ascii=False, indent=2))
