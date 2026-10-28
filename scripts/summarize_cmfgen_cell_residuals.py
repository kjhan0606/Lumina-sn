#!/usr/bin/env python3
"""저장 셀 분해를 원 SC 계수와 대조한다. 새 광선 적분/물질 갱신은 없다."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_cmfgen_fixed_blue_transport import load_packet
from check_cmfgen_local_sc_packet import extended_material, local_sc
from prepare_cmfgen_mom_capture import sha


def endpoint_source(chi, eta, b, blue, alpha, up, target, length):
    """s-선형 계수의 도착점 S와 첫 optically-thick 전개항(유효 tau)을 구한다."""
    effective = chi+(alpha+3.)*b
    emission = eta+alpha*b*blue
    kt, ku = effective[target], effective[up]
    if length <= 0 or kt <= 0 or ku <= 0:
        raise ValueError('양의 유효 소멸계수/길이 필요')
    source = emission[target]/kt
    derivative_emission = eta[target]-eta[up]+alpha*(
        (b[target]-b[up])*blue[target]+b[target]*(blue[target]-blue[up]))
    derivative_source_x = (derivative_emission-source*(kt-ku))/kt
    return dict(endpoint_source=float(source),
        upstream_source=float(emission[up]/ku),
        material_chi_upstream=float(chi[up]), material_chi_target=float(chi[target]),
        effective_opacity_upstream=float(ku), effective_opacity_target=float(kt),
        first_order_intensity=float(source-derivative_source_x/(kt*length)),
        tau_chi=float(.5*length*(chi[target]+chi[up])),
        tau_frequency_drift=float(.5*length*(alpha+3.)*(b[target]+b[up])),
        blue_emission_fraction=float(alpha*b[target]*blue[target]/emission[target]))


def summarize(analysis, out):
    if out.exists(): raise FileExistsError('기존 source 형태 분석 보존')
    d = json.loads(analysis.read_text())
    if d['status'] != 'FIXED_ETA_715_CELL_INJECTION_DIAGNOSTIC':
        raise ValueError('완성 셀 분해 증거 필요')
    paths = {Path(p): digest for p, digest in d['verified_packet_input_sha256'].items()}
    for path, digest in paths.items():
        if sha(path) != digest: raise ValueError('기존 포획 자료 변경: '+str(path))
    run = next(p.parent for p in paths if p.name == 'FORMAL_44360_02_in.bin')
    build = next(p.parent for p in paths if p.name == 'formal-build.json')
    entry, _, g, blue, actual, _, _ = load_packet(run, build, None, 'FORMAL_44360_02')
    a = entry['args']; chi, eta = extended_material(a, g); alpha = 1/a['dLOG_NU']
    node = d['target_native_node_zero_based']; rows = []
    for side in ('P', 'M'):
        adjacent = [x for x in d['cell_terms'] if x['direction'] == side and
            x['cell_index_zero_based'] == int(g['rays'][x['ray_1based']-1]['LNK'][node])-1-
            (1 if side == 'M' else 0)]
        for term in sorted(adjacent, key=lambda x: abs(x['weighted_delta_J']), reverse=True)[:5]:
            ip = term['ray_1based']-1; ray = g['rays'][ip]
            k = int(ray['LNK'][node]-1); n = len(ray['R'])
            if not ((2 <= k <= n-2) if side == 'P' else (1 <= k <= n-2)):
                continue
            up = k+1 if side == 'P' else k-1
            length = abs(float(ray['S_'+side][k]-ray['S_'+side][up]))
            state = actual['rays'][ip]['I_'+side]
            previous = blue['rays'][ip]['I_'+side]
            endpoint = endpoint_source(chi[:n], eta[:n], ray['B_'+side], previous,
                                       alpha, up, k, length)
            replay = local_sc(chi, eta, previous, ray, state, k, a['dLOG_NU'], side)
            delta = math.fsum(x['propagated_delta_I'] for x in d['cell_terms']
                if x['ray_1based'] == ip+1 and x['direction'] == side)
            delta += math.fsum(x['boundary_contribution'] for x in d['boundary_terms']
                if x['ray_1based'] == ip+1 and x['direction'] == side)
            tau = endpoint['tau_chi']+endpoint['tau_frequency_drift']
            if not np.isclose(tau, term['tau'], rtol=3.e-13, atol=0.):
                raise ValueError('원 SC와 선택 전달률의 셀 tau 불일치')
            if not np.isclose(replay, state[k], rtol=1.e-10, atol=0.):
                raise ValueError('원 SC 국소식이 저장 강도를 재현하지 못함')
            source = endpoint['endpoint_source']
            rows.append(dict(ray_1based=ip+1, direction=side, radial_class=term['radial_class'],
                effective_tau=tau, formal_intensity=float(state[k]),
                new_intensity_reconstructed=float(state[k]+delta),
                formal_replay_relative_error=float((replay-state[k])/state[k]),
                formal_minus_endpoint_over_endpoint=float((state[k]-source)/source),
                new_minus_endpoint_over_endpoint=float((state[k]+delta-source)/source), **endpoint))
    if not rows: raise ValueError('국소 SC와 대조할 내부 셀 표본이 없음')
    report = dict(status='READ_ONLY_CELL_SOURCE_SHAPE', analysis_sha256=sha(analysis),
        code_sha256=sha(Path(__file__)), samples=rows,
        interpretation='같은 셀 전달률에서 적분 구동항의 재구성 차이이며 국소 원자 η 결함의 증명이 아님',
        new_ray_integrations=0, new_material_updates=0, final_goal='NOT_MET')
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with out.open('x') as stream:
        stream.write(encoded)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.analysis.resolve(), args.out.resolve())
