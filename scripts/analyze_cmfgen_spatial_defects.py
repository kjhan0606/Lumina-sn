#!/usr/bin/env python3
"""고정-blue 공간 차이를 원 셀의 국소 주입과 경계 전달로 분해한다.

같은 선형 물질의 전달률로 차이 e의 e_down-T*e_up를 추적한다. 이는 원 SC와
명시적 연속물질 사이의 재구성 차이 위치이지 참해 오차나 빠진 물리율의 귀속이 아니다.
"""
import argparse
import csv
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from compare_cmfgen_fixed_blue_transport import (load_packet, build_kernel, transport,
    extended_material, angular_integral, FixedBlueKernel)
from check_cmfgen_local_sc_packet import local_sc
from compare_cmfgen_packet_cell_material import linear_material_reference
from prepare_cmfgen_mom_capture import sha


class QuadraticBlueKernel(FixedBlueKernel):
    """분할 없이 같은 선형 물질의 이차 유효 방출항을 적분하는 소형 진단."""
    def __init__(self, library):
        super().__init__(library)
        function = self.library.lumina_fixed_blue_quadratic_leg
        function.argtypes = self.function.argtypes
        function.restype = ct.c_int
        self.function = function

    def leg(self, length, chi, eta, b, blue, alpha, incident, pieces=1):
        if pieces != 1: raise ValueError('이차 source 진단은 원 셀당 직접 적분1회다')
        return super().leg(length, chi, eta, b, blue, alpha, incident, pieces)


def defect_terms(error, transmission):
    """방향을 따라 정렬된 e[0..n]의 각 셀 주입을 마지막 점까지 전달한다."""
    error = np.asarray(error, dtype='f8'); transmission = np.asarray(transmission, dtype='f8')
    if (error.ndim != 1 or transmission.shape != (len(error)-1,) or
            not len(error) or not np.isfinite(error).all() or
            not np.isfinite(transmission).all() or np.any(transmission < 0)):
        raise ValueError('유한 차이·비음수 전달률의 같은 경로 필요')
    result = np.empty(len(transmission)); factor = 1.
    for i in range(len(transmission)-1, -1, -1):
        result[i] = factor*(error[i+1]-transmission[i]*error[i])
        factor *= transmission[i]
    boundary = factor*error[0]
    reconstructed = math.fsum([boundary, *result])
    scale = math.fsum([abs(boundary), *abs(result)])
    closure = abs(reconstructed-error[-1])
    return result, float(boundary), float(closure/scale if scale else closure)


def probe_cell(kernel, args, g, blue, actual, chi, eta, ip, j, side, pieces):
    """한 지배 셀에서 같은 실제 상류 강도로 원 SC와 독립 연속 해를 비교한다."""
    ray = g['rays'][ip]; n = len(ray['R']); down = j if side == 'P' else j+1
    if not (2 <= down <= n-2 if side == 'P' else 1 <= down <= n-2):
        return dict(status='BOUNDARY_CELL_NOT_LOCAL_INTERIOR_PROBE')
    ids = np.array([j+1, j]) if side == 'P' else np.array([j, j+1])
    current = actual['rays'][ip]['I_'+side]; previous = blue['rays'][ip]['I_'+side]
    b = ray['B_'+side][ids]; length = float(np.diff(ray['S_'+side][ids])[0])
    incoming = float(current[ids[0]]); native = float(current[ids[1]])
    original = local_sc(chi, eta, previous, ray, current, down, args['dLOG_NU'], side)
    result = kernel.leg([length], chi[ids], eta[ids], b, previous[ids],
        1/args['dLOG_NU'], incoming, pieces)[-1]
    reference, accuracy = linear_material_reference(incoming, chi[ids], eta[ids], b,
        previous[ids], length, 1/args['dLOG_NU'])
    return dict(status='SAME_INCOMING_LOCAL_SC_AND_CONTINUOUS_REFERENCE', ray_1based=ip+1,
        direction=side, native_outer_zero_based=int(j-g['ND_ADD']), native_inner_zero_based=int(j+1-g['ND_ADD']),
        incoming=incoming, actual_downstream=native, original_SC_replayed=float(original),
        linear_material_downstream=float(result), independent_reference=float(reference),
        SC_replay_absolute_difference=abs(float(original)-native),
        production_core_reference_absolute_difference=abs(float(result)-float(reference)),
        local_injection=float(result)-native, optical_depth=float(accuracy['optical_depth']),
        chi_endpoints=chi[ids].tolist(), eta_endpoints=eta[ids].tolist(),
        B_endpoints=b.tolist(), blue_endpoints=previous[ids].tolist(),
        independent_quadrature=accuracy)


def analyze(run, formal_build, prior, out, node=49, pieces=64, seconds=150):
    if out.exists(): raise FileExistsError('기존 공간 주입 분해 보존')
    if not 0 < seconds <= 160 or type(pieces) is not int or not 1 <= pieces <= 64:
        raise ValueError('공간 주입 분석 예산 오류')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 진단 환경 필요')
    start = time.monotonic()
    entry, _, g, blue, actual, paths, _ = load_packet(run, formal_build, None, 'FORMAL_44360_02')
    a = entry['args']
    if type(node) is not int or not 0 <= node < g['ND']: raise ValueError('native 점 범위 오류')
    previous = json.loads((prior/'pair-history.json').read_text())
    for path in paths:
        if sha(path) != previous['input_sha256'][str(path)]: raise ValueError('직전 실행 입력 신원 변경')
    if sha(prior/'pair-moments.npz') != previous['output_sha256']:
        raise ValueError('직전 실행 모멘트 신원 변경')
    out.mkdir(); kernel, command, producers = build_kernel(out); computed = []
    moments, stat = transport(kernel, a, g, blue, actual, pieces, start+seconds,
        'extended', ray_output=computed)
    with np.load(prior/'pair-moments.npz', allow_pickle=False) as old:
        identical = np.array_equal(moments, old['first'])
        scale = float(np.max(abs(old['first'])))
        previous_difference = float(np.max(abs(moments-old['first']))/scale) if scale else 0.
    if previous_difference > 1e-12: raise ValueError('직전 첫 주파수 결과 재현 불일치')
    reference = sum(angular_integral(g, actual))
    chi, eta = extended_material(a, g); groups = {}; largest = {}; per_ray = []; closure = 0.; boundary_sum = 0.
    for ip in range(g['NP']-node):
        ray = g['rays'][ip]; n = len(ray['R']); k = int(ray['LNK'][node])-1
        e = {s: computed[ip]['I_'+s]-actual['rays'][ip]['I_'+s] for s in ('P', 'M')}
        tau = {}
        for s in ('P', 'M'):
            opacity = chi[:n]+(1/a['dLOG_NU']+3)*ray['B_'+s]
            length = np.diff(ray['S_M']) if s == 'M' else -np.diff(ray['S_P'])
            tau[s] = .5*(opacity[:-1]+opacity[1:])*length
        with np.errstate(over='raise', invalid='raise'):
            t = {s: np.exp(-tau[s]) for s in ('P', 'M')}
        for observed in ('P', 'M'):
            if observed == 'M':
                errors = e['M'][:k+1]; trans = t['M'][:k]
                labels = [('M', j) for j in range(k)]
            else:
                ids = np.arange(n-2, k-1, -1)
                if ip < g['NC']:
                    errors = e['P'][np.r_[n-1, ids]]; trans = t['P'][ids]
                    labels = [('P', j) for j in ids]
                else:
                    if e['P'][-1] != e['M'][-1]: raise ValueError('접선에서 차이의 연속성 불일치')
                    errors = np.r_[e['M'], e['P'][ids]]; trans = np.r_[t['M'], t['P'][ids]]
                    labels = [('M', j) for j in range(n-1)]+[('P', j) for j in ids]
            terms, boundary, residual = defect_terms(errors, trans)
            closure = max(closure, residual)
            weight = float(g['weights']['JQW_'+observed][node, ip]); boundary_sum += boundary*weight
            for (source_leg, j), value in zip(labels, terms):
                key = (int(j-g['ND_ADD']), int(j+1-g['ND_ADD']), source_leg)
                contribution = float(value)*weight
                groups[key] = groups.get(key, 0.)+contribution
                if key not in largest or abs(contribution) > abs(largest[key][-1]):
                    largest[key] = (ip, j, source_leg, contribution)
        per_ray.append(dict(ray_1based=ip+1, core=ip < g['NC'],
            delta_J_plus=float(e['P'][k]*g['weights']['JQW_P'][node, ip]),
            delta_J_minus=float(e['M'][k]*g['weights']['JQW_M'][node, ip])))
    cells = [dict(native_outer_zero_based=k[0], native_inner_zero_based=k[1],
                  source_leg=k[2], delta_J=v) for k, v in groups.items()]
    cells.sort(key=lambda row: abs(row['delta_J']), reverse=True)
    selected = cells[0]
    probe_key = (selected['native_outer_zero_based'], selected['native_inner_zero_based'], selected['source_leg'])
    ip, j, source_leg, _ = largest[probe_key]
    probe = probe_cell(kernel, a, g, blue, actual, chi, eta, ip, j, source_leg, pieces)
    total = float(moments[0, node]-reference[0, node])
    reconstructed = math.fsum([boundary_sum, *groups.values()])
    normalization = math.fsum([abs(boundary_sum), *map(abs, groups.values())])
    with (out/'cells.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    files = [*paths, prior/'pair-history.json', prior/'pair-moments.npz', *producers,
             Path(__file__), Path(transport.__code__.co_filename), Path(extended_material.__code__.co_filename),
             Path(linear_material_reference.__code__.co_filename)]
    source = out/'source_snapshot'; source.mkdir()
    for path in [Path(__file__), Path(transport.__code__.co_filename),
                 Path(extended_material.__code__.co_filename),
                 Path(linear_material_reference.__code__.co_filename), *producers[:-1]]:
        shutil.copy2(path, source/path.name)
    report = dict(status='FIXED_BLUE_SPATIAL_DEFECT_ATTRIBUTION', scope=__doc__,
        node_zero_based=node, frequency_Hz=a['FREQ']*1e15, pieces=pieces,
        prior_first_moments_byte_equal=identical, prior_difference=previous_difference,
        formal_J=float(reference[0, node]), linear_material_J=float(moments[0, node]),
        observed_delta_J=total, reconstructed_delta_J=reconstructed,
        boundary_contribution=boundary_sum,
        path_identity_scaled_error=closure,
        grouped_identity_scaled_error=abs(total-reconstructed)/normalization if normalization else abs(total-reconstructed),
        dominant_cells=cells[:10], dominant_cell_local_probe=probe, ray_contributions=per_ray, transport=stat,
        compiler_command=command, input_sha256={str(p): sha(p) for p in files},
        cells_sha256=sha(out/'cells.csv'), elapsed_seconds=time.monotonic()-start,
        production_changed=False, self_consistent_scattering=False,
        new_full_frequency_runs=0, new_material_updates=0, final_goal='NOT_MET')
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with (out/'spatial-defects.json').open('x') as stream:
        stream.write(encoded)
    print(json.dumps({k: report[k] for k in ('status', 'formal_J', 'linear_material_J',
        'observed_delta_J', 'reconstructed_delta_J', 'boundary_contribution',
        'path_identity_scaled_error', 'grouped_identity_scaled_error', 'dominant_cells',
        'dominant_cell_local_probe')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'formal-build', 'prior', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--node', type=int, default=49)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.formal_build.resolve(), args.prior.resolve(), args.out.resolve(), args.node)
