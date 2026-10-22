#!/usr/bin/env python3
"""실제 294Å 광선의 양의 SC와 같은 끝점 물질의 직접 formal 적분을 비교한다.

원 포획은 물질60의 원래 SC 경로다. 여기서 계산하는 양의 Bézier SC는
같은 고정 blue/입사·끝점으로 재평가한 반사실험이며, 물질70 전체장을
재현하거나 수송·인구를 갱신하지 않는다.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from analyze_cmfgen_formal_capture import compensated_moments, formal_state, geometry, rays
from compare_cmfgen_packet_cell_material import kernel, linear_material_reference
from prepare_cmfgen_mom_capture import sha


def angular_cell(packet, build, native, out):
    packet, build, native, out = map(Path, (packet, build, native, out))
    if out.exists():
        raise FileExistsError('기존 UV 연산자 결과 보존')
    stem = 'FORMAL_23732_00'
    names = ('_in.bin', '_out.bin', '_geometry.bin', '_rays_in.bin', '_rays_out.bin')
    paths = {name: packet / (stem + name) for name in names}
    previous = packet / 'FORMAL_23731_01_rays_out.bin'
    manifest = build / 'formal-build.json'
    for path in (*paths.values(), previous, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if previous.read_bytes() != paths['_rays_in.bin'].read_bytes():
        raise ValueError('23732 실제 blue가 직전 광선과 다름')
    inventory = json.loads(manifest.read_text())['ray_inventory']
    initial = formal_state(paths['_in.bin'], inventory)
    final = formal_state(paths['_out.bin'], inventory)
    geo = geometry(paths['_geometry.bin'])
    blue = rays(paths['_rays_in.bin'])
    actual = rays(paths['_rays_out.bin'])
    args = initial['args']
    if initial['phase'] != (60, 23732, 0, 0, 1) or geo['RAY_POINTS_INSERTED'] or \
       not 1.0e16 < args['FREQ']*1e15 < 1.1e16 or args['P'][0] != 0 or \
       any(x['NP'] != args['NP'] or x['ND'] != args['ND'] for x in (geo, blue, actual)):
        raise ValueError('예상한 원294Å·원격자·중앙광선 포획이 아님')
    measured, bound = compensated_moments(geo, actual)
    stored = np.asarray(final['ray']['JNU_STORE']['values'])
    if stored.shape != (args['ND'],) or np.any(abs(stored-measured[0]) > bound[0]):
        raise ValueError('원 각도 J 합이 저장된 formal J와 불일치')
    out.mkdir(parents=False)
    cell, compiler, producers = kernel(out, native)
    alpha = 1.0 / args['dLOG_NU']
    rows = []
    started = time.monotonic()
    for outer_depth in (50, 52):
        j = outer_depth - 1
        for direction in ('M', 'P'):
            downstream = j + 1 if direction == 'M' else j
            cells = []
            for ip, (raygeo, current, prior) in enumerate(zip(geo['rays'], actual['rays'], blue['rays'])):
                links = raygeo['LNK']
                if links[j] < 1 or links[j+1] < 1:
                    continue
                ids = np.array((j, j+1) if direction == 'M' else (j+1, j))
                rid = np.array((links[j]-1, links[j+1]-1) if direction == 'M' else
                               (links[j+1]-1, links[j]-1), dtype=int)
                intensity = current['I_'+direction]
                prior_intensity = prior['I_'+direction]
                if np.any(rid < 0) or np.any(rid >= len(intensity)) or abs(rid[1]-rid[0]) != 1:
                    raise ValueError('원 ray의 셀 깊이/연속성 불일치')
                chi, eta = (np.asarray(args[name][ids], dtype=float) for name in ('CHI', 'ETA'))
                b = np.asarray(raygeo['B_'+direction][rid], dtype=float)
                old = np.asarray(prior_intensity[rid], dtype=float)
                s = raygeo['S_'+direction]
                length = float(s[rid[1]]-s[rid[0]])
                effective = chi + (alpha+3)*b
                tau = 0.5*length*sum(effective)
                source = (eta+alpha*b*old)/effective
                if not length > 0 or not tau > 0 or np.any(effective <= 0) or \
                   np.any(source < 0) or not np.isfinite(source).all():
                    raise ValueError('지원하지 않는 실제 양의 cell')
                far_depth = j+2 if direction == 'M' else j-1
                if 0 <= far_depth < args['ND'] and links[far_depth] > 0:
                    far_rid = int(links[far_depth])-1
                    if not 0 <= far_rid < len(intensity) or abs(far_rid-rid[1]) != 1:
                        raise ValueError('다음 원 ray node의 불연속')
                    far_eff = args['CHI'][far_depth]+(alpha+3)*raygeo['B_'+direction][far_rid]
                    far_source = (args['ETA'][far_depth]+alpha*raygeo['B_'+direction][far_rid]*
                                  prior_intensity[far_rid])/far_eff
                    neighbor = 0.5*float(s[far_rid]-s[rid[1]])*(effective[1]+far_eff)
                    control = cell('control', source[0], source[1], far_source, tau, neighbor)
                    linear_boundary = False
                else:
                    control = 0.5*sum(source)
                    linear_boundary = True
                incident = float(intensity[rid[0]])
                original = float(intensity[rid[1]])
                positive = cell('point', incident, source[0], control, source[1], tau, tau)
                direct, estimate, evaluations, status = cell(
                    'material', incident, chi, eta, b, old, length, alpha, 1e-10)
                if status or not np.isfinite((positive, direct, estimate)).all():
                    raise ValueError('물질 formal 직접 적분 실패: '+str((outer_depth, direction, ip, status)))
                if ip == 0:
                    independent, accuracy = linear_material_reference(
                        incident, chi, eta, b, old, length, alpha)
                    if abs(independent-direct) > 5e-9*max(incident, positive, direct):
                        raise ValueError('중앙 ray 독립 formal 적분 불일치')
                else:
                    accuracy = None
                weight = float(geo['weights']['JQW_'+direction][downstream, ip])
                cells.append(dict(ray=ip, weight=weight, incident=incident, original_legacy=original,
                    positive_SC=positive, direct_material=direct, tau=tau,
                    difference_over_scale=(positive-direct)/max(incident, positive, direct),
                    material_estimate=estimate, material_evaluations=evaluations,
                    linear_boundary=linear_boundary, central_independent_accuracy=accuracy))
            if not cells:
                raise ValueError('대상 깊이에 원 angular ray가 없음')
            reference = float(measured[0, downstream])
            signed = sum(row['weight']*(row['positive_SC']-row['direct_material']) for row in cells)
            absolute = sum(abs(row['weight']*(row['positive_SC']-row['direct_material'])) for row in cells)
            rows.append(dict(outer_depth_1based=outer_depth, direction=direction,
                downstream_depth_1based=downstream+1, rays=len(cells),
                original_J_at_downstream=reference,
                signed_angular_delta_over_original_J=signed/reference,
                absolute_angular_delta_bound_over_original_J=absolute/reference,
                central_ray=next(row for row in cells if row['ray'] == 0),
                largest_weighted_cells=sorted(cells, key=lambda row: abs(row['weight']*(
                    row['positive_SC']-row['direct_material'])), reverse=True)[:5]))
    report = dict(scope=__doc__, material_outer=60, frequency_index=23732,
        frequency_Hz=args['FREQ']*1e15, wavelength_A=2.99792458e18/(args['FREQ']*1e15),
        original_packet_SC='pre-positive legacy; positive_SC is recomputed, not the stored formal ray',
        comparison='one-cell positive Bezier versus direct linear-material transfer at fixed original incident/blue',
        does_not_prove='material70/J71 or whole angular self-consistency, Si/S distribution, spectrum',
        elapsed_seconds=time.monotonic()-started, results=rows,
        input_sha256={str(path):sha(path) for path in (*paths.values(), previous, manifest, *producers)},
        compiler_command=compiler, production_changed=False, final_goal='NOT_MET')
    result = out/'uv-cell-operator.json'
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(result=str(result), wavelength_A=report['wavelength_A'],
                          summary=[{k:v for k,v in row.items() if k not in ('central_ray', 'largest_weighted_cells')}
                                   for row in rows]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'build', 'native', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    angular_cell(*(getattr(a, name) for name in ('packet', 'build', 'native', 'out')))
