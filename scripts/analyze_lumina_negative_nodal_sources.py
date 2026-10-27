#!/usr/bin/env python3
"""저장된 90점 signed 소광에서 음수 위치와 대표 원인 선을 추적한다.

계수를 재생산하거나 수정하지 않는다. 전 본문은 수송 제공자와 같은 합산
순서로 한 점씩 읽으며, 기록한 본문 해시는 생산 기록에서 계승한다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import struct

import numpy as np

from analyze_cmfgen_signed_sources import read_full
import cmfgen_model_continuum as B
from prepare_lumina_outer_input import read_native


def analyze(sources, material, c_full, reference_full, output, *, max_cases=12):
    sources, material, c_full, reference_full, output = map(
        lambda p: Path(p).resolve(), (sources, material, c_full, reference_full, output))
    if output.exists(): raise FileExistsError('기존 분석 결과 보존')
    report_path = sources/'sources.json'; report = json.loads(report_path.read_text())
    binary = sources/'sources.bin'; catalog = sources/'line_catalog.npz'
    if B.sha(catalog) != report['output_sha256']['line_catalog.npz']:
        raise ValueError('선 목록 변경')
    for path in (c_full, material/'native_material_nodes.bin', material/'atomic/levels.csv'):
        if B.sha(path) != report['input_sha256'][str(path)]:
            raise ValueError('계수 생산과 물질/원자/실제 C 인구 불일치')
    with binary.open('rb') as stream:
        magic, ns, nf = struct.unpack('<8sII', stream.read(16))
    base = 16+8*(nf+2*ns); expected_size = base+7*ns*nf*8
    before = binary.stat()
    if (magic != b'LMPHYS01' or (ns, nf) != (report['cells'], report['frequencies']) or
            before.st_size != expected_size): raise ValueError('계수 형상/길이 오류')
    physical, _ = read_native(material/'native_material_nodes.bin')
    full = read_full(c_full); reference = read_full(reference_full)
    levels = list(csv.DictReader((material/'atomic/levels.csv').open()))
    if (len(physical) != ns or full.shape != (len(levels), ns) or reference.shape != full.shape):
        raise ValueError('원90점/실제 C full 인구 주소 오류')
    fd = os.open(binary, os.O_RDONLY)
    def read_values(offset, count):
        raw = os.pread(fd, count*8, offset)
        if len(raw) != count*8: raise ValueError('계수 파일의 요청 범위 잘림')
        return np.frombuffer(raw, dtype='<f8')
    def plane_row(plane, node):
        return read_values(base+8*(plane*ns+node)*nf, nf)
    def plane_value(plane, node, index):
        return float(read_values(base+8*((plane*ns+node)*nf+index), 1)[0])
    per_node = []; candidates = []; total = 0
    try:
        frequency = read_values(16, nf)
        if (not np.isfinite(frequency).all() or np.any(frequency <= 0) or
                np.any(np.diff(frequency) <= 0) or
                not np.array_equal(read_values(16+8*nf, ns), physical[:, 1]) or
                not np.array_equal(read_values(16+8*(nf+ns), ns), physical[:, 2])):
            raise ValueError('계수 주파수 또는 같은 물질의 T/ne 불일치')
        for node in range(ns):
            # 실제 NativeSourceColumns.read와 동일한 합산 순서.
            chi = (plane_row(6, node)+(plane_row(0, node)+plane_row(2, node)))+plane_row(4, node)
            indices = np.flatnonzero(chi < 0); total += len(indices)
            if not len(indices): continue
            n = min(3, len(indices)); chosen = indices[np.argpartition(chi[indices], n-1)[:n]]
            candidates.extend((float(chi[i]), node, int(i)) for i in chosen)
            per_node.append(dict(node=node, radius_cm=float(physical[node, 0]),
                count=len(indices), minimum_chi=float(chi[indices].min()),
                wavelength_min_A=float(B.C/frequency[indices[-1]]*1.e8),
                wavelength_max_A=float(B.C/frequency[indices[0]]*1.e8)))
        with np.load(catalog, allow_pickle=False) as archive:
            lines = {name: archive[name] for name in archive.files}
        order = np.argsort(lines['nu0']); sorted_nu = lines['nu0'][order]
        profile = report['profile']; support = profile['support_fraction']; width_fraction = profile['width_fraction']
        opacity = np.pi*B.atomic.E_CGS**2/(B.ME*B.C); cases = []
        for total_chi, node, index in sorted(candidates)[:max_cases]:
            nu = float(frequency[index]); begin = np.searchsorted(sorted_nu, nu/(1+support), side='left')
            end = np.searchsorted(sorted_nu, nu/(1-support), side='right'); active = order[begin:end]
            center = lines['nu0'][active]; lo = lines['lower'][active]; up = lines['upper'][active]
            width = center*width_fraction; x = (nu-center)/width
            phi = np.exp(-x*x)/(1.77245385095516*width)
            contributions = opacity*lines['f'][active]*(full[lo, node]-lines['g_ratio'][active]*full[up, node])*phi
            actual_bb = plane_value(4, node, index); difference = abs(float(contributions.sum())-actual_bb)
            scale = float(abs(contributions).sum())
            if difference > 1.e-12*scale:
                raise ValueError('대표 위치의 전체 선 합이 저장 BB 소광과 다름')
            top = []
            for k in np.argsort(contributions)[:min(5, len(active))]:
                if contributions[k] >= 0: break
                lower, upper = int(lo[k]), int(up[k]); gl, gu = float(levels[lower]['g']), float(levels[upper]['g'])
                top.append(dict(Z=int(levels[lower]['atomic_number']), ion_number=int(levels[lower]['ion_number']),
                    lower_FL_1based=int(levels[lower]['level_number'])+1,
                    upper_FL_1based=int(levels[upper]['level_number'])+1,
                    wavelength_A=float(B.C/center[k]*1.e8), contribution=float(contributions[k]),
                    lower_n_over_g=float(full[lower, node]/gl), upper_n_over_g=float(full[upper, node]/gu),
                    original70_inverted=bool(reference[lower, node]/gl < reference[upper, node]/gu)))
            dr = max(abs(physical[node, 0]-physical[j, 0]) for j in (node-1, node+1) if 0 <= j < ns)
            cases.append(dict(node=node, bin=index, radius_cm=float(physical[node, 0]),
                frequency_Hz=nu, wavelength_A=float(B.C/nu*1.e8), total_chi=total_chi,
                components={name: plane_value(i, node, index) for i, name in
                            ((0, 'bf'), (2, 'ff'), (4, 'bb'), (6, 'electron_scattering'))},
                adjacent_spacing_cm=float(dr), local_constant_chi_times_spacing=float(-total_chi*dr),
                local_spacing_caveat='상수 소광을 인접점 거리만큼 적용한 크기 추정; 실제 광선 광학깊이가 아님',
                line_sum_relative_error=difference/scale if scale else 0., top_negative_lines=top))
    finally:
        os.close(fd)
    after = binary.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('분석 중 계수 본문 변경')
    result = dict(schema='lumina-negative-nodal-source-attribution-v1',
        negative_samples=total, producer_report_negative_samples=report['total_negative_samples'],
        negative_nodes=len(per_node), per_node=per_node, representative_cases=cases,
        source_binary_sha256=report['output_sha256']['sources.bin'],
        source_binary_hash_scope='생산 기록에서 계승; 이번 전수 순회에서 별도 해시하지 않음',
        input_sha256={str(p): B.sha(p) for p in (report_path, catalog, c_full, reference_full,
                                                material/'atomic/levels.csv', material/'native_material_nodes.bin')},
        original70_comparison='같은 반경에서 원70 C full 재구성 인구와 비교; native FL 저장 원본 아님',
        modified_opacity=False, new_transport=0, final_goal='NOT_JUDGED')
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('sources', 'material', 'c-full', 'reference-full', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    result = analyze(**vars(p.parse_args()))
    print(json.dumps({k: result[k] for k in ('negative_samples', 'producer_report_negative_samples', 'negative_nodes',
                                             'representative_cases')}, ensure_ascii=False, indent=2))
