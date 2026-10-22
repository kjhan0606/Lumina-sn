#!/usr/bin/env python3
"""같은 native 반경에서 CMFGEN formal 입력과 Lumina 준비 계수를 대조한다.

주파수 격자가 다르므로 Lumina 계수만 log(nu)에서 선형 보간한다.
이 결과는 계수 동일성 진단이지 수송/이온분포 수용 시험이 아니다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def formal(path, expected_sha, native_indices):
    if sha(path) != expected_sha:
        raise ValueError('참조 formal 파일 신원 불일치')
    with path.open('rb') as stream:
        tag = stream.read(16)
        kind, bits, ibits, nodes, count, fields = struct.unpack('<6i', stream.read(24))
        geometry = np.frombuffer(stream.read(4 * nodes * 8), dtype='<f8').reshape(4, nodes).copy()
    header, stride = 40 + 4 * nodes * 8, 24 + 8 * nodes * fields
    if (tag.rstrip() != b'FORMALPRIMARYV1' or (kind, bits, ibits, fields) != (8, 64, 32, 8)
            or nodes != 90 or count != 197291 or path.stat().st_size != header + count * stride):
        raise ValueError('참조 formal 형식/범위 불일치')
    record = np.memmap(path, mode='r', offset=header, shape=(count,), dtype=np.dtype([
        ('index', '<i4'), ('calls', '<i4'), ('freq', '<f8'), ('weight', '<f8'),
        ('field', '<f8', (8, nodes))]))
    freq = np.asarray(record['freq']).copy() * 1e15
    if not np.isfinite(freq).all() or not np.all(np.diff(freq) < 0):
        raise ValueError('참조 주파수 순서/값 오류')
    actual = np.empty((3, len(native_indices), count), dtype='f8')
    for start in range(0, count, 1024):
        stop = min(start + 1024, count)
        block = record[start:stop]
        if not np.array_equal(block['index'], np.arange(start + 1, stop + 1)):
            raise ValueError('참조 기록 순서 오류')
        for component in range(3):
            for at, node in enumerate(native_indices):
                actual[component, at, start:stop] = block['field'][:, component + 4, node] / 1e10
    if not np.isfinite(actual).all():
        raise ValueError('참조 계수 비유한')
    return geometry, freq, actual


def band_metrics(freq, reference, proposed, lo, hi):
    wavelength = 2.99792458e18 / freq
    selection = (wavelength >= lo) & (wavelength < hi)
    nu = freq[selection][::-1]
    ref = reference[selection][::-1]
    new = proposed[selection][::-1]
    if len(nu) < 2:
        raise ValueError('대역 주파수 표본 부족')
    scale = float(np.trapz(np.abs(ref), nu))
    if not scale > 0:
        raise ValueError('참조 계수 대역 적분이 0')
    return dict(points=int(len(nu)), signed_integral_difference_over_reference_abs=
                float(np.trapz(new - ref, nu) / scale),
                unsigned_integral_difference_over_reference_abs=
                float(np.trapz(np.abs(new - ref), nu) / scale),
                reference_negative=int(np.count_nonzero(ref < 0)),
                lumina_negative=int(np.count_nonzero(new < 0)))


def compare(reference_field, preparation, frequency_grid):
    report = json.loads(preparation.read_text())
    root = preparation.parent
    reference_sha = report['reference_field_sha256']
    if report['input_sha256'][str(frequency_grid)] != sha(frequency_grid):
        raise ValueError('Lumina 주파수 격자 신원 불일치')
    own_freq = np.load(frequency_grid, mmap_mode='r', allow_pickle=False)
    if own_freq.ndim != 1 or not np.all(np.diff(own_freq) > 0):
        raise ValueError('Lumina 주파수 격자 오류')
    indices = [int(x) for x in report['native_reference_indices']]
    if indices != [0, 14] or 'inner_chi.npy' not in report['output_sha256']:
        raise ValueError('J4 외삽 기준점/내부 경계가 아님')
    names = ('node_chi.npy', 'node_eta.npy', 'node_es.npy', 'inner_chi.npy')
    own = {}
    for name in names:
        path = root / name
        if sha(path) != report['output_sha256'][name]:
            raise ValueError('Lumina 준비 계수 신원 불일치: ' + name)
        own[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    for name in names[:3]:
        if own[name].shape != (2, len(own_freq)):
            raise ValueError('Lumina 보조점 계수 차원 오류')
    if own['inner_chi.npy'].shape != (len(own_freq),):
        raise ValueError('Lumina 내부경계 계수 차원 오류')
    geometry, ref_freq, ref = formal(reference_field, reference_sha, indices + [89])
    geometry_file = root / 'geometry.npz'
    if sha(geometry_file) != report['output_sha256']['geometry.npz']:
        raise ValueError('Lumina 기하 신원 불일치')
    with np.load(geometry_file, allow_pickle=False) as saved:
        radial_difference = float(np.max(np.abs(saved['native_radius'] - geometry[0] * 1e10)
                                         / np.maximum(np.abs(geometry[0] * 1e10), 1)))
    if radial_difference > 1e-12:
        raise ValueError('native 반경 대응 불일치')
    if ref_freq[-1] < own_freq[0] * (1 - 1e-12) or ref_freq[0] > own_freq[-1] * (1 + 1e-12):
        raise ValueError('참조 주파수가 Lumina 격자 밖으로 확장됨')
    log_own, log_ref = np.log(own_freq), np.log(ref_freq)
    bands = ((100, 400), (400, 800), (800, 1600), (1600, 3000))
    output = {}
    sources = ((0, 'node_chi.npy'), (1, 'node_eta.npy'), (2, 'node_es.npy'))
    for at, node in enumerate(indices + [89]):
        node_result = {}
        for component, source in sources:
            if node == 89 and component != 0:
                continue
            values = own['inner_chi.npy'] if node == 89 else own[source][at]
            projected = np.interp(log_ref, log_own, values)
            reference = ref[component, at]
            node_result[source] = {f'{lo}-{hi}A': band_metrics(ref_freq, reference, projected, lo, hi)
                                   for lo, hi in bands}
        output[str(node)] = node_result
    return dict(schema='lumina-native-formal-coefficient-comparison-v1',
                reference_field=str(reference_field), reference_sha256=reference_sha,
                preparation=str(preparation), preparation_sha256=sha(preparation),
                frequency_grid=str(frequency_grid), frequency_grid_sha256=sha(frequency_grid),
                native_radius_max_relative_difference=radial_difference,
                reference_frequency_count=int(len(ref_freq)), lumina_frequency_count=int(len(own_freq)),
                frequency_mapping='Lumina log(nu)-linear to native CMFGEN frequencies; no spatial interpolation',
                unit_mapping='CMFGEN formal chi/eta/scattering per 1e10 cm divided by 1e10',
                notes='Same native nodes only. Nonzero mismatch may include frequency-grid interpolation; not a shell-center or transport equivalence proof.',
                result=output, physical_RT_executions=0, final_goal='NOT_MET')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-field', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--frequency-grid', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('기존 결과를 덮어쓰지 않음')
    result = compare(args.reference_field, args.preparation, args.frequency_grid)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(out=str(args.out), result=result['result']), ensure_ascii=False))
