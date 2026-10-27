"""저장 소광의 원인 선 역추적. 합성 자료에서 실제 수송은 실행하지 않는다."""
import csv
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_lumina_negative_nodal_sources as A


def test_signed_line_is_attributed_and_reference_inversion_is_separate(tmp_path):
    sources = tmp_path/'sources'; sources.mkdir()
    material = tmp_path/'material'; atomic = material/'atomic'; atomic.mkdir(parents=True)
    ns = 24; nu = np.array([1.e15, 1.99e15, 2.e15, 2.01e15]); nf = len(nu)
    radius = np.linspace(3.e14, 1.e14, ns)
    physical = np.column_stack((radius, np.full(ns, 1.e4), np.full(ns, 1.e8), np.full(ns, 1.e-14)))
    with (material/'native_material_nodes.bin').open('wb') as f:
        f.write(struct.pack('<8sIIId', b'LUMNOD01', ns, 1, 2, 1.e6))
        f.write(np.array([[14, 1, 0]], dtype='<i4').tobytes())
        f.write(physical.astype('<f8').tobytes()); f.write(np.ones((ns, 1), dtype='<f8').tobytes())
    with (atomic/'levels.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['atomic_number', 'ion_number', 'level_number', 'g'])
        writer.writeheader(); writer.writerows([dict(atomic_number=14, ion_number=1, level_number=i, g=1)
                                                for i in range(2)])
    full = np.vstack([np.ones(ns), np.full(ns, 100.)]); ref = full[::-1].copy()
    def write_full(path, array):
        path.write_bytes(struct.pack('<8sII', b'LMATFL01', 2, ns)+array.astype('<f8').tobytes())
    c_full = tmp_path/'full.bin'; old = tmp_path/'old.bin'; write_full(c_full, full); write_full(old, ref)
    lines = dict(lower=np.array([0]), upper=np.array([1]), nu0=np.array([2.e15]),
                 f=np.array([1.]), A=np.array([1.]), g_ratio=np.array([1.]))
    np.savez(sources/'line_catalog.npz', **lines)
    profile = dict(width_fraction=.005, support_fraction=.01)
    width = lines['nu0'][0]*profile['width_fraction']; x = (nu-lines['nu0'][0])/width
    line = np.pi*A.B.atomic.E_CGS**2/(A.B.ME*A.B.C)*(1-100)*np.exp(-x*x)/(1.77245385095516*width)
    line[abs(nu-lines['nu0'][0]) > lines['nu0'][0]*profile['support_fraction']] = 0.
    planes = np.zeros((7, ns, nf)); planes[4] = line; planes[6] = 1.e-18
    binary = sources/'sources.bin'
    with binary.open('wb') as f:
        f.write(struct.pack('<8sII', b'LMPHYS01', ns, nf)); f.write(nu.astype('<f8').tobytes())
        f.write(physical[:, 1].astype('<f8').tobytes()); f.write(physical[:, 2].astype('<f8').tobytes())
        f.write(planes.astype('<f8').tobytes())
    expected_count = int(np.count_nonzero(planes[4]+planes[6] < 0))
    report = dict(cells=ns, frequencies=nf, profile=profile, total_negative_samples=expected_count,
        input_sha256={str(p): A.B.sha(p) for p in (c_full, material/'native_material_nodes.bin', atomic/'levels.csv')},
        output_sha256={name: A.B.sha(sources/name) for name in ('sources.bin', 'line_catalog.npz')})
    (sources/'sources.json').write_text(json.dumps(report))
    output = tmp_path/'diagnosis.json'
    result = A.analyze(sources, material, c_full, old, output)
    assert result['negative_samples'] == expected_count and result['negative_nodes'] == ns
    case = result['representative_cases'][0]; line = case['top_negative_lines'][0]
    assert line['Z'] == 14 and line['ion_number'] == 1 and line['contribution'] < 0
    assert not line['original70_inverted'] and line['lower_n_over_g'] < line['upper_n_over_g']
    assert case['line_sum_relative_error'] < 1.e-12 and not result['modified_opacity']
    with pytest.raises(FileExistsError): A.analyze(sources, material, c_full, old, output)
    c_full.write_bytes(c_full.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='물질/원자/실제 C 인구'):
        A.analyze(sources, material, c_full, old, tmp_path/'other.json')
