import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_lumina_public_toy06 import compare, interp_columns, sha


def fixture(tmp_path):
    pub = tmp_path / 'public'
    run = tmp_path / 'run'
    pub.mkdir()
    run.mkdir()
    for name in ('ionfrac_si_toy06_cmfgen.txt', 'ionfrac_s_toy06_cmfgen.txt'):
        rows = ['#TIME: 19.480', '#NVEL: 100']
        rows += [f'{i + 1} 0 .1 .2 .3 .4' for i in range(100)]
        (pub / name).write_text('\n'.join(rows) + '\n')
    (pub / 'phys_toy06_cmfgen.txt').write_text(
        '#TIME: 19.480\n#NVEL: 100\n' +
        ''.join(f'{i + 1} 10000 1e-10 1e8 1e8\n' for i in range(100)))
    epoch = 19.48 * 86400
    radius = np.arange(1., 91.) * epoch * 1e5
    np.savez(run / 'physical-profiles.npz', radius_cm=radius,
             new_response_temperature=np.full(90, 9000.), new_response_ne=np.full(90, 9e7))
    table = run / 'Si-S-comparison.csv'
    with table.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['node', 'radius_cm', 'Z', 'charge',
                                           'new_J_response_fraction'])
        w.writeheader()
        for z in (14, 16):
            for q in (1, 2, 3, 4):
                for node in range(90):
                    w.writerow(dict(node=node, radius_cm=radius[node], Z=z, charge=q,
                                    new_J_response_fraction=q / 10 + .01))
    report = dict(state='COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE',
        fixed_accepted_temperature=True, source_identity={'epoch_seconds': epoch},
        selected_response={'response_sha256': 'a' * 64},
        output_sha256={name: sha(run / name) for name in
            ('Si-S-comparison.csv', 'physical-profiles.npz')})
    result = run / 'result.json'
    result.write_text(json.dumps(report))
    return result, pub


def test_comparison_is_stage_mapped_and_scientifically_conditional(tmp_path):
    result, pub = fixture(tmp_path)
    output = compare(result, pub)
    assert output['stages']['14:2']['compared_nodes'] == 90
    assert output['stages']['14:2']['public_fraction'] == pytest.approx(.2)
    assert output['stages']['14:2']['lumina_fraction'] == pytest.approx(.21)
    assert output['stages']['16:1']['max_abs_fraction'] == pytest.approx(.01)
    assert output['final_equivalence'] == 'NOT_JUDGED'
    assert output['new_physical_calculations'] == 0


def test_no_public_extrapolation():
    table = np.array([[1., 10.], [2., 20.]])
    result = interp_columns(table, [0., 1., 1.5, 2., 3.], 1)
    assert np.isnan(result[[0, 4]]).all()
    assert result[1:4].tolist() == [10., 15., 20.]


def test_reject_changed_csv(tmp_path):
    result, pub = fixture(tmp_path)
    with (result.parent / 'Si-S-comparison.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='해시'):
        compare(result, pub)


def test_reject_nonmatching_epoch(tmp_path):
    result, pub = fixture(tmp_path)
    report = json.loads(result.read_text())
    report['source_identity']['epoch_seconds'] += 1
    result.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='epoch'):
        compare(result, pub)
