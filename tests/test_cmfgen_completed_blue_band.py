"""미완료 구간을 완성된 광율/전체 장으로 가장하지 않는다."""
from pathlib import Path
import sys
import csv
import json

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_cmfgen_completed_blue_band import supported_rate, prefix_field


def test_rate_whose_entire_support_is_blue_of_cutoff():
    weights = np.array([0., 0., 1., 2., 3.])
    partial = np.ones((3, 2)); new = np.ones((5, 2)) * 2
    a, b = supported_rate(weights, partial, new)
    np.testing.assert_array_equal(a, [6., 6.]); np.testing.assert_array_equal(b, [12., 12.])


def test_even_tiny_nonzero_missing_support_is_refused():
    weights = np.array([1e-280, 0., 1., 2., 3.])
    with pytest.raises(ValueError, match='지지'):
        supported_rate(weights, np.ones((3, 2)), np.ones((5, 2)))


def test_only_completed_direct_records_are_read_even_with_corrupt_red_tail(tmp_path):
    seed = tmp_path / 'seed'; seed.mkdir(); (seed / 'SCRTEMP').write_bytes(b'matter59')
    run = tmp_path / 'legacy'; run.mkdir(); (run / 'seq_logs').mkdir()
    (run / 'SCRTEMP').write_bytes(b'matter59')
    (run / 'seq_logs/whole-runtime.json').write_text(json.dumps(dict(
        coverage=dict(count=3), diagnostic_complete=False, returncode=95)))
    (run / 'seq_logs/whole-inputs.json').write_text(json.dumps(dict(seed=str(seed))))
    (run / 'EDDFACTOR_INFO').write_text('date\ndate\n3 32 8 1 4 T\n')
    records = np.array([[4., 4., 4., 4.], [3., 3., 3., 3.], [2., 2., 2., 2.]], dtype='<f8')
    (run / 'EDDFACTOR').write_bytes(bytes(14 * 32) + records.tobytes() + b'bad uncomputed red tail')
    with (run / 'WHOLE_SWEEP.csv').open('x', newline='') as f:
        writer = csv.writer(f); writer.writerow(['outer', 'index', 'freq_native'])
        for i, frequency in enumerate((4., 3., 2.), 1):
            writer.writerow([60, i, frequency])
    field, count = prefix_field(run, seed, np.array([1., 2., 3., 4.]) * 1e15)
    assert count == 3
    np.testing.assert_array_equal(field[:, 0], [2., 3., 4.])
    (run / 'SCRTEMP').write_bytes(b'matter39')
    with pytest.raises(ValueError, match='59'):
        prefix_field(run, seed, np.array([1., 2., 3., 4.]) * 1e15)
