"""봉인 BF 기록에 Si/S 용해율만 합치는 byte 배선 직접시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_lumina_pseudo_candidate_overlay import HEADER, RECORD, merge_bound_pseudo, parse_candidate


def fixture():
    count = 24667
    bound = np.zeros((count, 6))
    bound[:, 0] = 1.
    pseudo = np.zeros_like(bound)
    pseudo[0] = [2, 3, 4, -1, -2, -3]
    roster = dict(lower=np.arange(count, dtype='i4'), target=np.full(count, -1, dtype='i4'),
                  Z=np.full(count, 26, dtype='i4'), probability=np.ones(count),
                  threshold=np.full(count, 5e15), sigma_edge=np.full(count, 1e-18),
                  saha=np.full(count, 1e-3))
    roster['Z'][0] = 14
    records = np.zeros(count, dtype=RECORD)
    records['ids'][:, 0] = roster['lower']
    records['ids'][:, 1] = roster['target']
    for key, column in (('probability', 0), ('threshold', 1), ('sigma_edge', 2), ('saha', 3)):
        records['values'][:, column] = roster[key]
    records['values'][:, 4:] = bound
    original = HEADER.pack(b'LMBFFUL1', count, count, 90, 49, 1e4, 1e8) + records.tobytes()
    return original, bound, pseudo, roster


def test_overlay_changes_only_pseudo_route():
    original, bound, pseudo, roster = fixture()
    merged = merge_bound_pseudo(original, bound, pseudo, roster, 49)
    got = np.frombuffer(merged, dtype=RECORD, offset=HEADER.size)
    np.testing.assert_array_equal(got['values'][0, 4:], bound[0] + pseudo[0])
    np.testing.assert_array_equal(got['values'][1:, 4:], bound[1:])
    assert merged[:HEADER.size] == original[:HEADER.size]


def test_overlay_rejects_other_element_and_route_mismatch():
    original, bound, pseudo, roster = fixture()
    roster['Z'][0] = 26
    with pytest.raises(ValueError, match='Si/S 이외'):
        merge_bound_pseudo(original, bound, pseudo, roster, 49)
    roster['Z'][0] = 14
    roster['threshold'][0] *= 2
    with pytest.raises(ValueError, match='정규화'):
        merge_bound_pseudo(original, bound, pseudo, roster, 49)


def test_parse_candidate_requires_exact_addresses():
    original = {(14, 1, 0): 2., (16, 2, -1): 3.}
    output = '\n'.join([
        '[MODEL-SE][FIXED-FIELD] Z=14 native_node=49 variables=1 state=PRIVATE_PARTIAL_CHANNEL_CANDIDATE residual=1e-16 physical_equivalence=NOT_PROVEN',
        '[MODEL-SE][FIXED-FIELD] Z=16 native_node=49 variables=1 state=PRIVATE_PARTIAL_CHANNEL_CANDIDATE residual=1e-16 physical_equivalence=NOT_PROVEN',
        '[MODEL-SE][FIXED-FIELD-ROW] Z=14 stage0=1 super0=0 original=2 candidate=2.1',
        '[MODEL-SE][FIXED-FIELD-ROW] Z=16 stage0=2 super0=-1 original=3 candidate=2.9'])
    got, residual = parse_candidate(output, original, 49)
    assert got == {(14, 1, 0): 2.1, (16, 2, -1): 2.9} and residual == 1e-16
    with pytest.raises(ValueError, match='누락'):
        parse_candidate(output.rsplit('\n', 1)[0], original, 49)
