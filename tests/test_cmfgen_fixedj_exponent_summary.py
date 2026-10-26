import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_cmfgen_fixedj_exponent70 import compare_printed, profile_metrics


def test_acceptance_is_per_level_not_cancelling_total():
    result = compare_printed([10000., 10000.], [10001., 9999.])
    assert result['residual_sum'] == 0.
    assert result['outside_print_bound'] == 2
    assert compare_printed([10000.], [10000.4])['outside_print_bound'] == 0


def test_printed_zero_does_not_hide_small_nonzero_prediction():
    assert compare_printed([0.], [1e-100])['outside_print_bound'] == 1
    with pytest.raises(ValueError, match='차원'):
        compare_printed([1.], [-1.])


def test_profile_L1_does_not_hide_level_cancellation():
    metrics = profile_metrics([100.,100.],[110.,90.])
    assert metrics['sum_relative'] == 0.
    assert metrics['normalized_L1'] == .1
    assert metrics['major_max_relative'] == .1
    assert profile_metrics([0.,1.],[1.,0.])['zero_reference_positive_candidate'] == 1
