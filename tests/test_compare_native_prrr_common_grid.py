import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_native_prrr_common_grid import (check_prior, continuum_mapping,
                                             printed_half_step,
                                             cached_exponent_frequency,
                                             cached_exponent_rates)


def test_continuum_mapping_preserves_native_order_and_rejects_missing(tmp_path):
    path = tmp_path/'CONT_FREQ'
    path.write_text('\n!NCF, 3\n! original grid\n1 3.0000000E+00 3.0000000E+00 1\n'
                    '2 2.0000000E+00 3.0000000E+00 2\n'
                    '3 1.0000000E+00 1.0000000E+00 3\n'
                    '1 9.000E+00 9.000E+00 9\n')
    np.testing.assert_array_equal(continuum_mapping(path, [1e15, 2e15, 3e15]),
                                  [1e15, 3e15, 3e15])
    path.write_text(path.read_text().replace('3 1.0000000E+00', '4 1.0000000E+00'))
    with pytest.raises(ValueError, match='주소'):
        continuum_mapping(path, [1e15, 2e15, 3e15])


def test_continuum_mapping_restores_exact_native_frequency(tmp_path):
    path = tmp_path/'CONT_FREQ'
    path.write_text('\n!NCF, 3\n! original grid\n1 3.0000000E+00 3.0000000E+00 1\n'
                    '2 2.0000000E+00 2.0000000E+00 2\n'
                    '3 1.0000000E+00 1.0000000E+00 3\n')
    original = np.array([1e15, 2e15+12345., 3e15])
    np.testing.assert_array_equal(continuum_mapping(path, original), original)


def test_prior_total_is_not_silently_replaced():
    prior = [dict(mode='mapped-native-SE', Z='14', stage='2', native_node='49',
                  **{f'{scope}_Jref_{process}': '2' for scope in ('bound', 'total')
                     for process in ('photo', 'spontaneous', 'stimulated', 'reverse')})]
    rates = dict.fromkeys(('photo', 'spontaneous', 'stimulated', 'reverse'), 2.)
    check_prior(prior, 14, 2, 49, rates, rates)
    with pytest.raises(ValueError, match='재현 실패'):
        check_prior(prior, 14, 2, 49, rates, {**rates, 'reverse': 3.})


def test_five_digit_print_bound():
    assert printed_half_step(1.2345e4) == .5
    assert printed_half_step(1.2345e-4) == 5e-9
    assert printed_half_step(0.) == 0.


def test_cached_exponent_uses_first_descending_point_per_contiguous_group():
    nu = np.arange(1., 8.)
    evaluation = np.array([2., 2., 4., 4., 4., 2., 2.])
    np.testing.assert_array_equal(cached_exponent_frequency(nu, evaluation),
                                  [2., 2., 5., 5., 5., 7., 7.])
    np.testing.assert_array_equal(cached_exponent_frequency(nu, nu), nu)
    with pytest.raises(ValueError, match='cache'):
        cached_exponent_frequency(nu[::-1], evaluation)


def test_cached_exponent_changes_reverse_not_photo_and_identity_is_exact():
    nu = np.array([1e15, 2e15])
    sampled = dict(chi_abs=np.ones((1, 2)), chi_stim=np.ones((1, 2)),
                   eta_spont=np.ones((1, 2)))
    radiation = {'CMFGEN': np.ones((1, 2))}
    exact = cached_exponent_rates(sampled, nu, nu, [1e4], radiation, np.ones(2))
    old = cached_exponent_rates(sampled, nu, [2e15, 2e15], [1e4], radiation, np.ones(2))
    assert old['photo'] == exact['photo'] == 2.
    assert old['reverse'] < exact['reverse'] == 4.
