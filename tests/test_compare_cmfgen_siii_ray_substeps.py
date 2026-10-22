"""실제 광선 주파수 연쇄 비교기의 작은 해석해 시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_cmfgen_siii_ray_substeps import (forced_bezier_one_step,
                                              linear_sc_sweep, sc_sweep)


def make_data(chi, eta, b, incoming, previous):
    n = len(incoming)
    return dict(incoming=np.asarray(incoming, dtype='f8'),
                previous_inner=np.asarray(previous, dtype='f8'),
                previous_outer=np.asarray(previous, dtype='f8'),
                observed=np.zeros(n), chi_inner=np.full(n, chi),
                chi_outer=np.full(n, chi), eta_inner=np.full(n, eta),
                eta_outer=np.full(n, eta), b_inner=b, b_outer=b,
                length=1., dlognu=np.full(n, .01))


def test_linear_sc_substeps_reduce_to_constant_source_analytic_solution():
    data = make_data(1., 1., 0., np.zeros(4), np.zeros(4))
    for pieces in (1, 2, 8, 32):
        got = linear_sc_sweep(data, pieces)
        np.testing.assert_allclose(got, 1. - np.exp(-1.), rtol=2e-13)


def test_linear_sc_substeps_preserve_stationary_frequency_coupled_field():
    chi, b, value = .1, .02, 2.
    data = make_data(chi, (chi + 3 * b) * value, b,
                     np.full(4, value), np.full(4, value))
    for pieces in (1, 4, 16):
        np.testing.assert_allclose(linear_sc_sweep(data, pieces), value,
                                   rtol=1e-13)


def test_bezier_sc_substeps_reduce_to_constant_source_and_forced_step():
    data = make_data(1., 1., 0., np.zeros(4), np.zeros(4))
    data['far_neighbor'] = dict(previous=np.zeros(4), chi=np.ones(4),
                                eta=np.ones(4), b=0., length=1.)
    expected = 1. - np.exp(-1.)
    assert forced_bezier_one_step(data, 2) == pytest.approx(expected, rel=1e-13)
    for pieces in (1, 2, 8):
        np.testing.assert_allclose(sc_sweep(data, pieces, mode='bezier'),
                                   expected, rtol=1e-12)


def test_linear_sc_substeps_reject_invalid_range():
    data = make_data(1., 1., 0., np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        linear_sc_sweep(data, 0)
    with pytest.raises(ValueError):
        linear_sc_sweep(data, 1, 4)
