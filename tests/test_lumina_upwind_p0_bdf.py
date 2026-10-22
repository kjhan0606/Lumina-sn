"""강성 이산 상풍 기준해의 독립 해석해·행렬지수 대조."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lumina_upwind_p0_bdf import bdf_p0_segment, bdf_p0_segment_nonuniform
from lumina_upwind_p0_oracle import exact_p0_segment


def test_stiff_two_frequency_transient_matches_analytic_solution():
    drift, h, length = .005, .01, 1.
    lam = drift / h
    chi = np.array([.1, 1e6])
    low, blue = chi[0] + 3 * drift + lam, chi[1] + 3 * drift
    expected = lam * (math.exp(-blue) - math.exp(-low)) / (low - blue)
    got, info = bdf_p0_segment(h, length, drift, chi, np.zeros(2),
                               np.array([0., 1.]), rtol=1e-9, atol=1e-14)
    assert info['nfev'] > 0 and info['nlu'] > 0
    np.testing.assert_allclose(got[0], expected, rtol=1e-6, atol=0)


def test_variable_chi_matches_sparse_matrix_exponential():
    n = 256
    h, length, drift = 1.1267110173509078e-5, 1e12, .0018759549664
    x = np.arange(n)
    chi = (.1 + 30 * np.exp(-((x - 100) / 5)**2)) / length
    eta = np.full(n, 1e-15 / length)
    incident = 1 + .3 * np.exp(-((x - 180) / 12)**2)
    expected = exact_p0_segment(h, length, drift, chi, eta, incident)
    got, _ = bdf_p0_segment(h, length, drift, chi, eta, incident,
                            rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(got, expected, rtol=3e-7, atol=1e-9)


def test_signed_chi_stationary_solution():
    n, length, drift, h = 24, 1e12, .04, .01
    chi = np.linspace(-.01, 1., n) / length
    source = np.full(n, .7)
    eta = (chi + 3 * drift / length) * source
    got, _ = bdf_p0_segment(h, length, drift, chi, eta, source)
    np.testing.assert_allclose(got, source, rtol=1e-11, atol=1e-11)


def test_invalid_negative_emission_rejected():
    with pytest.raises(ValueError):
        bdf_p0_segment(.01, 1., 0., np.zeros(3), -np.ones(3),
                       np.ones(3))


def test_nonuniform_bdf_reduces_to_existing_uniform_operator():
    n = 24
    x = np.arange(n, dtype='f8') * .01
    chi = np.linspace(.05, 2., n)
    eta = np.full(n, .02)
    incident = 1. + .2 * np.sin(np.arange(n))**2
    old, _ = bdf_p0_segment(.01, 1., .04, chi, eta, incident,
                            rtol=1e-10, atol=1e-14)
    new, _ = bdf_p0_segment_nonuniform(x, 1., .04, chi, eta, incident,
                                       rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(new, old, rtol=2e-12, atol=2e-12)


def test_nonuniform_two_bin_transient_matches_analytic_solution():
    x = np.array([0., .037])
    drift, length = .019, 1.
    chi = np.array([.1, 4.])
    lam = drift / (x[1] - x[0])
    low, blue = chi[0] + 3 * drift + lam, chi[1] + 3 * drift
    expected = lam * (math.exp(-blue) - math.exp(-low)) / (low - blue)
    got, _ = bdf_p0_segment_nonuniform(x, length, drift, chi,
                                       np.zeros(2), np.array([0., 1.]),
                                       rtol=1e-10, atol=1e-14)
    assert got[0] == pytest.approx(expected, rel=2e-8)


def test_nonuniform_frequency_order_is_required():
    with pytest.raises(ValueError, match='비등간격'):
        bdf_p0_segment_nonuniform([0., .02, .02], 1., .1,
                                  np.ones(3), np.ones(3), np.ones(3))
