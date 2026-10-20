"""관측 flux 없이 정의한 정적 경계층의 독립 적분/극한을 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_cmfgen_core_layer import exponential_cell_moments, boundary_mode, diffusion_boundary, weighted_quantiles


def test_exponential_moments_against_independent_gauss_quadrature():
    a = np.array([0., 1e-9, .49, .51, 1., 30., 100.])
    x, w = np.polynomial.legendre.leggauss(300)
    x = (x + 1) / 2; w = w / 2
    exact = np.array([np.exp(-a[:, None] * x) @ (w * x**m) for m in range(3)])
    np.testing.assert_allclose(exponential_cell_moments(a), exact, rtol=2e-11, atol=1e-15)
    np.testing.assert_allclose(exponential_cell_moments(np.array([1e6]))[:, 0], [1e-6, 1e-12, 2e-18], rtol=1e-15)


def test_thin_and_thick_modes_not_node_trapezoid():
    chi = np.array([1e-9, 1e6]); r = 1e10
    integral, trap, _, _ = boundary_mode(chi, np.zeros(2), np.ones(2), np.zeros(2), r, 1., 128)
    assert integral[0] == pytest.approx(trap[0], rel=2e-8)
    assert integral[1] / r**2 == pytest.approx(-.25, rel=1e-12)
    assert trap[1] / integral[1] == pytest.approx(1e6, rel=1e-12)


def test_radius_squared_and_linear_boundary_angular_term():
    chi = np.array([2.]); r, width = 1., 2.
    integral, _, _, _ = boundary_mode(chi, np.array([.4]), np.array([1.2]), np.array([.7]), r, width, 160)
    x, w = np.polynomial.legendre.leggauss(400)
    pos = (x + 1) * width / 2; mu = (x + 1) / 2
    angular = .5 * (np.exp(-chi[0] * pos[:, None] / mu) @ (w / 2 * (.8 + .7 * mu)))
    independent = -chi[0] * np.dot(w * width / 2, (r + pos)**2 * angular)
    assert integral[0] == pytest.approx(independent, rel=2e-9)


def test_diffusion_boundary_uses_native_temperature_and_frequency_units():
    nu = np.array([.2, 4., 1000.]); t, dt = .9, 1e-6
    b, dbdr = diffusion_boundary(nu, t, dt)
    delta = 1e-5
    plus = diffusion_boundary(nu, t + delta, 0)[0]
    minus = diffusion_boundary(nu, t - delta, 0)[0]
    np.testing.assert_allclose(dbdr[:2], ((plus - minus) / (2 * delta) * dt)[:2], rtol=2e-7)
    assert b[-1] == 0 and dbdr[-1] == 0


def test_structure_quantiles_retain_small_nonzero_weights():
    assert weighted_quantiles(np.array([1., 2.]), np.array([1e-280, 1e-280])) == pytest.approx([1., 1.9, 1.98])
    assert weighted_quantiles(np.array([1., 2.]), np.array([0., 0.])) is None
