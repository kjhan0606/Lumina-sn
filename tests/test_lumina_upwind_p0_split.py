"""흡수·방출/상풍 분할 후보를 독립 sparse 기준해에 대조."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lumina_upwind_p0_oracle import exact_p0_segment
from lumina_upwind_p0_split import split_p0_segment


def test_actual_drift_constant_chi_spike():
    n = 512
    h, length, drift = 1.1267110173509078e-5, 1e12, .0018759549664
    chi = np.full(n, 3.2 / length)
    eta = np.zeros(n)
    incident = np.zeros(n)
    incident[350] = 1
    expected = exact_p0_segment(h, length, drift, chi, eta, incident)
    got = split_p0_segment(h, length, drift, chi, eta, incident, 16)
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=1e-15)


def test_frequency_varying_chi_converges_to_exact_green():
    n = 256
    h, length, drift = 1.1267110173509078e-5, 1e12, .0018759549664
    x = np.arange(n)
    chi = (.1 + 30 * np.exp(-((x - 100) / 5)**2)) / length
    eta = np.full(n, 1e-15 / length)
    incident = 1 + .3 * np.exp(-((x - 180) / 12)**2)
    expected = exact_p0_segment(h, length, drift, chi, eta, incident)
    errors = []
    for steps in (16, 64, 256, 1024, 4096):
        got = split_p0_segment(h, length, drift, chi, eta, incident, steps)
        errors.append(float(np.max(np.abs(got - expected))))
    assert all(a > 10 * b for a, b in zip(errors[:-1], errors[1:]))
    assert errors[-1] < 5e-6


def test_resolved_positive_opacity_stationary_source_is_finite():
    n = 64
    h, length, drift = .01, 1e12, .4
    chi_length = np.full(n, .1)
    chi_length[20:30] = 3000
    equilibrium = np.full(n, .7)
    chi = chi_length / length
    eta = (chi_length + 3 * drift) * equilibrium / length
    got = split_p0_segment(h, length, drift, chi, eta, equilibrium, 8192)
    np.testing.assert_allclose(got, equilibrium, rtol=1e-13, atol=1e-13)


def test_unresolved_stiff_blue_transient_fails_closed():
    # 두 빈 해석해: 출구 적색 강도는 2.703206098470242e-7이다.
    # 분할만 반복하면 겉으로 안정된 0을 얻는 반례이므로 실행을 거부한다.
    with pytest.raises(ValueError, match='청색 과도구간'):
        split_p0_segment(.01, 1., .005, np.array([.1, 1e6]),
                         np.zeros(2), np.array([0., 1.]), 16384)


def test_invalid_source_rejected():
    with pytest.raises(ValueError):
        split_p0_segment(.01, 1, 0, np.zeros(3), -np.ones(3),
                         np.ones(3), 1)
