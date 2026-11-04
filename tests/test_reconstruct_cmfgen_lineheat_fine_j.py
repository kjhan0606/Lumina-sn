"""Unit checks for the read-only fixed-state line-heat reweighting algebra."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reconstruct_cmfgen_lineheat_fine_j as reweight


def test_reweighted_line_heat_matches_native_equation():
    # Z=Q*(1-alpha*Jnative), with eta=H/Z and alpha=0.4.
    q_one, q_one_fine = reweight.reweighted_line_heat(
        native_heat=1.92,
        native_znet=0.96,
        native_mass=1.2,
        native_jbar=0.5,
        fine_jbar=0.75,
        fine_mass=1.0,
    )
    assert q_one == pytest.approx(1.6)
    assert q_one_fine == pytest.approx(1.4)


def test_profile_slice_recovers_unit_gaussian_mass():
    center = 2.0e15
    sigma = center * 4.5e-5
    frequency = np.linspace(center - 10 * sigma, center + 10 * sigma, 20001)
    delta = np.diff(frequency)
    weights = np.empty_like(frequency)
    weights[0] = delta[0] / 2
    weights[-1] = delta[-1] / 2
    weights[1:-1] = (frequency[2:] - frequency[:-2]) / 2
    _, _, mass = reweight.profile_slice(frequency, weights, center, 4.5e-5)
    assert mass == pytest.approx(1.0, abs=2e-8)


@pytest.mark.parametrize(
    "values",
    [
        (1.0, 0.0, 1.0, 0.5, 0.7, 1.0),
        (1.0, 1.0, 0.0, 0.5, 0.7, 1.0),
        (1.0, 1.0, 1.0, 0.0, 0.7, 1.0),
        (1.0, 1.0, 1.0, 0.5, 0.7, 0.0),
    ],
)
def test_degenerate_line_heat_inputs_fail_closed(values):
    with pytest.raises(ValueError):
        reweight.reweighted_line_heat(*values)
