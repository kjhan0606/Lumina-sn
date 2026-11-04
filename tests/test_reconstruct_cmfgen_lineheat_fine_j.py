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


def test_printed_coordinate_box_contains_perturbed_discrete_masses():
    center=4.9e15;width=4.5e-5
    printed=np.linspace(center-9*center*width,center+9*center*width,73)
    printed=np.array([float(f'{v:.7e}') for v in printed])
    low,high=reweight.profile_mass_printing_interval(printed,center,width)
    eps=.5*10.**(np.floor(np.log10(printed))-7)
    rng=np.random.default_rng(1234)
    for _ in range(50):
        actual=printed+rng.uniform(-1,1,len(printed))*eps
        w=reweight.native_trapuneq_weights(actual)
        phi=np.exp(-((actual-center)/(center*width))**2)/(1.77245385095516*center*width)
        q=np.sum(w*phi*(abs(actual-center)<=6*center*width))
        assert low-1e-14<=q<=high+1e-14
    assert 0<high-low<.01


@pytest.mark.parametrize('nu,width',[(np.array([1.,1.]),.1),(np.array([1.,2.]),0.),
                                    (np.array([np.nan,2.]),.1)])
def test_invalid_mass_printing_box_inputs_rejected(nu,width):
    with pytest.raises(ValueError):reweight.profile_mass_printing_interval(nu,1.,width)


def test_negative_center_and_width_do_not_cancel_validation():
    with pytest.raises(ValueError):
        reweight.profile_mass_printing_interval(np.array([1.,2.]),-1.,-.1)
