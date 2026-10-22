"""실제 광선 조건부 표본의 접점별 강성 해석 기준 직접시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_cmfgen_siii_ray_segment as C
from compare_cmfgen_siii_ray_segment import continuous_scalar_reference


def test_scalar_knot_split_reference_matches_constant_medium_analytic_solution():
    x = np.linspace(0., 1., 101)
    length, drift = 2., .07
    chi = np.full(len(x), .3)
    eta = np.full(len(x), .12)
    incident = np.full(len(x), .8)
    got, pieces = continuous_scalar_reference(x, length, drift, chi, eta,
                                               incident, 50, rtol=1e-10)
    tau = chi[0] * length + 3 * drift
    expected = incident[0] * np.exp(-tau) + eta[0] * length * (-np.expm1(-tau)) / tau
    assert pieces >= 7
    assert got == pytest.approx(expected, rel=2e-8)


def test_scalar_reference_uses_bluer_incident_and_3b_dilution():
    x = np.linspace(0., 1., 101)
    incident = 1. + x
    got, _ = continuous_scalar_reference(x, 1., .055, np.zeros(len(x)),
                                         np.zeros(len(x)), incident, 50, rtol=1e-10)
    assert got == pytest.approx((1. + x[50] + .055) * np.exp(-3 * .055), rel=2e-8)


def test_scalar_reference_rejects_missing_blue_boundary():
    x = np.linspace(0., 1., 11)
    with pytest.raises(ValueError, match='청색 경계'):
        continuous_scalar_reference(x, 1., .2, np.ones(len(x)),
                                    np.ones(len(x)), np.ones(len(x)), 10)


def test_spot_report_preserves_conditional_scope(tmp_path, monkeypatch):
    x = np.linspace(0., 1., 31)
    geom = tmp_path / 'geometry.bin'
    compact = tmp_path / 'compact.bin'
    geom.write_bytes(b'geometry')
    compact.write_bytes(b'compact')
    monkeypatch.setattr(C, 'actual_ray', lambda *_: dict(
        log_frequency=x, incoming=np.ones(len(x)), observed=np.ones(len(x)),
        chi=np.full(len(x), .1), eta=np.full(len(x), .1),
        length=1., drift=.05, geometry=geom, compact=compact))
    out = tmp_path / 'spot.json'
    C.compare_spot(tmp_path, out, 1, 15)
    report = __import__('json').loads(out.read_text())
    assert report['status'] == 'LOCAL_CONDITIONAL_ONLY'
    assert report['operator_contribution'] == 'NOT_DETERMINED'
    assert report['final_goal'] == 'NOT_MET'
    assert report['continuous_frequency_pieces'] >= 2
    with pytest.raises(FileExistsError):
        C.compare_spot(tmp_path, out, 1, 15)
