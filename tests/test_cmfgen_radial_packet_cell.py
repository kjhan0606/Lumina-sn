"""생산 반경 Reader·같은 연속장 세분·원 SC 연속 source를 독립식과 대조한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_cmfgen_local_sc_packet import bezier_control, bezier_source
from compare_cmfgen_radial_packet_cell import (ProductionSampler, RadialCell, build,
                                               radial_reference, sc_reference)
from analyze_cmfgen_spatial_defects import QuadraticBlueKernel


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp('radial-packet'))[0]


def test_exact_production_reader_keeps_ES_times_J_product(library):
    sample = ProductionSampler(library, [.2, .8], [.3, .5], [.1, .4], [1., 4.])
    chi, eta = sample.at(.5)
    assert chi == pytest.approx(.5)
    assert eta == pytest.approx(.4+.25*2.5)
    assert eta != pytest.approx(.5*(.3+.1*1)+.5*(.5+.4*4))
    np.testing.assert_allclose(sample.at(0.), [.2, .4])
    np.testing.assert_allclose(sample.at(1.), [.8, 2.1])
    with pytest.raises(ValueError): sample.at(-.1)


@pytest.mark.parametrize('tau', [1.e-5, .1, 10., 300.])
def test_native_SC_continuous_source_matches_independent_integral(tau):
    sup, target, far, incoming = 2., 4., 5., .7
    control = bezier_control(sup, target, far, tau, .3*tau)
    expected = incoming*np.exp(-tau)+bezier_source(sup, target, far, tau, .3*tau)
    assert sc_reference(incoming, sup, control, target, tau) == pytest.approx(expected, rel=2.e-12)


def test_constant_radial_material_has_the_gray_closed_form(library):
    sampler = ProductionSampler(library, [.7, .7], [.3, .3], [.1, .1], [0., 0.])
    cell = RadialCell(sampler, 2., 3., 1., np.zeros(2), np.zeros(2), 4.)
    value, accuracy = radial_reference(cell, .4)
    tau = .7*cell.length(0., 1.)
    expected = .4*np.exp(-tau)+.3/.7*(-np.expm1(-tau))
    assert value == pytest.approx(expected, rel=2.e-12)
    assert accuracy['effective_tau'] == pytest.approx(tau)


def test_refinement_resamples_one_fixed_radial_field_not_new_chords(library):
    sampler = ProductionSampler(library, [.3, .9], [.2, .4], [.1, .4], [2., 5.])
    cell = RadialCell(sampler, 2., 3., 1., np.array([.03, .05]), np.array([.8, .3]), 3.)
    exact, _ = radial_reference(cell, .4); kernel = QuadraticBlueKernel(library); errors = []
    for pieces in (4, 8, 16, 32):
        length, chi, eta, b, blue = cell.mesh(pieces)
        value = kernel.leg(length, chi, eta, b, blue, cell.alpha, .4)[-1]
        errors.append(abs(value-exact)/exact)
    assert errors[-1] < errors[0]/20 and errors[-1] < 2.e-4
    coarse = cell.mesh(4); fine = cell.mesh(8)
    for a, b in zip(coarse[1:], fine[1:]): np.testing.assert_array_equal(a, b[::2])
