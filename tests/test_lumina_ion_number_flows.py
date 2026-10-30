from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_model_response_balance import ion_number_densities, ion_number_flows


def test_number_flows_use_candidate_populations_route_weights_and_terminal():
    roster = {'Z': np.array([14, 14, 16, 16]), 'stage': np.array([2, 2, 2, 3]),
              'lower': np.array([0, 1, 2, 3]), 'target': np.array([1, -1, 3, -1]),
              'probability': np.array([.5, .25, 1., 1.])}
    group = np.array([0, 1, 2, 3]); index = {(14, 2, -1): 4, (16, 3, -1): 5}
    pop = np.array([[10.], [20.], [30.], [40.], [50.], [60.]])
    fraction = np.array([[.2], [.3], [.4], [.5]])
    rates = np.ones((4, 1, 6)); rates[:, :, 0] = 2; rates[:, :, 1] = 3; rates[:, :, 2] = 4
    labels, flows = ion_number_flows(roster, group, index, pop, fraction, rates)
    np.testing.assert_array_equal(labels, [[14, 2], [14, 3], [16, 2], [16, 3]])
    # Si II 두 경로와 terminal 대상. 상위 인구에 분율을 중복 적용하지 않는다.
    expected_photo = .5*10*.2*2 + .25*20*.3*2
    expected_spont = (.5*20 + .25*50)*3
    expected_stim = (.5*20 + .25*50)*4
    np.testing.assert_allclose(flows[0, :, 0], [expected_photo, expected_spont, expected_stim,
                                                expected_photo-expected_spont-expected_stim])
    np.testing.assert_array_equal(flows[1, :, 0], np.zeros(4))
    np.testing.assert_allclose(flows[2, :, 0], [24., 120., 160., -256.])


def test_number_flows_include_every_rostered_si_s_stage():
    roster = {'Z': np.array([14, 14, 16, 16, 16, 16]),
              'stage': np.array([2, 5, 1, 2, 3, 4]),
              'lower': np.zeros(6, dtype=int), 'target': np.full(6, -1, dtype=int),
              'probability': np.ones(6)}
    targets = {(14, 2), (14, 3), (16, 2), (16, 3), (14, 5),
               (16, 1), (16, 4)}
    index = {(z, stage, -1): i+1 for i, (z, stage) in enumerate(sorted(targets))}
    group = np.array([0]); pop = np.arange(1., len(index)+2)[:, None]
    fraction = np.array([[.5]])
    rates = np.ones((6, 1, 6)); rates[:, :, 0] = 2.; rates[:, :, 1] = 3.; rates[:, :, 2] = 4.
    labels, flows = ion_number_flows(roster, group, index, pop, fraction, rates)
    np.testing.assert_array_equal(labels, np.array(sorted(targets)))
    assert flows.shape == (len(targets), 4, 1)
    stage5_si = np.flatnonzero(np.all(labels == [14, 5], axis=1))[0]
    stage1_s = np.flatnonzero(np.all(labels == [16, 1], axis=1))[0]
    assert flows[stage5_si, 0, 0] > 0.
    assert flows[stage1_s, 0, 0] > 0.


def test_density_denominator_uses_charge_not_roman_stage():
    labels = np.array([[14, 2], [14, 3], [16, 2], [16, 3]])
    columns = np.array([[14, 1, 0], [14, 1, 1], [14, 2, 0], [14, 3, -1],
                        [16, 1, 0], [16, 2, 0], [16, 2, 1], [16, 3, -1]])
    pop = np.array([[1., 2.], [3., 4.], [100., 200.], [900., 1800.],
                    [5., 10.], [6., 12.], [7., 14.], [800., 1600.]])
    charges, densities = ion_number_densities(labels, columns, pop)
    np.testing.assert_array_equal(charges, [[14, 1], [14, 2], [16, 1], [16, 2]])
    np.testing.assert_array_equal(densities, [[4., 6.], [100., 200.], [5., 10.], [13., 26.]])
    np.testing.assert_array_equal(labels, [[14, 2], [14, 3], [16, 2], [16, 3]])
