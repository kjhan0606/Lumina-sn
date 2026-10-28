"""부분대역 가중치가 원 격자 폭과 같은 점의 이온 분모를 유지하는지 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reweight_cmfgen_native_band import band_weights, rates
from compare_lumina_checkpoint_rates import photon_weights


def test_band_edges_keep_outside_neighbor_quadrature_width():
    nu = np.array([1., 2., 4., 7., 11.])*1.e15
    mask = np.array([False, True, True, True, False])
    got = band_weights(nu, mask, np.ones(3))
    np.testing.assert_array_equal(got, photon_weights(nu)[mask])
    assert got[0] > photon_weights(nu[mask])[0]
    assert got[-1] > photon_weights(nu[mask])[-1]


def test_volume_and_ion_rates_use_one_native_population_not_cell_index():
    result = rates(np.array([2., 3.]), {'J': np.array([5., 7.])}, 31.)
    assert result['J']['photo_cm3_s'] == 31.
    assert result['J']['effective_photo_s'] == 1.
    with pytest.raises(ValueError): rates(np.ones(2), {'J': np.ones(2)}, 0.)
    with pytest.raises(ValueError): rates(np.ones(2), {'J': np.ones(3)}, 1.)


def test_band_weight_does_not_hide_zero_support_or_negative_coefficients():
    nu = np.array([1., 2., 3.])*1.e15; mask = np.array([True, True, False])
    weight = band_weights(nu, mask, np.array([0., 1.]))
    assert weight[0] == 0.
    with pytest.raises(ValueError): band_weights(nu, mask, np.array([-1., 1.]))
