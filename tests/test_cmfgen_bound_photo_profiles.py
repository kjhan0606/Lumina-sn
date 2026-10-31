"""전치 보간이 fine 격자 적분을 바꾸지 않는지 독립 직접합으로 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cmfgen_bound_photo_profiles import LogInterpolationAdjoint


def test_adjoint_matches_direct_interpolation_for_multiple_depths_and_edges():
    native = np.geomspace(1.e12, 1.e18, 57)
    fine = np.unique(np.r_[np.geomspace(native[0], native[-1], 1201), native])
    rng = np.random.default_rng(71)
    J = np.exp(rng.normal(size=(7, len(native))))
    weights = np.exp(rng.normal(size=len(fine)))
    operator = LogInterpolationAdjoint(native, fine)
    projected = operator.project(weights)
    expected = np.array([np.sum(np.asarray(weights, dtype=np.longdouble)*
                                 np.interp(np.log(fine), np.log(native), row), dtype=np.longdouble)
                         for row in J], dtype='f8')
    np.testing.assert_allclose(J@projected, expected, rtol=5.e-14, atol=0)
    assert np.all(projected >= 0)
    np.testing.assert_allclose(projected.sum(), weights.sum(), rtol=5.e-15)


def test_identity_grid_zero_weights_and_endpoint_support_are_exact():
    grid = np.array([1., 2., 4.])
    operator = LogInterpolationAdjoint(grid, grid)
    np.testing.assert_array_equal(operator.project([2., 3., 5.]), [2., 3., 5.])
    np.testing.assert_array_equal(operator.project(np.zeros(3)), np.zeros(3))
    edge = LogInterpolationAdjoint(grid, grid[[0, -1]])
    np.testing.assert_array_equal(edge.project([7., 11.]), [7., 0., 11.])


@pytest.mark.parametrize('native,fine', [([1.], [1., 2.]), ([1., 1.], [1., 2.]),
    ([2., 1.], [1., 2.]), ([0., 2.], [1., 2.]), ([1., np.inf], [1., 2.]),
    ([1.e18, np.nextafter(1.e18, np.inf)], [1.e18, np.nextafter(1.e18, np.inf)]),
    ([1., 4.], [.9, 2.]), ([1., 4.], [2., 4.1]), ([1., 4.], [2., 1.])])
def test_bad_grids_and_extrapolation_are_rejected(native, fine):
    with pytest.raises(ValueError):
        LogInterpolationAdjoint(native, fine)


@pytest.mark.parametrize('weights', [[1.], [1., -1.], [1., np.nan], [np.inf, 1.], [[1., 2.]]])
def test_bad_weights_are_not_clamped(weights):
    operator = LogInterpolationAdjoint([1., 4.], [1., 4.])
    with pytest.raises(ValueError):
        operator.project(weights)


def test_existing_output_is_preserved_before_any_input_is_opened(tmp_path):
    from compare_cmfgen_bound_photo_profiles import run
    output = tmp_path/'keep'; output.mkdir()
    marker = output/'result.json'; marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        run(tmp_path/'absent.json', '0'*64, output)
    assert marker.read_text() == 'preserve'


def test_changed_config_does_not_create_output(tmp_path):
    from compare_cmfgen_bound_photo_profiles import run
    config = tmp_path/'config.json'; config.write_text('{}')
    output = tmp_path/'not-created'
    with pytest.raises(ValueError, match='비교 입력 변경'):
        run(config, '0'*64, output)
    assert not output.exists()
