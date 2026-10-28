"""계수 규약 대조의 외삽·signed 값·분모 처리를 확인한다."""
import numpy as np
import pytest

from compare_cmfgen_band_contract import contrast, log_sample


def test_log_interpolation_preserves_native_values_and_signed_chi():
    nu = np.array([1., 4., 16.]); value = np.array([-1., 3., 1.])
    np.testing.assert_array_equal(log_sample(nu,value,nu),value)
    assert log_sample(nu,value,[2.])[0] == pytest.approx(1.)


@pytest.mark.parametrize('nu,value,requested', [
    ([1.,4.],[1.,2.],[.99]), ([1.,4.],[1.,2.],[4.01]),
    ([4.,1.],[1.,2.],[2.]), ([1.,4.],[1.],[2.]),
    ([1.,4.],[1.,np.nan],[2.]), ([1.,4.],[1.,2.],[]),
])
def test_invalid_interpolation_is_not_silent_extrapolation(nu,value,requested):
    with pytest.raises(ValueError): log_sample(nu,value,requested)


def test_zero_and_negative_pairs_are_not_given_positive_ratios():
    result = contrast([0.,-1.,2.,4.],[0.,1.,1.,2.])
    assert result['nonpositive_pairs'] == 2
    assert result['symmetric_max'] == 2.
    assert result['positive_ratio_min'] == result['positive_ratio_max'] == 2.
    assert contrast([0.,0.],[0.,0.])['positive_ratio_min'] is None


@pytest.mark.parametrize('x,y', [([],[]),([np.inf],[1.]),([1.,2.],[1.])])
def test_contrast_rejects_invalid_arrays(x,y):
    with pytest.raises(ValueError): contrast(x,y)
