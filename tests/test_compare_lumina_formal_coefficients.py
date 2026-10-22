"""계수 비교의 주파수 방향과 부호/절댓값 적분을 직접 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_lumina_native_formal_coefficients import band_metrics


def test_identical_and_scaled_coefficients():
    frequency = np.array([2.5e16, 2.0e16, 1.5e16])  # native 저장 방향: 고→저 ν
    reference = np.array([1., 2., 3.])
    same = band_metrics(frequency, reference, reference, 100, 400)
    doubled = band_metrics(frequency, reference, 2 * reference, 100, 400)
    assert same['points'] == 3
    assert same['signed_integral_difference_over_reference_abs'] == 0
    assert same['unsigned_integral_difference_over_reference_abs'] == 0
    assert doubled['signed_integral_difference_over_reference_abs'] == pytest.approx(1)
    assert doubled['unsigned_integral_difference_over_reference_abs'] == pytest.approx(1)


def test_signed_cancellation_does_not_hide_unsigned_error():
    frequency = np.array([2.5e16, 2.0e16, 1.5e16])
    reference = np.ones(3)
    compared = np.array([2., 0., 2.])
    result = band_metrics(frequency, reference, compared, 100, 400)
    assert result['signed_integral_difference_over_reference_abs'] == pytest.approx(0)
    assert result['unsigned_integral_difference_over_reference_abs'] == pytest.approx(1)
