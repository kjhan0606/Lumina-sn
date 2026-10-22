"""실제 S III 가중율 표본 선택 규칙의 소형 직접시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rank_cmfgen_siii_weighted_spots import top_spots


def test_top_spots_respects_signed_rate_tau_and_blue_margin():
    contribution = np.array([2., -100., 9., 8., 5., 7.])
    tau = np.array([2., 2., 200., 4., 5., 6.])
    margin = np.array([.2, .2, .2, .01, .2, .2])
    result = top_spots(contribution, tau, margin, .01, count=2)
    assert [spot['reverse_band_index'] for spot in result] == [5, 4]
    assert [spot['signed_photo_residual_cm3_s'] for spot in result] == [7., 5.]


def test_top_spots_rejects_invalid_depth_and_shape():
    with pytest.raises(ValueError):
        top_spots(np.ones(3), np.array([1., -1., 1.]), np.ones(3), .1)
    with pytest.raises(ValueError):
        top_spots(np.ones(3), np.ones(2), np.ones(3), .1)
