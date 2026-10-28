"""저장 잔차 판독의 유효 source 끝점/미분식을 독립 닫힌식과 대조한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_cmfgen_cell_residuals import endpoint_source


def test_constant_endpoint_source_separates_material_and_frequency_depth():
    result = endpoint_source(np.full(2, .2), np.full(2, .3), np.full(2, .1),
                             np.full(2, .7), 4., 0, 1, 2.)
    assert result['endpoint_source'] == pytest.approx(.58/.9)
    assert result['upstream_source'] == pytest.approx(.58/.9)
    assert result['material_chi_target'] == pytest.approx(.2)
    assert result['effective_opacity_upstream'] == pytest.approx(.9)
    assert result['first_order_intensity'] == pytest.approx(.58/.9)
    assert result['tau_chi'] == pytest.approx(.4)
    assert result['tau_frequency_drift'] == pytest.approx(1.4)
    assert result['blue_emission_fraction'] == pytest.approx(.28/.58)


@pytest.mark.parametrize('up,target', [(0, 1), (1, 0)])
def test_endpoint_derivative_has_the_correct_propagation_sign(up, target):
    result = endpoint_source(np.ones(2), np.array([2., 5.]), np.zeros(2),
                             np.zeros(2), 4., up, target, 2.)
    assert result['first_order_intensity'] == pytest.approx(3.5)
