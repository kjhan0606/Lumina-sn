"""공간격자 조사에서 순서·이전 수치오차·약한 값을 보존하는지 확인한다."""
import math
import pytest
from lumina_linear_native import Map, Positive
from lumina_spatial_curve_cpu import advance, ratio


def transfer(t, source, error=0., tau_error=0.):
    return Map(Positive.of(t), Positive.of(source), Positive.of(error), tau_error)


def test_upstream_then_downstream_source_order():
    value, error = advance(Positive(), Positive(), transfer(1., 3.))
    value, error = advance(value, error, transfer(.5, 2.))
    assert value.value() == 3.5
    assert error.value() > 0.


def test_upstream_error_uses_transmission_upper_bound():
    value, error = advance(Positive.of(2.), Positive.of(.1), transfer(2., 3., .25, .01))
    expected = .1*2.*math.exp(.01)+.25+4.*math.expm1(.01)
    assert value.value() == 7.
    assert error.value() == pytest.approx(expected, rel=1e-12)


def test_intermediate_underflow_does_not_destroy_recoverable_signal():
    value, error = advance(Positive(), Positive(), transfer(1., 1.))
    for transmission in (Positive(.5, -1599), Positive(.5, 1601)):
        value, error = advance(value, error, Map(transmission, Positive(), Positive(), 0.))
    assert value.value() == 1.
    assert ratio(error, value) < 1e-12


def test_each_weak_output_keeps_its_own_relative_scale():
    assert ratio(Positive(.5, -2010), Positive(.5, -2000)) == 2.**-10
    assert ratio(Positive(), Positive()) == 0.
    assert ratio(Positive.of(1.), Positive()) is None
