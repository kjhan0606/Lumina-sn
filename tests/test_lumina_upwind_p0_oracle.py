"""정확한 소형 P0 이산 상풍 연립 기준해의 독립 해석 사례."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_upwind_p0_oracle import exact_p0_segment


def test_spike_has_poisson_green_function_at_actual_cell_drift():
    nf=512;peak=350;h=1.1267110173509078e-5
    drift=.0018759549664;length=1e12;tau=3.2
    incident=np.zeros(nf);incident[peak]=1.
    got=exact_p0_segment(h,length,drift,np.full(nf,tau/length),
                         np.zeros(nf),incident)
    lam=drift/h
    expected=np.zeros(nf)
    for k in range(peak+1):
        r=peak-k
        expected[k]=math.exp(-tau-3*drift-lam+r*math.log(lam)-math.lgamma(r+1))
    np.testing.assert_allclose(got,expected,rtol=2e-10,atol=1e-15)


def test_stationary_solution_and_signed_chi_are_not_clipped():
    nf=24;length=1e12;drift=.04;h=.01;b=drift/length
    chi=np.linspace(-.01,1.,nf)/length
    source=np.full(nf,.7)
    eta=(chi+3*b)*source
    assert np.all(eta>=0)
    got=exact_p0_segment(h,length,drift,chi,eta,source)
    np.testing.assert_allclose(got,source,rtol=2e-13,atol=0)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        exact_p0_segment(.01,1.,0.,np.zeros(4),np.full(4,-1.),np.zeros(4))
