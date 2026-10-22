"""기각된 CRAM16을 음성 대조군으로 보존하는 반례 시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_upwind_p0_cram import cram16_p0_segment
from lumina_upwind_p0_oracle import exact_p0_segment


def test_actual_cell_spike_rejects_cram16():
    n=512;peak=350;h=1.1267110173509078e-5
    length=1e12;drift=.0018759549664
    chi=np.full(n,3.2/length);eta=np.zeros(n);incident=np.zeros(n)
    incident[peak]=1.
    exact=exact_p0_segment(h,length,drift,chi,eta,incident)
    assert np.isfinite(exact).all() and np.min(exact)>=0
    with pytest.raises(ValueError,match='CRAM16 후보 비유한 또는 음수'):
        cram16_p0_segment(h,length,drift,chi,eta,incident)


def test_frequency_varying_chi_exposes_cram16_error():
    n=256;length=1e12;drift=.0018759549664;h=1.1267110173509078e-5
    x=np.arange(n)
    chi=(.1+30*np.exp(-((x-100)/5)**2))/length
    eta=np.full(n,1e-15/length)
    incident=1.+.3*np.exp(-((x-180)/12)**2)
    exact=exact_p0_segment(h,length,drift,chi,eta,incident)
    candidate=cram16_p0_segment(h,length,drift,chi,eta,incident)
    relative=np.max(np.abs(candidate-exact)/np.maximum(exact,1e-100))
    assert relative>0.01
