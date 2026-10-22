"""공통 P0 구간 이산 상풍의 합성 직접시험; 물리 분포 수용은 아니다."""
import ctypes as ct
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def segment(tmp_path_factory):
    library=tmp_path_factory.mktemp('upwind-p0')/'upwind.so'
    subprocess.run(['cc','-O2','-std=c11','-Wall','-Wextra','-Werror','-fPIC','-shared',
        str(ROOT/'src/lumina_upwind_p0.c'),'-lm','-o',str(library)],
        check=True,capture_output=True,text=True,timeout=60)
    f=ct.CDLL(str(library)).lumina_upwind_p0_segment
    ptr=np.ctypeslib.ndpointer(dtype='f8',ndim=1,flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int,ct.c_double,ct.c_double,ct.c_double,*[ptr]*4]
    f.restype=ct.c_int

    def run(h,length,drift,chi,eta,incident):
        values=[np.ascontiguousarray(v,dtype='f8') for v in (chi,eta,incident)]
        if len({len(v) for v in values})!=1:
            raise ValueError('입력 주파수 차원 불일치')
        output=np.empty_like(values[0]);status=f(len(output),h,length,drift,*values,output)
        if status:raise ValueError(f'upwind status={status}')
        return output
    return run


@pytest.mark.parametrize('tau',[-2.,-1e-8,0.,1e-8,1.,100.])
def test_static_signed_chi_has_exact_constant_coefficient_solution(segment,tau):
    length=2e12;incident=np.linspace(.1,1.,8);eta=np.linspace(.2,.9,8)/length
    chi=np.full(8,tau/length)
    got=segment(.01,length,0.,chi,eta,incident)
    mean=-np.expm1(-tau)/tau if tau else 1.
    expected=np.exp(-tau)*incident+length*mean*eta
    np.testing.assert_allclose(got,expected,rtol=2e-13,atol=0.)


def test_coupled_constant_radiation_respects_three_b_and_blue_upwind(segment):
    length=1e12;h=.01;drift=.08;b=drift/length
    source=np.full(12,.7);chi=np.full(12,2/length)
    eta=(chi+3*b)*source
    got=segment(h,length,drift,chi,eta,source)
    np.testing.assert_allclose(got,source,rtol=2e-14,atol=0.)


def test_blue_first_bin_has_zero_previous_frequency_coupling(segment):
    length=1e12;h=.02;drift=.1;b=drift/length
    incident=np.linspace(.1,1.,8);chi=np.zeros(8);eta=np.zeros(8)
    got=segment(h,length,drift,chi,eta,incident)
    assert got[-1]==pytest.approx(np.exp(-3*drift)*incident[-1],rel=2e-14)
    assert np.all(np.isfinite(got)) and np.all(got>=0.)


def test_invalid_emission_is_not_clipped(segment):
    with pytest.raises(ValueError):
        segment(.01,1.,0.,np.zeros(8),np.full(8,-1.),np.zeros(8))


def test_vacuum_blue_to_red_drift_converges_under_frequency_refinement(segment):
    drift=.1;steps=256;length=1e12
    def profile(x):
        return 2.+.5*np.exp(-((x-.4)/.06)**2)
    errors=[]
    for nf in (64,128,256):
        x=np.linspace(0.,1.,nf);h=x[1]-x[0]
        value=profile(x)
        zero=np.zeros(nf)
        for _ in range(steps):
            value=segment(h,length/steps,drift/steps,zero,zero,value)
        expected=np.exp(-3*drift)*profile(x+drift)
        errors.append(float(np.max(np.abs(value-expected))/np.max(expected)))
    assert errors[0]>errors[1]>errors[2]
    assert errors[-1]<.03
