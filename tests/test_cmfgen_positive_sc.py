"""연속 SC 적분의 독립 수학 확인과 실제 실패 source의 재현."""
import csv
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_cmfgen_formal_primary import BUILD,NATIVE,native_link


@pytest.fixture(scope='module')
def sc(tmp_path_factory):
    root=tmp_path_factory.mktemp('positive_sc_math')
    wrapper=root/'wrapper.f90'
    wrapper.write_text('''subroutine weights(tau,w)bind(c)
use iso_c_binding
use lumina_positive_sc
real(c_double),value::tau
real(c_double)::w(3)
call sc_bernstein_weights(tau,w)
end subroutine
subroutine source(s0,s1,s2,t,h,value)bind(c)
use iso_c_binding
use lumina_positive_sc
real(c_double),value::s0,s1,s2,t,h
real(c_double)::value
value=sc_bezier_source(s0,s1,s2,t,h)
end subroutine
''')
    so=root/'sc.so'
    subprocess.run(['gfortran','-shared','-fPIC','-O','-ffree-line-length-none','-I'+str(NATIVE/'mod'),
        str(BUILD/'cmfgen_positive_sc.f90'),str(wrapper),'-o',str(so)],cwd=root,check=True,capture_output=True,timeout=30)
    lib=ctypes.CDLL(str(so));ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.weights.argtypes=[ctypes.c_double,ptr]
    lib.source.argtypes=[ctypes.c_double]*5+[ctypes.POINTER(ctypes.c_double)]
    def weights(tau):
        out=np.zeros(3);lib.weights(tau,out);return out
    def source(s0,s1,s2,t,h):
        out=ctypes.c_double();lib.source(s0,s1,s2,t,h,ctypes.byref(out));return out.value
    return weights,source


def test_bernstein_integrals_have_positive_weights_and_correct_sum(sc):
    weights,_=sc
    x,w=np.polynomial.legendre.leggauss(128);u=(x+1)/2;w=w/2
    basis=np.array([(1-u)**2,2*u*(1-u),u*u])
    for tau in np.geomspace(1e-14,100,61):
        exact=tau*(basis@(w*np.exp(-tau*(1-u))))
        actual=weights(tau)
        assert np.all(actual>=0)
        np.testing.assert_allclose(actual,exact,rtol=2e-11,atol=0)
        assert actual.sum()==pytest.approx(-np.expm1(-tau),rel=3e-14,abs=0)
    for tau in (0.,1e-280,1e10,1e280):
        actual=weights(tau)
        assert np.isfinite(actual).all() and np.all(actual>=0)
        assert actual.sum()==pytest.approx(-np.expm1(-tau),rel=3e-14,abs=0)


def test_constant_linear_and_unlimited_quadratic_source(sc):
    _,source=sc
    x,w=np.polynomial.legendre.leggauss(128);u=(x+1)/2;w=w/2
    for tau,h in ((.001,.1),(.5,1.),(1.,1.),(3.,.1),(100.,10.)):
        assert source(2.,2.,2.,tau,h)==pytest.approx(2*-np.expm1(-tau),rel=3e-14)
        s0,s1=1.,2.;s2=s1+h/tau*(s1-s0)
        exact=tau*np.dot(w*np.exp(-tau*(1-u)),s0+(s1-s0)*u)
        assert source(s0,s1,s2,tau,h)==pytest.approx(exact,rel=2e-11)
    # S(x)=1+x/2+x^2/2, x=0,1,2의 세 점. control point는 범위 내다.
    exact=np.dot(w*np.exp(-(1-u)),1+.5*u+.5*u*u)
    assert source(1.,2.,4.,1.,1.)==pytest.approx(exact,rel=1e-13)
    for t,h in ((1e-280,1e280),(1e280,1e-280)):
        value=source(1e-280,1e200,1e280,t,h)
        assert np.isfinite(value) and value>=0


def test_actual_native_zero_crossing_loses_its_discontinuous_jump(sc):
    _,source=sc
    path=Path('/gpfs/kjhan/cmfgen_runs/lumina_formal_source_64658_replay_20261019-v1/branch_probe/ACTUAL_SOURCE_BRANCHES.bin')
    dtype=np.dtype([(k,'<i4') for k in ('trial','ip','iz','dir')]+[(k,'<f8') for k in ('t1','s0','s1','s2','d0','d1','a','b','c')])
    data=np.fromfile(path,dtype=dtype)
    row=data[(data['trial']==18)&(data['ip']==51)&(data['iz']==74)&(data['dir']==1)][0]
    critical=-(row['a']*row['s0']+row['c']*row['s2'])/row['b']
    outputs=[];old=[]
    for sign in (-1,1):
        target=critical*(1+sign*1e-11)
        raw=row['a']*row['s0']+row['b']*target+row['c']*row['s2']
        linear=row['a']*(target+row['d0']*(target-row['s2'])/row['d1'])+row['b']*target+row['c']*row['s2']
        old.append(linear if raw<0 else raw)
        outputs.append(source(row['s2'],target,row['s0'],row['d1'],row['d0']))
    assert abs(old[1]-old[0])>1e-6
    assert abs(outputs[1]-outputs[0])<1e-14
    assert all(x>0 for x in outputs)


@pytest.fixture(scope='module')
def failure_exe(tmp_path_factory):
    root=tmp_path_factory.mktemp('positive_sc_packet');exe=root/'replay'
    native_link(Path(__file__).resolve().parents[1]/'scripts/replay_cmfgen_formal_source_failure.f90',exe)
    return exe


def test_feature_off_preserves_actual_native_last_call(failure_exe,tmp_path):
    root=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428766')
    result=subprocess.run([str(failure_exe),str(root),'original'],cwd=tmp_path,capture_output=True,timeout=15,
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='',LUMINA_FORMAL_PRIMARY_WHOLE='',LUMINA_WHOLE_MODE=''))
    assert result.returncode==0,result.stderr
    assert (tmp_path/'ACTUAL_LAST_CALL_RAYS.bin').read_bytes()==(root/'FORMAL_PRIMARY_RAYS.bin').read_bytes()


@pytest.mark.parametrize('method',['anderson','picard'])
def test_two_cycle_disappears_without_increasing_iteration_cap(failure_exe,tmp_path,method):
    root=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428766')
    result=subprocess.run([str(failure_exe),str(root),method],cwd=tmp_path,capture_output=True,timeout=20,
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='',LUMINA_WHOLE_MODE=''))
    assert result.returncode==0,result.stderr
    rows=list(csv.DictReader((tmp_path/'SOURCE_REPLAY.csv').open()))
    assert len(rows)<=20
    assert max(float(rows[-1][k]) for k in ('J_residual','source_residual','H_change_over_J'))<1e-8
    assert b'DERIVED_SOURCE_CAUSAL_REPLAY_NOT_EXACT_SOURCE_OR_ANCHOR' in result.stdout
