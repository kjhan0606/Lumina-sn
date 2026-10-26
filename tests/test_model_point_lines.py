"""동일 source Gaussian의 비재정규화 Jbar와 유효성 직접시험."""
import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    out=tmp_path_factory.mktemp('point-line')/'lines.so'
    subprocess.run(['gcc','-O2','-std=c11','-fPIC','-shared','-Wl,-z,defs',
        str(ROOT/'src/model_source_kernels.c'),'-lm','-o',str(out)],check=True)
    dll=C.CDLL(str(out));fn=dll.lumina_model_point_gaussian_jbar;ptr=C.POINTER(C.c_double)
    fn.argtypes=[C.c_size_t,C.c_size_t]+[ptr]*4+[C.c_double]*2+[ptr]*2+[C.POINTER(C.c_uint8)]
    fn.restype=C.c_int
    return dll,fn


def evaluate(kernel,nu,width,j,centers,wf=1.e-4,sf=6.e-4):
    arrays=[np.ascontiguousarray(a,dtype='f8') for a in (nu,width,j,centers)]
    out=np.full(len(centers),-123.);mass=out.copy();state=np.full(len(centers),99,dtype='u1')
    ptr=C.POINTER(C.c_double)
    rc=kernel[1](len(nu),len(centers),*[a.ctypes.data_as(ptr) for a in arrays],wf,sf,
        out.ctypes.data_as(ptr),mass.ctypes.data_as(ptr),state.ctypes.data_as(C.POINTER(C.c_uint8)))
    return rc,out,mass,state


def test_same_source_gaussian_integral_and_no_renormalization(kernel):
    nu=np.linspace(1.e15*(1-8.e-4),1.e15*(1+8.e-4),4001)
    width=np.zeros_like(nu);width[:-1]+=.5*np.diff(nu);width[1:]+=.5*np.diff(nu)
    x=(nu-1.e15)/1.e11;j=2.+.1*x+.3*x*x
    rc,out,mass,state=evaluate(kernel,nu,width,j,[1.e15])
    active=abs(nu-1.e15)<=6.e11
    phi=np.exp(-x*x)/(1.77245385095516*1.e11)
    expected=np.sum((phi*width*j)[active]);normalization=np.sum((phi*width)[active])
    assert rc==0 and state[0]==1
    assert out[0]==pytest.approx(expected,rel=3.e-15)
    assert mass[0]==pytest.approx(normalization,rel=3.e-15)
    assert out[0]!=pytest.approx(expected/normalization,rel=1.e-12)


def test_exact_dark_not_missing(kernel):
    nu=np.linspace(.999e15,1.001e15,1001);width=np.full_like(nu,nu[1]-nu[0])
    rc,out,mass,state=evaluate(kernel,nu,width,np.zeros_like(nu),[1.e15])
    assert rc==0 and state[0]==2 and out[0]==0 and mass[0]>0


def test_outside_and_unsampled_are_not_zero_field(kernel):
    nu=np.array([.999e15,1.001e15]);width=np.array([1.e12,1.e12])
    rc,out,mass,state=evaluate(kernel,nu,width,[1.,1.],[1.e15,2.e15])
    assert rc==0 and state.tolist()==[4,3]
    assert np.isnan(out).all() and mass[0]==0 and np.isnan(mass[1])


def test_bad_late_input_preserves_outputs(kernel):
    rc,out,mass,state=evaluate(kernel,[1.e15,2.e15],[.5e15,.5e15],[1.,np.nan],[1.5e15])
    assert rc!=0 and out[0]==-123 and mass[0]==-123 and state[0]==99
