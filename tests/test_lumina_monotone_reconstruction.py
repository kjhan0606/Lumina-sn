"""연속 보간 후보의 산술·이산화 차이를 구분한다. 실제 장 검증은 아니다."""
import ctypes as ct
from decimal import Decimal,localcontext
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    library=tmp_path_factory.mktemp('monotone-reconstruction')/'probe.so'
    p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),
        str(ROOT/'tests/monotone_reconstruction_probe.cpp'),'-o',str(library)],capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    lib=ct.CDLL(str(library));a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    lib.monotone_reconstruction_probe.argtypes=[ct.c_int,ct.c_double,a,a]
    lib.monotone_reconstruction_probe.restype=ct.c_int
    lib.monotone_reconstruction_scaled_probe.argtypes=[ct.c_int,ct.c_double,a,ct.c_int,a,
        np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS')]
    lib.monotone_reconstruction_scaled_probe.restype=ct.c_int
    return lib


def run(kernel,values,shift):
    values=np.ascontiguousarray(values,dtype='f8');out=np.full(len(values),-17.)
    return kernel.monotone_reconstruction_probe(len(values),shift,values,out),out


def reference(values,shift,i):
    """독립 Decimal Hermite: 내부 조화미분·양끝 secant이며 SciPy 끝공식과 다르다."""
    with localcontext() as c:
        c.prec=900;D=Decimal.from_float;v=[D(float(x)) for x in values]
        n=len(v);whole=int(np.floor(shift));j=i+whole;t=D(float(shift))-whole
        if j>=n-1:return v[-1]
        if t==0:return v[j]
        d=v[j+1]-v[j]
        def harmonic(a,b):return 2*a*b/(a+b) if a*b>0 else Decimal(0)
        ml=d if j==0 else harmonic(v[j]-v[j-1],d)
        mr=d if j+2>=n else harmonic(d,v[j+2]-v[j+1])
        # Hermite power 기저를 써서 후보의 Bernstein/Positive 연산과 별도 계산한다.
        return ((2*t**3-3*t*t+1)*v[j]+(t**3-2*t*t+t)*ml+
                (-2*t**3+3*t*t)*v[j+1]+(t**3-t*t)*mr)


@pytest.mark.parametrize('kind',['zero','flat','tiny_flat','linear','reverse','sharp','dynamic','near_max'])
def test_values_and_range_against_independent_high_precision(kernel,kind):
    rng=np.random.default_rng(1022226)
    values=dict(zero=np.zeros(17),flat=np.full(17,.7),tiny_flat=np.full(17,np.nextafter(0.,1.)),
        linear=np.arange(17,dtype='f8'),reverse=np.arange(17,dtype='f8')[::-1],
        sharp=np.r_[np.zeros(8),np.ones(9)],dynamic=10.**rng.uniform(-280,280,17),
        near_max=np.linspace(0.,1.,17)*np.finfo('f8').max)[kind]
    for shift in (0.,.125,.5,.875,3.375,17.,1e10):
        code,actual=run(kernel,values,shift);assert code==0,(kind,shift,code)
        for i,a in enumerate(actual):
            j=min(len(values)-1,i+int(np.floor(min(shift,len(values)))))
            lo=min(values[j],values[min(j+1,len(values)-1)]);hi=max(values[j],values[min(j+1,len(values)-1)])
            assert lo<=a<=hi
            wanted=reference(values,shift,i)
            # 이는 마지막 binary64 표현의 구적오차 허용이며 물리값 floor가 아니다.
            tolerance=abs(wanted)*Decimal(64*np.finfo('f8').eps)+2*Decimal.from_float(np.nextafter(0.,1.))
            assert abs(Decimal.from_float(float(a))-wanted)<=tolerance


def test_degenerate_stability_is_not_claimed_as_old_polynomial_exactness(kernel):
    values=np.array([144.,36.,4.,0.,0.,4.,36.,144.])
    for j in (2,5):
        for direction in (-np.inf,np.inf):
            v=values.copy();v[j]=np.nextafter(v[j],direction)
            code,out=run(kernel,v,.5);assert code==0 and out[3]==0.
    code,out=run(kernel,values,.5);assert code==0 and out[3]==0.
    assert out[3]!=.0625  # 옛7차의 정확한 다항식값을 복구한 것이 아니라 다른 이산화다.


def test_branch_neighborhood_is_continuous_and_scale_homogeneous(kernel):
    base=np.array([0.,1.,2.,2.,3.,5.,4.,1.]);results=[]
    for epsilon in (-1e-8,0.,1e-8):
        v=base.copy();v[3]+=epsilon;code,out=run(kernel,v,.375);assert code==0;results.append(out)
    assert max(abs(results[0]-results[1]).max(),abs(results[2]-results[1]).max())<2e-8
    for scale in (1e-80,1e80):
        code,out=run(kernel,base*scale,.375);assert code==0
        np.testing.assert_allclose(out/scale,results[1],rtol=2e-14,atol=0.)


def test_scaled_nonzero_is_preserved_if_final_double_cannot_represent_it(kernel):
    v=np.arange(4)*np.nextafter(0.,1.);m=np.full(1,-17.);e=np.full(1,-17,dtype='i8')
    assert kernel.monotone_reconstruction_scaled_probe(4,.5,v,0,m,e)==0
    assert m[0]>0 and int(e[0]) < -1073
    # 1100비트 이득 후 회복되는 값이다. double로0을 반환하는 대신 실패한다.
    assert np.ldexp(m[0],int(e[0])+1100)>0
    code,out=run(kernel,v,.5);assert code!=0
    np.testing.assert_array_equal(out,np.full(4,-17.))


@pytest.mark.parametrize('bad',['n','negative','nan','shift','infinite_shift'])
def test_invalid_or_unrepresentable_vectors_never_publish_partial_output(kernel,bad):
    v=np.ones(8);shift=.375
    if bad=='n':v=np.ones(3)
    elif bad=='negative':v[2]=-1.
    elif bad=='nan':v[2]=np.nan
    elif bad=='shift':shift=-1.
    else:shift=np.inf
    code,out=run(kernel,v,shift);assert code!=0
    np.testing.assert_array_equal(out,np.full(len(v),-17.))


@pytest.mark.parametrize('kind',['smooth','sharp'])
def test_analytic_frequency_grid_convergence_without_peak_normalization(kernel,kind,tmp_path):
    function=(lambda x:1.+.7*np.sin(2*np.pi*x)) if kind=='smooth' else (
        lambda x:1e-4+np.exp(-((x-.517)/.01)**2))
    sizes=(33,65,129,257,513) if kind=='smooth' else (129,257,513,1025,2049,4097)
    rows=[]
    for n in sizes:
        x=np.linspace(0.,1.,n);values=function(x);worst=0.;functional=0.
        for shift in (.125,.375,.875):
            code,out=run(kernel,values,shift);assert code==0
            target=(np.arange(n-1)+shift)/(n-1);exact=function(target);actual=out[:-1]
            worst=max(worst,float((abs(actual-exact)/exact).max()))
            weight=np.exp(-target)
            functional=max(functional,float(abs(np.sum(weight*actual)-np.sum(weight*exact))/np.sum(weight*exact)))
        rows.append(dict(points=n,maximum_each_output_relative=worst,positive_functional_relative=functional))
    root=Path(os.environ.get('LUMINA_MONOTONE_RESULT_ROOT',str(tmp_path)))
    with (root/f'convergence-{kind}.json').open('x') as stream:
        json.dump(dict(rows=rows,scope='합성 함수/양의 가중합이며 실제 Si/S PHOT이나 전체장 수용이 아니다.'),stream,ensure_ascii=False,indent=2)
    assert rows[-1]['maximum_each_output_relative']<rows[0]['maximum_each_output_relative']/10
    assert rows[-1]['maximum_each_output_relative']<=1e-4
    assert rows[-1]['positive_functional_relative']<=1e-4
