"""충돌 subset 하한과 평가된 local 잔차의 구간을 정확 유리수로 대조한다."""
import ctypes
from fractions import Fraction as F
from pathlib import Path
import subprocess

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/cmfgen_se_collision_bound.f90'


@pytest.fixture(scope='module')
def witness(tmp_path_factory):
    root = tmp_path_factory.mktemp('se_collision_bound')
    wrapper = root/'wrapper.f90'
    wrapper.write_text('''subroutine collision(n,c,h,hs,ne,inc,out,valid)bind(c)
use iso_c_binding
use lumina_se_collision_bound
integer(c_int),value::n
real(c_double)::c(n,n),h(n),hs(n),inc(n+1),out(n+1)
real(c_double),value::ne
integer(c_int)::valid
logical::ok
call collision_lower(n,c,h,hs,ne,inc,out,ok)
valid=merge(1,0,ok)
end subroutine
subroutine raw(n,terms,lo,hi,valid)bind(c)
use iso_c_binding
use lumina_se_collision_bound
integer(c_int),value::n
real(c_double)::terms(n),lo,hi
integer(c_int)::valid
integer::i
logical::ok
lo=0;hi=0;ok=.true.
do i=1,n
call raw_interval_add(lo,hi,terms(i),ok)
end do
valid=merge(1,0,ok)
end subroutine
subroutine certificate(lo,hi,inc,out,covered,valid,tol,upper,proven)bind(c)
use iso_c_binding
use lumina_se_collision_bound
real(c_double),value::lo,hi,inc,out,tol
integer(c_int),value::covered,valid
real(c_double)::upper
integer(c_int)::proven
logical::ok
call residual_upper(lo,hi,inc,out,covered==1,valid==1,tol,upper,ok)
proven=merge(1,0,ok)
end subroutine
subroutine unsafe_mode(upper,proven)bind(c)
use iso_c_binding
use ieee_arithmetic
use lumina_se_collision_bound
real(c_double)::upper
integer(c_int)::proven
logical::saved,ok
call ieee_get_underflow_mode(saved)
call ieee_set_underflow_mode(.false.)
call residual_upper(0.0_c_double,1e-300_c_double,1e300_c_double,0.0_c_double,.true.,.true.,1e-8_c_double,upper,ok)
call ieee_set_underflow_mode(saved)
proven=merge(1,0,ok)
end subroutine
''')
    libpath=root/'witness.so'
    subprocess.run(['gfortran','-shared','-fPIC','-O','-ffree-line-length-none',
                    str(SOURCE),str(wrapper),'-o',str(libpath)],cwd=root,
                   check=True,capture_output=True,timeout=30)
    lib=ctypes.CDLL(str(libpath))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    iptr=ctypes.POINTER(ctypes.c_int);dptr=ctypes.POINTER(ctypes.c_double)
    lib.collision.argtypes=[ctypes.c_int,ptr,ptr,ptr,ctypes.c_double,ptr,ptr,iptr]
    lib.raw.argtypes=[ctypes.c_int,ptr,dptr,dptr,iptr]
    lib.certificate.argtypes=[ctypes.c_double]*4+[ctypes.c_int]*2+[ctypes.c_double,dptr,iptr]
    lib.unsafe_mode.argtypes=[dptr,iptr]
    def collision(c,h,hs,ne):
        c,h,hs=[np.array(x,dtype=float) for x in (c,h,hs)]
        n=len(h);inc=np.zeros(n+1);out=np.zeros(n+1);valid=ctypes.c_int()
        snapshots=[x.tobytes() for x in (c,h,hs)]
        lib.collision(n,c.ravel(order='F'),h,hs,ne,inc,out,ctypes.byref(valid))
        assert snapshots==[x.tobytes() for x in (c,h,hs)]
        return inc,out,bool(valid.value)
    def raw(terms):
        terms=np.array(terms,dtype=float);lo=ctypes.c_double();hi=ctypes.c_double();valid=ctypes.c_int()
        lib.raw(len(terms),terms,ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(valid))
        return lo.value,hi.value,bool(valid.value)
    def certificate(lo,hi,inc,out,covered=True,valid=True,tol=1e-8):
        upper=ctypes.c_double();proven=ctypes.c_int()
        lib.certificate(lo,hi,inc,out,covered,valid,tol,ctypes.byref(upper),ctypes.byref(proven))
        return upper.value,bool(proven.value)
    def unsafe_mode():
        upper=ctypes.c_double();proven=ctypes.c_int()
        lib.unsafe_mode(ctypes.byref(upper),ctypes.byref(proven))
        return upper.value,bool(proven.value)
    return collision,raw,certificate,unsafe_mode


def exact_gross(c,h,hs,ne):
    c=[[F(float(x)) for x in row] for row in c];h=list(map(lambda x:F(float(x)),h))
    hs=list(map(lambda x:F(float(x)),hs));ne=F(float(ne));n=len(h)
    inc=[F(0) for _ in range(n+1)];out=inc.copy()
    for j in range(n):
        for l in range(n):
            if l!=j:
                inc[j]+=ne*h[l]*c[l][j];out[j]+=ne*h[j]*c[j][l]
        ion=ne*h[j]*c[j][j];rec=ne*hs[j]*c[j][j]
        inc[j]+=rec;out[j]+=ion;inc[-1]+=ion;out[-1]+=rec
    return inc,out


def test_bb_direction_and_upper_ground_ionization_recombination_are_not_swapped(witness):
    c=np.array([[.3,2.],[7.,.5]]);h=[3.,11.];hs=[5.,13.];ne=17.
    inc,out,valid=witness[0](c,h,hs,ne)
    exact_in,exact_out=exact_gross(c,h,hs,ne)
    assert valid
    for actual,expected in zip([*inc,*out],[*exact_in,*exact_out]):
        assert 0<F(actual)<=expected
        assert float(expected)==pytest.approx(actual,rel=2e-14)
    assert sum(exact_in)==sum(exact_out)
    assert inc[-1]==pytest.approx(17*(3*.3+11*.5),rel=2e-14)
    assert out[-1]==pytest.approx(17*(5*.3+13*.5),rel=2e-14)


def test_random_extreme_positive_products_and_sums_never_overestimate_subset(witness):
    rng=np.random.default_rng(20261020)
    for n in (1,2,5):
        for _ in range(12):
            c=10.**rng.uniform(-90,90,(n,n));h=10.**rng.uniform(-90,90,n)
            hs=10.**rng.uniform(-90,90,n);ne=float(10.**rng.uniform(-40,40))
            inc,out,valid=witness[0](c,h,hs,ne)
            assert valid
            exact_in,exact_out=exact_gross(c,h,hs,ne)
            for actual,expected in zip([*inc,*out],[*exact_in,*exact_out]):
                assert 0<=F(actual)<=expected
                assert float(expected)==pytest.approx(actual,rel=1e-13,abs=0)


@pytest.mark.parametrize('defect',['negative','nan','overflow','ne'])
def test_invalid_collision_terms_remain_unproven_not_repaired(witness,defect):
    c=np.ones((2,2));h=np.ones(2);hs=np.ones(2);ne=1.
    if defect=='negative':c[0,1]=-1
    if defect=='nan':c[1,0]=np.nan
    if defect=='overflow':h[0]=1e308;ne=1e308
    if defect=='ne':ne=0
    inc,out,valid=witness[0](c,h,hs,ne)
    assert not valid
    assert not witness[2](0.,0.,inc[0],out[0],valid=valid)[1]


def test_zero_and_underflow_are_lower_bounds_not_population_floors(witness):
    for c in (0.,1e-300):
        inc,out,valid=witness[0]([[c]],[1e-300],[1e-300],1.)
        assert valid and np.all(inc==0) and np.all(out==0)
        upper,proven=witness[2](0.,0.,inc[0],out[0])
        assert np.isnan(upper) and not proven


@pytest.mark.parametrize('terms',[[1e100,1.,-1e100],[1.,-1.,1e-300],[-1e100,1e100],
                                 [1e-310,1e-310,-1e-310],[0.,0.]])
def test_raw_global_interval_encloses_exact_sum_and_native_evaluated_sum(witness,terms):
    lo,hi,valid=witness[1](terms)
    assert valid
    exact=sum(map(F,terms),F(0))
    assert F(lo)<=exact<=F(hi)
    actual=0.
    for term in terms:actual+=term
    assert lo<=actual<=hi


def test_small_upper_bound_is_sufficient_for_any_larger_true_denominator(witness):
    upper,proven=witness[2](-2e-9,1e-9,4.,8.)
    assert proven and F(upper)>=F(2e-9)/F(8.)
    for real_denominator in (8.,100.,1e12):
        assert F(2e-9)/F(real_denominator)<=F(upper)
    assert not witness[2](-1.,1.,4.,8.)[1]
    assert not witness[2](0.,0.,4.,8.,covered=False)[1]
    assert not witness[2](0.,0.,4.,8.,valid=False)[1]


def test_ratio_underflow_is_rounded_up_not_certified_as_exact_zero(witness):
    upper,proven=witness[2](0.,1e-300,1e300,0.)
    assert proven and upper>0 and F(upper)>=F(1e-300)/F(1e300)


def test_flush_to_zero_cannot_produce_a_false_upper_certificate(witness):
    upper,proven=witness[3]()
    assert np.isnan(upper) and not proven
    # 시험에서 바꾼 mode는 복원됐으며 원래의 정상 증명은 여전히 가능하다.
    assert witness[2](0.,1e-300,1e300,0.)[1]
