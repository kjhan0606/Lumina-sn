"""외향 반올림의 명시적 경계를 binary64 전이율의 정확한 유리수 해와 비교한다."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from certify_cmfgen_owner_interval import load_library,enclose,candidate_error


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    root=tmp_path_factory.mktemp('interval_gth');path=root/'interval.so'
    source=Path(__file__).resolve().parents[1]/'scripts/certify_generator_interval.c'
    wrapper=root/'environment.c'
    wrapper.write_text('''#include <fenv.h>
#include <fpu_control.h>
int interval_test_bad_environment(int mode) {
 if(mode==1) return fesetround(FE_DOWNWARD);
 if(mode==0) return fesetround(FE_TONEAREST);
 fpu_control_t word; _FPU_GETCW(word);
 word=(word & ~0x300U) | (mode==2 ? 0x200U : 0x300U); _FPU_SETCW(word);
 return 0;
}
''')
    subprocess.run(['gcc','-shared','-fPIC','-O2','-fno-fast-math','-ffp-contract=off','-frounding-math',
        '-I'+str(source.parents[1]/'src'),
        str(source),str(wrapper),'-lm','-o',str(path)],check=True,capture_output=True,timeout=30)
    return load_library(path)


def rational(value):
    value=np.longdouble(value)
    mantissa,exponent=np.frexp(value);bits=np.finfo(np.longdouble).nmant+1
    integer=int(np.ldexp(mantissa,bits));power=int(exponent)-bits
    return Fraction(integer*2**power) if power>=0 else Fraction(integer,2**(-power))


@pytest.mark.parametrize('a,b',[(1.,2.),(1e-280,1.),(1e-280,1e280),(2.718281828,3.141592654)])
def test_two_state_exact_rational_root_is_enclosed(lib,a,b):
    matrix=np.array([[-a,b],[a,-b]])
    lower,upper=enclose(lib,matrix)
    denominator=Fraction.from_float(a)+Fraction.from_float(b)
    exact=[Fraction.from_float(b)/denominator,Fraction.from_float(a)/denominator]
    for lo,hi,target in zip(lower,upper,exact):
        assert rational(lo)<=target<=rational(hi)
        assert hi>lo>0


def test_weak_connection_is_not_deleted_and_group_sum_is_outward(lib):
    tiny=1e-280
    matrix=np.array([[-1.,1.,0.],[1.,-1.,tiny],[0.,tiny,-tiny]])
    lo,hi,gl,gh=enclose(lib,matrix,[0,0,1])
    for low,high in zip(lo,hi):assert rational(low)<=Fraction(1,3)<=rational(high)
    assert rational(gl[0])<=Fraction(2,3)<=rational(gh[0])
    matrix[1,2]=0;matrix[2,2]=0
    with pytest.raises(ValueError,match='인증 실패'):
        enclose(lib,matrix)


def test_invalid_diagonal_or_negative_rate_is_not_reinterpreted(lib):
    for matrix in (np.array([[-1.,-2.],[1.,2.]]),np.array([[-4.,2.],[1.,-2.]])):
        with pytest.raises(ValueError,match='인증 실패'):
            enclose(lib,matrix)


@pytest.mark.parametrize('mode',[1,2])
def test_wrong_rounding_or_x87_precision_is_refused(lib,mode):
    assert lib.interval_test_bad_environment(mode)==0
    try:
        with pytest.raises(ValueError,match='인증 실패'):
            enclose(lib,np.array([[-1.,2.],[1.,-2.]]))
    finally:
        assert lib.interval_test_bad_environment(0)==0
        assert lib.interval_test_bad_environment(3)==0


def test_multistate_enclosure_matches_independent_exact_spanning_tree_solution(lib):
    n=5;rng=np.random.default_rng(20261020)
    rates=rng.integers(1,10,size=(n,n)).astype(float)
    np.fill_diagonal(rates,0)
    rates[2,3]=1e-280;rates[4,1]=1e280
    matrix=rates.copy();np.fill_diagonal(matrix,-rates.sum(axis=0))
    weights=[]
    for root in range(n):
        sources=[s for s in range(n) if s!=root];total=Fraction(0)
        choices=[[d for d in range(n) if d!=s] for s in sources]
        for destinations in product(*choices):
            successor=dict(zip(sources,destinations));valid=True
            for start in sources:
                current=start
                for _ in range(n):
                    if current==root:break
                    current=successor[current]
                if current!=root:valid=False;break
            if valid:
                weight=Fraction(1)
                for s,d in successor.items():weight*=Fraction.from_float(rates[d,s])
                total+=weight
        weights.append(total)
    denominator=sum(weights);exact=[w/denominator for w in weights]
    for permutation in (np.arange(n),np.array([3,1,4,0,2])):
        lo,hi=enclose(lib,matrix[np.ix_(permutation,permutation)])
        for i,state in enumerate(permutation):
            assert rational(lo[i])<=exact[state]<=rational(hi[i])


def test_bound_covers_actual_binary64_density_including_bad_candidates(lib):
    matrix=np.array([[-1.,2.],[1.,-2.]])
    low,high=enclose(lib,matrix);mass=1234567.
    exact=[Fraction(2,3),Fraction(1,3)]
    for candidate in (np.array([mass*2/3,mass/3]),np.array([mass/2,mass/2])):
        absolute,relative=candidate_error(lib,low,high,candidate,mass)
        for i,target in enumerate(exact):
            nominal=Fraction.from_float(candidate[i])/Fraction.from_float(mass)
            assert abs(nominal-target)<=rational(absolute[i])
            assert abs(nominal-target)/target<=rational(relative[i])


def strict_certificate(lib,matrix,candidate,total):
    matrix=np.asfortranarray(matrix,dtype=float)
    candidate=np.ascontiguousarray(candidate,dtype=float);n=len(candidate)
    groups=np.arange(n,dtype=np.int32)
    low=np.zeros(n,dtype=np.longdouble);high=np.zeros_like(low)
    absolute=np.zeros(1,dtype=np.longdouble);relative=np.zeros_like(absolute)
    status=lib.generator_interval_certify_candidate(matrix.ctypes.data,n,candidate.ctypes.data,total,
        groups.ctypes.data,n,low.ctypes.data,high.ctypes.data,absolute.ctypes.data,relative.ctypes.data)
    return status,absolute[0],relative[0],low,high


def test_actual_production_certificate_rejects_tiny_level_error_without_absolute_cut(lib):
    tiny=1e-280;matrix=np.array([[-tiny,1.],[tiny,-1.]])
    good=strict_certificate(lib,matrix,[1.,tiny],1.)
    assert good[0]==0 and good[1]<1e-12 and good[2]<1e-10
    for candidate in ([1.,tiny*1.000001],[1.,0.]):
        bad=strict_certificate(lib,matrix,candidate,1.)
        assert bad[0]==60 and bad[1]<1e-12 and bad[2]>1e-10


@pytest.mark.parametrize('candidate,total', [([.5,.5],1.),([np.nan,.5],1.),
    ([-.1,1.1],1.),([2/3,1/3],0.)])
def test_actual_production_certificate_rejects_bad_candidate_or_total(lib,candidate,total):
    assert strict_certificate(lib,[[-1.,2.],[1.,-2.]],candidate,total)[0]!=0


def test_actual_production_certificate_does_not_bypass_environment_or_generator_errors(lib):
    for matrix in ([[-4.,2.],[1.,-2.]],[[-1.,0.],[1.,0.]]):
        assert strict_certificate(lib,matrix,[2/3,1/3],1.)[0]!=0
    lib.interval_test_bad_environment(1)
    try:
        assert strict_certificate(lib,[[-1.,2.],[1.,-2.]],[2/3,1/3],1.)[0]!=0
    finally:
        lib.interval_test_bad_environment(0)
