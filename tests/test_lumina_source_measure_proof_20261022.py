"""원천 측도 후보의 분리된 고정밀 수학 시험. 생산 합격 시험이 아니다."""
import mpmath as mp
import pytest

from lumina_source_measure_proof_20261022 import (
    NativeTree,bernstein_value,integrate_source,independent_integral,leaf,merge,project)


@pytest.fixture(autouse=True)
def precision():
    with mp.workdps(70):
        yield


def close(a,b,relative='1e-60'):
    assert abs(a-b) <= mp.mpf(relative)*max(abs(a),abs(b),mp.mpf('1e-900'))


def test_positive_leaf_merge_mass_and_lower_degree_projection():
    a,b = leaf('0.001','12',25),leaf('12','0.02',25)
    joined = merge(a,b,mp.mpf('0.375'))
    assert min(joined) > 0
    close(mp.fsum(joined),mp.mpf('.375')*(mp.mpf('.001')+12)/2
          +mp.mpf('.625')*(12+mp.mpf('.02'))/2)
    for degree in (0,1,3,12,25):
        ma = project(a,degree)
        for x,y in zip(ma,leaf('.001','12',degree)):
            close(x,y)


@pytest.mark.parametrize('begin,end',[('0.13','7.81'),('7.81','0.13'),('3.25','3.25')])
def test_clipped_query_direction_and_stationary_limit(begin,end):
    a = NativeTree([0,2,0,9,1,0,0,3,0])
    mom = a.query(begin,end)
    # 직접 기저 적분은 트리의 leaf/merge 공식을 사용하지 않는다.
    lo,hi = mp.mpf(begin),mp.mpf(end)
    for degree in (0,1,3):
        actual = project(mom,degree)
        for k in range(degree+1):
            if lo == hi:
                reference = a.value(lo)/(degree+1)
            else:
                cuts = [mp.mpf(0),mp.mpf(1)]
                cuts += [(i-lo)/(hi-lo) for i in range(len(a.values)) if 0 < (i-lo)/(hi-lo) < 1]
                cuts.sort()
                reference = mp.fsum(mp.quad(lambda t: mp.binomial(degree,k)*t**k*(1-t)**(degree-k)
                                            *a.value(lo+(hi-lo)*t),[x,y]) for x,y in zip(cuts,cuts[1:]))
            close(actual[k],reference)


@pytest.mark.parametrize('x0,q',[('0.17','11.6'),('11.77','-11.6'),('5.125','0')])
def test_general_cubic_attenuation_with_native_source_knots(x0,q):
    a = NativeTree([0,0,7,0,1,0,0,3,0,0,4,0,1])
    b = NativeTree([1,0,0,2,0,9,0,0,2,0,0,0,0])
    # 내부 극값과 부호 변화가 있는 위상: 투명/상수 감쇠만의 시험이 아니다.
    phase = tuple(map(mp.mpf,['-.03','.07','-.04','.02']))
    result = integrate_source(a,b,x0,q,'.13','.91',phase,length='2.7',log_scale='-.3')
    reference = independent_integral(a,b,x0,q,'.13','.91',phase,length='2.7',log_scale='-.3')
    margin = mp.mpf('1e-60')*reference
    assert result['model_lower']-margin <= reference <= result['model_upper']+margin
    assert abs(result['value']/reference-1) < mp.mpf('2.1e-14')
    assert result['roundoff_certified'] is False


def test_known_phase_residual_is_charged_not_ignored():
    a,b = NativeTree([0,8,0,3,0]),NativeTree([2,0,0,5,0])
    phase = tuple(map(mp.mpf,['.02','-.04','.07','-.01']))
    delta = mp.mpf('.0002')
    result = integrate_source(a,b,'.1','3.7',0,1,phase,delta=delta)
    for sign in (-1,1):
        reference = independent_integral(a,b,'.1','3.7',0,1,phase,
                                         perturbation=lambda t: sign*delta*mp.sin(mp.pi*t))
        assert result['model_lower'] <= reference <= result['model_upper']


def test_true_zero_and_varying_subnormal_source_recover_after_length_and_gain():
    phase = tuple(map(mp.mpf,['-.01','.03','-.04','.01']))
    zero = NativeTree([0,0,0,0])
    result = integrate_source(zero,zero,0,3,0,1,phase,length='1e300',log_scale=720)
    assert result['value'] == result['model_lower'] == result['model_upper'] == 0
    weak = NativeTree(['5e-324','1e-323',0,'5e-324'])
    result = integrate_source(weak,zero,0,3,0,1,phase,length='1e20',log_scale=720)
    reference = independent_integral(weak,zero,0,3,0,1,phase,length='1e20',log_scale=720)
    assert reference > 0 and result['value'] > 0
    assert result['model_lower'] <= reference <= result['model_upper']
    assert float(mp.exp(720)) == float('inf')  # exp를 먼저 만드는 binary64 구현은 실패한다.


def test_negative_endpoint_component_and_excessive_phase_width_are_rejected():
    with pytest.raises(ValueError):
        NativeTree([1,-1,2])
    a = NativeTree([1,2])
    with pytest.raises(ValueError):
        integrate_source(a,a,0,1,0,1,[0,0,0,1])


def test_function_nonnegative_is_not_bernstein_coefficient_nonnegative():
    phase = tuple(map(mp.mpf,['.025','-.0083333333333333333333333333333333333',
                             '-.0083333333333333333333333333333333333','.025']))
    assert min(phase) < 0
    assert bernstein_value(phase,mp.mpf('.5')) >= 0
    a = NativeTree([0,3,0,2,0])
    result = integrate_source(a,a,0,4,0,1,phase)
    reference = independent_integral(a,a,0,4,0,1,phase)
    assert result['model_lower'] <= reference <= result['model_upper']
