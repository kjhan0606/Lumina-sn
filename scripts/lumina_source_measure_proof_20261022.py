"""고정밀 CPU 수학 증명용 원형. 생산 수송·GPU·오차 인증 코드가 아니다.

원래 선형 원천의 Bernstein 모멘트를 양의 연산으로 병합한다. 원천을
secant로 다시 만들지 않고 cubic 감쇠의 양의 Taylor 다항식과 적분한다.
mpmath의 확장 정밀도는 binary64/device 반올림 보증을 대신하지 않는다.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb

import mpmath as mp


def number(x):
    return mp.mpf(x)


def leaf(a, b, degree):
    a, b = number(a), number(b)
    if a < 0 or b < 0 or degree < 0:
        raise ValueError('성분별 비음수 원천과 비음수 차수가 필요하다')
    denominator = (degree + 1) * (degree + 2)
    return tuple(((degree-k+1)*a+(k+1)*b)/denominator for k in range(degree+1))


def merge(left, right, fraction):
    """큰 prefix 차감 없이 자식 좌표를 부모 좌표로 옮긴다."""
    if len(left) != len(right):
        raise ValueError('차수가 다르다')
    a = number(fraction)
    b = 1-a
    if not 0 < a < 1:
        raise ValueError('분할 비율 범위 오류')
    n = len(left)-1
    return tuple(
        a*mp.fsum(comb(j,k)*a**k*b**(j-k)*left[j] for j in range(k,n+1))
        + b*mp.fsum(comb(n-j,k-j)*a**(k-j)*b**(n-k)*right[j] for j in range(k+1))
        for k in range(n+1))


def project(moment, degree):
    n = len(moment)-1
    if not 0 <= degree <= n:
        raise ValueError('저장 차수보다 높은 모멘트를 요청했다')
    return tuple(mp.fsum(number(comb(degree,j)*comb(n-degree,k-j))/comb(n,k)*moment[k]
                         for k in range(j,j+n-degree+1)) for j in range(degree+1))


@dataclass(frozen=True)
class Node:
    begin: object
    end: object
    moment: tuple
    children: tuple = ()


class NativeTree:
    """단일 비음수 endpoint 성분의 원래 선형 bin 트리. 끝 leaf만 제한한다."""
    def __init__(self, values, degree=25):
        self.values = tuple(map(number, values))
        if len(values) < 2 or any(v < 0 or not mp.isfinite(v) for v in self.values):
            raise ValueError('유한 비음수 native 자료가 필요하다')
        self.degree = degree
        self.root = self._build(0,len(values)-1)
        self.queries = 0
        self.covered_nodes = 0

    def _build(self, lo, hi):
        if hi-lo == 1:
            return Node(number(lo),number(hi),leaf(self.values[lo],self.values[hi],self.degree))
        mid = (lo+hi)//2
        a,b = self._build(lo,mid),self._build(mid,hi)
        return Node(number(lo),number(hi),merge(a.moment,b.moment,number(mid-lo)/(hi-lo)),(a,b))

    def value(self, x):
        x = number(x)
        if not 0 <= x <= len(self.values)-1:
            raise ValueError('이 증명용 원형은 native 구간 내부만 지원한다')
        i = min(int(mp.floor(x)),len(self.values)-2)
        t = x-i
        return (1-t)*self.values[i]+t*self.values[i+1]

    def _parts(self, node, lo, hi):
        if node.end <= lo or node.begin >= hi:
            return []
        if lo <= node.begin and node.end <= hi:
            return [(node.begin,node.end,node.moment)]
        if not node.children:
            a,b = max(lo,node.begin),min(hi,node.end)
            return [(a,b,leaf(self.value(a),self.value(b),self.degree))]
        return self._parts(node.children[0],lo,hi)+self._parts(node.children[1],lo,hi)

    def query(self, begin, end):
        begin,end = number(begin),number(end)
        self.queries += 1
        if begin == end:
            v = self.value(begin)/(self.degree+1)
            return (v,)*(self.degree+1)
        reverse = end < begin
        lo,hi = (end,begin) if reverse else (begin,end)
        if lo < 0 or hi > len(self.values)-1:
            raise ValueError('native 범위 밖 query')
        parts = self._parts(self.root,lo,hi)
        self.covered_nodes += len(parts)
        a,b,result = parts[0]
        for c,d,other in parts[1:]:
            if c != b:
                raise AssertionError('구간 덮개의 공백')
            result = merge(result,other,(b-a)/(d-a))
            b = d
        return tuple(reversed(result)) if reverse else result


def bernstein_value(coefficients, t):
    c = list(coefficients)
    while len(c) > 1:
        c = [(1-t)*a+t*b for a,b in zip(c,c[1:])]
    return c[0]


def multiply(a, b):
    n,m = len(a)-1,len(b)-1
    out = [mp.mpf(0)]*(n+m+1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            out[i+j] += number(comb(n,i)*comb(m,j))/comb(n+m,i+j)*x*y
    return tuple(out)


def elevate(coefficients, degree):
    n = len(coefficients)-1
    if degree < n:
        raise ValueError('다항식 차수를 내릴 수 없다')
    return tuple(mp.fsum(number(comb(n,i)*comb(degree-n,k-i))/comb(degree,k)*coefficients[i]
                         for i in range(max(0,k-degree+n),min(n,k)+1)) for k in range(degree+1))


def attenuation(phase, order):
    """함수 최솟값이 아니라 Bernstein 계수 최솟값으로 shift한다."""
    phase = tuple(map(number,phase))
    if len(phase) != 4 or order < 0:
        raise ValueError('cubic 위상과 비음수 Taylor 차수가 필요하다')
    low,high = min(phase),max(phase)
    width = high-low
    if width > mp.mpf('0.125'):
        raise ValueError('증명용 패널의 Bernstein 위상 폭을 넘었다')
    shifted = tuple(x-low for x in phase)
    degree = 3*order
    power = (mp.mpf(1),)
    total = [mp.mpf(0)]*(degree+1)
    for n in range(order+1):
        for k,v in enumerate(elevate(power,degree)):
            total[k] += v/mp.factorial(n)
        power = multiply(power,shifted)
    # 상대 Taylor 나머지 상계. 1에서 거의 같은 수를 빼지 않는다.
    tail = width**(order+1)/mp.factorial(order+1)
    return tuple(mp.exp(-width)*v for v in total),high,tail


def integrate_source(a, b, x0, q, ua, ub, phase, *, length=1, log_scale=0, order=8, delta=0):
    """알려진 cubic 위상에 대한 모멘트 contraction과 모델 오차만 반환한다.

    delta는 호출자가 독립적으로 보증한 실제 위상과 cubic의 최대 차이다.
    반환 구간에는 binary64/고정밀 산술의 인증된 반올림 상계가 없다.
    따라서 생산 수용 판단에 직접 사용하면 안 된다.
    """
    x0,q,ua,ub,length,log_scale,delta = map(number,(x0,q,ua,ub,length,log_scale,delta))
    if not 0 <= ua < ub <= 1 or length < 0 or delta < 0:
        raise ValueError('공간 범위·길이·오차 범위 오류')
    p,scale,tail = attenuation(phase,order)
    m = len(p)-1
    ma = project(a.query(x0+q*ua,x0+q*ub),m+1)
    mb = project(b.query(x0+q*ua,x0+q*ub),m+1)
    mass = mp.fsum(p[k]*(number(m+1-k)/(m+1)*(ua*ma[k]+(1-ua)*mb[k])
                         + number(k+1)/(m+1)*(ub*ma[k+1]+(1-ub)*mb[k+1])) for k in range(m+1))
    value = length*(ub-ua)*mp.exp(log_scale+scale)*mass
    return {'value':value,'model_lower':mp.exp(-delta)*value,
            'model_upper':mp.exp(delta)*value/(1-tail),'tail_relative_bound':tail,
            'roundoff_certified':False}


@lru_cache(maxsize=8)
def quadrature_rule(digits, nodes):
    # query/병합/다항식 급수를 사용하지 않는 독립 Gauss 기준의 절점이다.
    with mp.workdps(digits):
        x,w = mp.gauss_quadrature(nodes,'legendre')
        return tuple(x),tuple(w)


def independent_integral(a,b,x0,q,ua,ub,phase,*,length=1,log_scale=0,perturbation=None,nodes=24):
    """원래 모든 bin 교점에서 나누고 고정밀 Gauss로 직접 적분한다."""
    x0,q,ua,ub,length,log_scale = map(number,(x0,q,ua,ub,length,log_scale))
    cuts = [ua,ub]
    if q:
        for i in range(len(a.values)):
            u = (i-x0)/q
            if ua < u < ub:
                cuts.append(u)
    cuts.sort()
    xs,ws = quadrature_rule(mp.mp.dps,nodes)
    values = []
    for lo,hi in zip(cuts,cuts[1:]):
        center,half = (lo+hi)/2,(hi-lo)/2
        for x,w in zip(xs,ws):
            u = center+half*x
            t = (u-ua)/(ub-ua)
            eta = u*a.value(x0+q*u)+(1-u)*b.value(x0+q*u)
            exponent = log_scale+bernstein_value(phase,t)
            if perturbation is not None:
                exponent += perturbation(t)
            values.append(half*w*eta*mp.exp(exponent))
    return length*mp.fsum(values)
