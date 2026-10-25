"""주파수 이력 후보의 작은 공간 적응 연결. 실제 미세장 생산 실행기는 아니다.

같은 물리 구간의 이동 특성선 좌표에서만 비교하므로 내부 강도 재투영은 없다.
χ/총η는 매 공간 표본에서 다시 요청한다. 오차는 추정치이며 인증 상한이 아니다.
"""
import time

import numpy as np


class HistoryLimit(ValueError):
    def __init__(self, reason, partial, report):
        super().__init__('주파수 이력 공간 적응 한도 소진: '+reason)
        self.reason, self.partial, self.report = reason, partial, report


def propagate(kernel, h, incident, geometry, coefficients, remaining_drift, *,
              panels=8, rtol=1e-6, atol=0., max_leaves=4096,
              max_depth=32, max_calls=200000, max_seconds=60.,compiled_provider=None,radius=None):
    """한 물리 구간. geometry(a,b)=(공변 길이,drift), coefficients(x)=(χ,총η).

    전체/두 반구간을 같은 입사 특성선 장에서 비교하고, 각 주파수의 하류 응답으로
    오차를 전달한다. 출력 최대값으로 정규화한 전ν 최대 오차를 사용한다.
    이 작은 배열 연결의 guard를 풀어 실제 90×1.1M에 적용하면 안 된다.
    """
    original = np.asarray(incident, dtype='f8')
    if (original.ndim != 1 or not 8 <= original.size <= 512 or
            not np.isfinite(original).all() or np.any(original < 0) or
            not np.isfinite(h) or h <= 0 or type(panels) is not int or not 1 <= panels <= 16 or
            not np.isfinite(rtol) or rtol < 0 or not np.isfinite(atol) or atol < 0 or rtol+atol == 0 or
            type(max_leaves) is not int or not panels <= max_leaves <= 4096 or
            type(max_depth) is not int or not 0 <= max_depth <= 32 or
            type(max_calls) is not int or not 1 <= max_calls <= 2000000 or
            not np.isfinite(max_seconds) or not 0 < max_seconds <= 60):
        raise ValueError('소형 주파수 이력 공간 적응 입력/한도 오류')
    if compiled_provider is not None and (compiled_provider.nf!=original.size or not callable(radius)):
        raise ValueError('소형 C 연결은 전체 전역ν와 반경 표본 함수가 필요하다')
    nf = original.size; started = time.monotonic(); calls = 0; completed = None
    coefficient_cache, geometry_cache, delta_cache = {}, {}, {}
    report = dict(status='STARTED', reconstructions=0, interior_intensity_remappings=0,
        accepted_physical_field=False, componentwise_verified=False, final_goal='NOT_JUDGED')

    def limit(reason):
        detail = dict(report, status='LIMIT_REACHED', reason=reason, formal_vectors=calls,
            elapsed_seconds=time.monotonic()-started)
        raise HistoryLimit(reason, None if completed is None else completed.copy(), detail)

    def check_time():
        if time.monotonic()-started >= max_seconds: limit('max_seconds')

    def delta(x):
        if x not in delta_cache:
            value = float(remaining_drift(x))
            if not np.isfinite(value) or value < 0: raise ValueError('관측 Doppler 범위 오류')
            delta_cache[x] = value
        return delta_cache[x]

    def geo(a,b):
        if (a,b) not in geometry_cache:
            length, drift = map(float, geometry(a,b))
            if (not np.isfinite(length) or length < 0 or not np.isfinite(drift) or drift < 0 or
                    (length == 0 and drift != 0)):
                raise ValueError('공간 적응 구간 기하 오류')
            expected = delta(a)-delta(b)
            if expected < 0 or abs(drift-expected) > 2e-13*max(1.,abs(drift),abs(expected)):
                raise ValueError('기하와 관측 Doppler 차이 불일치')
            geometry_cache[a,b] = length
        return geometry_cache[a,b]

    def coeff(x):
        if x not in coefficient_cache:
            pair = tuple(np.array(v, dtype='f8', copy=True) for v in coefficients(x))
            if (len(pair) != 2 or any(v.shape != (nf,) or not np.isfinite(v).all() for v in pair)
                    or np.any(pair[1] < 0)):
                raise ValueError('공간 표본의 동결 χ/총η 오류')
            coefficient_cache[x] = pair
        return coefficient_cache[x]

    if delta(1.) != 0: raise ValueError('물리 구간 출구 Doppler offset은 0이어야 한다')
    initial, reconstruction = kernel.reconstruct(original,delta(0.)/h)
    report.update(reconstruction)
    leaves = [(i/panels,(i+1)/panels,0) for i in range(panels)]
    rounds = 0; branch_checks=0
    while True:
        check_time(); count = len(leaves)
        if calls+3*count > max_calls: limit('max_calls')
        lengths = np.empty((count,3)); deltas = np.empty_like(lengths)
        chi = np.empty((count,5,nf)) if compiled_provider is None else None
        eta = np.empty((count,3,nf)) if compiled_provider is None else None
        radii = np.empty((count,5)) if compiled_provider is not None else None
        for row,(a,b,depth) in enumerate(leaves):
            check_time(); m=.5*(a+b)
            lengths[row] = geo(a,b),geo(a,m),geo(m,b)
            deltas[row] = delta(a),delta(m),delta(b)
            if compiled_provider is None:
                chi[row] = [coeff(x)[0] for x in (a,.5*(a+m),m,.5*(m+b),b)]
                eta[row] = [coeff(x)[1] for x in (a,m,b)]
            else:radii[row]=[radius(x) for x in (a,.5*(a+m),m,.5*(m+b),b)]
        spans=np.stack(((deltas[:,0]-deltas[:,2]),(deltas[:,0]-deltas[:,1]),(deltas[:,1]-deltas[:,2])),axis=1)/h
        expected_calls=3*count+int(np.count_nonzero((spans>=.75)&(spans<=1.25)))
        if calls+expected_calls>max_calls:limit('max_calls')
        if compiled_provider is None:
            completed, errors, responses, detail = kernel.compare_hybrid(h,lengths,deltas,chi,eta,initial)
        else:
            completed,errors,responses,detail=kernel.compare_provider(compiled_provider,h,lengths,deltas,radii,initial)
        if detail['formal_vectors']!=expected_calls:raise ValueError('혼합 formal 호출수 명세 불일치')
        calls += detail['formal_vectors'];branch_checks+=detail['branch_checks'];rounds += 1; check_time()
        tail=np.ones(nf); total=np.zeros(nf); scores=np.zeros(count)
        with np.errstate(over='ignore',invalid='ignore'):
            for row in range(count-1,-1,-1):
                local=errors[row]*tail; total+=local; scores[row]=local.max()
                tail*=responses[row]
        if not all(np.isfinite(v).all() for v in (total,tail,scores)):
            raise ValueError('하류 감쇠/gain 오차 전달 수치 범위 초과')
        estimate=float(total.max()); tolerance=atol+rtol*float(completed.max())
        report.update(spatial_leaves=count, rounds=rounds, formal_vectors=calls,branch_checks=branch_checks,
            scalar_formals=calls*nf, estimated_output_error=estimate, tolerance=tolerance,
            coefficient_positions=len(coefficient_cache), geometry_entries=len(geometry_cache),
            coefficient_cache_bytes=sum(v.nbytes for pair in coefficient_cache.values() for v in pair),
            comparison_array_bytes=sum(v.nbytes for v in (lengths,deltas,chi,eta,radii,completed,errors,responses) if v is not None),
            coefficient_backend='Python reference arrays' if compiled_provider is None else 'C source provider; no Python coefficient callbacks',
            error_contract='estimate only; all frequencies; downstream diagonal response includes gain')
        if estimate <= tolerance:
            return completed,dict(report,status='ESTIMATE_MET',elapsed_seconds=time.monotonic()-started)
        selected = scores >= .5*float(scores.max())
        if count+int(selected.sum()) > max_leaves: limit('max_leaves')
        refined=[]
        for split,(a,b,depth) in zip(selected,leaves):
            if not split: refined.append((a,b,depth)); continue
            m=.5*(a+b)
            if depth >= max_depth or m==a or m==b: limit('max_depth')
            refined.extend(((a,m,depth+1),(m,b,depth+1)))
        leaves=refined
