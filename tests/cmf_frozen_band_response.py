"""국소 시험이 고정한 복사장 대역의 이온비 민감도. 생산 장의 보정이 아니다.

기존 실제 BB 평균/BF GL8 커널과 전체 원소 행렬을 사용한다. T/ne/비열과
다른 파장은 고정하며, 광전리 흐름 비중과 최종 정상 이온비 미분을 구별한다.
"""
import math

import numpy as np


def investigate(g):
    J, nu, N = g['J'], g['nu'], g['N']
    energy = nu*g['h']/1.602176634e-12
    broad = (g['depth'] > 100) & (g['error'] < 1e-3) & np.isfinite(g['error'])
    strict = (g['depth'] > 1e4) & (g['error'] < 1e-5) & np.isfinite(g['error'])
    states = []
    for b, (_, lo, hi, stages) in zip(g['blocks'], g['groups']):
        a = np.array(b['A'], dtype=np.longdouble, copy=True)
        np.fill_diagonal(a, 0)
        np.fill_diagonal(a, -a.sum(axis=0))
        x, residual = g['stationary'](a)
        scale = np.max(abs(a), axis=1)
        normalized = np.asarray(a/scale[:, None], dtype=float)
        pivot = int(np.argmax(x))
        normalized[pivot] = 1
        q = np.zeros(len(x))
        q[stages == 2] = 1/float(x[stages == 2].sum())
        q[stages == 1] = -1/float(x[stages == 1].sum())
        adjoint = np.linalg.solve(normalized.T, q)
        adjoint[pivot] = 0
        adjoint /= np.asarray(scale, dtype=float)
        states.append((b['z'], lo, hi, stages, a, x, adjoint, residual))
    starts = {z: lo for z, lo, hi, stages in g['groups']}

    def matrices(mask):
        field = J*mask
        line_field = g['P']@field
        coarse_field = g['W']@field
        bb = np.zeros((N, N), dtype=np.longdouble)
        bf = np.zeros_like(bb)
        np.add.at(bb, (g['upper_sl'], g['lower']), np.asarray(g['blu'])*line_field)
        np.add.at(bb, (g['lower'], g['upper_sl']), np.asarray(g['bul'])*line_field)
        for z, lower, upper, active, photo, stim in g['bf_response_channels']:
            lo, up = starts[z]+lower, starts[z]+upper
            bf[up, lo] += photo@coarse_field[active]
            bf[lo, up] += stim@coarse_field[active]
        return bb, bf

    all_bb, all_bf = matrices(np.ones(len(J), dtype=bool))
    result = dict(scope='셸39 실제 첫 장/고정T/ne/비열, BB와 BF의 같은 미세격자 J에 대한 조건부 민감도; 새 공간 수송 아님',
                  definition='d ln(nIII/nII) / d epsilon, J(mask) -> (1+epsilon) J(mask)',
                  masks=dict(strict_bins=int(strict.sum()), broad_bins=int(broad.sum())),
                  ions={}, bands=[])
    for z, lo, hi, stages, a, x, adjoint, residual in states:
        result['ions'][z] = dict(baseline_III_II=float(x[stages == 2].sum()/x[stages == 1].sum()),
                                normalized_rate_residual=residual)

    bands = [(0, 5), (5, 10), (10, 16), (16, 24), (24, None)]
    sums = {key: np.zeros_like(all_bb) for key in ('BB', 'BF')}
    for low, high in bands:
        band = energy >= low
        if high is not None:
            band &= energy < high
        for selection, mask in (('all', band), ('frozen_broad', band & ~broad)):
            bb, bf = matrices(mask)
            if selection == 'all':
                sums['BB'] += bb
                sums['BF'] += bf
            row = dict(eV=[low, high], selection=selection, ions={})
            for z, lo, hi, stages, a, x, adjoint, residual in states:
                values = {}
                for name, delta in (('BB', bb), ('BF', bf), ('joint', bb+bf)):
                    d = delta[lo:hi, lo:hi].copy()
                    np.fill_diagonal(d, 0)
                    np.fill_diagonal(d, -d.sum(axis=0))
                    predicted = float(-adjoint@(d@x))
                    eps = 1e-4
                    plus, rp = g['stationary'](a+eps*d)
                    minus, rm = g['stationary'](a-eps*d)
                    ratio = lambda v: float(v[stages == 2].sum()/v[stages == 1].sum())
                    measured = (math.log(ratio(plus))-math.log(ratio(minus)))/(2*eps)
                    difference = abs(predicted-measured)
                    assert difference < 2e-6*max(1, abs(predicted)), (z, low, selection, name, predicted, measured)
                    values[name] = dict(adjoint=predicted, finite_difference=measured,
                                        absolute_difference=difference, maximum_rate_residual=max(rp, rm))
                row['ions'][z] = values
            result['bands'].append(row)
            print('FROZEN_BAND', row, flush=True)
    # 같은 J의 대역 분해가 원래 모든 방사 유도 항의 합을 보존하는지 확인한다.
    for name, reference in (('BB', all_bb), ('BF', all_bf)):
        error = float(np.max(abs(sums[name]-reference))/np.max(abs(reference)))
        assert error < 1e-12
        result[name+'_partition_error'] = error
    return result
