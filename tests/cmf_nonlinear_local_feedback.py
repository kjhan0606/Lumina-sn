"""실제 저장 율의 국소 비선형 되먹임 진단. 전체 공간 공동해가 아니다.

T/ne/다른 원소/비열 및 비국소 수송 잔여항은 동결한다. 원래 물질에서
선택한 불투명 구간에만 J(x)=[chi_abs0*J0+delta_eta]/[chi_abs0+delta_chi]를
적용한다. 이 닫힘을 생산 수송식 또는 CMFGEN 동질성으로 대체하지 않는다.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse


def investigate(g):
    A0, x0, J0 = np.array(g['A'], dtype=np.longdouble, copy=True), g['x0'], g['J']
    old_diagonal = A0.diagonal().copy()
    np.fill_diagonal(A0, 0)
    exact_outflow = A0.sum(axis=0)
    diagonal_roundoff = float(np.max(abs(old_diagonal+exact_outflow)/exact_outflow))
    n_eps = len(A0)*np.finfo(float).eps
    diagonal_roundoff_bound = n_eps/(1-n_eps)
    assert diagonal_roundoff <= diagonal_roundoff_bound, diagonal_roundoff
    # 기존 binary64 대각합과 새 longdouble 합의 반올림 차이만 통일한다.
    # 양의 비대각 물리율은 변경하지 않는다.
    np.fill_diagonal(A0, -exact_outflow)
    groups, N, nf, nb = g['groups'], g['N'], g['nf'], g['nb']
    P, W, I = g['P'], g['W'], g['I']
    dc, de, Dcb, Deb = g['dc'], g['de'], g['Dcb'], g['Deb']
    q, red_ff, red_B = g['charge_weight'], g['red_ff'], g['red_B']
    lower, upper = np.asarray(g['lower']), np.asarray(g['upper_sl'])
    blu, bul = np.asarray(g['blu']), np.asarray(g['bul'])
    incidence = g['incidence']
    starts = {z: lo for z, lo, hi, stages in groups}
    br, bc, bp, bs, bl, bu = ([] for _ in range(6))
    for z, l, u, active, photo, stim in g['bf_response_channels']:
        row = len(bl)
        bl.append(starts[z]+l); bu.append(starts[z]+u)
        br.extend([row]*len(active)); bc.extend(active)
        bp.extend(photo); bs.extend(stim)
    bl, bu = np.asarray(bl), np.asarray(bu)
    Bp = sparse.csr_matrix((bp, (br, bc)), shape=(len(bl), nb))
    Bs = sparse.csr_matrix((bs, (br, bc)), shape=(len(bl), nb))
    Binc = sparse.csr_matrix((np.r_[-np.ones(len(bl)), np.ones(len(bl))],
        (np.r_[bl, bu], np.tile(np.arange(len(bl)), 2))), shape=(N, len(bl)))
    diagonal = np.diag_indices(N)

    def generator(field):
        dJ = field-J0
        result = np.array(A0, dtype=np.longdouble, copy=True)
        lj = P@dJ
        cj = W@dJ
        # 비방사 율은 유지하고 실제 BB/BF의 J 의존항만 교환한다.
        # 시험 전용이며 생산 입력에는 쓰지 않는다.
        np.add.at(result, (upper, lower), np.asarray(blu*lj, dtype=np.longdouble))
        np.add.at(result, (lower, upper), np.asarray(bul*lj, dtype=np.longdouble))
        np.add.at(result, (bu, bl), np.asarray(Bp@cj, dtype=np.longdouble))
        np.add.at(result, (bl, bu), np.asarray(Bs@cj, dtype=np.longdouble))
        result[diagonal] = 0
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise ValueError('새 장의 율에 음수/비유한 항 발생; 절단하지 않음')
        result[diagonal] = -result.sum(axis=0)
        return result

    def radiation_response(x):
        return (incidence@sparse.diags(blu*x[lower]-bul*x[upper])@P,
                Binc@(sparse.diags(x[bl])@Bp-sparse.diags(x[bu])@Bs))

    def composition(delta):
        change_chi = dc@delta+I@(Dcb@delta)+red_ff*(q@delta)
        change_eta = de@delta+I@(Deb@delta)+red_ff*red_B*(q@delta)
        return change_chi, change_eta

    def ratios(x):
        return {str(z): float(x[lo:hi][stages == 2].sum()/x[lo:hi][stages == 1].sum())
                for z, lo, hi, stages in groups}

    def residual(matrix, x):
        f = matrix@np.asarray(x, dtype=np.longdouble)
        gross = abs(matrix)@np.asarray(x, dtype=np.longdouble)
        if np.any(gross <= 0):
            raise ValueError('잔차 정규화의 양의 총율 필요')
        level = float(np.max(abs(f)/gross))
        ion = 0.
        for z, lo, hi, stages in groups:
            for stage in np.unique(stages):
                ids = np.arange(lo, hi)[stages == stage]
                rest = np.arange(lo, hi)[stages != stage]
                inflow = np.sum(matrix[np.ix_(ids, rest)]@x[rest], dtype=np.longdouble)
                outflow = np.sum(matrix[np.ix_(rest, ids)]@x[ids], dtype=np.longdouble)
                if inflow+outflow > 0:
                    ion = max(ion, float(abs(inflow-outflow)/(inflow+outflow)))
        return f, max(level, ion), level, ion

    assert np.max(abs(generator(J0)-A0)) < 1e-8
    result = dict(scope='셸39, 고정T/ne/비열/다른 원소/비국소 잔여항; 전체 공간 수송 아님',
                  initial_ratios=ratios(x0), tolerance=1e-9,
                  baseline_diagonal_relative_roundoff=diagonal_roundoff,
                  baseline_diagonal_summation_bound=diagonal_roundoff_bound, cases=[])
    for mindepth, maxerror in ((1e4, 1e-5), (1e2, 1e-3)):
        mask = (g['depth'] > mindepth) & (g['error'] < maxerror) & np.isfinite(g['error'])

        def evaluate(x, jacobian=False):
            if np.any(x < 0) or not np.all(np.isfinite(x)):
                raise ValueError('후보 개체수 음수/비유한; floor 없음')
            dchi, deta = composition(np.asarray(x-x0, dtype=float))
            denominator = g['chi_abs']+dchi
            if np.any(denominator[mask] <= 0):
                raise ValueError('선택된 국소 구간의 소광이 양수가 아님')
            field = J0.copy()
            field[mask] += (deta[mask]-J0[mask]*dchi[mask])/denominator[mask]
            if np.any(field < 0) or not np.all(np.isfinite(field)):
                raise ValueError('후보 복사장 음수/비유한; floor 없음')
            matrix = generator(field)
            f, merit, level, ion = residual(matrix, x)
            state = dict(merit=merit, level_residual=level, ion_flow_residual=ion,
                         III_II=ratios(x), minimum_population=float(min(x)),
                         normalization_error=float(max(abs(x[lo:hi].sum()-1) for z, lo, hi, stages in groups)),
                         minimum_selected_depth=float(min(denominator[mask]*min(864000000000., g['c']*172800*g['fdl']))))
            if not jacobian:
                return matrix, f, state
            inv = np.zeros(nf); inv[mask] = 1/denominator[mask]
            R, Rbf = radiation_response(np.asarray(x, dtype=float))
            L0, LJ = sparse.diags(inv), sparse.diags(inv*field)
            D = L0@de-LJ@dc
            C = (R@D).toarray()+Rbf@(W@D).toarray()
            C += ((R@L0@I).toarray()+Rbf@(W@L0@I).toarray())@Deb
            C -= ((R@LJ@I).toarray()+Rbf@(W@LJ@I).toarray())@Dcb
            red_response = inv*red_ff*(red_B-field)
            C += np.outer(R@red_response+Rbf@(W@red_response), q)
            return matrix, f, state, matrix+C

        # 전체 비선형 잔차의 중앙차분으로 결합 Jacobian을 대조한다.
        direction = x0*np.random.default_rng(20261017).normal(size=N)
        for z, lo, hi, stages in groups:
            direction[lo:hi] -= x0[lo:hi]*direction[lo:hi].sum()
        _, _, _, jac = evaluate(x0, True)
        step = 1e-4
        numerical = (evaluate(x0+step*direction)[1]-evaluate(x0-step*direction)[1])/(2*step)
        exact = jac@direction
        derivative_error = float(max(abs(numerical-exact))/max(abs(exact)))
        assert derivative_error < 2e-6, derivative_error

        x = np.asarray(x0, dtype=np.longdouble)
        newton = []; status = 'ITERATION_LIMIT'
        for iteration in range(21):
            matrix, f, state, jac = evaluate(x, True)
            state['iteration'] = iteration; newton.append(state)
            print('LOCAL_NEWTON', mindepth, iteration, state, flush=True)
            if state['merit'] < 1e-9:
                status = 'CONVERGED'; break
            if iteration == 20:
                break
            dx, _ = g['constrained'](jac, -f)
            alpha = 1.
            negative = dx < 0
            if np.any(negative):
                alpha = min(alpha, .99*float(min(-x[negative]/dx[negative])))
            accepted = False
            for backtrack in range(30):
                trial = x+alpha*dx
                try:
                    trial_state = evaluate(trial)[2]
                    accepted = trial_state['merit'] < state['merit']
                except ValueError:
                    accepted = False
                if accepted:
                    x = trial; state['accepted_step'] = alpha; break
                alpha *= .5
            if not accepted:
                status = 'LINE_SEARCH_STOP'; break

        picard_x = np.asarray(x0, dtype=np.longdouble)
        picard = []; picard_status = 'ITERATION_LIMIT'
        for iteration in range(81):
            matrix, _, state = evaluate(picard_x)
            state['iteration'] = iteration
            if iteration < 4 or iteration % 10 == 0 or state['merit'] < 1e-9:
                picard.append(state)
                print('LOCAL_PICARD', mindepth, iteration, state, flush=True)
            if state['merit'] < 1e-9:
                picard_status = 'CONVERGED'; break
            if iteration == 80:
                break
            for z, lo, hi, stages in groups:
                picard_x[lo:hi], _ = g['stationary'](matrix[lo:hi, lo:hi])
        saved_x = np.asarray(x, dtype=float)
        replay_state = evaluate(saved_x)[2]
        if status == 'CONVERGED':
            assert replay_state['merit'] < 1e-9
            assert replay_state['normalization_error'] < 1e-12
        result['cases'].append(dict(minimum_depth=mindepth, maximum_source_error=maxerror,
            bins=int(mask.sum()), derivative_error=derivative_error,
            newton_status=status, newton=newton, picard_status=picard_status, picard=picard,
            newton_populations=[float(v) for v in saved_x], stored_population_replay=replay_state,
            picard_populations=[float(v) for v in picard_x],
            final_ratio_relative_difference={z: ratios(picard_x)[z]/ratios(x)[z]-1 for z in ratios(x)}))
    g['identities'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result
