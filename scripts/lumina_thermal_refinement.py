"""저장한 centered 응답으로 제한 T 제안을 만들고 실제 전하/열 개선을 확인한다.

명시한 유한 근 탐색 경로다. 기존 물리 수용 기준이나 기본 조기 반환을
변경하지 않는다. 선형 예측만으로 물질을 수용하거나 온도 근을 인증하지 않는다.
"""
import numpy as np

from lumina_common_charge import solve as solve_charge
from lumina_coupled_material import packet, residual, jacobian, newton_direction


def solve_root(evaluate, first, adiabatic, lower, upper, *, policy,
               max_evaluations=12, on_evaluation=None, on_direction=None):
    """실제 수용 상태에서 box 보정과 새 centered 근 확인을 같은 예산 안에 수행한다.

    4회 centered 확인을 남겨두고 후보를 평가한다. 예산 부족은 미수렴이며,
    작은 제한 보폭이나 열잔차만으로 근을 인증하지 않는다.
    """
    from lumina_fixed_field_thermal import linearize
    if type(max_evaluations) is not int or not 4 <= max_evaluations <= 12:
        raise ValueError('box 근 탐색은 실제 응답4~12회 한도 필요')
    selected = first; used = 0; accepted = 0; history = []; directions = []
    root_max = None; converged = False; reason = 'EVALUATION_LIMIT'
    def actual(t, ne):
        nonlocal used
        if used >= max_evaluations: raise RuntimeError('box 근 탐색 실제평가 한도 초과')
        value = evaluate(t, ne); used += 1
        return value
    def decision(value, entry):
        item = dict(entry, evaluation=used)
        if history and history[-1]['evaluation'] == used: history[-1] = item
        else: history.append(item)
        if on_evaluation is not None: on_evaluation(value, item)
    while max_evaluations-used >= 4:
        probes = []; captured = []
        def probe(t, ne):
            value = actual(t, ne); probes.append(value); return value
        def direction(*args):
            captured.append(args)
            if on_direction is not None: on_direction(*args)
        linear = linearize(probe, selected, adiabatic, policy=policy,
                           on_evaluation=decision, on_direction=direction)
        directions.append(linear)
        root_max = max(linear['max_abs_log_temperature_direction'], linear['max_abs_log_ne_direction'])
        base = packet(*selected[:2], policy)
        physical = residual(base, adiabatic, base.ne, base.thermal_scale)
        n = len(base.ne)
        if root_max <= 1.e-3 and abs(physical[:n]).max() <= 1.e-10 and abs(physical[n:]).max() <= 1.e-3:
            converged = True; reason = 'FRESH_CENTERED_ROOT_AND_PHYSICAL_TOLERANCES'; break
        allowance = max_evaluations-used-4
        if allowance < 2: break
        _, matrix, f, delta, c, h = captured[0]
        basis = dict(temperature=base.temperature, ne=base.ne, jacobian=matrix,
                     scaled_residual=f, log_direction=delta, charge_scale=c, thermal_scale=h)
        correction = refine(actual, selected, probes, adiabatic, basis, lower, upper,
                            policy=policy, max_evaluations=allowance, max_temperature_trials=1,
                            on_evaluation=decision)
        if not correction['accepted_iterations']:
            reason = 'ACTUAL_BOX_CANDIDATE_NOT_ACCEPTED'; break
        selected = correction['selected']; accepted += 1; root_max = None
    final = packet(*selected[:2], policy); n = len(final.ne)
    physical = residual(final, adiabatic, final.ne, final.thermal_scale)
    return dict(selected=selected, history=history, centered_checks=directions,
        accepted_iterations=accepted, evaluations_used=used, fresh_evaluations_used=used,
        replayed_evaluations=0, fixed_field_numerical_converged=converged,
        fixed_field_root_converged=converged, root_search_requested=True,
        root_log_step_tolerance=1.e-3, root_fresh_direction_max=root_max,
        charge_relative=float(abs(final.charge/final.ne).max()),
        thermal_physical_scaled_max=float(abs(physical[n:]).max()),
        physical_tolerances_satisfied=bool(abs(physical[:n]).max()<=1.e-10 and abs(physical[n:]).max()<=1.e-3),
        charge_tolerance=1.e-10, thermal_tolerance=1.e-3,
        charge_projected_trials=True, bounded_root_restart=True, stop_reason=reason,
        new_transport=0, joint_convergence=False)


def bounded_direction(first, probes, adiabatic, basis, *, policy, max_log_step=.1):
    """동일 기점/네 응답/열행을 확인하고 전하 제거 후 box 선형 문제를 푼다."""
    from scipy.optimize import lsq_linear

    if len(probes) != 4 or not np.isfinite(max_log_step) or max_log_step <= 0:
        raise ValueError('제한 열방향에는 centered4응답과 양의 보폭 필요')
    base = packet(*first[:2], policy); n = len(base.ne)
    c, h = base.ne.copy(), base.thermal_scale.copy()
    f = residual(base, adiabatic, c, h)
    matrix = jacobian(base, *(packet(*p[:2], policy) for p in probes), 1.e-4,
                      adiabatic, charge_scale=c, thermal_scale=h)
    expected = dict(temperature=base.temperature, ne=base.ne, charge_scale=c,
                    thermal_scale=h, scaled_residual=f, jacobian=matrix)
    if any(not np.array_equal(basis.get(k), v) for k, v in expected.items()):
        raise ValueError('제한 열방향의 기점/네 응답/정책/선형계 불일치')
    direction = newton_direction(matrix, f)
    if not np.array_equal(basis.get('log_direction'), direction):
        raise ValueError('제한 열방향의 저장 Newton 방향 불일치')
    a, b, d, e = matrix[:n, :n], matrix[:n, n:], matrix[n:, :n], matrix[n:, n:]
    charge_T = np.linalg.solve(b, a); charge_f = np.linalg.solve(b, f[:n])
    schur = d-e@charge_T; rhs = f[n:]-e@charge_f
    row = np.max(abs(schur), axis=1)
    if np.any(row <= 0) or not np.isfinite(row).all():
        raise ValueError('제한 열방향의 전하 제거 열행에 유한 감도 없음')
    fit = lsq_linear(schur/row[:, None], -rhs/row,
                     bounds=(-max_log_step, max_log_step), method='bvls',
                     tol=1.e-12, max_iter=200)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('제한 열방향 선형 최소제곱 실패')
    limited = np.r_[fit.x, -charge_f-charge_T@fit.x]
    # ne의 한도도 넘지 않도록 전체 제안에 같은 인자를 적용한다.
    # 실제 T/ne/인구/소광을 잘라내거나 선형 예측을 실제 응답으로 대용하지 않는다.
    magnitude = float(abs(limited).max())
    if magnitude > max_log_step: limited *= max_log_step/magnitude
    predicted = f+matrix@limited
    return dict(direction=limited, matrix=matrix, residual=f, charge_scale=c,
                thermal_scale=h, predicted_heat_max=float(abs(predicted[n:]).max()),
                predicted_charge_max=float(abs(predicted[:n]).max()),
                box_iterations=int(fit.nit), box_optimality=float(fit.optimality))


def refine(evaluate, first, probes, adiabatic, basis, lower, upper, *, policy,
           max_evaluations=12, max_temperature_trials=2, max_log_step=.1,
           on_evaluation=None, on_proposal=None):
    """최대 두 온도 제안과 고정 T 전하 보정으로 실제 개선 한 번만 수용한다.

RE 통과는 후보의 필요조건일 뿐이다. 기점 척도의 실제 열잔차도 감소해야
수용한다. 기존 기준 안에 있다는 이유만으로 기점을 새 수정 물질로 반환하지 않는다.
"""
    if (type(max_evaluations) is not int or not 1 <= max_evaluations <= 12 or
            type(max_temperature_trials) is not int or not 1 <= max_temperature_trials <= 2 or
            max_evaluations < max_temperature_trials):
        raise ValueError('유한 온도수정의 실제평가1~12회/새 T1~2회 한도 오류')
    base = packet(*first[:2], policy); n = len(base.ne)
    lo, hi = (np.asarray(v, dtype='f8') for v in (lower, upper))
    if (lo.shape != base.ne.shape or hi.shape != base.ne.shape or
            not np.isfinite([lo, hi]).all() or np.any(lo <= 0) or np.any(hi <= lo) or
            np.any(base.ne < lo) or np.any(base.ne > hi) or
            float(abs(base.charge/base.ne).max()) > 1.e-10):
        raise ValueError('유한 온도수정의 원소 수밀도 경계/기준 전하 오류')
    model = bounded_direction(first, probes, adiabatic, basis,
                              policy=policy, max_log_step=max_log_step)
    c, h, f = (model[k] for k in ('charge_scale', 'thermal_scale', 'residual'))
    start = float(abs(f[n:]).max()); history = []; proposals = []; values = {}
    selected = first; accepted = False
    for trial in range(max_temperature_trials):
        alpha = 2.**-trial; step = alpha*model['direction']
        predicted = f+model['matrix']@step
        if start == 0 or float(abs(predicted[n:]).max()) >= start*(1.-1.e-4*alpha): break
        t = base.temperature*np.exp(step[:n]); ne = base.ne*np.exp(step[n:])
        if np.any(ne < lo) or np.any(ne > hi): continue
        # 각 새 T에 실제 전하 보정 예산을 남기되 총 callback은 넘기지 않는다.
        allowance = (max_evaluations-len(history))//(max_temperature_trials-trial)
        if allowance < 1: break
        proposal = dict(trial=trial+1, alpha=alpha,
            max_abs_log_T_step=float(abs(step[:n]).max()),
            max_abs_log_ne_step=float(abs(step[n:]).max()),
            predicted_heat_max=float(abs(predicted[n:]).max()), evaluated=False, accepted=False)
        proposals.append(proposal)
        if on_proposal is not None: on_proposal(proposal.copy())
        def actual(density):
            if len(history) >= max_evaluations: raise RuntimeError('유한 온도수정 실제평가 한도 초과')
            value = evaluate(t.copy(), density.copy()); p = packet(*value[:2], policy)
            if (not np.array_equal(p.temperature, t) or not np.array_equal(p.ne, density) or
                    not np.allclose(p.nuclei, base.nuclei, rtol=1.e-12, atol=0)):
                raise ValueError('유한 온도수정 callback의 T/ne/핵수 불일치')
            se = value[2].get('SE_residual')
            if se is None or not np.isfinite(se) or se > 1.e-8 or se < 0:
                raise ValueError('유한 온도수정 callback의 실제 SE 증거 오류')
            r = residual(p, adiabatic, c, h)
            entry = dict(evaluation=len(history)+1, trial=trial+1, alpha=alpha,
                phase='temperature_trial' if not any(e['trial']==trial+1 for e in history)
                      else 'charge_projection_ne', accepted=False,
                charge_relative=float(abs(p.charge/p.ne).max()),
                scaled_max=float(abs(r).max()), thermal_scaled_max=float(abs(r[n:]).max()))
            history.append(entry); values[id(p)] = (value, entry)
            if on_evaluation is not None: on_evaluation(value, entry.copy())
            return dict(packet=p, charge_residual=p.charge)
        initial = actual(ne); candidate = initial['packet']; proposal['evaluated'] = True
        charge_closed = float(abs(candidate.charge/candidate.ne).max()) <= 1.e-10
        if not charge_closed and allowance > 1:
            charge = solve_charge(actual, ne, candidate.charge, lo, hi,
                                   tolerance=1.e-10, max_evaluations=allowance-1)
            charge_closed = charge['converged']
            if charge['payload'] is not None: candidate = charge['payload']['packet']
        actual_heat = float(abs(residual(candidate, adiabatic, c, h)[n:]).max())
        physical = residual(candidate, adiabatic, candidate.ne, candidate.thermal_scale)
        accepted = bool(charge_closed and float(abs(physical[n:]).max()) <= 1.e-3 and
                        actual_heat < start*(1.-1.e-4*alpha))
        value, entry = values[id(candidate)]
        entry.update(phase='charge_projected_line', accepted=accepted, charge_solved=bool(charge_closed),
                     scaled_start_max=start, actual_heat_max=actual_heat,
                     thermal_physical_scaled_max=float(abs(physical[n:]).max()))
        if on_evaluation is not None: on_evaluation(value, entry.copy())
        proposal.update(accepted=accepted, actual_heat_max=actual_heat, charge_solved=bool(charge_closed))
        if on_proposal is not None: on_proposal(proposal.copy())
        if accepted:
            selected = value
            break
    final = packet(*selected[:2], policy)
    physical = residual(final, adiabatic, final.ne, final.thermal_scale)
    return dict(selected=selected, history=history, proposals=proposals,
        accepted_iterations=int(accepted), evaluations_used=len(history),
        fresh_evaluations_used=len(history), replayed_evaluations=0,
        saved_centered_probes_reused=4, temperature_trials=len(proposals),
        charge_projected_trials=True, temperature_refinement=True,
        fixed_field_numerical_converged=bool(accepted),
        charge_relative=float(abs(final.charge/final.ne).max()),
        charge_tolerance=1.e-10, thermal_tolerance=1.e-3,
        thermal_physical_scaled_max=float(abs(physical[n:]).max()),
        temperature_root_certified=False, new_transport=0, joint_convergence=False)
