"""기존 감쇠 T/ne 해법을 실제 새 복사장 물질응답 callback에 연결한다.

이는 고정장 물질 블록이며 수송/산란 또는 최종 공동수렴을 판정하지 않는다.
"""
import numpy as np

from lumina_coupled_material import packet, solve_coupled, residual, jacobian, newton_direction
import lumina_charge_projection as charge_projection
from lumina_common_charge import solve as solve_charge


def linearize(evaluate, first, adiabatic, *, policy, max_evaluations=4,
              on_evaluation=None, on_direction=None):
    """같은 J의 centered 네 응답만 소비한다. 근 탐색/후보 수용은 하지 않는다.

    기존 RE 허용값을 바꾸거나 수렴 판정을 우회하지 않는다. 허용 영역 안의
    기준점에서도 선형 온도방향을 별도로 추정하며 실제 물질을 게시하지 않는다.
    """
    if type(max_evaluations) is not int or max_evaluations != 4:
        raise ValueError('온도방향 진단은 centered 실제평가4회 필요')
    step = 1.e-4
    base = packet(*first[:2], policy)
    if float(np.max(abs(base.charge/base.ne))) > 1.e-10:
        raise ValueError('같은 J의 전하가 닫힌 기준 물질 필요')
    t, ne = base.temperature, base.ne
    history = []; probes = []
    for phase, temperature, density in (
            ('T+', t*np.exp(step), ne.copy()), ('T-', t*np.exp(-step), ne.copy()),
            ('ne+', t.copy(), ne*np.exp(step)), ('ne-', t.copy(), ne*np.exp(-step))):
        value = evaluate(temperature, density)
        probe = packet(*value[:2], policy)
        if (not np.array_equal(probe.temperature, temperature) or
                not np.array_equal(probe.ne, density)):
            raise ValueError('온도방향 진단이 요청 T/ne를 소비하지 않음')
        entry = dict(evaluation=len(probes)+1, phase=phase, purpose='CENTERED_LINEARIZATION_ONLY')
        probes.append(probe); history.append(entry)
        if on_evaluation is not None: on_evaluation(value, entry)
    c, h = base.ne.copy(), base.thermal_scale.copy()
    f = residual(base, adiabatic, c, h)
    matrix = jacobian(base, *probes, step, adiabatic, charge_scale=c, thermal_scale=h)
    direction = newton_direction(matrix, f)
    if on_direction is not None: on_direction(first, matrix, f, direction, c, h)
    n = len(t); norm = float(abs(f).max())
    error = float(abs(matrix@direction+f).max())
    return dict(state='CENTERED_DIRECTION_NOT_ROOT_OR_ACCEPTED_MATERIAL', history=history,
        evaluations_used=4, fresh_evaluations_used=4, centered_log_step=float(step),
        max_abs_log_temperature_direction=float(abs(direction[:n]).max()),
        max_abs_log_ne_direction=float(abs(direction[n:]).max()),
        linear_relative_residual=error/norm if norm else 0.,
        accepted_iterations=0, material_published=False, new_transport=0,
        joint_convergence=False)


def project(evaluate, first, saved_trial, adiabatic, basis, alpha, lower, upper, *,
            policy, max_evaluations, on_evaluation=None, on_replay=None):
    """원 제안 T를 고정해 실제 전하를 닫는다. 완료 첫 후보는 재평가하지 않는다."""
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 4:
        raise ValueError('고정 T 신규 전하평가 한도1~4회 필요')
    base = packet(*first[:2], policy); values = {id(base): first}; indices = {}
    history = []; fresh = 0
    def actual(t, ne):
        nonlocal fresh
        if not history:
            candidate = saved_trial
            if on_replay is not None: on_replay(candidate)
        else:
            if fresh >= max_evaluations:
                raise RuntimeError('고정 T 신규 전하평가 한도 초과')
            if not np.array_equal(t, saved_trial[0]['temperature']):
                raise ValueError('전하 보정에서 온도 변경 금지')
            candidate = evaluate(t, ne); fresh += 1
        data, heat, _ = candidate
        if not np.array_equal(data['temperature'], t) or not np.array_equal(data['ne'], ne):
            raise ValueError('완료 전하 후보의 원 제안 T/ne 불일치')
        value = packet(data, heat, policy)
        values[id(value)] = candidate; indices[id(value)] = len(history)
        history.append(dict(evaluation=len(history)+1,
            phase='charge_projection_initial' if not history else 'charge_projection_ne',
            scaled_max=float(abs(residual(value, adiabatic, basis['charge_scale'], basis['thermal_scale'])).max())))
        return value
    def decision(value, entry):
        record = history[indices[id(value)]]
        record.update({k: v for k, v in entry.items() if k != 'evaluation'})
        if on_evaluation is not None: on_evaluation(values[id(value)], record)
    outer = charge_projection.trial(actual, base, adiabatic, basis, alpha, lower, upper,
        max_evaluations=max_evaluations+1, on_evaluation=decision)
    selected = values[id(outer['packet'])]
    return dict(selected=selected, history=history, accepted_iterations=outer['iterations'],
        fixed_field_numerical_converged=outer['numerical_converged'],
        charge_relative=float(abs(selected[0]['charge_residual']/selected[0]['ne']).max()),
        evaluations_used=len(history), fresh_evaluations_used=fresh, replayed_evaluations=1,
        charge_projection=True, charge_projection_history=outer['charge_history'],
        new_transport=0, joint_convergence=False)


def solve(evaluate, first, adiabatic, *, policy, max_evaluations,
          numerical_tolerance=1.e-10, charge_tolerance=None, thermal_tolerance=None,
          max_log_step=.1, on_evaluation=None,
          on_direction=None, replay=(), on_replay=None, charge_bounds=None,
          secant_after_accept=False, root_log_step_tolerance=None):
    """evaluate(T,ne)는 같은 J에서 (전체 SE 결과, 전체 열장부, 기록)을 반환한다."""
    replay = tuple(replay)
    centered_minimum = 4 if root_log_step_tolerance is not None else 5
    minimum = 1 if replay else centered_minimum
    evaluation_limit = (48 if secant_after_accept and root_log_step_tolerance is not None
                        else 12)
    if (type(max_evaluations) is not int or not minimum <= max_evaluations <= evaluation_limit or
            len(replay)+max_evaluations < centered_minimum):
        raise ValueError('고정장 온도 보정은 secant root 최대48, 그 외 최대12 실제평가와 완전한 섭동/후보 예산 필요')
    data, energy, metadata = first
    initial = packet(data, energy, policy)
    values = {id(initial): first}
    used = []; fresh = 0
    projector = None
    if charge_bounds is not None:
        lower, upper = [np.asarray(a, dtype='f8') for a in charge_bounds]
        if (lower.shape != initial.ne.shape or upper.shape != initial.ne.shape or
                not np.isfinite([lower, upper]).all() or np.any(lower <= 0) or np.any(upper <= lower)):
            raise ValueError('정상 열반복의 원소 수밀도 경계 오류')
        def projector(value, at_fixed_T, budget):
            if np.any(value.ne < lower) or np.any(value.ne > upper): return None
            if float(abs(value.charge/value.ne).max()) <= 1.e-10: return value
            if budget < 1: return None
            def evaluate_ne(ne):
                candidate = at_fixed_T(ne)
                return dict(packet=candidate, charge_residual=candidate.charge)
            solved = solve_charge(evaluate_ne, value.ne, value.charge, lower, upper,
                                  tolerance=1.e-10, max_evaluations=budget)
            if not solved['converged']: return None
            return value if solved['payload'] is None else solved['payload']['packet']

    def actual(temperature, ne):
        nonlocal fresh
        if len(used) < len(replay):
            candidate = replay[len(used)]
            old = candidate[0]
            if (not np.array_equal(old['temperature'], temperature) or
                    not np.array_equal(old['ne'], ne)):
                raise ValueError('보존 열평가의 호출 순서/T/ne 불일치; 새 적분으로 조용히 대체하지 않음')
            if on_replay is not None: on_replay(candidate)
        else:
            if fresh >= max_evaluations:
                raise RuntimeError('신규 열평가 예산 초과')
            candidate = evaluate(temperature, ne)
            fresh += 1
        result, heat, record = candidate
        value = packet(result, heat, policy)
        if (not np.array_equal(value.temperature, temperature) or
                not np.array_equal(value.ne, ne)):
            raise ValueError('새 장 열보정 callback의 T/ne 불일치')
        values[id(value)] = candidate
        used.append(candidate)
        return value

    def decision(value, entry):
        if on_evaluation is not None:
            on_evaluation(values[id(value)], entry)

    def direction(value, matrix, residual, delta, charge_scale, thermal_scale, kind='centered'):
        if on_direction is not None:
            arguments=(values[id(value)], matrix, residual, delta, charge_scale, thermal_scale)
            if secant_after_accept:
                on_direction(*arguments, kind)
            else:
                on_direction(*arguments)

    outer = solve_coupled(actual, initial, adiabatic,
        max_evaluations=len(replay)+max_evaluations, numerical_tolerance=numerical_tolerance,
        max_log_step=max_log_step, on_evaluation=decision, on_direction=direction,
        project_candidate=projector,secant_after_accept=secant_after_accept,
        charge_tolerance=charge_tolerance,thermal_tolerance=thermal_tolerance,
        root_log_step_tolerance=root_log_step_tolerance)
    selected = values[id(outer['packet'])]
    # 기준 물질 자체가 물리 문턱을 만족하면 섭동은 필요 없다. 사용한 섭동의
    # 순서 검증은 그대로 두고, 미사용 섭동을 소비한 것으로 보고하지 않는다.
    converged_without_replay = (not used and outer['iterations'] == 0 and
                                outer['numerical_converged'])
    if len(used) < len(replay) and not converged_without_replay:
        raise ValueError('현재 해법에서 보존 열평가 순서가 끝까지 재현되지 않음')
    result = selected[0]
    charge = float(np.max(abs(result['charge_residual']/result['ne'])))
    return dict(selected=selected, history=outer['history'],
        accepted_iterations=outer['iterations'],
        fixed_field_numerical_converged=outer['numerical_converged'],
        fixed_field_root_converged=outer['fixed_field_root_converged'],
        physical_tolerances_satisfied=outer['physical_tolerances_satisfied'],
        root_search_requested=outer['root_search_requested'],
        root_log_step_tolerance=outer['root_log_step_tolerance'],
        root_fresh_direction_max=outer['root_fresh_direction_max'],
        charge_scaled_max=outer['charge_scaled_max'],
        thermal_scaled_max=outer['thermal_scaled_max'],
        charge_physical_scaled_max=outer['charge_physical_scaled_max'],
        thermal_physical_scaled_max=outer['thermal_physical_scaled_max'],
        charge_tolerance=outer['charge_tolerance'],
        thermal_tolerance=outer['thermal_tolerance'],
        charge_relative=charge, evaluations_used=len(outer['history']),
        fresh_evaluations_used=fresh, replayed_evaluations=len(used)-fresh,
        charge_projected_trials=projector is not None,
        secant_after_accept=secant_after_accept, new_transport=0, joint_convergence=False)
