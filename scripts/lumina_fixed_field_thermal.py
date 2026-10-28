"""기존 감쇠 T/ne 해법을 실제 새 복사장 물질응답 callback에 연결한다.

이는 고정장 물질 블록이며 수송/산란 또는 최종 공동수렴을 판정하지 않는다.
"""
import numpy as np

from lumina_coupled_material import packet, solve_coupled


def solve(evaluate, first, adiabatic, *, policy, max_evaluations,
          numerical_tolerance=1.e-10, max_log_step=.1, on_evaluation=None,
          on_direction=None, replay=(), on_replay=None):
    """evaluate(T,ne)는 같은 J에서 (전체 SE 결과, 전체 열장부, 기록)을 반환한다."""
    replay = tuple(replay)
    minimum = 1 if replay else 5
    if (type(max_evaluations) is not int or not minimum <= max_evaluations <= 12 or
            len(replay)+max_evaluations < 5):
        raise ValueError('고정장 온도 보정은 신규 실제평가 한도와 완전한 섭동/후보 예산 필요')
    data, energy, metadata = first
    initial = packet(data, energy, policy)
    values = {id(initial): first}
    used = []; fresh = 0

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

    def direction(value, matrix, residual, delta, charge_scale, thermal_scale):
        if on_direction is not None:
            on_direction(values[id(value)], matrix, residual, delta, charge_scale, thermal_scale)

    outer = solve_coupled(actual, initial, adiabatic,
        max_evaluations=len(replay)+max_evaluations, numerical_tolerance=numerical_tolerance,
        max_log_step=max_log_step, on_evaluation=decision, on_direction=direction)
    selected = values[id(outer['packet'])]
    if len(used) < len(replay):
        raise ValueError('현재 해법에서 보존 열평가 순서가 끝까지 재현되지 않음')
    result = selected[0]
    charge = float(np.max(abs(result['charge_residual']/result['ne'])))
    return dict(selected=selected, history=outer['history'],
        accepted_iterations=outer['iterations'],
        fixed_field_numerical_converged=outer['numerical_converged'],
        charge_relative=charge, evaluations_used=len(outer['history']),
        fresh_evaluations_used=fresh, replayed_evaluations=len(used)-fresh,
        new_transport=0, joint_convergence=False)
