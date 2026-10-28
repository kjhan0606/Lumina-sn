"""기존 감쇠 T/ne 해법을 실제 새 복사장 물질응답 callback에 연결한다.

이는 고정장 물질 블록이며 수송/산란 또는 최종 공동수렴을 판정하지 않는다.
"""
import numpy as np

from lumina_coupled_material import packet, solve_coupled


def solve(evaluate, first, adiabatic, *, policy, max_evaluations,
          numerical_tolerance=1.e-10, max_log_step=.1, on_evaluation=None):
    """evaluate(T,ne)는 같은 J에서 (전체 SE 결과, 전체 열장부, 기록)을 반환한다."""
    if type(max_evaluations) is not int or not 5 <= max_evaluations <= 12:
        raise ValueError('고정장 온도 보정은 실제평가 5~12회 한도를 명시해야 함')
    data, energy, metadata = first
    initial = packet(data, energy, policy)
    values = {id(initial): first}

    def actual(temperature, ne):
        candidate = evaluate(temperature, ne)
        result, heat, record = candidate
        value = packet(result, heat, policy)
        if (not np.array_equal(value.temperature, temperature) or
                not np.array_equal(value.ne, ne)):
            raise ValueError('새 장 열보정 callback의 T/ne 불일치')
        values[id(value)] = candidate
        return value

    def decision(value, entry):
        if on_evaluation is not None:
            on_evaluation(values[id(value)], entry)

    outer = solve_coupled(actual, initial, adiabatic,
        max_evaluations=max_evaluations, numerical_tolerance=numerical_tolerance,
        max_log_step=max_log_step, on_evaluation=decision)
    selected = values[id(outer['packet'])]
    result = selected[0]
    charge = float(np.max(abs(result['charge_residual']/result['ne'])))
    return dict(selected=selected, history=outer['history'],
        accepted_iterations=outer['iterations'],
        fixed_field_numerical_converged=outer['numerical_converged'],
        charge_relative=charge, evaluations_used=len(outer['history']),
        new_transport=0, joint_convergence=False)
