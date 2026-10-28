"""봉인된 동일 고정장의 완료 물질 평가를 읽는다. 새 평가/실행은 하지 않는다."""
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
from lumina_coupled_material import packet

ENERGY_POLICIES = ('physical_stored', 'physical_instantaneous_scenario',
                   'native_stored', 'native_instantaneous_scenario')


def load(path, digest, *, field, source_identity, implementation, state,
         library, line_library, inputs, policy='physical_stored', prefix=None):
    path = Path(path).resolve()
    if B.sha(path) != digest:
        raise ValueError('보존 열평가 보고서 변경')
    report = json.loads(path.read_text())
    previous_policy = report.get('thermal_energy_policy', 'physical_stored')
    if prefix is not None and (type(prefix) is not int or not 1 <= prefix <= 4):
        raise ValueError('다른 열정책 미분에는 초기 섭동1~4개만 명시적으로 재사용 가능')
    if (policy not in ENERGY_POLICIES or previous_policy not in ENERGY_POLICIES or
            (previous_policy != policy and prefix is None)):
        raise ValueError('보존 열평가의 열방정식 정책 불일치; 다른 정책의 방향 후보를 재생하지 않음')
    if report.get('state') not in ('INCOMPLETE_PRESERVED',
            'COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE'):
        raise ValueError('종료된 물질 실행의 보존 평가만 재개 가능')
    if (report.get('field') != field or report.get('source_identity') != source_identity or
            report.get('thermal_implementation_sha256') != implementation):
        raise ValueError('보존 열평가의 장/물질/실제 물리 해법 불일치')
    old_inputs = report['input_sha256']
    for current, name in ((library, 'bridge.so'), (line_library, 'lines.so')):
        if B.sha(Path(current)) != old_inputs[str(path.parent.parent/'build'/name)]:
            raise ValueError('보존 열평가의 실제 바이너리 불일치')
    consumed = {str(path): digest}
    for filename, expected in old_inputs.items():
        if B.sha(Path(filename)) != expected:
            raise ValueError('보존 열평가의 원 입력 변경: '+Path(filename).name)
        if filename in inputs and inputs[filename] != expected:
            raise ValueError('보존 열평가와 현재 원 입력 불일치')
        consumed[filename] = expected

    def read(entry):
        root = Path(entry['root'])
        arrays = []
        for name, key in (('response.npz', 'response_sha256'), ('balance.npz', 'balance_sha256')):
            file = root/name
            if B.sha(file) != entry[key]:
                raise ValueError('보존 완료 열평가 변경: '+name)
            consumed[str(file)] = entry[key]
            with np.load(file, allow_pickle=False) as values:
                arrays.append({k: values[k] for k in values.files})
        data, heat = arrays
        if (not np.array_equal(data['columns'], state['columns']) or
                data['population'].shape != state['superlevel'].shape or
                not np.isfinite(data['population']).all() or np.any(data['population'] <= 0) or
                not np.allclose(data['population'].sum(axis=0), state['superlevel'].sum(axis=0), rtol=1.e-10, atol=0)):
            raise ValueError('보존 완료 열평가의 원자주소/핵수 불일치')
        for z in np.unique(state['columns'][:, 0]):
            selected = state['columns'][:, 0] == z
            if not np.allclose(data['population'][selected].sum(axis=0),
                    state['superlevel'][selected].sum(axis=0), rtol=1.e-10, atol=0):
                raise ValueError('보존 완료 열평가의 깊이별 원소/핵종 총수 변경')
        packet(data, heat, policy).validate()
        return data, heat, entry

    first = read(report['thermal_initial_response'])
    order = report.get('thermal_evaluation_order', [])
    if prefix is not None:
        if prefix > len(order): raise ValueError('요청한 초기 섭동이 아직 완료되지 않음')
        order = order[:prefix]
    replay = []
    for entry in order:
        if entry.get('fixed_field_thermal_callback') is not True:
            raise ValueError('보존 호출 순서의 비열평가 혼합')
        replay.append(read(entry))
    if not replay:
        raise ValueError('재사용할 완료 열평가 없음')
    if previous_policy != policy:
        for name in ('stored_decay', 'instantaneous_scenario'):
            reference = first[1][name]
            if (reference.shape != first[0]['ne'].shape or not np.isfinite(reference).all() or
                    np.any(reference < 0) or any(not np.array_equal(value[1][name], reference) for value in replay)):
                raise ValueError('다른 열정책 재사용의 고정 국소 침적열 프로필 불일치')
    inputs.update(consumed)
    return dict(first=first, replay=replay, charge_history=report.get('charge_history', []),
                bound_temperature_tables=report.get('temperature_bound_cache', {}).get('completed_bound_tables', []),
                provenance=dict(result=str(path), result_sha256=digest,
                    previous_energy_policy=previous_policy, energy_policy=policy,
                    initial_probe_prefix=prefix, previous_direction_or_acceptance_reused=False,
                    saved_evaluations=len(replay),
                    new_material_evaluations=0, new_transport=0))
