"""봉인된 동일 고정장의 완료 물질 평가를 읽는다. 새 평가/실행은 하지 않는다."""
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
from lumina_coupled_material import packet


def load(path, digest, *, field, source_identity, implementation, state,
         library, line_library, inputs):
    path = Path(path).resolve()
    if B.sha(path) != digest:
        raise ValueError('보존 열평가 보고서 변경')
    report = json.loads(path.read_text())
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
        packet(data, heat, 'physical_stored').validate()
        return data, heat, entry

    first = read(report['thermal_initial_response'])
    replay = []
    for entry in report.get('thermal_evaluation_order', []):
        if entry.get('fixed_field_thermal_callback') is not True:
            raise ValueError('보존 호출 순서의 비열평가 혼합')
        replay.append(read(entry))
    if not replay:
        raise ValueError('재사용할 완료 열평가 없음')
    inputs.update(consumed)
    return dict(first=first, replay=replay, charge_history=report.get('charge_history', []),
                provenance=dict(result=str(path), result_sha256=digest,
                    saved_evaluations=len(replay), new_material_evaluations=0, new_transport=0))
