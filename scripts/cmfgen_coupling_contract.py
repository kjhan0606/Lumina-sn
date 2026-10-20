"""명시 등록된 유한 장 결합 계약. 기존 full-step 기본 규약은 바꾸지 않는다."""
import json
from pathlib import Path
import re

import numpy as np

from ptf_gated_metrics import read_dimensions, parse_solutions
from run_cmfgen_lambda_probe import sha

PURPOSE = 'CMFGEN_COUPLING_68_69_2026-10-20'
FOLLOWUP_PURPOSE = 'CMFGEN_COUPLING_70_71_2026-10-20'
INITIAL_RECORDS = {PURPOSE: 67, FOLLOWUP_PURPOSE: 69}


def load_contract(path, expected_sha256):
    if not expected_sha256 or sha(path) != expected_sha256:
        raise ValueError('부분-step 계약 신원 불일치')
    c = json.loads(path.read_text())
    initial = INITIAL_RECORDS.get(c['purpose_id'])
    if (c['schema'] != 1 or initial is None or c['initial_record'] != initial or
        c['allowed_source_records'] != [initial, initial+1] or c['allowed_new_records'] != [initial+1, initial+2] or
        any(c[k] != 2 for k in ('maximum_outer_executions', 'maximum_RT_executions', 'maximum_native_executions')) or
        c['automatic_extension'] is not False or c['T_release'] is not False or
        c['final_goal_complete'] is not False):
        raise ValueError('이번 최대2outer의 범위와 다른 계약')
    if initial == 69 and (c['initial_next_outer_eligible_unchanged'] is not True or
        c['maximum_parent_reservation_seconds'] != 5400 or
        c['quantitative_observation'] != dict(reference_record=69,
            each_new_record_ne_max_relative_step_below=.005,
            final_F_P_all_components_below_reference=True, physical_equivalence_acceptance=False)):
        raise ValueError('70/71 관찰 목표 또는 자원 범위 변경')
    expected = dict(FIX_T='T', FIX_NE='F', DO_DDT='F', DO_LAM_IT='T', NUM_ITS='1',
        MAX_LAM='1.0D6', SCALE_OPT='LOCAL', LAM_SCALE_OPT='NONE', DO_NG='F',
        DO_LAM_AUTO='F', DO_T_AUTO='F', WRITE_RATES='F')
    if c['controls'] != expected:
        raise ValueError('이번 부분-step의 물리/수치 제어 불일치')
    if c['resources_per_outer'] != dict(CPU=16, OMP=16, memory_GiB=32, GPU=0,
        parent_seconds=2700, shared_child_seconds=2400, retry=0, node_pinned=False):
        raise ValueError('계약 자원 한도 불일치')
    c.update(_path=str(path), _sha256=expected_sha256)
    for name, key in (('noninterference_evidence', 'noninterference_evidence_sha256'),
                      ('C_binary', 'C_binary_sha256')):
        if sha(Path(c[name])) != c[key]:
            raise ValueError('계약 근거 변경: '+name)
    evidence = json.loads(Path(c['noninterference_evidence']).read_text())
    if (not evidence.get('evaluation_complete') or not evidence.get('source_unchanged') or
        evidence.get('noninterference') != 'REPRODUCED' or evidence.get('failure') or
        evidence['native_executions'] != 2 or evidence['RT_executions'] != 0 or evidence['outer_executions'] != 0):
        raise ValueError('완료된 실제 비간섭 근거가 아님')
    return c


def witness_executable(c):
    build = Path(c['native_witness_build'])
    manifest = build/'se-witness-build.json'
    executable = build/'cmfgen_se_witness.exe'
    if sha(manifest) != c['native_witness_manifest_sha256'] or sha(executable) != c['native_witness_binary_sha256']:
        raise ValueError('비간섭 확인 witness build 변경')
    m = json.loads(manifest.read_text())
    for name, expected in {**m['input_sha256'], **m['output_sha256']}.items():
        if sha(Path(name)) != expected:
            raise ValueError('witness build 입력 변경: '+name)
    return executable


def classify_step(correction, minimum_scale, actual_full_step):
    """ES12.4의 반올림 폭을 사용해 증가 상한과 양수성 감소 제한을 구분한다."""
    a = np.asarray(correction, dtype=float)
    result = dict(allowed=False, physical_SE_convergence='NOT_JUDGED')
    if (a.ndim != 2 or a.shape[0] < 3 or not np.isfinite(a).all() or
        minimum_scale is None or not np.isfinite(minimum_scale) or not 0 < minimum_scale <= 1):
        return dict(result, reason='STEP_EVIDENCE_UNAVAILABLE')
    if np.any(a[-1] != 0):
        return dict(result, reason='FIXED_T_CORRECTION_NOT_ZERO')
    # native LOCAL은 NT-1까지 검사하므로 전자밀도도 증가 제한에 포함한다.
    scaled = a[:-1]
    x = abs(scaled)
    radius = np.zeros_like(x)
    nonzero = x > 0
    radius[nonzero] = .5*np.power(10., np.floor(np.log10(x[nonzero]))-4)
    growth_upper = float(np.nextafter(np.max(1-scaled+radius), np.inf))
    decrease_lower = float(np.nextafter(np.max(scaled-radius), -np.inf))
    result.update(maximum_scaled_variable_growth_upper=growth_upper,
        maximum_scaled_variable_correction_lower=decrease_lower,
        printed_minimum_scale=float(minimum_scale), actual_full_step=bool(actual_full_step))
    if growth_upper >= 1e6:
        return dict(result, reason='RELATIVE_GROWTH_CAP_NOT_EXCLUDED')
    if minimum_scale == 1 and actual_full_step:
        return dict(result, allowed=True, reason='FULL_STEP_REPRODUCED')
    if minimum_scale < 1 and decrease_lower > 1-1e-6:
        return dict(result, allowed=True, reason='POSITIVITY_DECREASE_LIMIT_WITH_GROWTH_CAP_EXCLUDED')
    return dict(result, reason='PARTIAL_STEP_CAUSE_UNPROVEN')


def step_evidence(source, runtime):
    nd, nt = read_dimensions(source)
    solutions = parse_solutions(source/'STEQ_VALS', nt, nd)
    if not solutions:
        raise ValueError('실제 보정 배열 없음')
    scales = re.findall(r'The local minimum value of scale is:\s*(\S+)', (source/'OUTGEN').read_text())
    if not scales:
        raise ValueError('실제 적용 배율 없음')
    minimum = float(scales[-1])
    if minimum != runtime.get('printed_minimum_scale'):
        raise ValueError('원시 출력과 runtime의 배율 불일치')
    return classify_step(solutions[-1], minimum, runtime.get('actual_full_step_within_print_precision') is True)


def authorize_source(source, record, inputs, runtime, c, selection=None):
    source = source.resolve()
    initial = c['initial_record']
    if record not in c['allowed_source_records'] or source == Path(c['observation68_checkpoint_forbidden']).resolve():
        raise ValueError('계약 밖 세대 또는 관찰 부산물 checkpoint 금지')
    if record == initial:
        if source != Path(c['initial_source']).resolve() or sha(source.parent/'seq_logs/feedback-runtime.json') != c['initial_runtime_sha256']:
            raise ValueError(f'이번 계약의 원래 완료{initial}이 아님')
        if runtime.get('next_outer_eligible') is not c['initial_next_outer_eligible_unchanged']:
            raise ValueError(f'원래{initial}의 기본 적격 메타데이터 변경')
    else:
        if (inputs.get('coupling_contract_sha256') != c['_sha256'] or inputs.get('coupling_purpose') != c['purpose_id'] or
            Path(inputs['source']).resolve() != Path(c['initial_source']).resolve() or
            inputs['source_state']['record'] != initial or runtime.get('coupling_next_input_permitted') is not True):
            raise ValueError(f'이번 새 J{initial+1}/물질{initial+1}의 계보가 아님')
        if selection is None:
            raise ValueError(f'{initial+2} 실행에는 첫 실제 outer의 묶음 판단 필요')
        chosen = json.loads(selection.read_text())
        if (chosen.get('decision') != f'CONTINUE{initial+2}_WITHIN_LIMIT' or
            chosen.get('coupling_contract_sha256') != c['_sha256'] or
            Path(chosen.get('selected_material', '')).resolve() != source):
            raise ValueError(f'이번 물질{initial+1}의 후속{initial+2}가 선택되지 않음')
        for name, expected in chosen['input_sha256'].items():
            if sha(Path(name)) != expected:
                raise ValueError('첫 outer 선택 근거 변경')
    evidence = step_evidence(source, runtime)
    if not evidence['allowed']:
        raise ValueError('부분-step 계약의 후속 입력 부적격: '+evidence['reason'])
    return evidence


def claim_generation(c, source_record, root, *, claims_root=None):
    """한 계약의 같은 세대 중복 제출을 막아 retry0/최대2회를 집행한다."""
    if source_record not in c['allowed_source_records']:
        raise ValueError('이번 계약 밖 실행 청구')
    folder = claims_root or Path('/gpfs/kjhan/cmfgen_runs')/('lumina_'+c['purpose_id']+'_claims')
    folder.mkdir(exist_ok=True)
    path = folder/f'{source_record+1}.json'
    with path.open('x') as stream:
        json.dump(dict(coupling_contract_sha256=c['_sha256'], input_record=source_record,
            new_record=source_record+1, run=str(root), retry=0), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return str(path)
