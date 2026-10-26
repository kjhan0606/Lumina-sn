#!/usr/bin/env python3
"""교정 full native의 자체장 fixed-T 응답 최대4회. 앵커·온도 해제가 아니다."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np

from configure_cmfgen_stageA import set_control
from run_cmfgen_field_feedback import clone_material, verify_material, RESTART_OVERLAY
from run_cmfgen_fixed_field import scan, output_bytes
from run_cmfgen_lambda_probe import control, sha, stop_child
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables, parse_outgen, validate_rvtj
from lumina_native_heat_sources import vector


VADAT_CHANGES = dict(USE_FIXED_J='F', MAX_LAM='10.0D0', COMP_BA='T', N_FIX_BA='0',
                     STORE_BA='F', STORE_BA_INV='F', WR_PRT_INV='F')


def configure(vadat, in_its):
    expected = dict(FIX_T='T', FIX_T_AUTO='F', FIX_NE='F', DO_DDT='F', DO_NG='F',
                    TRT_NON_TE='F', INC_GRID='F', SCL_LN='T', INS_DEP='T',
                    GAMRAY_TRANS='LOCAL', MAX_LIN='3.0D0', SCALE_OPT='LOCAL',
                    LAM_VAL='400.0D0', NUM_LAM='2', WRITE_RATES='F', WRITE_JH='T')
    for key, value in expected.items():
        if control(vadat, key) != value:
            raise ValueError('봉인 제어 불일치: ' + key)
    for key in ('DO_LAM_AUTO', 'DO_T_AUTO'):
        if control(in_its, key) != 'F':
            raise ValueError('자동 전환 제어 불일치: ' + key)
    for key, value in VADAT_CHANGES.items():
        vadat = set_control(vadat, key, value)
    in_its = ('4 [NUM_ITS] ! 자체장 fixed-T 응답의 추가 반복 상한\n'
              'F [DO_LAM_IT] ! 첫 full BA 요청; native fallback은 별도로 유지\n'
              'F [DO_LAM_AUTO] ! 명시 LAMBDA 모드의 자동 해제만 금지\n'
              'F [DO_T_AUTO] ! 온도 자동 해제 금지\n')
    return vadat, in_its


def ba_policy_violation(text):
    if 'BA matrix **NOT** computed' in text:
        return 'requested_fresh_BA_not_computed'
    prefix = text.split('Current great iteration count', 1)[0]
    if re.search(r'read BAMATPNT\s+T', prefix):
        return 'requested_fresh_BA_initial_cache_read'
    return None


def state_summary(run, input_state, first=70):
    nd, nt = read_dimensions(run)
    reader = ScrtempReader(run / 'SCRTEMP', nd, nt)
    last_pointer = int((run / 'POINT1').read_text().splitlines()[1].split()[0])
    if not first <= last_pointer <= first + 4:
        raise ValueError('완료 record가 실행 범위를 벗어남')
    variables = read_variables(run, nt - 2)
    indices = {}
    for index, item in enumerate(variables):
        indices.setdefault(item.species, {}).setdefault(item.ion, []).append(index)
    original = reader.state(first)
    if not np.array_equal(original, input_state):
        raise ValueError('입력 record가 실행 중 변경됨')
    rows = []
    for iteration in range(first + 1, last_pointer + 1):
        current, previous = reader.state(iteration), reader.state(iteration - 1)
        if not np.isfinite(current).all() or np.any(current < 0):
            raise ValueError('비유한/음수 후보')
        if not np.array_equal(current[-1], input_state[-1]):
            raise ValueError('fixed-T 온도가 바뀜')
        stages = {}
        for species in ('Si', 'S'):
            groups = indices[species]
            total = sum(current[group].sum(axis=0) for group in groups.values())
            stages[species] = {ion: (current[group].sum(axis=0) / total).tolist()
                               for ion, group in groups.items()}
        support = previous[:-2] > 0
        rel = np.abs(current[:-2][support] / previous[:-2][support] - 1)
        rows.append(dict(iteration=iteration, population_positive_support_p95=float(np.quantile(rel, .95)),
            population_positive_support_max=float(rel.max()),
            zero_to_positive_count=int(np.count_nonzero((previous[:-2] == 0) & (current[:-2] > 0))),
            max_ne_relative_change=float(np.max(abs(current[-2] / previous[-2] - 1))),
            stages=stages))
    rvtj = None
    if last_pointer > first:
        try:
            rvtj = validate_rvtj(reader, run / 'RVTJ')
        except (ValueError, RuntimeError, OSError) as error:
            rvtj = dict(complete=False, evidence_error=str(error))
    return dict(last_completed_record=last_pointer, iterations=rows, final_RVTJ=rvtj,
        active_population_statistics='미판정: 위 통계는 양수 support 전체이며 v2 active 기준 대체 아님')


def check_resources(memory_gib, environment=None):
    env = os.environ if environment is None else environment
    if type(memory_gib) is not int or memory_gib not in (32, 128):
        raise ValueError('native 메모리는 명시한32/128GiB만 허용')
    if (env.get('OMP_NUM_THREADS') != '16' or env.get('SLURM_CPUS_PER_TASK') != '16' or
        env.get('SLURM_MEM_PER_NODE') != str(memory_gib * 1024) or not env.get('SLURM_JOB_ID')):
        raise ValueError(f'Slurm CPU16/OMP16/{memory_gib}GiB 할당과 명시 인수 일치 필요')
    return f'CPU16/OMP16/{memory_gib}GiB/2h/GPU0/retry0; child6600s'


def run(args):
    resources = check_resources(args.memory_gib)
    if args.out.exists():
        raise FileExistsError('기존 실행 보존')
    build = args.build.resolve()
    report_path = build / 'result.json'
    if not re.fullmatch(r'[0-9a-f]{64}', args.build_report_sha256) or sha(report_path) != args.build_report_sha256:
        raise ValueError('완료 full build 신원 불일치')
    manifest = json.loads(report_path.read_text())
    variant = manifest['variants']['instantaneous_decay']
    executable = Path(variant['executable'])
    if sha(executable) != variant['executable_sha256']:
        raise ValueError('교정 실행파일 변경')
    if (manifest['test_stubs_linked'] or manifest['state'] != 'COMPLETE_FULL_NATIVE_COMPARISON_BUILD' or
        manifest.get('restart_BA_initialization_corrected') is not True):
        raise ValueError('full native 빌드가 아님')
    input_state, identity, _ = verify_material(args.source70, args.original, 70)
    seed_hashes = {str(args.source70 / n): sha(args.source70 / n) for n in RESTART_OVERLAY}
    args.out.mkdir()
    material = args.out / 'material'
    inherited = clone_material(args.original, args.source70, material, args.native, False, 70, '1.0D6')
    vadat, in_its = configure((material / 'VADAT').read_text(), (material / 'IN_ITS').read_text())
    (material / 'VADAT').write_text(vadat)
    (material / 'IN_ITS').write_text(in_its)
    if (material / 'IT_SPECIFIER').exists():
        raise ValueError('재조립/온도 제어를 덮어쓰는 IT_SPECIFIER 금지')
    expected_controls = {n: sha(material / n) for n in ('VADAT', 'IN_ITS')}
    inputs = dict(scope=__doc__, source70=str(args.source70), input_identity=identity,
        source_sha256=seed_hashes, full_build=str(build), build_report_sha256=args.build_report_sha256,
        executable=variant, source_commit=args.source_commit,
        changed_vadat=VADAT_CHANGES, DO_LAM_IT='F: 최초 full, 이후 native fallback 유지',
        unchanged_fallback=dict(LAM_VAL=400, NUM_LAM=2), controls_sha256=expected_controls,
        inherited_stage=inherited, resources=resources, memory_gib=args.memory_gib,
        end_result='자체장 fixed-T 안정화 방향 판단; certified anchor 아님',
        J_history='native는 최종 J/H만 저장; 마지막3회 J 안정성은 이 실행만으로 인증 불가',
        final_goal='NOT_MET')
    (args.out / 'inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')
    if output_bytes(material) > 16 * 1024**3:
        raise ValueError('시작 복제본16GiB 초과')
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
    env = {k: v for k, v in os.environ.items() if not k.startswith('LUMINA_')}
    started = time.monotonic()
    deadline = started + 6600
    cursors, reason = {}, None
    with (material / 'seq_logs/cmfgen.stdout').open('xb') as out, \
         (material / 'seq_logs/cmfgen.stderr').open('xb') as err:
        child = subprocess.Popen(['srun', '--cpu-bind=cores', str(executable)], cwd=material,
            env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        while child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = 'child_budget_6600s'
                stop_child(child)
                break
            try:
                child.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                log = material / 'OUTGEN'
                reason = scan(material, cursors) or ba_policy_violation(log.read_text() if log.exists() else '')
                if reason:
                    stop_child(child)
                    break
    log = material / 'OUTGEN'
    reason = reason or scan(material, cursors) or ba_policy_violation(log.read_text() if log.exists() else '')
    result = dict(returncode=child.returncode, stop_reason=reason,
        elapsed_seconds=time.monotonic() - started, released_temperature=False,
        anchor_status='NOT_AVAILABLE', final_goal='NOT_MET')
    try:
        result.update(state_summary(material, input_state))
        text = (material / 'OUTGEN').read_text()
        raw = parse_outgen(material / 'OUTGEN')
        result['raw_logged_corrections'] = {i: vars(row) for i, row in raw.items()}
        result['BA_computed_count'] = len(re.findall(r'^\s*BA matrix computed\s*$', text, re.M))
        result['BA_not_computed_count'] = text.count('BA matrix **NOT** computed')
        result['lambda_count'] = text.count('LAMBDA iteration used')
        result['luminosity_lines'] = [line.strip() for line in text.splitlines() if 'Luminosity of star' in line]
        result['radioactive_heating'] = vector((material / 'RVTJ').read_text(),
            'Heating: radioactive decay (ergs/cm^3/s)', input_state.shape[1]).tolist()
        result['source_unchanged'] = all(sha(Path(p)) == h for p, h in seed_hashes.items())
        result['controls_unchanged'] = all(sha(material / n) == h for n, h in expected_controls.items())
        result['executable_unchanged'] = sha(executable) == variant['executable_sha256']
        result['four_responses_complete'] = (child.returncode == 0 and reason is None and
            result['last_completed_record'] == 74 and result['BA_computed_count'] >= 4 and
            result['BA_not_computed_count'] == 0 and result['source_unchanged'] and
            result['controls_unchanged'] and result['executable_unchanged'] and
            result['final_RVTJ'] is not None and result['final_RVTJ']['complete'])
    except (ValueError, OSError, RuntimeError) as error:
        result['evidence_error'] = str(error)
        result['four_responses_complete'] = False
    (args.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'iterations'}, ensure_ascii=False))
    return 0 if result['four_responses_complete'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'source70', 'native', 'build', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--build-report-sha256', required=True)
    p.add_argument('--memory-gib', type=int, choices=(32, 128), default=32,
                   help='기본32GiB 유지.128GiB는 별도 사용자 승인과 같은 Slurm 할당 필요')
    raise SystemExit(run(p.parse_args()))
