#!/usr/bin/env python3
"""같은63/J64의 native64B: MAX_LAM만 바꿔 trace 성장 제한의 억제를 실증한다.

기존64A와 완주장은 보존한다. 새로운 수송·추가 반복·T-release는 하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import time

import numpy as np

from configure_cmfgen_stageA import set_control
from ptf_gated_metrics import parse_solutions, read_dimensions, read_variables, ScrtempReader
from run_cmfgen_field_feedback import clone_material, verify_material, child, total_bytes
from prepare_cmfgen_fixed_field_owner import point_records
from run_cmfgen_lambda_probe import sha


def density_interval_error(pre, printed_correction):
    """ES12.4의 반올림 폭과 density 곱의 IEEE 오차만 포함한다."""
    x = np.abs(printed_correction)
    radius = np.zeros_like(x)
    positive = x > 0
    radius[positive] = .5*np.power(10., np.floor(np.log10(x[positive]))-4)
    eps = np.finfo(float).eps
    return abs(pre)*radius + 8*eps*(abs(pre)+abs(pre*printed_correction))


def full_step_check(pre, after, sol, reference_sol):
    if not (pre.shape == after.shape == sol.shape == reference_sol.shape):
        raise ValueError('A/B 물질 차원 불일치')
    proposed = pre*(1-sol)
    reference = pre*(1-reference_sol)
    bound = density_interval_error(pre, sol)
    same = np.all(abs(proposed-reference) <= bound+density_interval_error(pre, reference_sol))
    applied = np.all(abs(after-proposed) <= bound)
    return dict(raw_candidate_same_within_print_precision=bool(same),
        raw_corrections_printed_equal=bool(np.array_equal(sol, reference_sol)),
        actual_full_step_within_print_precision=bool(applied),
        tolerance='ES12.4 half decimal ULP plus 8*binary64-eps density-operation bound; no population floor')


def run(reference, root, native):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부만 허용')
    if root.exists():
        raise ValueError('기존 A/B 실행 보존')
    started = time.monotonic()
    old_inputs = json.loads((reference/'seq_logs/feedback-inputs.json').read_text())
    old_runtime = json.loads((reference/'seq_logs/feedback-runtime.json').read_text())
    if not (old_runtime['field_complete'] and old_runtime['material_complete'] and
            old_runtime['material_record'] == 64 and old_runtime['radiation_produced_from_record'] == 63):
        raise ValueError('완료 대조64A/J64가 아님')
    original, source = Path(old_inputs['original']), Path(old_inputs['source'])
    pre, identity, source_manifest = verify_material(source, original)
    if sha(native/'exe/cmfgen_dev.exe') != source_manifest['binary']['sha256']:
        raise ValueError('원래 native 변경')
    for name, expected in old_runtime['field_evidence_sha256'].items():
        if sha(reference/'field'/name) != expected:
            raise ValueError('대조 완주장 변경')
    source_sha = {p:sha(Path(p)) for p in old_inputs['source_sha256']}
    if source_sha != old_inputs['source_sha256']:
        raise ValueError('완료63 입력 변경')
    root.mkdir();(root/'seq_logs').mkdir()
    (root/'field').symlink_to(reference/'field', target_is_directory=True)
    inputs = dict(old_inputs, scope=__doc__, ab_reference=str(reference), ab_case='64B',
        source_state=identity, numerical_change={'MAX_LAM':'2.0D0 -> 1.0D6'},
        transport_recomputed=False, runner_sha256=sha(Path(__file__)),
        resources='CPU16/OMP16/32GiB/GPU0 parent2700s shared2400s one native update retry0')
    (root/'seq_logs/feedback-inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
    material=root/'material'
    stage=clone_material(original,source,material,native,True)
    for name in ('EDDFACTOR','EDDFACTOR_INFO'):
        shutil.copy2(reference/'field'/name,material/name)
    v=set_control((material/'VADAT').read_text(),'MAX_LAM','1.0D6')
    if (set_control(v,'MAX_LAM','2.0D0') != (reference/'material/VADAT').read_text() or
        (material/'IN_ITS').read_bytes() != (reference/'material/IN_ITS').read_bytes()):
        raise ValueError('MAX_LAM 이외 A/B 제어 차이')
    (material/'VADAT').write_text(v)
    stage['controls_sha256']['VADAT']=sha(material/'VADAT')
    stage['numerical_controls']['MAX_LAM']=1e6
    stage['ab_case']='64B'
    stage['restart_sha256']={n:sha(material/n) for n in stage['restart_sha256']}
    if sha(material/'SCRTEMP') != sha(reference/'field/SCRTEMP'):
        raise ValueError('같은63 restart가 아님')
    (material/'seq_logs/feedback-stage.json').write_text(json.dumps(stage,ensure_ascii=False,indent=2)+'\n')
    fixed={n:sha(material/n) for n in ('EDDFACTOR','EDDFACTOR_INFO')}
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    env=dict(os.environ,LUMINA_FORMAL_PRIMARY='',LUMINA_FORMAL_PRIMARY_WHOLE='',
        LUMINA_FORMAL_CELL_BUDGET='',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='',LUMINA_FORMAL_CAPTURE_MODE='')
    report=dict(field_complete=True,field_reused_from=str(reference/'field'),
        field_evidence_sha256=old_runtime['field_evidence_sha256'],
        material_complete=False,material_record=64,radiation_produced_from_record=63,
        ab_case='64B',joint_convergence=False,released_T=False,final_goal='NOT_JUDGED')
    status=1
    try:
        report['material']=child(root,material,native/'exe/cmfgen_dev.exe',env,started+2400)
        if report['material']['returncode'] != 0 or report['material']['stop_reason']:
            raise ValueError('native64B 실행 실패')
        nd,nt=read_dimensions(material);reader=ScrtempReader(material/'SCRTEMP',nd,nt)
        if reader.last_it != 64 or point_records(material) != dict(POINT1=64,POINT2=64):
            raise ValueError('완료64B 없음')
        after=reader.state(64)
        if (not np.isfinite(after).all() or np.any(after[:-2]<0) or
            np.any(after[-2:]<=0) or np.any(after[pre>0]<=0) or not np.array_equal(after[-1],pre[-1])):
            raise ValueError('양수 support/고정T/유한 상태 위반')
        if any(sha(material/n) != h for n,h in fixed.items()):
            raise ValueError('물질 갱신 중 J64 변경')
        sol=parse_solutions(material/'STEQ_VALS',nt,nd)[-1]
        ref=parse_solutions(reference/'material/STEQ_VALS',nt,nd)[-1]
        report['full_step_check']=full_step_check(pre,after,sol,ref)
        match=re.findall(r'The local minimum value of scale is:\s*(\S+)',(material/'OUTGEN').read_text())
        report['printed_minimum_scale']=float(match[-1]) if match else None
        if (not report['full_step_check']['raw_candidate_same_within_print_precision'] or
            not report['full_step_check']['actual_full_step_within_print_precision'] or
            report['printed_minimum_scale'] != 1.0):
            raise ValueError('예측한 비활성 제한/전체 step 실증 실패')
        variables=read_variables(material,nt-2)
        charge=sum(v.charge*after[v.index-1] for v in variables)
        report['max_charge_residual']=float(np.max(abs(charge-after[-2])/after[-2]))
        if report['max_charge_residual']>1e-10:
            raise ValueError('전하보존 기준 위반')
        report['element_total_change_over63']={}
        for species in sorted({v.species for v in variables}):
            ix=[v.index-1 for v in variables if v.species==species]
            report['element_total_change_over63'][species]=float(np.max(abs(after[ix].sum(axis=0)/pre[ix].sum(axis=0)-1)))
        if max(report['element_total_change_over63'].values())>1e-10:
            raise ValueError('봉인63의 원소총량 보존 위반; 원래 NC식 잔차 인증과 별도')
        report.update(material_complete=True,max_ne_relative_step=float(np.max(abs(after[-2]/pre[-2]-1))),
            native_full_SE_residual='NOT_CERTIFIED; full step is not convergence',
            original_element_equation_residual='NOT_REPLACED_BY_CHANGE_OVER63',
            material_evidence_sha256={n:sha(material/n) for n in ('SCRTEMP','POINT1','POINT2','RVTJ')})
        status=0
    except (ValueError,OSError) as error:
        report['failure']=str(error)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,root_bytes=total_bytes(root),
            source_unchanged=all(sha(Path(p))==h for p,h in source_sha.items()),
            field_unchanged=all(sha(reference/'field'/n)==h for n,h in old_runtime['field_evidence_sha256'].items()))
        if not report['source_unchanged'] or not report['field_unchanged'] or report['root_bytes']>16*1024**3:
            status=1
        (root/'seq_logs/feedback-runtime.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False));return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','run','native'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.reference.resolve(),a.run.resolve(),a.native.resolve()))
