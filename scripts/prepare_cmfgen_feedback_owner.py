#!/usr/bin/env python3
"""완료 물질의 새 장에서 갱신 전/후 실제 C owner 입력을 만든다.

완료 장을 직접 읽으며 예전 J60/행렬을 재사용하지 않는다. 외부장 조건부
비교 입력이고, 장의 생산 물질과 갱신 후 물질의 세대를 구분한다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_cmfgen_fixed_field_owner import snapshot, record_bytes, replace_material
from prepare_cmfgen_owner_response import write_input
from run_cmfgen_formal_primary import coverage, fields, validate_edd
from run_cmfgen_lambda_probe import sha


def choose_step(root, record, runtime=None, source_record=63):
    if source_record < 63:
        raise ValueError('등록된 완료 물질 이전 세대')
    if record == source_record:
        return root/'field'
    if record != source_record+1:
        raise ValueError('이번 피드백의 갱신 전/후 물질만 허용')
    if (not runtime or not runtime.get('field_complete') or
        not runtime.get('material_complete') or runtime.get('material_record') != source_record+1 or
        runtime.get('radiation_produced_from_record') != source_record):
        raise ValueError('갱신 후 물질 미완료')
    return root/'material'


def material_arrays(state, previous, original_profiles, indexes, columns):
    if state.shape != previous.shape or original_profiles.shape != (state.shape[1], 5):
        raise ValueError('물질/원래 profile 차원 오류')
    if (len(indexes) != len(columns) or len(set(map(tuple, columns))) != len(columns) or
        len(set(map(int, indexes))) != len(indexes)):
        raise ValueError('원래 인구 열 대응 오류')
    if not np.array_equal(state[-1], previous[-1]):
        raise ValueError('고정T 범위 밖')
    profiles = original_profiles.copy()
    profiles[:, 1] = state[-1]*1e4
    profiles[:, 2] = state[-2]
    return profiles, state[indexes].T.copy()


def radiation_arrays(array):
    if not np.all(np.diff(array['freq']) < 0) or np.any(array['freq'] <= 0):
        raise ValueError('실제 장의 주파수 순서 오류')
    order = np.argsort(array['freq'])
    # 실제 새 angular J만 선택한다. 과거 owner_input의 J block은 읽지 않는다.
    return array['freq'][order]*1e15, np.asarray(array['field'][order, 0, :])


def same_native_inputs(stage, expected):
    if set(stage) != set(expected):
        raise ValueError('원래 native 입력 역할 집합 변경')
    for name, item in stage.items():
        if item['sha256'] != expected[name]['sha256'] or sha(Path(item['target'])) != item['sha256']:
            raise ValueError('실제 native 원자료 변경: '+name)


def same_field_material_input(pre_root, root, source_record, state, parent, indexes, columns, profiles, output, field_hash, epoch):
    """동일한 새 장의 block만 복사한다. ne/인구와 C 율·행렬을 재사용하지 않는다."""
    path=pre_root/'fixed-field-owner-inputs.json';m=json.loads(path.read_text())
    if (m['feedback_root']!=str(root) or m['material_snapshot']['record']!=source_record or
        m['radiation_record']!=source_record+1 or m['radiation_material_record']!=source_record):
        raise ValueError('다른 물질/복사장의 입력 block 재사용 금지')
    for name,expected in m['output_sha256'].items():
        if sha(pre_root/name)!=expected:raise ValueError('같은 장 입력 증거 변경')
    actual=[h for p,h in m['input_sha256'].items() if Path(p).name=='FORMAL_PRIMARY_FIELDS.bin']
    if actual!=[field_hash]:raise ValueError('현재 완주장과 다른 복사장')
    data=np.load(pre_root/'completed-state.npz')
    if (not np.array_equal(data['state'],parent) or
        not np.array_equal(data['columns'],columns) or
        not np.array_equal(data['owner_variable_indexes'],indexes) or
        not np.array_equal(data['profiles'][:,[0,3,4]],profiles[:,[0,3,4]])):
        raise ValueError('입력 block의 부모 인구/원자대응/질량 격자 변경')
    copied_columns,copied_profiles,identity=replace_material(pre_root/'owner_input.bin',output,state,parent,indexes)
    if (not np.array_equal(copied_columns,columns) or not np.array_equal(copied_profiles,profiles) or
        identity['epoch_seconds']!=epoch):
        raise ValueError('새 물질/T/ne 전달 오류')
    return dict(identity,same_current_field_blocks_reused=True,rates_and_matrices_reused=False,
                source_manifest=str(path))


def prepare(root, reference63, record, out, same_field_input=None):
    if out.exists():
        raise ValueError('기존 피드백 비교 입력 보존')
    inputs_path = root/'seq_logs/feedback-inputs.json'
    inputs = json.loads(inputs_path.read_text())
    source_record = inputs['source_state']['record']
    next_record = source_record+1
    reference_path = reference63/'fixed-field-owner-inputs.json'
    baseline = json.loads(reference_path.read_text())
    if (baseline['material_snapshot']['record'] != source_record or
        inputs['source'] != baseline['run'] or
        inputs['source_state']['record_storage_sha256'] != baseline['material_snapshot']['record_storage_sha256']):
        raise ValueError('서로 다른 완료 부모 물질 계보')
    for name, expected in baseline['output_sha256'].items():
        if sha(reference63/name) != expected:
            raise ValueError('원래 완료63 증거 변경: '+name)
    pair_path = Path(baseline['base_input'])/'pair-inputs.json'
    pair = json.loads(pair_path.read_text())
    original = Path(inputs['original'])
    if str(original) != pair['seed']:
        raise ValueError('원래 물리 seed 불일치')
    for name, expected in baseline['unchanged_atomic_view_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('봉인 원자 view 변경: '+name)
    runtime_path = root/'seq_logs/feedback-runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    material = choose_step(root, record, runtime, source_record)
    stage_path = material/'seq_logs/feedback-stage.json'
    stage = json.loads(stage_path.read_text())
    if (stage['material_source'] != inputs['source'] or stage['input_state_record'] != source_record or
        stage['next_native_iteration'] != next_record or stage['USE_FIXED_J'] != (record == next_record)):
        raise ValueError('다른 교대 단계')
    for name, expected in stage['controls_sha256'].items():
        if sha(material/name) != expected:
            raise ValueError('단계 제어 변경')
    parent_manifest = 'fixed-field-inputs.json' if source_record == 63 else 'feedback-stage.json'
    original_native = json.loads((Path(inputs['source'])/'seq_logs'/parent_manifest).read_text())
    same_native_inputs(stage['linked_inputs'], original_native['linked_inputs'])
    for name in ('MODEL_SPEC', 'LEVEL_SL_STEQ_LINKS'):
        if sha(material/name) != pair['input_sha256'][str(original/name)]:
            raise ValueError('물질의 원자/준위 대응 변경')
    state, previous, identity = snapshot(material, original, record)
    _, saved_parent, _ = record_bytes(material/'SCRTEMP', source_record, state.shape[1], state.shape[0])
    if hashlib.sha256(saved_parent).hexdigest() != baseline['material_snapshot']['record_storage_sha256']:
        raise ValueError('교대 실행의 초기 부모 물질 변경')
    data = np.load(reference63/'completed-state.npz')
    indexes, columns = data['owner_variable_indexes'], data['columns']
    if not np.array_equal(previous, data['previous59']):
        raise ValueError('서로 다른 이전59 물질')
    profiles, populations = material_arrays(state, previous, data['profiles'], indexes, columns)
    field = root/'field'
    if record == next_record:
        field_stage = json.loads((field/'seq_logs/feedback-stage.json').read_text())
        same_native_inputs(field_stage['linked_inputs'], original_native['linked_inputs'])
    complete = field/'FORMAL_PRIMARY_COMPLETE.txt'
    if complete.read_text().splitlines()[0].split()[:3] != [str(next_record), '197291', '197291']:
        raise ValueError('새 장 미완료/다른 세대')
    cov = coverage(field/'FORMAL_PRIMARY_SWEEP.csv')
    geometry, array = fields(field/'FORMAL_PRIMARY_FIELDS.bin')
    if cov['count'] != 197291 or array['field'].shape != (197291, 8, 90):
        raise ValueError('새 장 부분 coverage')
    validate_edd(field, array)
    if sha(field/'SCRTEMP') != inputs['source_sha256'][str(Path(inputs['source'])/'SCRTEMP')]:
        raise ValueError('장 생산 중 부모 물질 변경')
    out.mkdir()
    epoch = pair['epoch_seconds']
    field_identity=dict(native_frequencies=len(array),actual_new_angular_J=True,
        legacy_J60_reused=False,epoch_seconds=epoch)
    if same_field_input is not None:
        if record!=next_record:raise ValueError('같은 장 block 복사는 갱신 후 물질에만 허용')
        field_identity.update(same_field_material_input(same_field_input,root,source_record,state,
            data['state'],indexes,columns,profiles,out/'owner_input.bin',sha(field/'FORMAL_PRIMARY_FIELDS.bin'),epoch))
    else:
        nu, radiation = radiation_arrays(array)
        if radiation.shape != (len(nu), len(profiles)):
            raise ValueError('새 장/물질 차원 불일치')
        write_input(out/'owner_input.bin', nu, profiles, columns, populations, radiation, epoch)
    np.savez(out/'completed-state.npz', state=state, previous59=previous, previous_parent=data['state'],
        owner_variable_indexes=indexes, profiles=profiles, columns=columns)
    with (out/'reference.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['depth_1based', 'Z', 'stage', 'population_cm3', 'fraction', 'Te_K', 'ne_cm3'])
        for d in range(len(profiles)):
            for Z in (14, 16):
                chosen = columns[:, 0] == Z
                total = populations[d, chosen].sum()
                for charge in range(1, 6):
                    density = populations[d, chosen & (columns[:, 1] == charge)].sum()
                    writer.writerow([d+1, Z, charge+1, density, density/total, profiles[d, 1], profiles[d, 2]])
    # 재현용 생산자 사본이며 새로운 판정 게이트를 만들지 않는다.
    producer = Path(__file__).resolve()
    shutil.copy2(producer, out/producer.name)
    from ptf_gated_metrics import read_variables
    variables = read_variables(material, state.shape[0]-2)
    expected_ne = sum(v.charge*state[v.index-1] for v in variables)
    totals = {}
    for species in sorted({v.species for v in variables}):
        selected = [v.index-1 for v in variables if v.species == species]
        totals[species] = float(np.max(abs(state[selected].sum(axis=0)/previous[selected].sum(axis=0)-1)))
    evidence = [inputs_path, reference_path, pair_path, stage_path, complete,
                field/'FORMAL_PRIMARY_FIELDS.bin', field/'EDDFACTOR', producer,
                Path(write_input.__code__.co_filename).resolve(),
                Path(snapshot.__code__.co_filename).resolve()]
    if record == next_record:
        evidence.append(runtime_path)
    if same_field_input is not None:
        evidence.extend([same_field_input/'fixed-field-owner-inputs.json',same_field_input/'owner_input.bin'])
    report = dict(scope=__doc__, material_snapshot=identity, run=str(material),
        feedback_root=str(root), base_input=baseline['base_input'], deck=baseline['deck'],
        material_parent_record=source_record,
        radiation_record=next_record, radiation_material_record=source_record,
        field_phase_warning=f'J{next_record}는 물질{source_record}의 장; 갱신 후 공동수렴은 미입증',
        field_identity=field_identity,
        max_ne_change_over59=float(np.max(abs(state[-2]/previous[-2]-1))),
        max_native_charge_residual=float(np.max(abs(expected_ne-state[-2])/state[-2])),
        element_total_max_relative_change=totals,
        unchanged_atomic_view_sha256=baseline['unchanged_atomic_view_sha256'],
        input_sha256={str(p): sha(p) for p in evidence},
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        native_full_SE_residual='NOT_EXTRACTED_NOT_REPLACED_BY_CORRECTION',
        physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')
    (out/'fixed-field-owner-inputs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(out), material_record=record, radiation_record=next_record,
        radiation_material_record=source_record, final_goal='NOT_JUDGED'), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-parent', '--reference63', dest='reference63', type=Path, required=True)
    parser.add_argument('--record', type=int, required=True)
    parser.add_argument('--same-field-input', type=Path)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.reference63.resolve(), args.record, args.out.resolve(),
        args.same_field_input.resolve() if args.same_field_input else None)
