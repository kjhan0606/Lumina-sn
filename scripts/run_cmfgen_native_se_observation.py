#!/usr/bin/env python3
"""기존 66/J67→67 재현 후에만 67/J67의 갱신 전 원시 SE를 관찰한다.

두 native에 합산 2400초를 적용한다. RT/outer 추가는 없고, 두 번째 실행의
갱신 record는 관찰 부산물이다. 기존 next_outer_eligible 규칙은 바꾸지 않는다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import time

import numpy as np

from prepare_cmfgen_fixed_field_owner import point_records, record_bytes, snapshot
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables, parse_solutions, validate_rvtj
from run_cmfgen_field_feedback import clone_material, controls, verify_material
from run_cmfgen_fixed_field import output_bytes, scan
from run_cmfgen_lambda_probe import sha, stop_child


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_hashes(entries):
    for name, expected in entries.items():
        if sha(Path(name)) != expected:
            raise ValueError('봉인 입력 변경: '+name)


def verify_observation_source(source, original, parent_source):
    """완료67만 관찰에 허용한다. 새 장/outer 적격성을 부여하지 않는다."""
    root = source.parent
    inputs = json.loads((root/'seq_logs/feedback-inputs.json').read_text())
    runtime = json.loads((root/'seq_logs/feedback-runtime.json').read_text())
    stage = json.loads((source/'seq_logs/feedback-stage.json').read_text())
    if (source.name != 'material' or Path(inputs['original']) != original or
        Path(inputs['source']) != parent_source or not runtime.get('material_complete') or
        not runtime.get('field_complete') or not runtime.get('source_unchanged') or
        runtime.get('failure') or runtime['material']['returncode'] != 0 or
        runtime['material'].get('stop_reason') or runtime['material_record'] != 67 or
        runtime['radiation_produced_from_record'] != 66 or stage['input_state_record'] != 66 or
        stage['next_native_iteration'] != 67 or not stage['USE_FIXED_J'] or
        point_records(source) != dict(POINT1=67, POINT2=67)):
        raise ValueError('관찰용 완료67의 계보/상태 불일치')
    # next_outer_eligible=false를 허용하는 범위는 이 읽기 전용 관찰 검증뿐이다.
    hashes = dict(inputs['source_sha256'])
    hashes.update({str(source/n): h for n, h in runtime['material_evidence_sha256'].items()})
    hashes.update({str(root/'field'/n): h for n, h in runtime['field_evidence_sha256'].items()})
    hashes.update({str(source/n): h for n, h in stage['controls_sha256'].items()})
    for name, item in stage['linked_inputs'].items():
        hashes[str(source/name)] = item['sha256']
        hashes[item['target']] = item['sha256']
    check_hashes(hashes)
    expected_v, expected_i = controls((original/'VADAT').read_text(), (original/'IN_ITS').read_text(), True, '1.0D6')
    if (source/'VADAT').read_text() != expected_v or (source/'IN_ITS').read_text() != expected_i:
        raise ValueError('원래67의 제어 불일치')
    state, _, identity = snapshot(source, original, 67)
    nd, nt = read_dimensions(source)
    reader = ScrtempReader(source/'SCRTEMP', nd, nt)
    if reader.last_it != 67 or not validate_rvtj(reader, source/'RVTJ')['complete']:
        raise ValueError('완료67/RVTJ 연결 불완전')
    for name in ('BAMAT', 'BAMATPNT', 'CONT_FREQ', 'JEW', 'MODEL_SPEC', 'SN_HYDRO_DATA', 'LEVEL_SL_STEQ_LINKS'):
        if sha(source/name) != sha(original/name):
            raise ValueError('관찰 입력의 예상 밖 restart 변경: '+name)
    return state, identity, hashes, runtime


def read_witness(path):
    if path.stat().st_size > 32*1024**2:
        raise ValueError('관찰 출력 크기 불일치')
    data = path.read_bytes()
    if len(data) < 52 or data[:16].rstrip() != b'NATIVESEBOUNDV1':
        raise ValueError('원시 SE 형식 불일치')
    values = struct.unpack_from('<9i', data, 16)
    schema, bits, nr, nd, nt, ni, counter, parent, ns = values
    if (schema != 1 or bits != 64 or nr != nt-2 or min(nr, nd, ni, ns) < 1 or
        nr*nd > 1000000 or counter != parent+1):
        raise ValueError('원시 SE 차원/phase 불일치')
    offset = 52

    def array(dtype, shape):
        nonlocal offset
        count = int(np.prod(shape))
        nbytes = count*np.dtype(dtype).itemsize
        if offset+nbytes > len(data):
            raise ValueError('불완전 원시 SE 배열')
        result = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
        offset += nbytes
        return result.reshape(shape, order='F')

    fields = dict(pops=array('<f8', (nt, nd)))
    for key in ('raw', 'lo', 'hi', 'inc', 'out'):
        fields[key] = array('<f8', (nr, nd))
    fields['raw_count'] = array('<i4', (nr,))
    for key in ('raw_valid', 'coll_valid', 'coll_count'):
        fields[key] = array('<i4', (nr, nd))
    for key in ('nc', 'nc_lo', 'nc_hi', 'nc_target'):
        fields[key] = array('<f8', (ns, nd))
    fields['nc_valid'] = array('<i4', (ns, nd))
    fields['charge_raw'] = array('<f8', (nd,))
    if offset != len(data):
        raise ValueError('원시 SE 후행 데이터')
    return dict(counter=counter, parent=parent, nr=nr, nd=nd, nt=nt, nion=ni, nspecies=ns), fields


def sufficient_bounds(fields, tolerance=1e-8):
    """큰 상한은 실제 SE 실패가 아닌 UNPROVEN이다. 미량 행도 제외하지 않는다."""
    lo, hi, inc, out = (fields[k] for k in ('lo', 'hi', 'inc', 'out'))
    covered = ((fields['raw_count'][:, None] > 0) & (fields['raw_valid'] == 1) &
               (fields['coll_valid'] == 1) & (fields['coll_count'] > 0))
    valid = (covered & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(inc) &
             np.isfinite(out) & (lo <= hi) & (inc >= 0) & (out >= 0))
    denominator = np.maximum(inc, out)
    usable = valid & (denominator > 0)
    upper = np.full(lo.shape, np.nan)
    numerator = np.maximum(abs(lo), abs(hi))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        upper[usable] = np.nextafter(numerator[usable]/denominator[usable], np.inf)
    upper[usable & (numerator == 0)] = 0
    proven = usable & np.isfinite(upper) & (upper <= tolerance)
    reasons = np.full(lo.shape, 'INVALID_OR_MISSING_COLLISION_OR_RAW', dtype=object)
    reasons[valid & (denominator == 0)] = 'ZERO_COLLISION_LOWER_BOUND'
    reasons[usable] = 'SUFFICIENT_UPPER_NOT_SMALL_ENOUGH'
    reasons[proven] = 'PROVEN_EVALUATED_SE_SUFFICIENT_BOUND'
    return upper, proven, reasons


def assess(step, expected_state, record, *, radiation_produced_from_record=66):
    path = step/f'NATIVE_SE_BOUND_{record+1:06d}.bin'
    header, fields = read_witness(path)
    if (header['parent'] != record or fields['pops'].shape != expected_state.shape or
        not np.array_equal(fields['pops'], expected_state)):
        raise ValueError('원시 SE의 갱신 전 물질이 명시한 완료 record와 다름')
    upper, proven, reasons = sufficient_bounds(fields)
    variables = read_variables(step, header['nr'])
    with (step/'seq_logs/population-se-rows.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variable_1based', 'depth_1based', 'species', 'ion', 'superlevel',
                         'population', 'raw', 'raw_lo', 'raw_hi', 'collision_in_lower',
                         'collision_out_lower', 'scaled_upper', 'status', 'reason'])
        for row, var in enumerate(variables):
            for depth in range(header['nd']):
                writer.writerow([var.index, depth+1, var.species, var.ion, var.superlevel,
                    fields['pops'][row, depth], *[fields[k][row, depth] for k in ('raw', 'lo', 'hi', 'inc', 'out')],
                    float(upper[row, depth]) if np.isfinite(upper[row, depth]) else '',
                    'PROVEN' if proven[row, depth] else 'UNPROVEN', reasons[row, depth]])
    by_species = {}
    for species in sorted({v.species for v in variables}):
        ix = [i for i, v in enumerate(variables) if v.species == species]
        block = proven[ix]
        counts = {str(reason): int(np.count_nonzero(reasons[ix] == reason)) for reason in np.unique(reasons[ix])}
        by_species[species] = dict(rows=int(block.size), proven=int(block.sum()), reasons=counts)
    charge = sum(v.charge*fields['pops'][v.index-1] for v in variables)
    charge_residual = float(np.max(abs(charge-fields['pops'][-2])/fields['pops'][-2]))
    nc_den = abs(fields['nc_target'])
    nc_valid = ((fields['nc_valid'] == 1) & np.isfinite(fields['nc_lo']) &
                np.isfinite(fields['nc_hi']) & np.isfinite(nc_den) & (nc_den > 0))
    nc_bound = np.full(nc_den.shape, np.nan)
    nc_bound[nc_valid] = np.nextafter(np.maximum(abs(fields['nc_lo'][nc_valid]),
        abs(fields['nc_hi'][nc_valid]))/nc_den[nc_valid], np.inf)
    report = dict(phase=header, input_population_byte_equal=True,
        radiation_record=radiation_produced_from_record+1,
        radiation_produced_from_record=radiation_produced_from_record,
        tolerance=1e-8, population_rows=int(proven.size), proven=int(proven.sum()),
        unproven=int((~proven).sum()), by_species=by_species,
        all_population_rows_sufficiently_proven=bool(proven.all()),
        original_denominator='max(total incoming,total outgoing,n/t_ref); collision subset is a lower bound, t_ref not invented',
        large_upper_is_not_actual_SE_failure=True, max_population_charge_residual=charge_residual,
        nc_valid_rows=int(nc_valid.sum()), nc_rows=int(nc_valid.size),
        nc_evaluated_scaled_upper_max=float(np.max(nc_bound[nc_valid])) if nc_valid.any() else None,
        witness_sha256=sha(path), rows_sha256=sha(step/'seq_logs/population-se-rows.csv'),
        full_native_internal_arithmetic_certificate=False, RE_certified=False,
        released_T=False, physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')
    write_json(step/'seq_logs/native-se-assessment.json', report)
    return report


def reproduction(actual, reference):
    nd, nt = read_dimensions(reference)
    ah, ar, ap = record_bytes(actual/'SCRTEMP', 67, nd, nt)
    rh, rr, rp = record_bytes(reference/'SCRTEMP', 67, nd, nt)
    a_sol, r_sol = (parse_solutions(p/'STEQ_VALS', nt, nd) for p in (actual, reference))
    scale = lambda p: re.findall(r'The local minimum value of scale is:\s*(\S+)', (p/'OUTGEN').read_text())
    a_scale, r_scale = scale(actual), scale(reference)
    report = dict(completed_record_storage_byte_equal=ah == rh and ar == rr,
        population_ne_T_byte_equal=ap.tobytes() == rp.tobytes(),
        correction_array_equal=bool(a_sol and r_sol and np.array_equal(a_sol[-1], r_sol[-1])),
        printed_scale_equal=bool(a_scale and r_scale and a_scale[-1] == r_scale[-1]),
        observed_scale=float(a_scale[-1]) if a_scale else None,
        original_scale=float(r_scale[-1]) if r_scale else None,
        point_records_equal=point_records(actual) == point_records(reference) == dict(POINT1=67, POINT2=67),
        controls_byte_equal=all((actual/n).read_bytes() == (reference/n).read_bytes() for n in ('VADAT', 'IN_ITS')),
        tolerance_enlarged=False, final_goal='NOT_JUDGED')
    report['reproduced'] = all(report[k] for k in (
        'completed_record_storage_byte_equal', 'population_ne_T_byte_equal', 'correction_array_equal',
        'printed_scale_equal', 'point_records_equal', 'controls_byte_equal'))
    return report


def run_child(root, step, executable, deadline):
    started = time.monotonic()
    if deadline-started < 540:
        raise ValueError('공유 예산에 native 9분 미만; 추가 실행하지 않음')
    env = dict(os.environ, LUMINA_NATIVE_SE_WITNESS='1', LUMINA_FORMAL_PRIMARY='',
        LUMINA_FORMAL_PRIMARY_WHOLE='', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='', LUMINA_FORMAL_CAPTURE_MODE='')
    cursors, reason = {}, None
    with (step/'seq_logs/cmfgen.stdout').open('xb') as out, (step/'seq_logs/cmfgen.stderr').open('xb') as err:
        proc = subprocess.Popen(['srun', '--cpu-bind=cores', str(executable)], cwd=step, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        while proc.poll() is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = 'shared_child_budget_2400s'
                stop_child(proc)
                break
            try:
                proc.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                reason = scan(step, cursors)
                if sum(output_bytes(root/n) for n in ('reproduction', 'evaluation') if (root/n).exists()) > 16*1024**3:
                    reason = 'bundle_root_16GiB'
                if reason:
                    stop_child(proc)
                    break
    return dict(returncode=proc.returncode, stop_reason=reason or scan(step, cursors),
                elapsed_seconds=time.monotonic()-started)


def run(original, source66, source67, root, native, build):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부에서만 실행')
    if root.exists():
        raise ValueError('기존 관찰 계산 보존')
    started, deadline = time.monotonic(), time.monotonic()+2400
    state66, identity66, parent = verify_material(source66, original, 66)
    state67, identity67, hashes, runtime67 = verify_observation_source(source67, original, source66)
    bm = json.loads((build/'se-witness-build.json').read_text())
    if Path(bm['native']) != native or bm['activation'] != 'LUMINA_NATIVE_SE_WITNESS=1; 기본 비활성':
        raise ValueError('관찰 build 정체성/활성 조건 불일치')
    hashes.update(bm['input_sha256'])
    hashes.update(bm['output_sha256'])
    hashes[str(build/'se-witness-build.json')] = sha(build/'se-witness-build.json')
    hashes[str(source67/'STEQ_VALS')] = sha(source67/'STEQ_VALS')
    hashes[str(source67/'OUTGEN')] = sha(source67/'OUTGEN')
    check_hashes(hashes)
    if sha(native/'exe/cmfgen_dev.exe') != parent['binary']['sha256']:
        raise ValueError('원래 native 변경')
    root.mkdir()
    (root/'seq_logs').mkdir()
    write_json(root/'seq_logs/observation-inputs.json', dict(scope=__doc__, original=str(original),
        source66=str(source66), source67=str(source67), source66_identity=identity66, source67_identity=identity67,
        source67_next_outer_eligible=runtime67['next_outer_eligible'], input_sha256=hashes,
        build=str(build), resources='CPU16/OMP16/32GiB/GPU0 parent2700s shared2400s retry0 no pin',
        RT_executions=0, outer_executions=0, maximum_native_executions=2,
        byproduct_record68_adopted_as_outer_checkpoint=False, final_goal='NOT_JUDGED'))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    report = dict(noninterference='NOT_YET_REPRODUCED', evaluation_complete=False,
        native_executions=0, RT_executions=0, outer_executions=0,
        byproduct_record68_adopted_as_outer_checkpoint=False, final_goal='NOT_JUDGED')
    status = 1
    try:
        for name, source, record, state in (('reproduction', source66, 66, state66), ('evaluation', source67, 67, state67)):
            step = root/name
            clone_material(original, source, step, native, True, record, '1.0D6')
            for item in ('EDDFACTOR', 'EDDFACTOR_INFO'):
                shutil.copy2(source67.parent/'field'/item, step/item)
            fixed = {item: sha(step/item) for item in ('EDDFACTOR', 'EDDFACTOR_INFO')}
            write_json(step/'seq_logs/observation-stage.json', dict(input_record=record,
                source=str(source), purpose=name, radiation_produced_from_record=66,
                native_binary=str(build/'cmfgen_se_witness.exe'), fixed_J_sha256=fixed,
                observer_enabled=True, new_outer_permission=False, produced_record_is_observation_byproduct=True))
            report['native_executions'] += 1
            report[name] = run_child(root, step, build/'cmfgen_se_witness.exe', deadline)
            if report[name]['returncode'] != 0 or report[name]['stop_reason']:
                raise ValueError(name+': native 실행 실패')
            if any(sha(step/n) != h for n, h in fixed.items()):
                raise ValueError(name+': 고정 J 변경')
            nd, nt = read_dimensions(step)
            if (ScrtempReader(step/'SCRTEMP', nd, nt).last_it != record+1 or
                point_records(step) != dict(POINT1=record+1, POINT2=record+1)):
                raise ValueError(name+': 완료 포인터 불일치')
            if name == 'reproduction':
                report['reproduction_comparison'] = reproduction(step, source67)
                write_json(step/'seq_logs/reproduction.json', report['reproduction_comparison'])
                if not report['reproduction_comparison']['reproduced']:
                    raise ValueError('비간섭 재현 실패; 두 번째 native 실행 금지')
                report['noninterference'] = 'REPRODUCED'
            report[name+'_assessment'] = assess(step, state, record)
            write_json(root/'seq_logs'/('after-'+name+'.json'), report)
        report['evaluation_complete'] = True
        status = 0
    except (ValueError, OSError) as error:
        report['failure'] = str(error)
    finally:
        try:
            check_hashes(hashes)
            report['source_unchanged'] = True
        except (ValueError, OSError) as error:
            report['source_unchanged'] = False
            report['preservation_failure'] = str(error)
            status = 1
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(root/'seq_logs/observation-runtime.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'source66', 'source67', 'run', 'native', 'build'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(*(getattr(args, key).resolve() for key in ('original', 'source66', 'source67', 'run', 'native', 'build'))))
