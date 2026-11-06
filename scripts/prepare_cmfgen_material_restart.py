#!/usr/bin/env python3
"""검증된 수용 물질 입력을 CMFGEN의 새 SCRTEMP 초기값으로 직렬화한다.

기존 prepare_cmfgen_fixedj_heat.prepare_state의 원자 주소/단위 변환을 재사용한다.
원 run이나 과거 반복 기록은 수정하지 않는다. 새 장·계수·BA·RVTJ를 가장하거나
계산을 제출하지 않으며, 생성물은 독립 계산을 위한 초기값일 뿐 수렴해가 아니다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from ptf_gated_metrics import REC_BYTES, DOUBLES_PER_REC, SCRTEMP_HEADER_RECS
from prepare_cmfgen_fixed_field_owner import point_records
from run_cmfgen_lambda_probe import sha


def restart_bytes(header, radius, state):
    """고정 기하의 첫 상태만 만든다. 물리 배열을 보간·정규화하지 않는다."""
    radius, state = np.asarray(radius), np.asarray(state)
    if (radius.ndim != 1 or state.ndim != 2 or state.shape[1] != len(radius)
            or state.shape[0] < 3 or len(radius) < 2
            or 3*len(radius) > DOUBLES_PER_REC
            or len(header) != SCRTEMP_HEADER_RECS*REC_BYTES):
        raise ValueError('지원하는 고정 기하 restart 차원/헤더가 아님')
    geometry = np.frombuffer(header, dtype='<f8', count=3*len(radius))
    if (not np.isfinite(geometry).all() or not np.isfinite(state).all()
            or np.any(radius <= 0) or np.any(np.diff(radius) >= 0)
            or not np.array_equal(geometry[:len(radius)], radius)
            or np.any(state[:-2] < 0) or np.any(state[-2:] <= 0)):
        raise ValueError('restart 기하/반경/인구/T/ne 불일치')
    payload = np.asarray(state, dtype='<f8').tobytes(order='F')
    padded = payload + b'\0'*((-len(payload)) % REC_BYTES)
    # 새 초기값을 native state1로 읽는다. 이전 run의 성공 반복 수는 물려받지 않는다.
    pointer = '29-Feb-2004                                                  !Format date\n1 1 1 -1000 F\n'
    return header+padded, pointer


def prepare(state_input, state_input_sha256, out):
    state_input, out = Path(state_input).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('기존 restart 초기값 보존')
    if sha(state_input) != state_input_sha256:
        raise ValueError('봉인 물질 입력 manifest 변경')
    report = json.loads(state_input.read_text())
    if (report.get('schema') != 'lumina-native-heat-state-input-v1'
            or report.get('SL_and_ne_exact') is not True
            or report.get('temperature_conversion') != 'K -> 1e4 K; no equilibration'
            or report.get('actual_native_evaluation') is not False):
        raise ValueError('검증된 수용 물질 직렬화 입력이 아님')
    native = Path(report['native_run']).resolve()
    input_hashes = dict(report['input_sha256'])
    protected = {native, state_input.parent, *(Path(p).resolve().parent for p in input_hashes)}
    if any(out == p or p in out.parents for p in protected):
        raise ValueError('원 native/수용 입력 트리 안에 출력 금지')
    payload_path = state_input.parent/'NATIVE_HEAT_INPUT.bin'
    input_hashes[str(payload_path)] = report['output_sha256'][payload_path.name]
    input_hashes[str(state_input)] = state_input_sha256
    if any(sha(Path(p)) != h for p, h in input_hashes.items()):
        raise ValueError('봉인 수용 물질/원자모형 입력 변경')
    nd, nt = report['points'], report['variables']
    if (type(nd) is not int or type(nt) is not int or nd < 2 or nt < 3
            or 3*nd > DOUBLES_PER_REC or 32+8*nd*(nt+1) > 256*1024**2):
        raise ValueError('지원하는 고정 기하/물질 차원 초과')
    raw = payload_path.read_bytes()
    if (len(raw) != 32+8*nd*(nt+1) or raw[:16] != b'NATIVEHEATINPV1 '
            or struct.unpack_from('<4i', raw, 16) != (1, 64, nt, nd)):
        raise ValueError('수용 native 물질 입력 헤더/길이 오류')
    radius = np.frombuffer(raw, dtype='<f8', count=nd, offset=32)
    state = np.frombuffer(raw, dtype='<f8', offset=32+8*nd).reshape((nt, nd), order='F')
    points = point_records(native)
    if points['POINT1'] != points['POINT2']:
        raise ValueError('원 native 완료 포인터가 서로 다름')
    for name in ('POINT1', 'POINT2'):
        path = native/name
        if path.read_text().splitlines()[1].split()[4].upper() != 'F':
            raise ValueError('반복별 이동 기하는 지원하지 않음')
        input_hashes[str(path)] = sha(path)
    with (native/'SCRTEMP').open('rb') as stream:
        header = stream.read(SCRTEMP_HEADER_RECS*REC_BYTES)
    geometry_sha = hashlib.sha256(header).hexdigest()
    if not np.array_equal(radius, np.frombuffer(header, dtype='<f8', count=nd)):
        raise ValueError('수용 물질과 native 원 반경이 다름')
    blob, pointer = restart_bytes(header, radius, state)
    if any(sha(Path(p)) != h for p, h in input_hashes.items()):
        raise ValueError('restart 준비 중 원 입력 변경')
    with (native/'SCRTEMP').open('rb') as stream:
        if stream.read(SCRTEMP_HEADER_RECS*REC_BYTES) != header:
            raise ValueError('restart 준비 중 native 기하 변경')
    out.mkdir()
    with (out/'SCRTEMP').open('xb') as stream:
        stream.write(blob)
    for name in ('POINT1', 'POINT2'):
        with (out/name).open('x') as stream:
            stream.write(pointer)
    result = dict(schema='lumina-cmfgen-material-restart-v1',
        status='INITIAL_STATE_ONLY_NOT_RUN_READY', points=nd, variables=nt,
        input_sha256=input_hashes,
        geometry_header=dict(path=str(native/'SCRTEMP'), bytes=len(header), sha256=geometry_sha),
        accepted_result_sha256=report['accepted_result_sha256'],
        accepted_response_sha256=report['accepted_response_sha256'],
        initial_native_record=1, inherited_native_iterations=0, physical_iterations=0,
        population_T_ne_unchanged=True, geometry_header_unchanged=True,
        copied_radiation=False, copied_opacities=False, copied_jacobian=False,
        fabricated_RVTJ=False, released_temperature=False, computation_started=False,
        next='동일 물리 계약의 독립 native 실행을 별도 구성; 장/계수/BA 재계산 및 수렴 검증 필요',
        final_goal='NOT_MET', source_script_sha256=sha(Path(__file__)),
        output_sha256={n: sha(out/n) for n in ('SCRTEMP', 'POINT1', 'POINT2')})
    with (out/'restart-seed.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-input', type=Path, required=True)
    parser.add_argument('--state-input-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.state_input, args.state_input_sha256, args.out),
                     ensure_ascii=False, indent=2))
