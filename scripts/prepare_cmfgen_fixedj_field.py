#!/usr/bin/env python3
"""완료된 현재 Lumina J를 native fixed-J EDD 입력으로 옮긴다. 새 수송은 없다.

현재 장의 모든 주파수/90점 값을 보존한다. native 자체의 적분 격자를 바꾸거나
두 프로그램의 quadrature/열행 일치를 인증하지 않는다. H/K/관측광도 입력이 아니다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from prepare_cmfgen_mom_capture import sha


def write_field(target, header, frequency, values, *, block=16384):
    """native 14개 header record 뒤에 내림차순 (J_1..J_ND, nu/1e15)을 쓴다."""
    nu, j = np.asarray(frequency), values
    if (nu.ndim != 1 or len(nu) < 2 or j.ndim != 2 or j.shape[1] != len(nu)
            or j.dtype != np.dtype('f8') or nu.dtype != np.dtype('f8')
            or not np.isfinite(nu).all() or np.any(nu <= 0) or np.any(np.diff(nu) <= 0)
            or block < 1):
        raise ValueError('완료 J의 주파수/배열 형식 오류')
    nd, nf = j.shape
    if nd < 2 or np.any(np.diff(nu*1.e-15) <= 0) or np.any(nu*1.e-15 <= 0):
        raise ValueError('native 주파수/깊이 표현 범위 오류')
    recl = 8*(nd+1)
    if len(header) != 14*recl or struct.unpack_from('<i', header, recl)[0] != 10:
        raise ValueError('검토한 native EDD header/RV 위치가 아님')
    header = bytearray(header)
    for record in (1, 3, 4):
        struct.pack_into('<iii', header, (record-1)*recl, 15, nf, nd)
    struct.pack_into('<d', header, 4*recl, 0.)  # 부분 출력은 완료로 읽지 못한다.
    with Path(target).open('xb') as stream:
        stream.write(header)
        for first in range(0, nf, block):
            last = min(first+block, nf)
            columns = np.asarray(j[:, nf-last:nf-first][:, ::-1])
            if not np.isfinite(columns).all() or np.any(columns < 0):
                raise ValueError('완료 J에 비유한/음수 값')
            record = np.empty((last-first, nd+1), dtype='<f8')
            record[:, :nd] = columns.T
            record[:, nd] = nu[nf-last:nf-first][::-1]*1.e-15
            stream.write(record.tobytes())
        if stream.tell() != (14+nf)*recl:
            raise ValueError('현재 J의 EDD 직렬화 길이 불일치')
        stream.seek(4*recl)
        stream.write(struct.pack('<d', 1.))
    return dict(depths=nd, frequencies=nf, record_bytes=recl, data_first_record=15,
                J_values_preserved=True, native_quadrature_equal=False)


def prepare(preparation, preparation_sha, feedback, feedback_sha, native_run, out):
    paths = [Path(p).resolve() for p in (preparation, feedback, native_run, out)]
    preparation, feedback, native_run, out = paths
    if out.exists():
        raise FileExistsError('기존 현재 J 입력 보존')
    if out == native_run or native_run in out.parents:
        raise ValueError('원 native 트리 안에 출력 금지')
    if sha(preparation) != preparation_sha or sha(feedback) != feedback_sha:
        raise ValueError('완료 자체장 계보 변경')
    p, f = (json.loads(path.read_text()) for path in (preparation, feedback))
    if (p['schema'] != 'lumina-accepted-field-preparation-v1'
            or f['schema'] != 'lumina-resident-fixed-material-feedback-v1'
            or f['accepted_preparation_sha256'] != preparation_sha
            or f['source_identity']['preparation_sha256'] != preparation_sha):
        raise ValueError('현재 완료 장과 준비 물질의 계보 불일치')
    last = f['last_complete']; entry = f['sweeps'][last['sweep']-1]
    child = feedback.parent/('F-'+str(last['sweep']))
    result_path = child/'resident-field.json'
    if (Path(last['J']['path']) != child/'J.npy' or entry['directory'] != str(child)
            or entry['state'] != 'COMPLETE_F_NOT_CONVERGENCE'
            or sha(result_path) != entry['result_sha256']):
        raise ValueError('실제 완료한 마지막 수송장이 아님')
    result = json.loads(result_path.read_text())
    from lumina_fullsr_sc_actual import completed_transport
    if (result['schema'] != 'lumina-resident-fullsr-native90-v1'
            or not completed_transport(result, p['frequencies'])
            or result['source_identity'] != f['source_identity'] or result['J'] != last['J']
            or result['original_layout']['frequencies'] != p['frequencies']):
        raise ValueError('현재 자체장의 실제 완료/원격자 증거 불일치')
    grid, J = Path(p['grid']), Path(last['J']['path'])
    inputs = {str(preparation): preparation_sha, str(feedback): feedback_sha,
              str(result_path): entry['result_sha256'], str(grid): p['input_sha256'][str(grid)],
              str(J): last['J']['sha256']}
    if f['source_identity']['grid_sha256'] != inputs[str(grid)]:
        raise ValueError('J와 다른 주파수 격자')
    if any(sha(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError('완료 장 본문/격자 변경')
    nu = np.load(grid, mmap_mode='r', allow_pickle=False)
    values = np.load(J, mmap_mode='r', allow_pickle=False)
    try:
        nd, nf = p['native_nodes'], p['frequencies']
        if nu.shape != (nf,) or values.shape != (nd, nf):
            raise ValueError('현재 J 깊이/주파수 수 불일치')
        info = native_run/'EDDFACTOR_INFO'
        info_bytes = info.read_bytes()
        description = info_bytes.decode().splitlines()[2].split()
        if (list(map(int, description[:5])) != [nd, 8*(nd+1), 8, 1, 4]
                or description[5] != 'T'):
            raise ValueError('다른 native EDD 정밀도/크기')
        recl = 8*(nd+1)
        with (native_run/'EDDFACTOR').open('rb') as stream:
            header = stream.read(14*recl)
        with (native_run/'SCRTEMP').open('rb') as stream:
            radius = stream.read(nd*8)
        if header[9*recl:9*recl+nd*8] != radius:
            raise ValueError('native EDD와 물질의 반경 불일치')
        from prepare_lumina_material_cells import read_nodes
        material = Path(p['material'])
        if sha(material) != p['input_sha256'][str(material)]:
            raise ValueError('현재 장 생산 물질 변경')
        _, physical, _, _, epoch = read_nodes(material)
        if (not np.array_equal(physical[:, 0], np.frombuffer(radius, dtype='<f8')*1.e10)
                or epoch != p['epoch_seconds']):
            raise ValueError('현재 장과 native 입력 반경/시각 불일치')
        inputs[str(material)] = p['input_sha256'][str(material)]
        inputs[str(info)] = hashlib.sha256(info_bytes).hexdigest()
        out.mkdir()
        state = write_field(out/'EDDFACTOR', header, nu, values)
        (out/'EDDFACTOR_INFO').write_bytes(info_bytes)
    finally:
        nu._mmap.close(); values._mmap.close()
    if any(sha(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError('J 직렬화 중 원본 변경')
    record = dict(schema='lumina-native-current-fixedj-input-v1', scope=__doc__,
                  input_sha256=inputs, layout=state,
                  native_header_sha256=hashlib.sha256(header).hexdigest(),
                  output_sha256={n: sha(out/n) for n in ('EDDFACTOR', 'EDDFACTOR_INFO')},
                  interpolation='original native RD_CONT_J linear-frequency interpolation',
                  quadrature='native-generated grid; not asserted equal to Lumina fine grid',
                  original_native_edited=False, new_transport=0, new_material_response=0,
                  actual_native_physics_evaluation=False, final_goal='NOT_MET')
    (out/'fixedj-field-input.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'preparation-sha', 'feedback', 'feedback-sha', 'native-run', 'out'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    prepare(a.preparation, a.preparation_sha, a.feedback, a.feedback_sha, a.native_run, a.out)
