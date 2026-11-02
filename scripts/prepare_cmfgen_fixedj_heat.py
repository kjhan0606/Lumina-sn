#!/usr/bin/env python3
"""J71 링크 계보를 보존한 공통-J 실제 RE 평가기를 준비한다. 계산은 하지 않는다.

고정-J SE 경로가 건너뛰던 COMP_OPAC을 다시 호출하며, 약한 선까지 합산한
실제 STEQ_T/입력 물질을 물질 갱신 전에 저장하고 STOP96으로 종료한다.
이는 실제 공통 상태 주입·J 소비 확인·native/Lumina 열행 일치의 증명이 아니다.
"""
import argparse
import json
from pathlib import Path
import struct
import shutil

import numpy as np

from prepare_cmfgen_fixedj_exponent import OLD
from prepare_cmfgen_mom_capture import FLAGS, command, once, sha


CALLER_SHA = '84e3b5ae0c6f44f531b4c1cd6aa83cde9cd51bc051a0f98cddd8fcd7fd17d54c'
ARCHIVE_SHA = 'e4d5ec0df548d9e98e50b1d3ebfc17e35f5e6bc582c1ce12ab7da97f8d49c4f7'
PARENT_SHA = '5210f73a0fdb7b16195dbb02b5fffdc8619055f2f49dbbff3d70d0c38b5dfc5f'
COMP_OPAC = ("\t      CALL TUNE(IONE,'C_OPAC')\n"
             '\t      CALL COMP_OPAC(POPS,NU_EVAL_CONT,FQW,\n'
             '\t1                FL,CONT_FREQ,FREQ_INDX,NCF,\n'
             '\t1                SECTION,ND,NT,LST_DEPTH_ONlY)\n'
             "\t      CALL TUNE(ITWO,'C_OPAC')\n")
FIXED_SKIP = OLD + '\n\t    ELSE\n' + COMP_OPAC + '\t    END IF\n'
GUARD_POINT = ('\tCALL SCR_READ_V2(R,V,SIGMA,POPS,IREC,NITSF,RITE_N_TIMES,LAST_NG,\n'
               '\t1                 WRITE_RVSIG,NT,ND,LUSCR,NEWMOD)\n')
GUARD = ('\tCALL LUMINA_FIXEDJ_HEAT_GUARD(USE_FIXED_J,IMPURITY_CODE,\n'
         '\t1   RD_COHERENT_ES,USE_ELEC_HEAT_BAL,COMP_STEQ_T_EHB,\n'
         '\t1   DO_CO_MOV_DDT,FLUX_CAL_ONLY,NO_VEL_FOR_CONTINUUM,\n'
         '\t1   FP_ENABLED(),GLOBAL_LINE_SWITCH,NEWMOD,NDIETOT,\n'
         '\t1   GAMRAY_TRANS)\n')
FINISH_POINT = "\tCALL TUNE(IONE,'SOLVE_FOR_POPS')\n"
FINISH = ('\tCALL LUMINA_FIXEDJ_HEAT_FINISH(POPS,R,V,SIGMA,T,ED,\n'
          '\t1   POP_ATOM,CLUMP_FAC,STEQ_T,STEQ_ED,NT,ND,NCF,\n'
          '\t1   MAIN_COUNTER,NUM_OF_WEAK_LINES)\n')
VECTOR_NAMES = ('radius_native', 'velocity_native', 'sigma', 'temperature_native',
                'ne', 'atoms', 'clump', 'steq_t_native', 'steq_ed_native')


def patched(source):
    """관찰 대상은 최종 적분 RE이며 EHB/GENCOOL/dummy H 출력이 아니다."""
    source = once(source, FIXED_SKIP, COMP_OPAC)
    source = once(source, GUARD_POINT, GUARD_POINT + GUARD)
    return once(source, FINISH_POINT, FINISH + FINISH_POINT)


def read_heat_row(path):
    """작은 native 결과만 읽는다. 단위 변환/상태 일치/합격 판정은 별개다."""
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(48)
        if len(header) != 48 or header[:16] != b'NATIVEHEATROWV1 ':
            raise ValueError('실제 native RE 행 헤더 누락')
        version, bits, nt, nd, ncf, counter, nv, weak = struct.unpack('<8i', header[16:])
        if (version != 1 or bits != 64 or nt < 3 or nd < 2 or ncf < 1 or counter < 1
                or nv != len(VECTOR_NAMES) or weak < 0):
            raise ValueError('실제 native RE 행 차원/형식 오류')
        expected = 48 + 8*nd*(nv+nt)
        if expected > 256*1024**2 or path.stat().st_size != expected:
            raise ValueError('실제 native RE 행 길이 오류')
        data = np.frombuffer(stream.read(), dtype='<f8').copy()
    if not np.isfinite(data).all():
        raise ValueError('실제 native RE 행 비유한')
    vectors = dict(zip(VECTOR_NAMES, data[:nv*nd].reshape(nv, nd), strict=True))
    pops = data[nv*nd:].reshape((nt, nd), order='F')
    r = vectors['radius_native']
    if (np.any(r <= 0) or np.any(np.diff(r) >= 0) or np.any(pops < 0)
            or np.any(vectors['atoms'] <= 0) or np.any(vectors['clump'] <= 0)
            or np.any(vectors['temperature_native'] <= 0) or np.any(vectors['ne'] <= 0)
            or not np.array_equal(pops[-1], vectors['temperature_native'])
            or not np.array_equal(pops[-2], vectors['ne'])):
        raise ValueError('실제 native RE 행 상태/반경 오류')
    return dict(vectors, pops=pops, frequencies=ncf, evaluation_counter=counter,
                weak_lines=weak, kind='INTEGRAL_RE_NOT_EHB',
                phase='BEFORE_POP_UPDATE', joint_convergence=False)


def prepare(parent, out):
    """원 J71의 실제 객체/라이브러리를 사용한다. 설치 native archive로 바꾸지 않는다."""
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('기존 native RE 평가기 보존')
    manifest_path = parent/'feedback-build.json'
    if sha(manifest_path) != PARENT_SHA:
        raise ValueError('봉인 J71 실행 계보 manifest가 아님')
    model = json.loads(manifest_path.read_text())
    native = Path(model['native'])
    if out == native or native in out.parents or parent in out.parents:
        raise ValueError('원 native/J71 build 트리 안에 출력 금지')
    inputs = {Path(p): digest for p, digest in model['input_sha256'].items()}
    caller = next(p for p in inputs if p.name == 'cmfgen_sub.o')
    source = caller.with_suffix('.f')
    archives = [p for p in inputs if p.name == 'libsubs.a']
    if (len(archives) != 1 or inputs[archives[0]] != ARCHIVE_SHA
            or sha(source) != CALLER_SHA or model.get('expected_iteration') != 71):
        raise ValueError('실제 J71 caller/stock archive 계보가 아님')
    objects = [p for p in inputs if p.suffix == '.o' and p != caller]
    # whole module은 J71의 다음 세대 표기를 가진 실제 object를 선택한다.
    whole = parent/'cmfgen_formal_primary_whole.o'
    if str(whole) not in model['output_sha256']:
        raise ValueError('J71 whole object 증거 누락')
    objects.append(whole)
    inputs[whole] = model['output_sha256'][str(whole)]
    includes = list(dict.fromkeys([caller.parent, *[p.parent for p in objects], native/'mod']))
    if any(folder == out or folder in out.parents for folder in includes):
        raise ValueError('원 J71 지원 객체 트리 안에 출력 금지')
    for folder in includes:
        modules = list(folder.glob('*.mod'))
        if len(modules) > 500:
            raise ValueError('native module 입력 범위 초과')
        for path in modules:
            inputs[path] = sha(path)
    observer = Path(__file__).with_name('cmfgen_fixedj_heat.f90').resolve()
    include = native/'new_main/GET_J_CHI.INC'
    if sha(include) != 'aa44ca3bfa346f4d79f2efe21337d8bb0a1249268ca0f78938b164b9cc460775':
        raise ValueError('봉인 GET_J_CHI include 변경')
    for path in (source, include, observer, Path(__file__).resolve(), manifest_path):
        inputs[path] = sha(path)
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('봉인 J71 native 링크 입력 변경')
    modified = patched(source.read_text())
    out.mkdir()
    (out/'cmfgen_sub.f').write_text(modified)
    shutil.copy2(include, out/'GET_J_CHI.INC')
    shutil.copy2(observer, out/observer.name)
    options = ['-I.', *['-I'+str(p) for p in includes]]
    command(['gfortran', *FLAGS, *options, '-c', 'cmfgen_sub.f'], out)
    command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none', *options,
             '-c', observer.name], out)
    command(['gfortran', '-fopenmp', '-m64', *objects, out/'cmfgen_sub.o',
             out/'cmfgen_fixedj_heat.o', archives[0],
             *[p for p in inputs if p.name in ('libpgplt.a', 'libmod.a',
                                              'libmy_lapack.a', 'libmy_blas.a')],
             '-o', out/'cmfgen_fixedj_heat.exe'], out)
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('빌드 중 원본 native 링크 입력 변경')
    report = dict(schema='lumina-native-fixedj-integral-re-build-v1', scope=__doc__,
                  input_sha256={str(p): d for p, d in inputs.items()},
                  output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                  activation='LUMINA_FIXEDJ_HEAT_ONLY=1; incompatible controls STOP94',
                  terminal='STOP96 before original population solver; no material update',
                  actual_native_evaluation=False, common_state_injection='NOT_PREPARED',
                  actual_fixed_J_consumption='NOT_YET_VERIFIED',
                  same_full_level_state='NOT_YET_VERIFIED',
                  heat_equivalence='NOT_YET_EVALUATED', final_goal='NOT_MET')
    (out/'fixedj-heat-build.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.out)
