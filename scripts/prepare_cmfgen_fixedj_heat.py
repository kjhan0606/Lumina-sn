#!/usr/bin/env python3
"""J71 링크 계보를 보존한 공통-J 실제 RE 평가기를 준비한다. 계산은 하지 않는다.

고정-J SE 경로가 건너뛰던 COMP_OPAC을 다시 호출하며, 약한 선까지 합산한
실제 STEQ_T/입력 물질을 물질 갱신 전에 저장하고 STOP96으로 종료한다.
이는 실제 공통 상태 주입·J 소비 확인·native/Lumina 열행 일치의 증명이 아니다.
"""
import argparse
import hashlib
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
         '\t1   GAMRAY_TRANS)\n'
         '\tCALL LUMINA_FIXEDJ_HEAT_LOAD(POPS,R,NT,ND)\n')
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


def native_state(columns, physical, population, variables, name_to_z):
    """native 변수 번호로 SL/terminal을 직접 대응한다. 평형·정규화·보간은 없다."""
    columns = np.asarray(columns)
    physical = np.asarray(physical)
    population = np.asarray(population)
    nd, nc = len(physical), len(columns)
    if (nd < 2 or nc < 1 or columns.shape != (nc, 3) or columns.dtype.kind not in 'iu'
            or physical.shape != (nd, 4) or population.shape != (nd, nc)
            or len(variables) != nc or not np.isfinite(physical).all()
            or not np.isfinite(population).all() or np.any(physical <= 0)
            or np.any(population < 0) or np.any(np.diff(physical[:, 0]) >= 0)):
        raise ValueError('수용 native 입력의 차원/반경/인구 오류')
    lookup = {tuple(key): i for i, key in enumerate(columns)}
    if len(lookup) != nc or [v.index for v in variables] != list(range(1, nc+1)):
        raise ValueError('중복 SL 또는 native 변수 번호 오류')
    try:
        addresses = [(name_to_z[v.species], v.charge, -1 if v.terminal else v.superlevel-1)
                     for v in variables]
    except KeyError as error:
        raise ValueError('등록되지 않은 native 원소') from error
    if len(set(addresses)) != nc or set(addresses) != set(lookup):
        raise ValueError('수용 SL/terminal과 native 변수 주소 집합 불일치')
    indexes = [lookup[key] for key in addresses]
    return np.vstack((population[:, indexes].T, physical[:, 2], physical[:, 1]*1.e-4))


def prepare_state(material, material_sha, native_run, out):
    """완료 수용 물질을 격리된 native 입력으로 직렬화한다. 계산/원본 편집은 없다."""
    from analyze_cmfgen_material_inventory import ELEMENTS
    from prepare_lumina_material_cells import read_nodes
    from ptf_gated_metrics import read_dimensions, read_variables
    material, native_run, out = [Path(p).resolve() for p in (material, native_run, out)]
    if out.exists():
        raise FileExistsError('기존 native 입력 보존')
    if out == native_run or native_run in out.parents or out == material.parent or material.parent in out.parents:
        raise ValueError('원 native/수용 물질 트리 안에 입력 출력 금지')
    if sha(material) != material_sha:
        raise ValueError('수용 물질 manifest 변경')
    manifest = json.loads(material.read_text())
    if (manifest['schema'] != 'lumina-accepted-nodal-material-v1'
            or manifest['status'] != 'EXPORTED_NOT_TRANSPORT_READY'
            or manifest['coordinate_role'] != 'original outer-to-inner point samples; no averaging, interpolation or remap'):
        raise ValueError('등록된 원점 수용 물질이 아님')
    nodes = material.parent/'native_material_nodes.bin'
    if sha(nodes) != manifest['output_sha256']['native_material_nodes.bin']:
        raise ValueError('수용 물질 node 파일 변경')
    # 같은 MODEL_SPEC/원자 링크를 요구하되 복사된 별도 native 실행 디렉터리는 허용한다.
    inputs = {str(material): material_sha, str(nodes): sha(nodes)}
    for name in ('MODEL', 'MODEL_SPEC', 'atomic_links.txt'):
        expected = [digest for path, digest in manifest['input_sha256'].items() if Path(path).name == name]
        path = native_run/name
        if len(expected) != 1 or sha(path) != expected[0]:
            raise ValueError('수용 물질과 다른 native 원자모형: '+name)
        inputs[str(path)] = expected[0]
    columns, physical, population, nl, epoch = read_nodes(nodes)
    nd, nt = read_dimensions(native_run)
    links = native_run/'LEVEL_SL_STEQ_LINKS'
    inputs[str(links)] = sha(links)
    variables = read_variables(native_run, nt-2)
    state = native_state(columns, physical, population, variables,
                         {name: z for z, (name, _) in ELEMENTS.items()})
    if len(physical) != nd or state.shape != (nt, nd):
        raise ValueError('native 깊이/변수 수 변경')
    # SCRTEMP의 첫 ND double은 원 native 반경이다. 상태 record/시간은 재사용하지 않는다.
    with (native_run/'SCRTEMP').open('rb') as stream:
        raw_radius = stream.read(nd*8)
    radius = np.frombuffer(raw_radius, dtype='<f8')
    if len(radius) != nd or not np.array_equal(radius*1.e10, physical[:, 0]):
        raise ValueError('수용 물질과 native 원 반경이 다름')
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError('native 상태 준비 중 봉인 입력 변경')
    with (native_run/'SCRTEMP').open('rb') as stream:
        if stream.read(nd*8) != raw_radius:
            raise ValueError('native 반경 구간이 준비 중 변경됨')
    out.mkdir()
    with (out/'NATIVE_HEAT_INPUT.bin').open('xb') as stream:
        stream.write(b'NATIVEHEATINPV1 '+struct.pack('<4i', 1, 64, nt, nd))
        stream.write(raw_radius)
        stream.write(np.asarray(state, dtype='<f8').tobytes(order='F'))
    report = dict(schema='lumina-native-heat-state-input-v1',
                  input_sha256=inputs, native_run=str(native_run), points=nd, variables=nt,
                  full_levels=nl, epoch_seconds=epoch,
                  accepted_result_sha256=manifest['accepted_result_sha256'],
                  accepted_response_sha256=manifest['accepted_response_sha256'],
                  output_sha256={'NATIVE_HEAT_INPUT.bin': sha(out/'NATIVE_HEAT_INPUT.bin')},
                  SL_and_ne_exact=True, temperature_conversion='K -> 1e4 K; no equilibration',
                  geometry='original native radius bytes; no remap, V/sigma not changed',
                  radius_source=dict(path=str(native_run/'SCRTEMP'), offset=0, bytes=len(raw_radius),
                                     sha256=hashlib.sha256(raw_radius).hexdigest()),
                  native_full_level_reconstruction='NOT_YET_EVALUATED',
                  radiation_input='NOT_PREPARED', heat_policy='NOT_YET_MATCHED',
                  actual_native_evaluation=False, final_goal='NOT_MET')
    (out/'heat-state-input.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


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
                  actual_native_evaluation=False, common_state_injection='EXPLICIT_INPUT_REQUIRED_BEFORE_POPTOION',
                  decay_implementation='UNCHANGED_J71_ARCHIVE; not an instantaneous-decay-patch build',
                  actual_fixed_J_consumption='NOT_YET_VERIFIED',
                  same_full_level_state='NOT_YET_VERIFIED',
                  heat_equivalence='NOT_YET_EVALUATED', final_goal='NOT_MET')
    (out/'fixedj-heat-build.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--state-material', type=Path)
    parser.add_argument('--state-material-sha')
    parser.add_argument('--native-run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if any((args.state_material, args.state_material_sha, args.native_run)):
        if args.parent or not all((args.state_material, args.state_material_sha, args.native_run)):
            parser.error('상태 준비는 --state-material/--state-material-sha/--native-run만 함께 지정')
        prepare_state(args.state_material, args.state_material_sha, args.native_run, args.out)
    elif args.parent:
        prepare(args.parent, args.out)
    else:
        parser.error('--parent 빌드 또는 명시한 수용 상태 준비가 필요')
