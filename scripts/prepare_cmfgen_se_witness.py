#!/usr/bin/env python3
"""원래 native 식은 보존하고 세 관찰 호출만 private 사본에 추가한다.

빌드 준비는 실제 잔차·원래 실행과의 비간섭·물리 동질성 증명이 아니다.
실제 native 적용은 별도로 정한 상태/장/phase와 자원 계약을 필요로 한다.
"""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS,command,once,sha

NATIVE_SOURCES = dict(main='new_main/cmfgen_sub.f',
    collision='new_main/subs/steq_multi_v10.f',solve='new_main/mod_subs/solve_for_pops.f')
INSERTIONS = dict(
    main=("\tWRITE(6,*)'Zeroed BA matrices'\n",
          "\tCALL LUMINA_NATIVE_SE_RESET(NT,ND,NION)\n"),
    collision=("          CALL TUNE(2,'SUBCOL')\n",
          "\t  CALL LUMINA_NATIVE_SE_COLLECT(ID,N_S,ND,I,CNM,HN_S(1,I),\n\t1         HNST_S(1,I),ED(I))\n"),
    solve=("\tCALL WR_ASCI_STEQ(NION,ND,'STEQ ARRAY',LU_SE)\n",
          "\tCALL LUMINA_NATIVE_SE_WRITE(POPS,NT,ND,NION,MAIN_COUNTER)\n"))


def patched(source,role):
    marker,insertion=INSERTIONS[role]
    return once(source,marker,(insertion+marker) if role=='main' else (marker+insertion))


def prepare(native,reference,out):
    if out.exists():raise ValueError('기존 관찰 build 보존')
    parent=json.loads(reference.read_text())['native_binary']
    executable=native/'exe/cmfgen_dev.exe'
    if Path(parent['path'])!=executable or sha(executable)!=parent['sha256']:
        raise ValueError('기존 native 실행파일과 다른 부모')
    local=[Path(__file__).with_name(n).resolve() for n in (
        'cmfgen_se_collision_bound.f90','cmfgen_native_se_witness.f90')]
    libraries=[native/'lib'/n for n in ('libsubs.a','libpgplt.a','libmod.a','libmy_lapack.a','libmy_blas.a')]
    inputs=[reference,executable,native/'new_main/cmfgen.o',native/'new_main/GET_J_CHI.INC',*libraries,
            *[native/p for p in NATIVE_SOURCES.values()],*local,Path(__file__).resolve()]
    before={str(p):sha(p) for p in inputs}
    out.mkdir()
    shutil.copy2(native/'new_main/GET_J_CHI.INC',out/'GET_J_CHI.INC')
    includes=['-I.','-I'+str(native/'mod'),'-I'+str(native/'new_main')]
    for path in local:
        shutil.copy2(path,out/path.name)
        command(['gfortran','-O','-fopenmp','-ffree-line-length-none',*includes,'-c',path.name],out)
    for role,name in NATIVE_SOURCES.items():
        path=out/Path(name).name
        path.write_text(patched((native/name).read_text(),role))
        command(['gfortran',*FLAGS,*includes,'-c',path.name],out)
    shutil.copy2(native/'lib/libsubs.a',out/'libsubs.a')
    command(['ar','r','libsubs.a','steq_multi_v10.o','solve_for_pops.o'],out)
    command(['gfortran','-fopenmp','-m64',native/'new_main/cmfgen.o',out/'cmfgen_sub.o',
        out/'cmfgen_se_collision_bound.o',out/'cmfgen_native_se_witness.o',
        '-L.','-L'+str(native/'lib'),'-lsubs','-lpgplt','-lmod',
        native/'lib/libmy_lapack.a',native/'lib/libmy_blas.a',
        '-o',out/'cmfgen_se_witness.exe'],out)
    if any(sha(Path(p))!=h for p,h in before.items()):raise ValueError('빌드 중 원래 입력 변경')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    outputs=[out/n for n in ('cmfgen_se_witness.exe','libsubs.a',
        'cmfgen_sub.f','steq_multi_v10.f','solve_for_pops.f','GET_J_CHI.INC',
        'cmfgen_se_collision_bound.f90','cmfgen_native_se_witness.f90',Path(__file__).name)]
    report=dict(scope=__doc__,native=str(native),reference=str(reference),
        input_sha256=before,output_sha256={str(p):sha(p) for p in outputs},
        changes='세 관찰 호출과 별도 read-only 누적 모듈; 원래 식/제어는 불변',
        activation='LUMINA_NATIVE_SE_WITNESS=1; 기본 비활성',
        phase='갱신 직전 입력 물질; counter-1, 실행 manifest와 확인 필요',
        residual_scope='native가 이미 평가한 local SE의 global 합산 enclosure; 내부 산술의 실수잔차 미인증',
        actual_native_execution=False,noninterference='NOT_YET_REPRODUCED',
        physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'se-witness-build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(build=str(out),actual_native_execution=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('native','reference','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.native.resolve(),a.reference.resolve(),a.out.resolve())
