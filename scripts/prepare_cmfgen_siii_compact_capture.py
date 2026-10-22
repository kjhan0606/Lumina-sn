#!/usr/bin/env python3
"""물질70/J71 S III 대역의 실제 광선 이력을 단일 stream으로 포획하는 격리 빌드.

원 formal 산술 커널은 바꾸지 않는다. 포획은 실행이 아니라 빌드만 수행한다.
"""
import argparse
import json
from pathlib import Path
import re

from prepare_cmfgen_center_packet import OLD_SELECTION
from prepare_cmfgen_mom_capture import command, once, sha


FIRST, LAST = 40286, 45481
SELECTION = ("FORMAL_SELECTED=MODE=='CAPTURE_FORMAL'.AND.LUMINA_OUTER==71."
             f"AND.LUMINA_FREQ_INDEX>={FIRST}.AND.LUMINA_FREQ_INDEX<={LAST}")


def replace_subroutine(source, name, replacement):
    pattern = re.compile(rf'(?ms)^SUBROUTINE {name}\(.*?^END SUBROUTINE$')
    source, count = pattern.subn(lambda _: replacement.strip(), source)
    if count != 1:
        raise ValueError(f'{name} 포획 지원 함수 신원 변경: {count}')
    return source


STATE_NOOP = """
SUBROUTINE FORMAL_STATE_IO(PATH,A,S,READING)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_FORMAL_ARGS) A
TYPE(LUMINA_RAY_STATE) S
LOGICAL READING
IF(READING)STOP 86
END SUBROUTINE
"""


COMPACT_RAYS = """
SUBROUTINE FORMAL_RAYS_IO(PATH,A,READING,IS_OUTPUT)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_FORMAL_ARGS) A
LOGICAL READING,IS_OUTPUT
INTEGER LU,IP,ID,K,NZ,NP_LIMIT
INTEGER(KIND=8) BYTES
CHARACTER(LEN=16) TAG
IF(READING)STOP 86
IF(.NOT.IS_OUTPUT)RETURN
IF(A%ND<52.OR.A%NP>500.OR.ND_EXT>A%ND+1000)STOP 88
NP_LIMIT=A%NP-1
IF(A%THICK_OB)NP_LIMIT=A%NP
IF(.NOT.COMPACT_BLUE_WRITTEN)THEN
 OPEN(NEWUNIT=LU,FILE='siii_blue_in.bin',STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED')
 TAG='LUMINASIIIBLUE1'
 WRITE(LU) TAG,LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,A%NP,A%ND,NP_LIMIT
 DO IP=1,NP_LIMIT
  NZ=RAY(IP)%NZ
  WRITE(LU) IP,NZ,RAY(IP)%I_P_PREV(1:NZ),RAY(IP)%I_M_PREV(1:NZ)
 END DO
 TAG='LUMINASIIIBLEND'
 WRITE(LU) TAG
 CLOSE(LU)
 COMPACT_BLUE_WRITTEN=.TRUE.
END IF
OPEN(NEWUNIT=LU,FILE='siii_compact.bin',STATUS='UNKNOWN',ACCESS='STREAM', &
     FORM='UNFORMATTED',POSITION='APPEND')
TAG='LUMINASIIIRAYV1'
WRITE(LU) TAG,LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,A%NP,A%ND,ND_EXT, &
          NP_LIMIT,MERGE(1,0,A%NEW_FREQ),MERGE(1,0,RAY_POINTS_INSERTED), &
          A%FREQ,A%dLOG_NU
WRITE(LU) A%JNU(49:52),CHI_EXT(1:ND_EXT),ETA_EXT(1:ND_EXT)
DO IP=1,NP_LIMIT
 NZ=RAY(IP)%NZ
 WRITE(LU) IP,NZ,RAY(IP)%I_P(1),RAY(IP)%I_M(1), &
           RAY(IP)%I_P(NZ),RAY(IP)%I_M(NZ)
END DO
DO ID=49,52
 DO IP=1,MIN(NP_LIMIT,A%NP-ID+1)
  K=RAY(IP)%LNK(ID)
  IF(K<1.OR.K>RAY(IP)%NZ)STOP 88
  WRITE(LU) IP,ID,K,RAY(IP)%I_P(K),RAY(IP)%I_M(K), &
            RAY(IP)%I_P_PREV(K),RAY(IP)%I_M_PREV(K), &
            RAY(IP)%B_P(K),RAY(IP)%B_M(K)
 END DO
END DO
TAG='LUMINASIIIEND1'
WRITE(LU) TAG
CLOSE(LU)
INQUIRE(FILE='siii_compact.bin',SIZE=BYTES)
IF(BYTES>536870912_8)STOP 88
END SUBROUTINE
"""


def compact_source(original):
    source = once(original, OLD_SELECTION, SELECTION)
    source = once(source, 'INTEGER(KIND=8),SAVE :: TOTAL_BYTES=0',
                  'INTEGER(KIND=8),SAVE :: TOTAL_BYTES=0\n'
                  'LOGICAL,SAVE :: COMPACT_BLUE_WRITTEN=.FALSE.,COMPACT_GEOMETRY_WRITTEN=.FALSE.')
    source = replace_subroutine(source, 'FORMAL_STATE_IO', STATE_NOOP)
    source = replace_subroutine(source, 'FORMAL_RAYS_IO', COMPACT_RAYS)
    marker = ('SUBROUTINE FORMAL_GEOMETRY(PATH,A)\n'
              'CHARACTER(LEN=*) PATH\nTYPE(LUMINA_FORMAL_ARGS) A\n'
              'INTEGER LU,IP,ID,NZ,J\nINTEGER(KIND=8) ESTIMATE\n'
              'CHARACTER(LEN=16) TAG')
    source = once(source, marker, marker+'\nIF(COMPACT_GEOMETRY_WRITTEN)RETURN')
    source = once(source, "CALL FORMAL_BUDGET(PATH)\nEND SUBROUTINE\nEND MODULE",
                  "CALL FORMAL_BUDGET(PATH)\nCOMPACT_GEOMETRY_WRITTEN=.TRUE.\n"
                  "END SUBROUTINE\nEND MODULE")
    if source.count(SELECTION) != 1 or source.count('COMPACT_GEOMETRY_WRITTEN=.TRUE.') != 1:
        raise ValueError('압축 포획 지점 불완전')
    return source


def build(feedback, out):
    if out.exists():
        raise FileExistsError('격리 빌드 대상이 이미 존재함')
    manifest_path = feedback/'feedback-build.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['expected_iteration'] != 71 or manifest['pre_population_stop'] != 97:
        raise ValueError('봉인 물질70/J71 pre-pop 빌드가 아님')
    parent = Path(manifest['parent'])
    original = json.loads((parent/'formal-primary-build.json').read_text())
    support = Path(original['parent'])
    native = Path(manifest['native'])
    if support != Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/whole_accel_v1'):
        raise ValueError('검증하지 않은 formal 계측 지원 빌드')
    diag_source = support/'lumina_formal_diag.f90'
    kernel = parent/'cmf_formal_rel_v4.f'
    kernel_source = kernel.read_text()
    if (kernel_source.count('LUM_CAPTURE=FORMAL_SELECTED()') != 1 or
            kernel_source.count('IF(LUMINA_FREQ_INDEX==23732)THEN') != 1 or
            kernel_source.count('STOP 96') != 1):
        raise ValueError('원 formal 포획 커널 변경')
    objects = [native/'new_main/cmfgen.o', parent/'cmfgen_sub.o',
               parent/'cmfgen_formal_primary.o', parent/'cmfgen_positive_sc.o',
               feedback/'cmfgen_formal_primary_whole.o']
    objects += [support/(name+'.o') for name in (
        'cmfgen_mom_acceleration', 'lumina_affine_boundary', 'lumina_mom_diag')]
    objects += [support/'lumina_whole_monitor.o',
                support.parent/'formal_v2/ext_rel_grid_v4.o',
                support.parent/'capture_first_v1/mod_jrel_v9.o']
    libraries = [parent/'libsubs.a'] + [native/'lib'/name for name in (
        'libpgplt.a', 'libmod.a', 'libmy_lapack.a', 'libmy_blas.a')]
    inputs = [manifest_path, parent/'formal-primary-build.json',
              support/'sweep-build.json', diag_source, kernel, *objects, *libraries]
    expected = {**manifest['input_sha256'], **manifest['output_sha256']}
    if any(sha(Path(name)) != digest for name, digest in expected.items()):
        raise ValueError('feedback 봉인 빌드 신원 변경')
    before = {str(path): sha(path) for path in inputs}
    source = compact_source(diag_source.read_text())
    out.mkdir(parents=False)
    (out/'lumina_formal_diag.f90').write_text(source)
    includes = ['-I'+str(path) for path in (
        parent, support, support.parent/'formal_v2',
        support.parent/'capture_first_v1', native/'mod')]
    command(['gfortran','-O','-fopenmp','-ffree-line-length-none',*includes,
             '-I.','-c','lumina_formal_diag.f90'],out)
    executable = out/'cmfgen_siii_compact.exe'
    command(['gfortran','-fopenmp','-m64',*objects[:8],
             out/'lumina_formal_diag.o',*objects[8:],
             '-L'+str(parent),'-L'+str(native/'lib'),
             '-lsubs','-lpgplt','-lmod',
             native/'lib/libmy_lapack.a',native/'lib/libmy_blas.a',
             '-o',executable],out)
    if any(sha(Path(path)) != digest for path,digest in before.items()):
        raise ValueError('빌드 중 봉인 입력 변경')
    report = dict(scope=__doc__, feedback=str(feedback), outer=71,
                  frequency_indices=[FIRST,LAST],formal_kernel_unchanged=True,
                  pre_population_stop=97,input_sha256=before,
                  output_sha256={str(path):sha(path) for path in (
                      out/'lumina_formal_diag.f90',out/'lumina_formal_diag.o',executable)},
                  final_goal='NOT_MET')
    (out/'siii-compact-build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(build=str(out),executable_sha256=report['output_sha256'][str(executable)])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.feedback.resolve(),args.out.resolve())
