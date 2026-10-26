#!/usr/bin/env python3
"""고정-J 경로의 실제ν 열적 지수 갱신만 private native build에서 수정한다."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS, command, once, sha


OLD = '''\t    IF(USE_FIXED_J)THEN
\t      IF(COMPUTE_NEW_CROSS)THEN
                T1=-HDKT*CONT_FREQ
                EMHNUKT_CONT(1:ND)=EXP(T1/T(1:ND))
                T1=-HDKT*FL
                EMHNUKT(1:ND)=EXP(T1/T(1:ND))
\t      END IF'''
NEW = '''\t    IF(USE_FIXED_J)THEN
\t      IF(COMPUTE_NEW_CROSS)THEN
                T1=-HDKT*CONT_FREQ
                EMHNUKT_CONT(1:ND)=EXP(T1/T(1:ND))
\t      END IF
! Actual-frequency thermal factor is not a continuum-cross-section cache.
              T1=-HDKT*FL
              EMHNUKT(1:ND)=EXP(T1/T(1:ND))'''


def patched(source):
    return once(source, OLD, NEW)


def prepare(parent, out):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('기존 수정 build 보존')
    parent_manifest = parent/'se-witness-build.json'
    manifest = json.loads(parent_manifest.read_text())
    native = Path(manifest['native'])
    fixed = [parent_manifest, *map(Path, manifest['output_sha256']),
             native/'new_main/cmfgen.o', *[native/'lib'/name for name in
             ('libpgplt.a', 'libmod.a', 'libmy_lapack.a', 'libmy_blas.a')],
             Path(__file__).resolve()]
    before = {str(path):sha(path) for path in fixed}
    for path, digest in manifest['output_sha256'].items():
        if before[path] != digest:
            raise ValueError('부모 build 변경: '+path)
    for path in fixed[-6:-1]:
        if before[str(path)] != manifest['input_sha256'].get(str(path)):
            raise ValueError('부모 native 링크 입력 변경: '+str(path))
    source = patched((parent/'cmfgen_sub.f').read_text())
    out.mkdir()
    shutil.copy2(parent/'GET_J_CHI.INC', out/'GET_J_CHI.INC')
    includes = ['-I.', '-I'+str(native/'mod'), '-I'+str(native/'new_main')]
    for name in ('cmfgen_se_collision_bound.f90', 'cmfgen_native_se_witness.f90'):
        shutil.copy2(parent/name, out/name)
        command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none',
                 *includes, '-c', name], out)
    (out/'cmfgen_sub.f').write_text(source)
    command(['gfortran', *FLAGS, *includes, '-c', 'cmfgen_sub.f'], out)
    executable = out/'cmfgen_fixedj_exponent.exe'
    command(['gfortran', '-fopenmp', '-m64', native/'new_main/cmfgen.o', out/'cmfgen_sub.o',
             out/'cmfgen_se_collision_bound.o', out/'cmfgen_native_se_witness.o',
             '-L'+str(parent), '-L'+str(native/'lib'), '-lsubs', '-lpgplt', '-lmod',
             native/'lib/libmy_lapack.a', native/'lib/libmy_blas.a', '-o', executable], out)
    if any(sha(Path(path)) != digest for path, digest in before.items()):
        raise ValueError('빌드 도중 원본/부모 입력 변경')
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    outputs = [executable, out/'cmfgen_sub.f', out/'GET_J_CHI.INC',
               out/Path(__file__).name]
    report = dict(schema='lumina-native-fixedj-exponent-build-v1', parent=str(parent),
                  native=str(native), change='EMHNUKT actual FL updated at every frequency; '
                  'EMHNUKT_CONT remains COMPUTE_NEW_CROSS cached',
                  input_sha256=before, output_sha256={str(p):sha(p) for p in outputs},
                  production_C_changed=False, native_original_changed=False,
                  physical_equivalence='NOT_YET_TESTED', final_goal='NOT_MET')
    (out/'fixedj-exponent-build.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.out)
