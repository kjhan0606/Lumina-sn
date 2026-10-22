#!/usr/bin/env python3
"""물질70/J71의 1450Å formal 광선만 포획하는 격리 CMFGEN 빌드.

원 formal 커널·수송 지원 객체는 재사용하고 계측 선택 함수만 바꾼다.
"""
import argparse
import json
from pathlib import Path

from prepare_cmfgen_mom_capture import command, once, sha


OLD_SELECTION = ("FORMAL_SELECTED=MODE=='CAPTURE_FORMAL'.AND.LUMINA_OUTER==60."
                 "AND.LUMINA_FREQ_INDEX>=23731.AND.LUMINA_FREQ_INDEX<=23732")
NEW_SELECTION = ("FORMAL_SELECTED=MODE=='CAPTURE_FORMAL'.AND.LUMINA_OUTER==71."
                 "AND.LUMINA_FREQ_INDEX>=60714.AND.LUMINA_FREQ_INDEX<=60715")


def selected_source(source):
    return once(source, OLD_SELECTION, NEW_SELECTION)


def build(feedback, out):
    if out.exists():
        raise FileExistsError('격리 빌드 대상이 이미 존재함')
    manifest_path = feedback / 'feedback-build.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['expected_iteration'] != 71 or manifest['pre_population_stop'] != 97:
        raise ValueError('요구한 물질70/J71의 pre-pop 장 빌드가 아님')
    parent = Path(manifest['parent'])
    original = json.loads((parent / 'formal-primary-build.json').read_text())
    support = Path(original['parent'])
    native = Path(manifest['native'])
    if support != Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/whole_accel_v1'):
        raise ValueError('검증하지 않은 formal 계측 지원 빌드')
    diag_source = support / 'lumina_formal_diag.f90'
    kernel = parent / 'cmf_formal_rel_v4.f'
    kernel_source = kernel.read_text()
    if kernel_source.count('IF(LUMINA_FREQ_INDEX==23732)THEN') != 1 or \
       kernel_source.count('STOP 96') != 1 or \
       kernel_source.count('LUM_CAPTURE=FORMAL_SELECTED()') != 1:
        raise ValueError('원 formal 포획 커널의 조건부 종료 경로 변경')
    source = selected_source(diag_source.read_text())
    objects = [native/'new_main/cmfgen.o', parent/'cmfgen_sub.o',
               parent/'cmfgen_formal_primary.o', parent/'cmfgen_positive_sc.o',
               feedback/'cmfgen_formal_primary_whole.o']
    objects += [support/(name+'.o') for name in (
        'cmfgen_mom_acceleration', 'lumina_affine_boundary', 'lumina_mom_diag')]
    objects += [support/'lumina_whole_monitor.o']
    objects += [Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v2/ext_rel_grid_v4.o'),
                Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/capture_first_v1/mod_jrel_v9.o')]
    libraries = [parent/'libsubs.a'] + [native/'lib'/name for name in (
        'libpgplt.a', 'libmod.a', 'libmy_lapack.a', 'libmy_blas.a')]
    inputs = [manifest_path, parent/'formal-primary-build.json',
              support/'sweep-build.json', diag_source, kernel, *objects, *libraries]
    expected = {**manifest['input_sha256'], **manifest['output_sha256']}
    for name, digest in expected.items():
        if sha(Path(name)) != digest:
            raise ValueError('기존 feedback 장 빌드 신원 변경: ' + name)
    before = {str(path): sha(path) for path in inputs}
    out.mkdir(parents=False)
    (out/'lumina_formal_diag.f90').write_text(source)
    includes = ['-I'+str(path) for path in (
        parent, support, support.parent/'formal_v2', support.parent/'capture_first_v1', native/'mod')]
    command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none', *includes,
             '-I.', '-c', 'lumina_formal_diag.f90'], out)
    executable = out/'cmfgen_center_packet.exe'
    command(['gfortran', '-fopenmp', '-m64', *objects[:8],
             out/'lumina_formal_diag.o', *objects[8:],
             '-L'+str(parent), '-L'+str(native/'lib'),
             '-lsubs', '-lpgplt', '-lmod',
             native/'lib/libmy_lapack.a', native/'lib/libmy_blas.a',
             '-o', executable], out)
    if any(sha(Path(path)) != digest for path, digest in before.items()):
        raise ValueError('빌드 중 봉인 입력 변경')
    report = dict(scope=__doc__, feedback=str(feedback), outer=71,
                  frequency_indices=[60714, 60715], formal_kernel_unchanged=True,
                  pre_population_stop=97, input_sha256=before,
                  output_sha256={str(path): sha(path) for path in (
                      out/'lumina_formal_diag.f90', out/'lumina_formal_diag.o', executable)},
                  final_goal='NOT_MET')
    (out/'center-packet-build.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(build=str(out), executable_sha256=report['output_sha256'][str(executable)])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.feedback.resolve(), args.out.resolve())
