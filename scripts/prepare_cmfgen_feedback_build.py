#!/usr/bin/env python3
"""봉인 수송의 다음 물질 세대만 바꾼다. 커널과 pre-pop STOP97은 유지한다."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import command, once, sha


def next_generation(source, iteration):
    if not isinstance(iteration, int) or iteration <= 60:
        raise ValueError('완료59 이후의 명시적인 다음 세대 필요')
    if source.count('stop 97') != 1:
        raise ValueError('원래 pre-pop 종료가 아님')
    result = once(source, '.or.lumina_outer/=60)', f'.or.lumina_outer/={iteration})')
    return once(result, 'write(lu,*)60,pw_count,pw_expected,pw_calls',
                f'write(lu,*){iteration},pw_count,pw_expected,pw_calls')


def prepare(parent, out, iteration):
    if out.exists():
        raise ValueError('기존 feedback build 보존')
    manifest_path = parent / 'formal-primary-build.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['whole_executable_released']:
        raise ValueError('미검증 수송 부모')
    for name, expected in manifest['output_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('봉인 수송 부모 변경: ' + name)
    support = Path(manifest['parent'])
    support_manifest = support / 'sweep-build.json'
    if sha(support_manifest) != manifest['parent_manifest_sha256']:
        raise ValueError('수송 지원 객체의 계보 변경')
    model = json.loads(support_manifest.read_text())
    native, formal, mom = (Path(model[k]) for k in ('native', 'formal_build', 'mom_build'))
    # 기존 객체/라이브러리도 실행파일의 실제 입력으로 봉인한다.
    objects = [native/'new_main/cmfgen.o', parent/'cmfgen_sub.o',
               parent/'cmfgen_formal_primary.o', parent/'cmfgen_positive_sc.o']
    objects += [support/(name+'.o') for name in (
        'cmfgen_mom_acceleration', 'lumina_affine_boundary', 'lumina_mom_diag',
        'lumina_formal_diag', 'lumina_whole_monitor')]
    objects += [formal/'ext_rel_grid_v4.o', mom/'mod_jrel_v9.o']
    libraries = [parent/'libsubs.a'] + [native/'lib'/name for name in (
        'libpgplt.a', 'libmod.a', 'libmy_lapack.a', 'libmy_blas.a')]
    source_path = parent/'cmfgen_formal_primary_whole.f90'
    source = next_generation(source_path.read_text(), iteration)
    inputs = [manifest_path, support_manifest, source_path, *objects, *libraries]
    before = {str(p): sha(p) for p in inputs}
    out.mkdir()
    name = 'cmfgen_formal_primary_whole.f90'
    (out/name).write_text(source)
    includes = ['-I'+str(p) for p in (parent, support, formal, mom, native/'mod')]
    command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none', *includes,
             '-I.', '-c', name], out)
    executable = out/'cmfgen_feedback_field.exe'
    command(['gfortran', '-fopenmp', '-m64', *objects,
             out/'cmfgen_formal_primary_whole.o', '-L'+str(parent),
             '-L'+str(native/'lib'), '-lsubs', '-lpgplt', '-lmod',
             native/'lib/libmy_lapack.a', native/'lib/libmy_blas.a',
             '-o', executable], out)
    if any(sha(Path(p)) != h for p, h in before.items()):
        raise ValueError('빌드 중 봉인 입력 변경')
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    outputs = [out/name, out/'cmfgen_formal_primary_whole.o', executable,
               out/Path(__file__).name]
    report = dict(scope=__doc__, parent=str(parent), expected_iteration=iteration,
        native=str(native), pre_population_stop=97, transport_kernel_unchanged=True,
        source_interpolation=manifest['source_interpolation'],
        input_sha256=before, output_sha256={str(p): sha(p) for p in outputs},
        final_goal='NOT_JUDGED')
    (out/'feedback-build.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(build=str(out), expected_iteration=iteration), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.out.resolve(), args.iteration)
