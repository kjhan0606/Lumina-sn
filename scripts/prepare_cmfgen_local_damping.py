#!/usr/bin/env python3
"""검증된 witness 사본에 LOCAL 갱신 직전의 공통 감쇠만 추가한다.

원본 native와 기존 실행파일은 변경하지 않는다. 감쇠 실험의 실제 실행은
별도 유한 계획과 원래 입력·장의 신원 확인 후에만 수행한다.
"""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS, command, once, sha

MARKER = ('\t    DO J=1,NT\n'
          '\t      POPS(J,I)=POPS(J,I)*(1.0_LDP-STEQ(J,I)*SCALE)\n'
          '\t    END DO\n\t    MINSCALE=MIN(SCALE,MINSCALE)\n')
INSERTION = '\t    CALL LUMINA_LOCAL_STEP_DAMPING(SCALE,I,LUER)\n'


def patched(source):
    return once(source, MARKER, INSERTION + MARKER)


def prepare(native, witness, out):
    if out.exists():
        raise ValueError('기존 감쇠 build 보존')
    manifest = witness / 'se-witness-build.json'
    parent = json.loads(manifest.read_text())
    if Path(parent['native']) != native:
        raise ValueError('다른 native 부모')
    for name, expected in {**parent['input_sha256'], **parent['output_sha256']}.items():
        if sha(Path(name)) != expected:
            raise ValueError('기존 witness 입력·출력 변경: ' + name)
    source = native / 'new_main/subs/solveba_v13.f'
    helper = Path(__file__).with_name('cmfgen_local_step_damping.f90').resolve()
    objects = [native / 'new_main/cmfgen.o'] + [witness / (name + '.o') for name in
        ('cmfgen_sub', 'cmfgen_se_collision_bound', 'cmfgen_native_se_witness')]
    libraries = [native / 'lib' / name for name in
        ('libpgplt.a', 'libmod.a', 'libmy_lapack.a', 'libmy_blas.a')]
    inputs = [manifest, source, helper, witness / 'libsubs.a', *objects, *libraries,
              Path(__file__).resolve()]
    before = {str(p): sha(p) for p in inputs}
    changed = patched(source.read_text())
    out.mkdir()
    (out / source.name).write_text(changed)
    shutil.copy2(helper, out / helper.name)
    includes = ['-I' + str(native / 'mod'), '-I' + str(native / 'new_main')]
    command(['gfortran', *FLAGS, *includes, '-c', source.name], out)
    command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none', *includes,
             '-c', helper.name], out)
    shutil.copy2(witness / 'libsubs.a', out / 'libsubs.a')
    command(['ar', 'r', 'libsubs.a', 'solveba_v13.o'], out)
    executable = out / 'cmfgen_local_damping.exe'
    command(['gfortran', '-fopenmp', '-m64', *objects, out / 'cmfgen_local_step_damping.o',
        '-L.', '-L' + str(native / 'lib'), '-lsubs', '-lpgplt', '-lmod',
        native / 'lib/libmy_lapack.a', native / 'lib/libmy_blas.a', '-o', executable], out)
    if any(sha(Path(p)) != h for p, h in before.items()):
        raise ValueError('감쇠 build 중 원래 입력 변경')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    outputs = [out / n for n in (source.name, helper.name, 'solveba_v13.o',
        'cmfgen_local_step_damping.o', 'libsubs.a', executable.name, Path(__file__).name)]
    report = dict(scope=__doc__, native=str(native), witness=str(witness),
        input_sha256=before, output_sha256={str(p): sha(p) for p in outputs},
        activation='LUMINA_LOCAL_STEP_OMEGA=0.5; 미설정 또는 1은 기존 배율 그대로',
        supported_omega=[1.0, 0.5], insertion=INSERTION.strip(),
        changes='LOCAL 제한 계산 후 인구·전자밀도·고정T의 공통 SCALE만 변경',
        individual_clipping_added=False, original_native_changed=False,
        actual_native_execution=False, physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')
    (out / 'local-damping-build.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(build=str(out), executable_sha256=sha(executable),
                         actual_native_execution=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'witness', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(*(getattr(args, name).resolve() for name in ('native', 'witness', 'out')))
