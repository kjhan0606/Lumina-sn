#!/usr/bin/env python3
"""원본을 보존한 full CMFGEN 비교 빌드. 계산·설치·온도 해제는 하지 않는다.

공통 actual-nu fixed-J 수정과 같은 native 링크 입력을 사용하며 두 실행파일의
차이는 검증한 순간 침적열 분기뿐이다. 단독시험의 stub은 링크하지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from prepare_cmfgen_fixedj_exponent import patched
from prepare_cmfgen_instantaneous_decay import ORIGINAL_SHA256, corrected_source
from prepare_cmfgen_mom_capture import FLAGS


SEALED = {
    'new_main/cmfgen_sub.f': '092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374',
    'new_main/GET_J_CHI.INC': 'aa44ca3bfa346f4d79f2efe21337d8bb0a1249268ca0f78938b164b9cc460775',
    'new_main/cmfgen.o': '2383ffc10c3a5bc1fe605c31f81e863e9e2c76fc2fb37f096c687321e6b24f00',
    'new_main/subs/do_species_decays_v2.f': ORIGINAL_SHA256,
    'lib/libsubs.a': 'f719c7393dcb49fca6dd603c43db8616533fc9b757c980ebe7588eb82184048a',
    'lib/libmod.a': 'de1c4841c533bb5ff80848ad41d8b446eecf646907b5f47969dc71d20f0ca6b1',
    'lib/libpgplt.a': '9cb57dc4e374685d7442a31c2e34953cb4f42f38f823cb51316d05d1831d539c',
    'lib/libmy_lapack.a': '49b1f986af37711e0bf968ba397f4745e7d5f251a1e5cd012129f9c3184b71a3',
    'lib/libmy_blas.a': '83bb4c947e2ba385b30130f0d08e67ebcebfa0561feeb7e4d9483c66efa5819b',
}
SYMBOL = 'do_species_decays_v2_'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot(native):
    """고정 목록과 mod 한 디렉터리만 읽는다. 원본 트리는 순회하지 않는다."""
    files = {name: native / name for name in SEALED}
    for name, path in files.items():
        if sha(path) != SEALED[name]:
            raise ValueError('검토한 native 입력 변경: ' + name)
    modules = sorted((native / 'mod').glob('*.mod'))
    if not modules or len(modules) > 500:
        raise ValueError('native module 목록 누락/범위 초과')
    for path in modules:
        if path.is_symlink() or not path.is_file():
            raise ValueError('외부 module 링크 금지: ' + path.name)
        files['mod/' + path.name] = path
    return {name: sha(path) for name, path in files.items()}


def definition_count(nm_text, symbol=SYMBOL):
    return sum(len(row) >= 3 and row[-1] == symbol and row[-2].upper() == 'T'
               for row in (line.split() for line in nm_text.splitlines()))


def run(command, cwd, log):
    result = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True,
                            text=True, timeout=180)
    with log.open('a') as stream:
        stream.write(json.dumps(list(map(str, command))) + '\n')
        stream.write(result.stdout + result.stderr)
    result.check_returncode()
    return result.stdout, result.stderr


def build(native, out):
    native, out = native.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError('기존 빌드 보존')
    if native == out or native in out.parents:
        raise ValueError('원 native 트리 안에 출력 금지')
    started = time.monotonic()
    before = snapshot(native)
    original_decay = (native / 'new_main/subs/do_species_decays_v2.f').read_bytes()
    common_source = patched((native / 'new_main/cmfgen_sub.f').read_text())
    fixed_decay = corrected_source(original_decay)
    out.mkdir()
    inputs = out / 'inputs'
    inputs.mkdir()
    # 원 입력을 먼저 봉인 사본으로 고정한다. 후속 compile/link는 이 사본만 읽는다.
    for name, digest in before.items():
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native / name, target)
        if sha(target) != digest:
            raise ValueError('복사 중 입력 변경: ' + name)
    common = out / 'common'
    common.mkdir()
    (common / 'cmfgen_sub.f').write_text(common_source)
    shutil.copyfile(inputs / 'new_main/GET_J_CHI.INC', common / 'GET_J_CHI.INC')
    log = out / 'build.log'
    version, _ = run(['gfortran', '--version'], out, log)
    run(['gfortran', *FLAGS, '-I' + str(inputs / 'mod'), '-c', 'cmfgen_sub.f'], common, log)
    variants = {}
    for name, decay in (('original_decay', original_decay), ('instantaneous_decay', fixed_decay)):
        variant = out / name
        variant.mkdir()
        (variant / 'do_species_decays_v2.f').write_bytes(decay)
        run(['gfortran', *FLAGS, '-I' + str(inputs / 'mod'), '-c', 'do_species_decays_v2.f'], variant, log)
        executable = variant / 'cmfgen.exe'
        mapfile = variant / 'link.map'
        # 두 decay object를 모두 명시한다. archive의 구형 정의에 의존하지 않는다.
        _, trace = run(['gfortran', '-fopenmp', '-m64', inputs / 'new_main/cmfgen.o',
            common / 'cmfgen_sub.o', variant / 'do_species_decays_v2.o',
            '-L' + str(inputs / 'lib'), '-lsubs', '-lpgplt', '-lmod',
            inputs / 'lib/libmy_lapack.a', inputs / 'lib/libmy_blas.a',
            '-Wl,-Map=' + str(mapfile), '-Wl,--trace-symbol=' + SYMBOL,
            '-o', executable], variant, log)
        symbols, _ = run(['nm', '--defined-only', str(executable)], variant, log)
        if definition_count(symbols) != 1:
            raise ValueError('붕괴 함수 정의가 하나가 아님')
        definitions = [line for line in trace.splitlines() if 'definition of ' + SYMBOL in line]
        if len(definitions) != 1 or str(variant / 'do_species_decays_v2.o') not in definitions[0]:
            raise ValueError('의도한 private decay object가 선택되지 않음')
        if 'libsubs.a(do_species_decays_v2.o)' in mapfile.read_text():
            raise ValueError('구형 archive decay object가 중복 포함됨')
        (variant / 'decay-symbol.txt').write_text('\n'.join(
            line for line in symbols.splitlines() if line.split()[-1] == SYMBOL) + '\n' + trace)
        variants[name] = dict(executable=str(executable), executable_sha256=sha(executable),
            decay_source_sha256=sha(variant / 'do_species_decays_v2.f'),
            decay_object_sha256=sha(variant / 'do_species_decays_v2.o'),
            link_map_sha256=sha(mapfile), definition_count=1, explicit_object_selected=True)
    if before != snapshot(native):
        raise ValueError('빌드 중 원 native 입력 변경')
    if any(sha(inputs / name) != digest for name, digest in before.items()):
        raise ValueError('봉인 사본이 빌드 중 변경됨')
    script_root = Path(__file__).resolve().parent
    script_hashes = {str(script_root / name): sha(script_root / name) for name in
        ('build_cmfgen_instantaneous_reference.py', 'prepare_cmfgen_instantaneous_decay.py',
         'prepare_cmfgen_fixedj_exponent.py', 'prepare_cmfgen_mom_capture.py')}
    report = dict(schema='lumina-native-instantaneous-full-build-v1',
        state='COMPLETE_FULL_NATIVE_COMPARISON_BUILD', native=str(native),
        input_sha256=before, builder_sha256=script_hashes, compiler=version,
        common_source_sha256=sha(common / 'cmfgen_sub.f'),
        common_object_sha256=sha(common / 'cmfgen_sub.o'), variants=variants,
        comparison='양쪽 actual-nu fixed-J 수정은 동일; 순간 침적열 분기만 다름',
        test_stubs_linked=False, native_original_changed=False, installed=False,
        new_full_native_calculations=0, new_transport=0, released_temperature=False,
        anchor_status='NOT_AVAILABLE', final_goal='NOT_MET',
        elapsed_seconds=time.monotonic() - started)
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.native, args.out), ensure_ascii=False))
