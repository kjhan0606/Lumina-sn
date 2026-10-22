#!/usr/bin/env python3
"""기존 물질70을 새 격리 장에 복제하여 1450Å 실제 CMFGEN formal 광선을 포획한다.

원 장/물질은 read-only 입력이며 새 native 갱신이나 SE/RE 반복은 없다.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from prepare_cmfgen_mom_capture import sha
from run_cmfgen_field_feedback import clone_material
from run_cmfgen_lambda_probe import stop_child


EXPECTED_ORIGINAL_FIELD = 'd9073ac830d8a70beffc731dc919140b2b97ccff7e8d960060d42a7f7ca60563'
PACKET_PREFIXES = ('FORMAL_60714_', 'FORMAL_60715_')


def run(previous, build, root):
    if root.exists():
        raise FileExistsError('기존 포획 실행 보존')
    if os.environ.get('OMP_NUM_THREADS') != '16':
        raise ValueError('CMFGEN formal은 OMP16으로 실행')
    prior_inputs = json.loads((previous/'seq_logs/feedback-inputs.json').read_text())
    prior_runtime = json.loads((previous/'seq_logs/feedback-runtime.json').read_text())
    if not prior_runtime['field_complete'] or prior_runtime['field_evidence_sha256'][
        'FORMAL_PRIMARY_FIELDS.bin'] != EXPECTED_ORIGINAL_FIELD:
        raise ValueError('완료된 물질70/J71 기준 장이 아님')
    source = Path(prior_inputs['source'])
    original = Path(prior_inputs['original'])
    native = Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
    if prior_inputs['source_state']['record'] != 70 or \
       source != Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material'):
        raise ValueError('기준 물질70 선택 오류')
    for name, digest in prior_inputs['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('기준 물질 또는 계약 신원 변경: ' + name)
    bm = json.loads((build/'center-packet-build.json').read_text())
    if bm['outer'] != 71 or bm['frequency_indices'] != [60714, 60715] or \
       not bm['formal_kernel_unchanged'] or bm['pre_population_stop'] != 97:
        raise ValueError('1450Å formal 전용 빌드가 아님')
    for name, digest in {**bm['input_sha256'], **bm['output_sha256']}.items():
        if sha(Path(name)) != digest:
            raise ValueError('포획 빌드 신원 변경: ' + name)
    start = time.monotonic()
    root.mkdir(parents=False)
    field = root/'field'
    report = dict(scope=__doc__, original_field=str(previous/'field'), source=str(source),
                  build=str(build), CPU=16, GPU=0, wall_limit_seconds=2400,
                  physical_field_calls=1, native_updates=0, SE_RE_updates=0,
                  baseline_field_sha256=EXPECTED_ORIGINAL_FIELD, final_goal='NOT_MET')
    try:
        stage = clone_material(original, source, field, native, False, 70, '1.0D6')
        old_field = previous/'field'
        for name in ('VADAT', 'IN_ITS', 'SCRTEMP', 'POINT1', 'POINT2', 'RVTJ'):
            if sha(field/name) != sha(old_field/name):
                raise ValueError('기존 물질70/J71 장과 입력 불일치: ' + name)
        report['staged_restart_sha256'] = stage['restart_sha256']
        env = dict(os.environ, LUMINA_FORMAL_PRIMARY='1', LUMINA_FORMAL_PRIMARY_WHOLE='1',
                   LUMINA_FORMAL_CAPTURE_MODE='CAPTURE_FORMAL', LUMINA_MOM_CAPTURE_MODE='',
                   LUMINA_WHOLE_MODE='', LUMINA_FORMAL_CELL_BUDGET='', LUMINA_NATIVE_SE_WITNESS='')
        resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
        with (field/'seq_logs/cmfgen.stdout').open('xb') as stdout, \
             (field/'seq_logs/cmfgen.stderr').open('xb') as stderr:
            process = subprocess.Popen([str(build/'cmfgen_center_packet.exe')], cwd=field,
                                       env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                returncode = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                stop_child(process)
                returncode = process.returncode
                report['timeout'] = True
        report['returncode'] = returncode
        if returncode != 97:
            raise ValueError('CMFGEN formal 전체 sweep가 의도한 STOP97에 이르지 않음')
        done = (field/'FORMAL_PRIMARY_COMPLETE.txt').read_text().splitlines()[0].split()
        if done[:3] != ['71', '197291', '197291']:
            raise ValueError('실제 물질70/J71 주파수 장 불완전')
        report['new_field_sha256'] = sha(field/'FORMAL_PRIMARY_FIELDS.bin')
        report['field_byte_identical'] = report['new_field_sha256'] == EXPECTED_ORIGINAL_FIELD
        if not report['field_byte_identical']:
            raise ValueError('계측 활성화가 원래 완성 formal 장을 바꿈')
        packets = sorted(p for p in field.iterdir() if p.is_file() and p.name.startswith(PACKET_PREFIXES))
        for index in (60714, 60715):
            if not any(p.name.startswith(f'FORMAL_{index}_') and p.name.endswith('_rays_out.bin')
                       for p in packets):
                raise ValueError(f'{index} 실제 ray packet 누락')
        report['packets_sha256'] = {p.name:sha(p) for p in packets}
        report['status'] = 'CAPTURED_WITH_IDENTICAL_COMPLETE_FIELD_NOT_CENTER_J_YET'
    except (OSError, ValueError) as error:
        report['failure'] = str(error)
        report['status'] = 'FAILED_PRESERVE_ROOT'
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        (root/'center-packet-run.json').write_text(json.dumps(report, ensure_ascii=False,
                                                         indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(root=str(root), status=report['status'],
                          elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False))
    return 0 if report['status'].startswith('CAPTURED_') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'build', 'root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(*(getattr(args, name).resolve() for name in ('previous', 'build', 'root'))))
