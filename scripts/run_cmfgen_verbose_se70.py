#!/usr/bin/env python3
"""물질70/J70의 충돌·연속항 SE를 원본 보존 복제본에서 한 번 출력한다."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from configure_cmfgen_stageA import set_control
from prepare_cmfgen_fixed_field_owner import point_records, record_bytes
from ptf_gated_metrics import read_dimensions
from run_cmfgen_field_feedback import clone_material, verify_material
from run_cmfgen_fixed_field import output_bytes, scan
from run_cmfgen_lambda_probe import control, sha, stop_child


def run(original, source70, reference71, native, build, root):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 작업만 허용')
    if root.exists():
        raise ValueError('기존 실행 root 보존')
    state70, identity70, _ = verify_material(source70, original, 70)
    state71, identity71, _ = verify_material(reference71, original, 71)
    runtime71 = json.loads((reference71.parent/'seq_logs/feedback-runtime.json').read_text())
    inputs71 = json.loads((reference71.parent/'seq_logs/feedback-inputs.json').read_text())
    if (state70.shape != state71.shape or
            runtime71['radiation_produced_from_record'] != 70 or
            Path(inputs71['source']) != source70):
        raise ValueError('물질70→71 계보 불일치')
    execution = json.loads((reference71/'seq_logs/material-execution.json').read_text())
    executable = build/'cmfgen_se_witness.exe'
    if (Path(execution['actual_executable']['path']) != executable or
            sha(executable) != execution['actual_executable']['sha256']):
        raise ValueError('원71 실행 바이너리 불일치')
    field = reference71.parent/'field'
    for name in ('EDDFACTOR', 'EDDFACTOR_INFO'):
        if sha(field/name) != runtime71['field_evidence_sha256'][name]:
            raise ValueError('원71 복사장 EDD 불일치: '+name)
    root.mkdir()
    (root/'seq_logs').mkdir()
    step = root/'material'
    clone_material(original, source70, step, native, True, 70, '1.0D6')
    for name in ('EDDFACTOR', 'EDDFACTOR_INFO'):
        shutil.copy2(field/name, step/name)
    vadat = (step/'VADAT').read_text()
    if control(vadat, 'WRITE_RATES') != 'F':
        raise ValueError('대량 RATE 출력 금지')
    vadat = set_control(vadat, 'VERBOSE_OUT', 'T', append=True)
    (step/'VADAT').write_text(vadat)
    inputs = dict(scope=__doc__, original=str(original), source70=str(source70),
                  reference71=str(reference71), input_identity70=identity70,
                  reference_identity71=identity71,
                  input_record=70, radiation_produced_from_record=70,
                  changed_control='VERBOSE_OUT=T only versus original 70→71 material replay',
                  unchanged_controls={name: sha(step/name) for name in ('IN_ITS',)},
                  stage_controls_sha256={name: sha(step/name) for name in ('VADAT', 'IN_ITS')},
                  fixed_field_sha256={name: sha(step/name) for name in ('EDDFACTOR', 'EDDFACTOR_INFO')},
                  executable=dict(path=str(executable), sha256=sha(executable)),
                  resources='grammar normal CPU16/OMP16 32GiB GPU0 2h; child 100min; retry0',
                  file_cap_bytes=8*1024**3, root_cap_bytes=16*1024**3,
                  physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')
    (root/'seq_logs/verbose-se-inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2)+'\n')
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    started = time.monotonic()
    deadline = started+6000
    cursors = {}
    reason = None
    env = dict(os.environ, LUMINA_NATIVE_SE_WITNESS='1', LUMINA_FORMAL_PRIMARY='',
               LUMINA_FORMAL_PRIMARY_WHOLE='', LUMINA_WHOLE_MODE='',
               LUMINA_MOM_CAPTURE_MODE='', LUMINA_FORMAL_CAPTURE_MODE='')
    with (step/'seq_logs/cmfgen.stdout').open('xb') as out, (step/'seq_logs/cmfgen.stderr').open('xb') as err:
        proc = subprocess.Popen(['srun', '--cpu-bind=cores', str(executable)], cwd=step,
                                env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=err, start_new_session=True)
        while proc.poll() is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = 'child_6000s'
                stop_child(proc)
                break
            try:
                proc.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                reason = scan(step, cursors)
                if output_bytes(step) > 16*1024**3:
                    reason = 'root_16GiB'
                if reason:
                    stop_child(proc)
                    break
    reason = reason or scan(step, cursors)
    matching_record = False
    matching_population = False
    if proc.returncode == 0 and not reason:
        nd, nt = read_dimensions(step)
        a_header, a_storage, a_population = record_bytes(step/'SCRTEMP', 71, nd, nt)
        r_header, r_storage, r_population = record_bytes(reference71/'SCRTEMP', 71, nd, nt)
        matching_record = a_header == r_header and a_storage == r_storage
        matching_population = a_population.tobytes() == r_population.tobytes()
    witness = 'NATIVE_SE_BOUND_000071.bin'
    report = dict(returncode=proc.returncode, stop_reason=reason,
                  elapsed_seconds=time.monotonic()-started,
                  point_records=point_records(step),
                  record71_storage_equal=matching_record,
                  record71_population_equal=matching_population,
                  witness_equal=(step/witness).is_file() and
                                sha(step/witness) == sha(reference71/witness),
                  diagnostic_file=str(step/'fort.19'),
                  diagnostic_file_exists=(step/'fort.19').is_file(),
                  raw_source_unchanged=all(sha(step/name) == expected for name, expected in
                                           inputs['fixed_field_sha256'].items()),
                  final_goal='NOT_JUDGED')
    report['reproduced'] = (report['returncode'] == 0 and not reason and
                            report['record71_storage_equal'] and
                            report['record71_population_equal'] and
                            report['witness_equal'] and
                            report['diagnostic_file_exists'] and
                            report['raw_source_unchanged'] and
                            report['point_records'] == dict(POINT1=71, POINT2=71))
    (root/'seq_logs/verbose-se-runtime.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['reproduced'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original', 'source70', 'reference71', 'native', 'build', 'root'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(*(getattr(args, key).resolve() for key in
                           ('original', 'source70', 'reference71', 'native', 'build', 'root'))))
