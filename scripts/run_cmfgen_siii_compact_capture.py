#!/usr/bin/env python3
"""봉인 물질70/J71의 S III 대역만 광선별 compact 이력으로 포획한다.

원 완성 장과 물질은 읽기 전용이다. 새 root에서 formal sweep 한 번만
수행하며 native 갱신·SE/RE 반복은 하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST
from run_cmfgen_center_packet import EXPECTED_ORIGINAL_FIELD
from run_cmfgen_field_feedback import clone_material
from run_cmfgen_lambda_probe import stop_child


def run(previous, build, root):
    if root.exists():
        raise FileExistsError('기존 실행 root 보존')
    if os.environ.get('OMP_NUM_THREADS')!='16':
        raise ValueError('원 CMFGEN formal OMP16 계약')
    prior_inputs=json.loads((previous/'seq_logs/feedback-inputs.json').read_text())
    prior_runtime=json.loads((previous/'seq_logs/feedback-runtime.json').read_text())
    if (not prior_runtime['field_complete'] or
            prior_runtime['field_evidence_sha256']['FORMAL_PRIMARY_FIELDS.bin']!=EXPECTED_ORIGINAL_FIELD):
        raise ValueError('봉인 물질70/J71 완료 장 불일치')
    source=Path(prior_inputs['source'])
    original=Path(prior_inputs['original'])
    native=Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
    if (prior_inputs['source_state']['record']!=70 or
            source!=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material')):
        raise ValueError('기준 물질70 선택 오류')
    for name,digest in prior_inputs['source_sha256'].items():
        if sha(Path(name))!=digest:
            raise ValueError('봉인 물질 신원 변경: '+name)
    bm=json.loads((build/'siii-compact-build.json').read_text())
    if (bm['outer']!=71 or bm['frequency_indices']!=[FIRST,LAST] or
            not bm['formal_kernel_unchanged'] or bm['pre_population_stop']!=97):
        raise ValueError('S III 압축 포획 빌드 계약 불일치')
    for name,digest in {**bm['input_sha256'],**bm['output_sha256']}.items():
        if sha(Path(name))!=digest:
            raise ValueError('압축 포획 빌드 신원 변경: '+name)
    start=time.monotonic()
    root.mkdir(parents=False)
    field=root/'field'
    report=dict(scope=__doc__,original_field=str(previous/'field'),source=str(source),
                build=str(build),cpu=16,gpu=0,memory_limit_bytes=64*1024**3,
                wall_limit_seconds=3600,compact_limit_bytes=512*1024**2,
                frequency_indices=[FIRST,LAST],physical_field_calls=1,
                native_updates=0,SE_RE_updates=0,
                baseline_field_sha256=EXPECTED_ORIGINAL_FIELD,final_goal='NOT_MET')
    try:
        stage=clone_material(original,source,field,native,False,70,'1.0D6')
        old_field=previous/'field'
        for name in ('VADAT','IN_ITS','SCRTEMP','POINT1','POINT2','RVTJ'):
            if sha(field/name)!=sha(old_field/name):
                raise ValueError('기존 물질70/J71 입력 불일치: '+name)
        report['staged_restart_sha256']=stage['restart_sha256']
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='1',
                 LUMINA_FORMAL_CAPTURE_MODE='CAPTURE_FORMAL',LUMINA_MOM_CAPTURE_MODE='',
                 LUMINA_WHOLE_MODE='',LUMINA_FORMAL_CELL_BUDGET='',
                 LUMINA_NATIVE_SE_WITNESS='')
        with (field/'seq_logs/cmfgen.stdout').open('xb') as stdout, \
             (field/'seq_logs/cmfgen.stderr').open('xb') as stderr:
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
                resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
            process=subprocess.Popen([str(build/'cmfgen_siii_compact.exe')],cwd=field,
                                     env=env,stdin=subprocess.DEVNULL,stdout=stdout,
                                     stderr=stderr,start_new_session=True,preexec_fn=limits)
            try:
                returncode=process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                stop_child(process)
                returncode=process.returncode
                report['timeout']=True
        report['returncode']=returncode
        if returncode!=97:
            raise ValueError('formal 전체 sweep가 의도한 STOP97에 이르지 않음')
        done=(field/'FORMAL_PRIMARY_COMPLETE.txt').read_text().splitlines()[0].split()
        if done[:3]!=['71','197291','197291']:
            raise ValueError('완성 formal 장의 주파수 coverage 불완전')
        report['new_field_sha256']=sha(field/'FORMAL_PRIMARY_FIELDS.bin')
        report['field_byte_identical']=report['new_field_sha256']==EXPECTED_ORIGINAL_FIELD
        if not report['field_byte_identical']:
            raise ValueError('계측으로 원 완성 formal 장이 변경됨')
        geometry=list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
        if len(geometry)!=1:
            raise ValueError('첫 대역 주파수의 단일 기하 누락')
        compact=[field/'siii_compact.bin',field/'siii_blue_in.bin',geometry[0]]
        if any(not p.is_file() for p in compact):
            raise ValueError('압축 광선 자료 누락')
        sizes={p.name:p.stat().st_size for p in compact}
        if sizes['siii_compact.bin']>512*1024**2 or sum(sizes.values())>512*1024**2:
            raise ValueError('압축 광선 저장량 한도 초과')
        report['compact_sizes_bytes']=sizes
        report['compact_sha256']={p.name:sha(p) for p in compact}
        report['status']='CAPTURED_WITH_IDENTICAL_COMPLETE_FIELD_NOT_ANALYZED'
    except (OSError,ValueError) as error:
        report['failure']=str(error)
        report['status']='FAILED_PRESERVE_ROOT'
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        (root/'siii-compact-run.json').write_text(json.dumps(report,ensure_ascii=False,
                                                        indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(root=str(root),status=report['status'],
                          elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False))
    return 0 if report['status'].startswith('CAPTURED_') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','build','root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(*(getattr(args,name).resolve() for name in ('previous','build','root'))))
