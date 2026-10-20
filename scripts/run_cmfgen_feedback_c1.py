#!/usr/bin/env python3
"""명시한 실제 물질과 현재 장을 수정 C1으로 전90깊이 비교한다.

기존 입력 준비·C 실행·요약을 연결하는 경량 실행기다. 새 RT/native는 없고
출력 충돌과 실패한 분기는 거부한다. 결과는 공통 입력의 조건부 비교다.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from prepare_cmfgen_feedback_owner import prepare
from summarize_cmfgen_fixed_field_owner import summarize
from run_cmfgen_lambda_probe import sha

BINARY = Path('/gpfs/kjhan/cmfgen_runs/lumina_inverse_bf_20261020-9tBN4q/bench_cmfgen_owner')
BINARY_SHA = '5447bd091db9e47f001a6bb0183d145940fb9ef2de35ff7af7574c6811c4da46'
ATOMIC = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic')


def run(root, parent, record, out, same_field_input=None):
    if out.exists():
        raise ValueError('기존 C1 비교 보존')
    runtime = root/'seq_logs/feedback-runtime.json'
    if runtime.exists():
        value = json.loads(runtime.read_text())
        if (value.get('failure') or not value.get('source_unchanged') or
            (value.get('branch') == '70B' and not value.get('branch_next_permitted'))):
            raise ValueError('실패하거나 보존/분기 증명이 없는 실행')
    if sha(BINARY) != BINARY_SHA:
        raise ValueError('수정 C1 실행파일 신원 불일치')
    env = dict(PATH='/usr/bin:/bin',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        LUMINA_TOPION_LEVELS_FILE=str(ATOMIC/'topion_levels.csv'),
        LUMINA_IONIZATION_REFERENCE_FILE=str(ATOMIC/'ionization_reference.csv'))
    evidence = [BINARY,Path(__file__).resolve(),Path(prepare.__code__.co_filename).resolve(),
                Path(summarize.__code__.co_filename).resolve()]
    evidence.extend(Path(v) for k,v in env.items() if k.endswith('_FILE'))
    before = {str(p):sha(p) for p in evidence}
    prepare(root,parent,record,out,same_field_input)
    response = out/'C_response'
    response.mkdir()
    deck = json.loads((out/'fixed-field-owner-inputs.json').read_text())['deck']
    command = [str(BINARY),deck,str(out/'owner_input.bin'),str(response/'response.csv')]
    with (response/'owner.log').open('xb') as stream:
        subprocess.run(command,cwd=response,env=env,check=True,timeout=180,
                       stdout=stream,stderr=subprocess.STDOUT)
    if any(sha(Path(p)) != h for p,h in before.items()):
        raise ValueError('C1 실행 중 입력/생산자 변경')
    summarize(out,response,BINARY,out/'comparison')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    report = dict(scope=__doc__,record=record,root=str(root),parent_C1=str(parent),
        command=command,environment=env,input_sha256=before,
        output_sha256={n:sha(out/n) for n in ('fixed-field-owner-inputs.json',
            'comparison/fixed-field-owner.json','comparison/fractions.csv')},
        RT_executions=0,native_executions=0,final_goal='NOT_JUDGED')
    with (out/'execution.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(execution=str(out/'execution.json'),
                         sha256=sha(out/'execution.json')),ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--record',type=int,required=True)
    parser.add_argument('--same-field-input',type=Path)
    args = parser.parse_args()
    run(args.root.resolve(),args.parent.resolve(),args.record,args.out.resolve(),
        args.same_field_input.resolve() if args.same_field_input else None)
