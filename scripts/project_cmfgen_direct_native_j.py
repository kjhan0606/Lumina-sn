#!/usr/bin/env python3
"""완성 CMFGEN 직접장 J를 같은 물질70의 native90×자체 ν 격자에 보간한다.

원 CMFGEN 197291점과 자체 1115020점의 양끝은 같아야 하며 logν 선형
보간만 허용한다. 새 수송·물질·SE/RE 계산은 하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import compare_cmfgen_direct_native_rates as D
import lumina_native_columns as N
import run_cmfgen_formal_primary as F


def project(base_path,direct_root,out):
    started=time.monotonic()
    base_path,direct_root,out=map(Path,(base_path,direct_root,out))
    if out.exists():raise FileExistsError('기존 직접장 보간 출력 보존')
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='1' or
        os.environ.get('SLURM_MEM_PER_NODE')!='4096'):
        raise ValueError('grammar Slurm CPU1/4GiB 필요')
    code_paths=[Path(m.__file__) for m in (B,D,N,F)]+[Path(__file__)]
    code_sha={str(p):B.sha(p) for p in code_paths}
    if B.sha(base_path)!=D.BASE_SHA256:raise ValueError('다른 물질70 계획')
    base=json.loads(base_path.read_text())
    config_path=Path(base['config']);config=json.loads(config_path.read_text())
    plan_path=Path(config['plan'])
    if B.sha(plan_path)!=config['plan_sha256']:raise ValueError('다른 봉인 계획')
    plan=json.loads(plan_path.read_text())
    inputs_path=direct_root/'run/direct-inputs.json'
    runtime_path=direct_root/'run/direct-runtime.json'
    inputs=json.loads(inputs_path.read_text());runtime=json.loads(runtime_path.read_text())
    D.verify_direct(inputs,runtime,plan)
    direct_path=direct_root/'run/field/FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(direct_path)!=runtime['evidence_sha256'].get('FORMAL_PRIMARY_FIELDS.bin'):
        raise ValueError('완성 직접장 byte 신원 변경')
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        if source.identity!=base['source_identity'] or (source.ns,source.nf)!=(90,1115020):
            raise ValueError('다른 native90 물질/격자')
        geometry,reference=F.fields(direct_path)
        try:
            if not np.array_equal(geometry[0]*1.e10,source.radius):
                raise ValueError('직접장과 자체 원90점 반경 불일치')
            native_nu=source.frequency
            direct_nu=np.asarray(reference['freq'][::-1])*1.e15
            if (len(direct_nu)!=197291 or not np.isfinite(direct_nu).all() or
                np.any(np.diff(direct_nu)<=0.) or
                direct_nu[0]!=native_nu[0] or direct_nu[-1]!=native_nu[-1]):
                raise ValueError('양끝이 다른 격자: 외삽 금지')
            log_native=np.log(native_nu);log_direct=np.log(direct_nu)
            out.mkdir()
            path=out/'J_direct_native.npy'
            projected=np.lib.format.open_memmap(path,mode='w+',dtype='<f8',shape=(source.ns,source.nf))
            for shell in range(source.ns):
                row=np.asarray(reference['field'][::-1,0,shell])
                if not np.isfinite(row).all() or np.any(row<0.):
                    raise ValueError('직접장 J 비유한/음수')
                values=np.interp(log_native,log_direct,row)
                if not np.isfinite(values).all() or np.any(values<0.):
                    raise ValueError('보간 J 비유한/음수')
                projected[shell]=values
            projected.flush();del projected
            if any(B.sha(p)!=code_sha[str(p)] for p in code_paths):
                raise ValueError('보간 중 생산 코드 변경')
            result=dict(schema='lumina-cmfgen-direct-native90-j-v1',status='FIELD_PROJECTED_NOT_SELF_CONSISTENT',
                material_record=70,native_shells=source.ns,native_frequencies=source.nf,
                direct_frequencies=len(direct_nu),interpolation='linear_in_log_frequency',
                extrapolated_frequency_count=0,direct_field_sha256=runtime['evidence_sha256']['FORMAL_PRIMARY_FIELDS.bin'],
                source_identity=dict(source.identity),code_sha256=code_sha,
                input_sha256={str(p):B.sha(p) for p in (base_path,config_path,plan_path,inputs_path,runtime_path)},
                J_path=str(path.resolve()),J_sha256=B.sha(path),
                material_updates=0,RT_executions=0,SE_RE_solution=False,
                wall_seconds=time.monotonic()-started,final_goal='NOT_MET')
            (out/'projection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ('status','native_shells','native_frequencies',
                                                   'direct_frequencies','J_sha256','wall_seconds')}),flush=True)
            return result
        finally:
            reference._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--direct-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    project(args.base,args.direct_root,args.out)
