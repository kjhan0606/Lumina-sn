#!/usr/bin/env python3
"""완료된 물질70 직접장·기존 참조장·자체 J_A의 Si/S BF율을 같은 원래점에서 비교한다.

기존 PHOT/Milne/용해 생산 적분을 재사용한다. 고정 물질의 율이며 새 SE/RE 해가 아니다.
"""
import argparse
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np

import cmfgen_model_continuum as B
import compare_lumina_checkpoint_rates as C
from compare_lumina_checkpoint_rates import TARGETS, ion_rates_stream, production_frequency, _ratio
import lumina_native90_rates as R
import lumina_native_columns as N
import run_cmfgen_formal_primary as F
from run_cmfgen_formal_primary import fields


BASE_SHA256='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab'


def verify_direct(inputs,runtime,plan):
    """미완성/다른 물질의 직접장을 비교 입력으로 승격하지 않는다."""
    if (inputs.get('schema')!='lumina-direct-continuum-reference-v1' or
            inputs.get('input_state_record')!=70 or
            inputs.get('next_native_iteration')!=71 or
            inputs.get('expected_depths')!=90 or
            inputs.get('expected_frequencies')!=197291 or
            inputs.get('changed_controls')!={'CROSS':'F -> T'} or
            inputs.get('baseline_field_sha256')!=plan['reference_field_sha256'] or
            runtime.get('state')!='FIELD_COMPLETE' or
            runtime.get('field_complete') is not True or
            runtime.get('coverage',{}).get('count')!=197291 or
            runtime.get('execution',{}).get('returncode')!=97 or
            runtime.get('execution',{}).get('stop_reason') is not None or
            runtime.get('material_byte_unchanged') is not True or
            runtime.get('geometry_and_frequency_unchanged') is not True):
        raise ValueError('동일 물질70의 완성된 직접 CMFGEN 장이 아님')


def direct_radiation_block(frequency,reference_nu,original,direct,own_nu,own,
                           reference_log,own_log):
    """두 CMFGEN 장에는 같은 logν 보간을, 자체 생산점에는 정확한 값을 쓴다."""
    result=R.radiation_block(frequency,reference_nu,original,own_nu,{'J_A':own},
                             reference_log=reference_log,own_log=own_log)
    logs=np.log(frequency)
    result['CMFGEN_DIRECT']=np.stack([np.interp(logs,reference_log,row) for row in direct])
    return result


def compare(base_path,direct_root,resident_path,out):
    started=time.monotonic()
    base_path,direct_root,resident_path,out=map(Path,(base_path,direct_root,resident_path,out))
    if out.exists():raise FileExistsError('기존 직접장 율 비교 보존')
    code_paths=[Path(m.__file__) for m in (B,C,R,N,F)] + [Path(__file__)]
    code_sha256={str(path):B.sha(path) for path in code_paths}
    if B.sha(base_path)!=BASE_SHA256:raise ValueError('다른 native90 계획')
    base=json.loads(base_path.read_text());config_path=Path(base['config'])
    c=json.loads(config_path.read_text());plan_path=Path(c['plan'])
    if B.sha(plan_path)!=c['plan_sha256']:raise ValueError('봉인 물질 계획 변경')
    plan=json.loads(plan_path.read_text())
    if plan['reference_radiation_material_record']!=70 or plan['reference_radiation_record']!=71:
        raise ValueError('다른 물질/참조 세대')
    input_path=direct_root/'run/direct-inputs.json'
    runtime_path=direct_root/'run/direct-runtime.json'
    inputs=json.loads(input_path.read_text());runtime=json.loads(runtime_path.read_text())
    verify_direct(inputs,runtime,plan)
    original_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    direct_path=direct_root/'run/field/FORMAL_PRIMARY_FIELDS.bin'
    resident=json.loads(resident_path.read_text());own_path=Path(resident['J']['path'])
    if (resident.get('status')!='COMPLETE_TRANSPORT_NOT_PHYSICAL_ACCEPTANCE' or
            resident['source_identity']!=base['source_identity'] or
            B.sha(Path(inputs['source'])/'SCRTEMP')!=inputs['input_restart_sha256']['SCRTEMP'] or
            B.sha(original_path)!=plan['reference_field_sha256'] or
            B.sha(direct_path)!=runtime['evidence_sha256'].get('FORMAL_PRIMARY_FIELDS.bin') or
            B.sha(own_path)!=resident['J']['sha256']):
        raise ValueError('세 복사장/물질 신원 불일치')
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        if source.identity!=base['source_identity']:raise ValueError('native90 계수 세대 변경')
        g0,a0=fields(original_path);g1,a1=fields(direct_path)
        own=np.load(own_path,mmap_mode='r',allow_pickle=False)
        try:
            if (not np.array_equal(g0,g1) or not np.array_equal(a0['freq'],a1['freq']) or
                    not np.array_equal(g0[0]*1e10,source.radius) or
                    own.shape!=(90,source.nf) or own.dtype!=np.dtype('f8')):
                raise ValueError('직접장/원 참조/native 좌표·주파수·차원 불일치')
            reference_nu=np.asarray(a0['freq'][::-1])*1e15
            original=np.ascontiguousarray(a0['field'][::-1,0,:].T)
            direct=np.ascontiguousarray(a1['field'][::-1,0,:].T)
            report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
            if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
                raise ValueError('native90 물리 계수 보고서 변경')
            report=json.loads(report_path.read_text())
            rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,report)
            frequency=production_frequency(source.frequency,reference_nu)
            reference_log,own_log=np.log(reference_nu),np.log(source.frequency)
            def block(sl):
                return direct_radiation_block(frequency[sl],reference_nu,original,direct,
                                              source.frequency,own,reference_log,own_log)
            output=[];completed=[]
            for (Z,stage),result in ion_rates_stream(frequency,block,routes,rows,models,state,do_dis):
                members=[i for i,row in enumerate(rows) if
                         (int(row['atomic_number']),int(row['ion_number'])+1)==(Z,stage)]
                population=state['full'][members].sum(axis=0)
                for node in range(90):
                    record=dict(Z=Z,stage=stage,native_node=node,radius_cm=float(source.radius[node]),
                                lower_ion_population_cm3=float(population[node]),routes=result['routes'])
                    for name in ('CMFGEN','CMFGEN_DIRECT','J_A'):
                        for scope in ('bound','total'):
                            rates=result[scope][name]
                            for process in ('photo','spontaneous','stimulated','reverse'):
                                record[f'{name}_{scope}_{process}_cm3_s']=float(rates[process][node])
                    for name,denominator in (('direct_over_original','CMFGEN'),
                                             ('lumina_over_direct','CMFGEN_DIRECT')):
                        numerator='CMFGEN_DIRECT' if name=='direct_over_original' else 'J_A'
                        for process in ('photo','reverse'):
                            record[f'{name}_{process}']=_ratio(
                                result['total'][numerator][process][node],
                                result['total'][denominator][process][node])
                    output.append(record)
                completed.append((Z,stage))
                print(json.dumps(dict(status='DIRECT_ION_RATES',Z=Z,stage=stage,
                                      elapsed_seconds=time.monotonic()-started)),flush=True)
            if tuple(completed)!=TARGETS or len(output)!=90*len(TARGETS):
                raise ValueError('Si/S 네 이온 또는90점율 누락')
            if any(B.sha(path)!=code_sha256[str(path)] for path in code_paths):
                raise ValueError('율 적분 도중 생산 코드 변경')
            out.mkdir()
            with (out/'rates.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(output[0]))
                writer.writeheader();writer.writerows(output)
            result=dict(schema='lumina-direct-m70-native90-rates-v1',
                status='FIXED_MATERIAL_RATE_COMPARISON_NOT_SE',material_record=70,
                native_nodes=90,targets=list(TARGETS),frequencies=len(frequency),
                fields=['CMFGEN','CMFGEN_DIRECT','J_A'],
                integration='동일 원래점의 전 생산 ν와 참조 양끝점; 원 PHOT/Milne/용해, 전역 사다리꼴',
                full_population=full_check,do_lev_dissolution=do_dis,phot_preprocessing=smoothing,
                input_sha256={str(p):B.sha(p) for p in (base_path,config_path,plan_path,
                    input_path,runtime_path,resident_path)},code_sha256=code_sha256,
                field_sha256={'CMFGEN':plan['reference_field_sha256'],
                    'CMFGEN_DIRECT':runtime['evidence_sha256']['FORMAL_PRIMARY_FIELDS.bin'],
                    'J_A':resident['J']['sha256']},
                rates_csv_sha256=B.sha(out/'rates.csv'),
                elapsed_seconds=time.monotonic()-started,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physical_RT_executions=0,SE_RE_solved=False,spectrum_compared=False,final_goal='NOT_MET')
            (out/'rates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            return result
        finally:
            own._mmap.close();a0._mmap.close();a1._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','direct-root','resident','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(compare(a.base,a.direct_root,a.resident,a.out),ensure_ascii=False))
