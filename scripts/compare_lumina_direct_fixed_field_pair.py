#!/usr/bin/env python3
"""물질70 원49점의 자체 J_A/직접 CMFGEN J private 후보를 짝비교한다.

두 입력은 BF·BB·충돌의 부분채널 고정장 후보이다. 전체 SE/RE, 새 수송,
스펙트럼의 CMFGEN 동질성 판정에는 사용하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256')
    return digest.hexdigest()


def compare(old_path,direct_path,rates_path,out):
    old_path,direct_path,rates_path,out=map(Path,(old_path,direct_path,rates_path,out))
    if out.exists():raise FileExistsError('기존 짝비교 보존')
    old=json.loads(old_path.read_text());direct=json.loads(direct_path.read_text())
    expected_key={'14:2','14:3','16:2','16:3'}
    for label,record in (('J_A',old),('DIRECT',direct)):
        full=record.get('full_bound_test') or {}
        if (record.get('state')!='PASS' or record.get('final_goal')!='NOT_MET' or
            record.get('public_material_updates')!=0 or record.get('new_transport')!=0 or
            full.get('native_node')!=49 or full.get('frequencies')!=1115020 or
            full.get('full_frequency_C_assembly') is not True or
            full.get('private_partial_channel_SE_candidate') is not True or
            full.get('complete_SE_RE_solution') is not False or
            set(full.get('Si_S_II_III_photo_spont_stim',{}))!=expected_key):
            raise ValueError(label+'는 완성 원49점 부분채널 후보가 아님')
    if (old['input_identity']!=direct['input_identity'] or
        old['point_collision_test']['source_result_sha256']!=
            direct['point_collision_test']['source_result_sha256'] or
        direct.get('fixed_radiation_field')!='CMFGEN_DIRECT_NATIVE90_LOGNU_FIXED_MATERIAL' or
        direct['full_bound_test'].get('radiation_J_sha256')!=direct['fixed_radiation_J_sha256'] or
        direct['full_bound_test'].get('radiation_provenance')!=direct['fixed_radiation_field']):
        raise ValueError('비복사 입력 또는 직접장 신원 불일치')
    with rates_path.open(newline='') as stream:
        rows={(int(row['native_node']),int(row['Z']),int(row['stage'])):row
              for row in csv.DictReader(stream)}
    if len(rows)!=360:raise ValueError('독립 율 비교의 전90점×4이온 누락')
    max_rate_relative=0.
    for key in sorted(expected_key):
        z,stage=map(int,key.split(':'));row=rows[49,z,stage]
        for record,name in ((old,'J_A'),(direct,'CMFGEN_DIRECT')):
            measured=record['full_bound_test']['Si_S_II_III_photo_spont_stim'][key]
            for i,process in enumerate(('photo','spontaneous','stimulated')):
                expected=float(row[f'{name}_bound_{process}_cm3_s'])
                error=abs(measured[i]-expected)/max(abs(expected),1.e-300)
                if not math.isfinite(error):raise ValueError('비유한 율 비교')
                max_rate_relative=max(max_rate_relative,error)
    if max_rate_relative>5.e-11:
        raise ValueError('독립 BF율과 C 소비율이 다름: '+str(max_rate_relative))
    profiles=[]
    for z in ('14','16'):
        a={row['stage0']:row for row in old['full_bound_test']['fixed_field_stage_profile'][z]}
        b={row['stage0']:row for row in direct['full_bound_test']['fixed_field_stage_profile'][z]}
        if set(a)!=set(b):raise ValueError('후보 이온 단계 범위 불일치')
        for stage in sorted(a):
            original=a[stage]['original'];old_value=a[stage]['candidate']
            new_original=b[stage]['original'];new_value=b[stage]['candidate']
            if not math.isclose(original,new_original,rel_tol=1.e-12,abs_tol=0.):
                raise ValueError('원 CMFGEN 물질 개체수 불일치')
            if stage not in (1,2,3):continue
            profiles.append(dict(Z=int(z),stage0=stage,original_cm3=original,
                J_A_candidate_cm3=old_value,direct_J_candidate_cm3=new_value,
                J_A_over_original=old_value/original,
                direct_J_over_original=new_value/original,
                log10_error_J_A=abs(math.log10(old_value/original)),
                log10_error_direct_J=abs(math.log10(new_value/original))))
    report=dict(schema='lumina-direct-J-fixed-field-pair-v1',
        status='PARTIAL_CHANNEL_FIXED_FIELD_COMPARISON_NOT_PHYSICAL_EQUIVALENCE',
        material_record=70,native_node=49,changed_input='radiation J for BF and BB only',
        profile=profiles,independent_BF_rate_max_relative=max_rate_relative,
        input_sha256={str(p.resolve()):sha(p) for p in (old_path,direct_path,rates_path)},
        direct_J_sha256=direct['fixed_radiation_J_sha256'],
        omitted_channels='pseudo and other unassembled processes',
        SE_charge_RE_transport_spectrum_complete=False,final_goal='NOT_MET')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],max_rate_relative=max_rate_relative,
        profile=profiles),ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True);p.add_argument('--direct',required=True)
    p.add_argument('--rates',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare(a.old,a.direct,a.rates,a.out)
