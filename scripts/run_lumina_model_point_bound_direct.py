#!/usr/bin/env python3
"""완성 CMFGEN 직접 J의 원49점 BF 생산 입력을 원 runner에 결박한다."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cmfgen_model_continuum as B
from run_lumina_model_point_bound import run as run_bound


def run(args):
    projection=Path(args.field_projection)
    seal=json.loads(projection.read_text())
    if (seal.get('schema')!='lumina-cmfgen-direct-native90-j-v1' or
        seal.get('status')!='FIELD_PROJECTED_NOT_SELF_CONSISTENT' or
        seal.get('material_record')!=70 or seal.get('native_shells')!=90 or
        seal.get('native_frequencies')!=1115020 or
        seal.get('direct_frequencies')!=197291 or
        seal.get('extrapolated_frequency_count')!=0):
        raise ValueError('동일 물질70의 완료된 직접 J 보간 결과가 아님')
    j_path=Path(seal['J_path'])
    j_sha=B.sha(j_path)
    if j_sha!=seal['J_sha256']:
        raise ValueError('직접 J byte 신원 변경')
    helper_sha=B.sha(Path(__file__))
    result=run_bound(SimpleNamespace(base=args.base,j=str(j_path),library=args.library,
                                      out=args.out,wall_seconds=7200))
    final_j_sha=B.sha(j_path)
    attestation=dict(schema='lumina-bound-direct-input-v1',status='COMPLETE_INPUT_SEALED'
                     if result==0 and final_j_sha==j_sha else 'INCOMPLETE',
                     projection_path=str(projection.resolve()),projection_sha256=B.sha(projection),
                     J_path=str(j_path.resolve()),J_sha256_before=j_sha,J_sha256_after=final_j_sha,
                     bound_result_sha256=B.sha(Path(args.out)/'result.json'),
                     helper_sha256=helper_sha,base_path=str(Path(args.base).resolve()),
                     library_sha256=B.sha(Path(args.library)),
                     material_updates=0,SE_RE_solution=False,final_goal='NOT_MET')
    (Path(args.out)/'radiation-input.json').write_text(json.dumps(attestation,indent=2)+'\n')
    return 0 if attestation['status']=='COMPLETE_INPUT_SEALED' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True)
    p.add_argument('--field-projection',required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--out',required=True)
    raise SystemExit(run(p.parse_args()))
