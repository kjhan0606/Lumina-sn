#!/usr/bin/env python3
"""전활성 원자모델의 동적 충돌 생산자 실물 확인. 새 SE/RE·수송은 없다."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_collision_source import ModelCollisionSource
from run_lumina_pseudo_candidate_overlay import HEADER


def node_temperatures(path,nlevels):
    """원90점의 물리 열만 읽는다. 셀 평균/새 온도장을 만들지 않는다."""
    with path.open('rb') as stream:
        tag,ns,nc,nl,epoch=struct.unpack('<8sIIId',stream.read(28))
        stride=4 if tag==b'LUMNOD01' else 0
        if stride!=4 or ns!=90 or nl!=nlevels or nc<1 or not np.isfinite(epoch) or epoch<=0:
            raise ValueError('원90점 물질의 형상/epoch 오류')
        stream.seek(12*nc,1)
        physical=np.frombuffer(stream.read(8*stride*ns),dtype='<f8').reshape(ns,stride)
    if (path.stat().st_size!=28+12*nc+8*ns*(stride+nc) or
        not np.isfinite(physical).all() or np.any(physical[:,1:3]<=0)):
        raise ValueError('원90점 물질의 온도/전자밀도/크기 오류')
    return physical[:,1].copy(),physical[:,2].copy()


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096'):
        raise ValueError('CPU1·4GiB Slurm 전용')
    if args.out.exists():raise FileExistsError('기존 전원소 충돌 확인 보존')
    started=time.monotonic()
    rows=list(csv.DictReader(args.levels.open()))
    source=ModelCollisionSource(args.material,rows)
    legacy=Path('/gpfs/kjhan/cmfgen_runs/lumina_fullpair8_20261026-qXSzbE/fullpair.bin')
    legacy_manifest=legacy.with_suffix('.bin.json')
    sealed=json.loads(legacy_manifest.read_text())
    if B.sha(legacy)!=sealed['output_sha256']:raise ValueError('기존 Si/S 전쌍 기준 변경')
    blob=legacy.read_bytes();tag,nl,ns,node,count,temp,ne=HEADER.unpack_from(blob)
    if (tag,nl,ns,node)!=(b'LMCBPTS1',len(rows),90,49):raise ValueError('Si/S 원49점 기준 주소 오류')
    reference=np.frombuffer(blob,dtype=[('lower','<i4'),('upper','<i4'),('omega','<f8')],offset=HEADER.size)
    fixed=[args.levels,legacy,legacy_manifest,Path(__file__),Path(__file__).with_name('lumina_model_collision_source.py'),
        Path(__file__).with_name('build_cmfgen_fullpair_omega.py'),Path(__file__).with_name('cmfgen_model_sources.py'),
        Path(__file__).with_name('build_cmfgen_coldata_all.py'),Path(__file__).with_name('cmfgen_parser.py'),
        Path(__file__).with_name('cmfgen_model_continuum.py'),Path(__file__).with_name('expand_atomic_data_cmfgen.py')]
    if args.material_nodes:fixed.append(args.material_nodes)
    identity={str(p):B.sha(p) for p in fixed};identity.update(source.identity)
    if args.material_nodes:
        if identity[str(args.material_nodes)]!=sealed['input_sha256'][str(args.material_nodes)]:
            raise ValueError('원70 물질 입력 해시 불일치')
        temperatures,densities=node_temperatures(args.material_nodes,len(rows))
        if temperatures[49]!=temp or densities[49]!=ne:
            raise ValueError('원49점과 저장 충돌 기준의 T/ne 불일치')
        reference_column=49
    else:temperatures=np.array([temp*.95,temp,temp*1.05]);reference_column=1
    result=[];legacy_cursor=0
    for Z,stage in sorted(source.models):
        before=time.monotonic();ion=source.load(Z,stage);identity.update(ion.identity)
        counts=dict(TABULATED=0,VR=0,OMEGA_SET=0);pair_count=0;changed=0;maximum_error=0.
        digest=hashlib.sha256();minimum=np.inf;maximum=0.;peak=0
        for lower,upper,omega,kinds in ion.blocks(temperatures,maximum=args.block):
            for key,value in kinds.items():counts[key]+=value
            pair_count+=len(lower);changed+=int(np.count_nonzero(np.any(omega!=omega[0],axis=0)))
            minimum=min(minimum,float(omega.min()));maximum=max(maximum,float(omega.max()));peak=max(peak,int(omega.nbytes))
            digest.update(lower.tobytes());digest.update(upper.tobytes());digest.update(omega.tobytes())
            if Z in (14,16):
                old=reference[legacy_cursor:legacy_cursor+len(lower)]
                if not np.array_equal(old['lower'],lower) or not np.array_equal(old['upper'],upper):
                    raise ValueError('기존 Si/S 순서/전쌍 누락')
                relative=abs(omega[reference_column]/old['omega']-1.)
                maximum_error=max(maximum_error,float(relative.max(initial=0.)))
                if np.any(relative>64*np.finfo(float).eps):
                    raise ValueError('기존 Si/S Ω 불일치')
                legacy_cursor+=len(lower)
        if pair_count!=ion.n*(ion.n-1)//2 or sum(counts.values())!=pair_count:
            raise ValueError('전활성 이온의 FL쌍 누락/추가')
        ion.verify_identity()
        result.append(dict(Z=Z,stage=stage,full_levels=ion.n,pairs=pair_count,branches=counts,
            source_counts=ion.counts,temperature_dependent_pairs=changed,
            minimum_omega=minimum if pair_count else None,maximum_omega=maximum if pair_count else None,
            maximum_output_block_bytes=peak,legacy_SiS_max_relative=maximum_error if Z in (14,16) else None,
            streamed_values_sha256=digest.hexdigest(),wall_seconds=time.monotonic()-before))
        print(json.dumps(dict(Z=Z,stage=stage,pairs=pair_count,seconds=time.monotonic()-before)),flush=True)
    if legacy_cursor!=len(reference):raise ValueError('기존 Si/S 전쌍 기준 소비 누락')
    if any(B.sha(Path(p))!=h for p,h in identity.items()):raise ValueError('동적 충돌 확인 중 입력 변경')
    report=dict(schema='lumina-full-model-dynamic-collision-source-v1',state='COMPLETE_SOURCE_COVERAGE',
        active_ions=len(source.models),full_levels=len(rows),ions=result,input_sha256=identity,
        tested_temperatures_K=temperatures.tolist(),legacy_SiS_pairs=len(reference),
        actual_native_points=len(temperatures) if args.material_nodes else 0,
        evaluated_pair_point_values=sum(r['pairs'] for r in result)*len(temperatures),
        new_SE_RE_solves=0,new_native=0,new_transport=0,public_material_updates=0,
        full_90_point_material_response_complete=False,final_goal='NOT_MET',source_unchanged=True,
        wall_seconds=time.monotonic()-started)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with args.out.open('x') as stream:stream.write(payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=Path,required=True)
    parser.add_argument('--material',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--block',type=int,default=16384)
    parser.add_argument('--material-nodes',type=Path)
    run(parser.parse_args())
