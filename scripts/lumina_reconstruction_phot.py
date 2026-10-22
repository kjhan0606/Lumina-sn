#!/usr/bin/env python3
"""재구성 편차의 실제 PHOT 가중 영향. 참조 시험 함수이며 자체장/SE/RE가 아니다."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import resource
import time

import numpy as np

import compare_lumina_checkpoint_rates as P
import lumina_native90_rates as R
import lumina_native_columns as N
import lumina_reconstruction_native_sample as S

NODES=np.array([0,15,30,45,60,75,89],dtype='i4')
PREPARATION=Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_reuse_20261021-uTVbnT/boundary/preparation.json')
PREPARATION_SHA='1fad50444de5db7a583befd8d59ae7970c5ebf63c3cd423273c341ddb21582f2'
PROOF=Path('/gpfs/kjhan/cmfgen_runs/lumina_reconstruction_detail_20261022-JpUGVS/native-sample.json')
PROOF_SHA='53aadec09a79aecda7b9297375bb0124bdacd0fbae71eafdccb6bc5e0d89cefa'


def select_nodes(state,nodes):
    """전체 FL/SL·terminal 위상은 유지하고 원래점 축만 선택한다. 평균은 없다."""
    nodes=np.asarray(nodes)
    ns=len(state['temperature'])
    if nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or not len(nodes) or \
       len(np.unique(nodes))!=len(nodes) or np.any(nodes<0) or np.any(nodes>=ns):
        raise ValueError('중복 없는 원래점 정수 목록 필요')
    result=dict(state)
    for name in ('temperature','ne'):result[name]=state[name][nodes].copy()
    result['physical']=state['physical'][nodes].copy()
    for name in ('full','superlevel','occupation'):result[name]=state[name][:,nodes].copy()
    return result


def error_fields(reference,original,candidate):
    return dict(reference=reference,original=original,candidate=candidate,
        abs_original_error=abs(original-reference),abs_candidate_error=abs(candidate-reference))


def run(root):
    root=Path(root).resolve();output=root/'phot-impact.json'
    if output.exists() or (root/'phot-impact.started').exists():raise FileExistsError('기존 PHOT 영향 대조 보존')
    with (root/'phot-impact.started').open('x') as f:f.write(str(time.time())+'\n')
    started=time.monotonic();deadline=started+1700.
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('PHOT 영향 대조 내부1700초')
    if S.sha(S.CONFIG)!=S.CONFIG_SHA or S.sha(PROOF)!=PROOF_SHA:raise ValueError('원 config/재구성 증거 변경')
    proof=json.loads(PROOF.read_text());library=PROOF.with_name('probe.so')
    if S.sha(library)!=proof['library_sha256']:raise ValueError('실제 시험 CPU 재구성 바이너리 변경')
    for path,digest in proof['source_sha256'].items():
        if S.sha(path)!=digest:raise ValueError('재구성 검증 소스 변경')
    c=json.loads(S.CONFIG.read_text());plan_path=Path(c['plan'])
    if S.sha(plan_path)!=c['plan_sha256']:raise ValueError('원 계획 변경')
    plan=json.loads(plan_path.read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if S.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:raise ValueError('계수 보고서 변경')
    report=json.loads(report_path.read_text())
    code=[Path(__file__),Path(P.__file__),Path(R.__file__),Path(R.B.__file__),Path(N.__file__)]
    identities={str(p):S.sha(p) for p in code}
    fun=ct.CDLL(str(library)).reconstruction_native_sample_probe
    a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,ct.c_int,i,a,a,a,i];fun.restype=ct.c_int
    reference_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if S.sha(reference_path)!=plan['reference_field_sha256']:raise ValueError('원 CMFGEN J 변경')
    geometry,records=R.reference_fields(reference_path)
    try:
        with N.open_sealed_native(S.CONFIG,PREPARATION,PREPARATION_SHA) as source:
            rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,report)
            if not np.array_equal(geometry[0]*1e10,source.radius):raise ValueError('원래점 반경 불일치')
            state=select_nodes(state,NODES)
            nu=source.frequency;lognu=np.log(nu)
            refnu=np.asarray(records['freq'][::-1])*1e15;reflog=np.log(refnu)
            refj=np.ascontiguousarray(records['field'][::-1,0,:][:,NODES].T)
            if not (nu[0]<=refnu[0]<refnu[-1]<=nu[-1]):raise ValueError('참조 대역 밖 보간 금지')
            midpoint=np.exp(.5*(lognu[:-1]+lognu[1:]))
            columns=np.ascontiguousarray(np.flatnonzero((midpoint>refnu[0])&(midpoint<refnu[-1])),dtype='i4')
            frequency=np.r_[refnu[0],midpoint[columns],refnu[-1]]
            log_frequency=np.log(frequency)
            original=np.empty((len(NODES),len(frequency)));candidate=np.empty_like(original)
            limited=0
            for node,j in enumerate(refj):
                values=np.ascontiguousarray(np.interp(lognu,reflog,j))
                original[node,[0,-1]]=candidate[node,[0,-1]]=j[[0,-1]]
                for begin in range(0,len(columns),8192):
                    check();index=columns[begin:begin+8192];count=len(index)
                    old=np.empty(count);new=np.empty(count);flags=np.zeros(count,dtype='i4')
                    if fun(len(nu),count,index,values,old,new,flags) or np.any(flags&3):
                        raise ValueError('전대역 재구성 실패: native_node='+str(NODES[node]))
                    original[node,begin+1:begin+count+1]=old;candidate[node,begin+1:begin+count+1]=new
                    limited+=int(np.sum((flags&4)!=0))
            print(json.dumps(dict(phase='RECONSTRUCTION_FINISHED',frequencies=len(frequency),seconds=time.monotonic()-started)),flush=True)
            def field_block(sl):
                check()
                reference=np.stack([np.interp(log_frequency[sl],reflog,j) for j in refj])
                return error_fields(reference,original[:,sl],candidate[:,sl])
            peaks={}
            def attribution(key,sl,bound,total):
                if key not in peaks:peaks[key]={name:[None]*len(NODES) for name in ('original','candidate')}
                for name in peaks[key]:
                    for node,value in enumerate(total['abs_'+name+'_error']['photo']):
                        old=peaks[key][name][node]
                        if old is None or value>old['absolute_photo_error_cm3_s']:
                            peaks[key][name][node]=dict(absolute_photo_error_cm3_s=float(value),
                                frequency_Hz=[float(frequency[sl.start]),float(frequency[sl.stop-1])],
                                wavelength_A=[float(R.B.C*1e8/frequency[sl.stop-1]),float(R.B.C*1e8/frequency[sl.start])],
                                points=sl.stop-sl.start)
            results=[]
            for key,rates in P.ion_rates_stream(frequency,field_block,routes,rows,models,state,do_dis,
                    targets=P.TARGETS,integration_block_bins=8192,block_report=attribution):
                for node,native in enumerate(NODES):
                    row=dict(Z=key[0],stage=key[1],native_node=int(native),radius_cm=float(source.radius[native]),
                        routes=rates['routes'],photo_reference_cm3_s=float(rates['total']['reference']['photo'][node]))
                    for name in ('original','candidate'):
                        actual=rates['total'][name];reference=rates['total']['reference']
                        photo=float(actual['photo'][node]);ref=float(reference['photo'][node])
                        error=float(rates['total']['abs_'+name+'_error']['photo'][node])
                        row[name]=dict(photo_cm3_s=photo,photo_relative_bias=P._ratio(photo-ref,ref),
                            photo_absolute_weighted_error_cm3_s=error,photo_absolute_weighted_error_relative=P._ratio(error,ref),
                            stimulated_cm3_s=float(actual['stimulated'][node]),spontaneous_cm3_s=float(actual['spontaneous'][node]),
                            bound_photo_cm3_s=float(rates['bound'][name]['photo'][node]),
                            dominant_error_block=peaks[key][name][node])
                    results.append(row)
                print(json.dumps(dict(phase='ION_FINISHED',Z=key[0],stage=key[1],seconds=time.monotonic()-started)),flush=True)
    finally:records._mmap.close()
    check()
    if any(S.sha(p)!=h for p,h in identities.items()):raise ValueError('수행 중 생산 코드 변경')
    result=dict(status='REFERENCE_RECONSTRUCTION_PHOT_IMPACT_NOT_OWN_FIELD',rows=results,
        native_nodes=NODES.tolist(),frequencies=len(frequency),original_limited=limited,
        input_sha256={str(S.CONFIG):S.CONFIG_SHA,str(plan_path):c['plan_sha256'],str(PROOF):PROOF_SHA,
            str(PREPARATION):PREPARATION_SHA,str(report_path):c['reuse_sources']['boundary_sources']['report_sha256'],
            str(reference_path):plan['reference_field_sha256']},source_sha256=identities,
        full_population=full_check,phot_preprocessing=smoothing,do_lev_dissolution=do_dis,
        physical_scope='고정 물질70의 실제 전 route/용해/FL 인구, 원래점7개. 참조 J의 재구성 영향이며 자체장·SE_RE·이온분포 수용이 아니다.',
        quadrature='동일 반bin 주파수 전대역+원참조 양끝점에서 기존 생산 사다리꼴 PHOT 가중치. 원 CMFGEN 내부 적분 정답은 아니다.',
        boundary='적분은 참조대역 안이며 양끝은 원값. stencil에 필요한 참조대역 밖 입력은 끝값 상수 연장한 시험 함수다.',
        absolute_error='비음수 abs(delta J)를 같은 실제 chi_abs로 수축. 이 보조 배열의 spontaneous/reverse는 물리 오차로 사용하지 않는다.',
        elapsed_seconds=time.monotonic()-started,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        GPU_seconds=0,actual_model_F_calls=0,SE_RE_updates=0,final_goal='NOT_MET')
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();run(args.root)
