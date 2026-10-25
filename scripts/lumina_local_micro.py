#!/usr/bin/env python3
"""승인된 실제 국소 시험: J=0, 최대2구간/512출력, 전체 주파수 이력.

전체 J/H/K장이나 물질을 갱신하지 않는다. 준비와 실제 실행을 구분하고,
실행은 고정 계획/새 출력에서 한 번만 한다. 국소 정확성은 최종 물리 정합이 아니다.
"""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import quad

import lumina_outer_atmosphere as A
import lumina_native_columns as N
from lumina_compiled_input import CompiledInput
from lumina_nodal_driver import CompiledDriverPlan
from lumina_nodal_history import NodalKernel


class MicroLimit(ValueError):
    def __init__(self,reason,detail=None,*,kind='OTHER',partial=None):
        super().__init__(reason);self.detail={} if detail is None else dict(detail);self.kind=kind
        self.partial=None if partial is None else np.array(partial,dtype='f8',copy=True)
        if self.partial is not None:
            if (self.partial.ndim!=1 or not 1<=self.partial.size<=512 or
                    not np.isfinite(self.partial).all() or np.any(self.partial<0)):
                raise ValueError('미승인 후보의 수치/범위 오류')
            self.partial.flags.writeable=False


class Budget:
    def __init__(self,maximum=2000000,seconds=270.):
        if type(maximum) is not int or not 1<=maximum<=32000000 or not 0<seconds<=270:
            raise ValueError('국소 시험 예산 오류')
        self.maximum,self.seconds,self.used,self.started=maximum,seconds,0,time.monotonic()

    def check(self):
        if time.monotonic()-self.started>=self.seconds:raise MicroLimit('전체 시간 한도',kind='GLOBAL_TIME')

    def take(self,count):
        self.check()
        if type(count) is not int or count<0 or count>self.maximum-self.used:
            raise MicroLimit('전체 scalar formal 한도',dict(used=self.used,requested=count,maximum=self.maximum),kind='GLOBAL_WORK')
        self.used+=count


class ZeroTrial:
    def __init__(self,source):
        self.ns,self.nf=source.ns,source.nf;self.radius=source.radius;self.frequency=source.frequency;self.closed=False
        self.identity=dict(role='EXPLICIT_ZERO_J_MICRO_TRIAL_ONLY',value=0.,self_consistent=False)
    def read(self,rows,columns):
        rows,columns=N.request(rows,columns,self.ns,self.nf)
        return np.zeros((len(rows),len(columns)))


class Interval:
    def __init__(self,g,step):
        self.g=g;self.step=tuple(map(float,step));p,z0,z1,r0,r1=self.step
        self.observer=float(A.doppler(g,p,z1,r1))

    @lru_cache(maxsize=65536)
    def point(self,x):
        p,z0,z1,r0,r1=self.step;z=z0 if x==0 else z1 if x==1 else z0+(z1-z0)*x
        native=r0 if x==0 else r1 if x==1 else None
        r=native if native is not None else float(np.hypot(p,z))
        d=float(np.log(A.doppler(self.g,p,z,native)/self.observer))
        if not np.isfinite(d) or d<0:raise ValueError('음수/비유한 관측 Doppler offset')
        return r,d,z,native

    def geometry(self,a,b):
        left,right=self.point(a),self.point(b)
        return float(A.segment_geometry(self.g,self.step[0],left[2],right[2],
            start_radius=left[3],end_radius=right[3])[0])


def adaptive(kernel,provider,interval,begin,width,budget,*,panels=16,comparison=None):
    """선택 출력만의 오차 추정. 전체ν 장의 수용 판정으로 사용하지 않는다."""
    if type(begin) is not int or type(width) is not int or begin<0 or not 1<=width<=512 or begin+width>provider.nf:
        raise ValueError('국소 출력 범위 오류')
    started=time.monotonic();h=provider._source.dlognu;leaves=[(i/panels,(i+1)/panels,0) for i in range(panels)]
    initial=np.zeros(width);stats={};rounds=0;last_completed=None;last_value=None;local_work=0
    def fail(reason,*,kind='OTHER'):
        raise MicroLimit(reason,dict(last_completed_round=last_completed,current_mesh_leaves=len(leaves)),kind=kind,partial=last_value)
    def check():
        try:budget.check()
        except MicroLimit as error:
            error.detail.update(last_completed_round=last_completed,current_mesh_leaves=len(leaves))
            if last_value is not None:
                error.partial=np.array(last_value,copy=True);error.partial.flags.writeable=False
            raise
    while True:
        check()
        if time.monotonic()-started>=60:fail('국소 적응 구간60초 한도',kind='INTERVAL_TIME')
        n=len(leaves)
        if n*width>2097152:fail('leaf 곱 작업공간 한도')
        lengths=np.empty((n,3));deltas=np.empty((n,3));radii=np.empty((n,5))
        for row,(a,b,depth) in enumerate(leaves):
            m=.5*(a+b);points=[interval.point(x) for x in (a,.5*(a+m),m,.5*(m+b),b)]
            radii[row]=[x[0] for x in points];deltas[row]=[points[i][1] for i in (0,2,4)]
            lengths[row]=[interval.geometry(x,y) for x,y in ((a,b),(a,m),(m,b))]
        spans=np.stack((deltas[:,0]-deltas[:,2],deltas[:,0]-deltas[:,1],deltas[:,1]-deltas[:,2]),axis=1)/h
        expected=3*n+int(np.count_nonzero((spans>=.75)&(spans<=1.25)))
        try:
            if comparison is None:
                budget.take(expected*width)
                value,errors,responses,detail=kernel.compare_provider(provider,h,lengths,deltas,radii,initial,begin=begin)
            else:
                value,errors,responses,detail=comparison.evaluate(lengths,deltas,radii,initial,budget)
        except MicroLimit as error:
            error.detail.update(last_completed_round=last_completed,pending_mesh_leaves=n,
                pending_formal_vectors=expected,pending_scalar_formals=expected*width)
            if last_value is not None:
                error.partial=np.array(last_value,copy=True);error.partial.flags.writeable=False
            raise
        if comparison is None and detail['formal_vectors']!=expected:raise ValueError('국소 가중 작업량 불일치')
        for key in ('quadrature_nodes','quadrature_panels','full_bin_maps','partial_bin_maps','compositions'):
            stats[key]=stats.get(key,0)+detail[key]
        local_work+=detail.get('scalar_formals',detail['formal_vectors']*width);check()
        if time.monotonic()-started>=60:fail('국소 적응 구간60초 한도',kind='INTERVAL_TIME')
        rounds+=1;tail=np.ones(width);total=np.zeros(width);score=np.zeros(n)
        with np.errstate(over='raise',invalid='raise'):
            for row in range(n-1,-1,-1):
                local=errors[row]*tail;total+=local;score[row]=local.max();tail*=responses[row]
        estimate=float(total.max());tolerance=1e-6*float(value.max())
        last_completed=dict(round=rounds,leaves=n,estimated_error=estimate,tolerance=tolerance,
            scalar_formals=local_work,elapsed_seconds=time.monotonic()-started,
            mesh=dict(boundaries=[leaf[0] for leaf in leaves]+[leaves[-1][1]],depths=[leaf[2] for leaf in leaves]))
        if comparison is not None:
            last_completed['cache']=dict(built_leaves=comparison.built_leaves,
                replayed_leaf_columns=comparison.replayed_leaf_columns,
                fallback_vectors=comparison.fallback_vectors,workspace_peak_bytes=comparison.peak_workspace_bytes)
            last_completed['cache'].update(fallback_scalar_formals=comparison.fallback_scalar_formals,
                unsafe_leaf_columns_built=comparison.unsafe_leaf_columns_built)
        last_value=np.array(value,copy=True);last_value.flags.writeable=False
        if estimate<=tolerance:
            return value,dict(rounds=rounds,leaves=n,estimated_error=estimate,tolerance=tolerance,
                elapsed_seconds=time.monotonic()-started,kernel_stats=stats,
                normalization='selected output slab maximum ONLY; not global-field acceptance',
                incident='identically zero; exact under the unchanged 8-point reconstruction',
                error_certificate=False,mesh=last_completed['mesh'],status='ADAPTIVE_MET')
        selected=score>=.5*float(score.max())
        if n+int(selected.sum())>4096:fail('공간 leaf4096 한도')
        updated=[]
        for choose,(a,b,depth) in zip(selected,leaves):
            if not choose:updated.append((a,b,depth));continue
            m=.5*(a+b)
            if depth>=32 or m==a or m==b:fail('공간 분할 깊이 한도')
            updated.extend(((a,m,depth+1),(m,b,depth+1)))
        leaves=updated


def independent_bin_integral(length,k0,k1,source,budget):
    """C 모멘트/rolling을 쓰지 않는 양의 exp(이차식) 직접 적분."""
    budget.take(1);a=.5*length*(k0-k1);b=length*k1;points=[0.,1.]
    if a and 0<-b/(2*a)<1:points.append(-b/(2*a))
    for level in (-512.,-128.,-32.,-8.,-1.,1.,8.,32.,128.,512.):
        if a==0:
            roots=[] if b==0 else [level/b]
        else:
            discriminant=b*b+4*a*level
            if discriminant<0:continue
            q=-.5*(b+math.copysign(math.sqrt(discriminant),b))
            roots=[q/a]+([] if q==0 else [-level/q])
        points.extend(x for x in roots if 0<x<1)
    def integrand(t):return source(t)*math.exp(-a*t*t-b*t)
    answer=quad(integrand,0.,1.,points=sorted(set(points)),epsabs=0.,epsrel=2e-12,limit=100,full_output=1)
    if len(answer)!=3:raise ValueError('독립 구적의 오차 제어 미완료')
    integral,error,information=answer
    value=length*integral;transmission=math.exp(-a-b)
    if not all(np.isfinite(x) for x in (value,transmission,error)) or value<0:raise ValueError('독립 적분 수치 범위 오류')
    return value,transmission,int(information['neval'])


def projected_oracle(kernel,provider,interval,columns,budget):
    """같은 공간동결χ 투영의 모든 bin: 독립 적분 대 C rolling.

    이 대조만으로 원래 공간변화 전체의 정확성을 주장하지 않는다.
    """
    h=provider._source.dlognu;drift=interval.point(0.)[1];span=drift/h;length=interval.geometry(0.,1.)
    if not span>0 or not length>0:raise ValueError('양의 국소 이동/길이 필요')
    radii=[interval.point(x)[0] for x in (0.,.5,1.)]
    lo=int(min(columns));hi=min(provider.nf-1,math.ceil(max(columns)+span))
    grid=np.arange(lo,hi+1,dtype='f8');chi,eta=provider.at(radii,grid)
    expected=[];actual=[];nodes=0;bin_maps=0;gamma=3*drift/length
    for column in columns:
        budget.take(1)
        value,_,_=kernel.range(provider,h,length,drift,0.,radii,np.zeros(1),begin=int(column),mode='history')
        actual.append(float(value[0]));start=float(column)+span;top=start;reference=0.
        while top>column:
            bottom=max(float(column),provider.nf-1. if top>provider.nf-1 else math.ceil(top)-1.)
            c0,c1=np.interp([top,bottom],grid,chi[1]);aa,ab=np.interp([top,bottom],grid,eta[0]);ba,bb=np.interp([top,bottom],grid,eta[2])
            def source(t):
                q=bottom+(top-bottom)*t;u=(start-q)/span
                return (1-u)*(ab+(aa-ab)*t)+u*(bb+(ba-bb)*t)
            emitted,transmission,count=independent_bin_integral(length*(top-bottom)/span,c0+gamma,c1+gamma,source,budget)
            reference=transmission*reference+emitted;nodes+=count;bin_maps+=1;top=bottom
        expected.append(reference)
    expected=np.array(expected);actual=np.array(actual);scale=float(abs(expected).max());difference=float(abs(actual-expected).max())
    relative=difference/scale if scale else (0. if difference==0 else None)
    if relative is None or relative>3e-12:raise ValueError('동일 투영의 독립 bin 적분 대조 실패')
    return dict(columns=list(map(int,columns)),C_value=actual.tolist(),independent_value=expected.tolist(),
        max_norm_difference=relative,criterion=3e-12,quadrature_evaluations=nodes,direct_bin_maps=bin_maps,
        scope='same frozen-midpoint chi and endpoint eta projection ONLY')


def fixed_reference(kernel,provider,interval,columns,panels,budget):
    """고정 미세 공간 격자. 입사 재투영 없이 같은 특성선을 끝까지 진행한다."""
    h=provider._source.dlognu;values=np.zeros(len(columns))
    for i in range(panels):
        budget.check();a=i/panels;b=(i+1)/panels;m=.5*(a+b)
        p=[interval.point(x) for x in (a,m,b)];length=interval.geometry(a,b);da,db=p[0][1],p[2][1]
        if (da-db)/h>2:raise MicroLimit('미세 참조 구간의 이동2bin 초과')
        for j,column in enumerate(columns):
            budget.take(1)
            value,_,_=kernel.range(provider,h,length,da,db,[x[0] for x in p],values[j:j+1],begin=int(column),mode='small')
            values[j]=value[0]
    return values


def open_geometry(preparation):
    p=json.loads(Path(preparation).read_text());path=Path(preparation).parent/'geometry.npz'
    if N.sha(path)!=p['output_sha256']['geometry.npz']:raise ValueError('기하 신원 변경')
    with np.load(path,allow_pickle=False) as a:
        return A.OuterGeometry(a['radius'].copy(),a['velocity'].copy(),a['sigma'].copy(),
            len(a['radius'])-len(a['native_radius']),tuple(a['reference_indices']),a['native_radius'].copy())


def prepare(config,preparation,digest,library,output,*,max_scalar_formals=2000000,cached_affine=False):
    if type(max_scalar_formals) is not int or max_scalar_formals not in (2000000,32000000):
        raise ValueError('승인된 국소 작업 상한만 지정할 수 있다')
    with N.open_sealed_native(config,preparation,digest) as source:
        g=open_geometry(preparation);plan=CompiledDriverPlan(g,source,panels=16,block_width=512);steps=plan.arrays['steps']
        begin=[int(np.searchsorted(source.frequency,nu))-256 for nu in (4e15,8e15)]
        if any(x<0 or x+512>source.nf for x in begin):raise ValueError('사전 지정 UV 창이 실제 격자 밖')
        gradient=[];columns=np.arange(begin[0],begin[0]+512,32)
        for row in (36,37,38):
            chi,_,_=source.read(np.array([row,row+1]),columns);scale=float(abs(chi).max())
            gradient.append(float(abs(chi[0]-chi[1]).max()/scale) if scale else 0.)
        row=36+int(np.argmax(gradient));r0,r1=source.radius[row:row+2]
        candidates=np.flatnonzero((steps[:,3]==r1)&(steps[:,4]==r0))
        if not len(candidates):raise ValueError('선택 native 이웃 구간 누락')
        def drift(i):
            s=steps[i];return float(np.log(A.doppler(g,s[0],s[1],s[3])/A.doppler(g,s[0],s[2],s[4])))
        first=max(map(int,candidates),key=drift);second=max((i for i in range(len(steps)) if i!=first),key=drift)
        record=dict(config=str(config),preparation=str(preparation),preparation_sha256=digest,
            library=str(library),library_sha256=N.sha(library),script_sha256=N.sha(Path(__file__)),
            source_identity=source.identity,selected_native_pair_zero_based=[row,row+1],gradient_scores=gradient,
            selection_coefficient_bytes_read=source.coefficient_bytes_read,
            intervals=[dict(index=i,step=steps[i].tolist(),begin=base,width=512,drift_bins=drift(i)/source.dlognu,
                frequency_first_Hz=float(source.frequency[base]),frequency_last_Hz=float(source.frequency[base+511]))
                for i,base in zip((first,second),begin)],
            limits=dict(cpus=1,memory_GiB=4,wall_seconds=300,internal_seconds=270,max_scalar_formals=max_scalar_formals,
                actual_invocations=1,automatic_retries=0,adaptive_interval_seconds=60,adaptive_rtol=1e-6,
                projection_oracle_rtol=3e-12,spatial_point_criterion=1e-4,reference_panels=[4096,8192],
                output_file_bytes=1048576),actual_model_F_calls=0,formal_calls=0,
            scope='explicit J=0 and incident=0; local selected outputs ONLY; no SE/RE/full field')
        if cached_affine:
            record.update(cached_affine=True,cached_affine_script_sha256=N.sha(Path(__file__).with_name('lumina_nodal_cached.py')))
    write_new(output,record)


def write_new(path,record):
    with Path(path).open('x') as out:json.dump(record,out,indent=2,allow_nan=False)


def load_projection_reuse(plan,result_path,result_sha):
    """같은 물리 투영의 고정 증거만 계승한다. 실패한 공간해는 계승하지 않는다."""
    if result_path is None and result_sha is None:return {}
    if not result_path or not result_sha:raise ValueError('계승 결과 경로와 SHA가 함께 필요하다')
    result_path=Path(result_path)
    if N.sha(result_path)!=result_sha:raise ValueError('계승 결과 신원 변경')
    prior=json.loads(result_path.read_text());prior_plan_path=result_path.parent/'micro-plan.json'
    prior_plan_sha=prior.get('plan_sha256')
    if N.sha(prior_plan_path)!=prior_plan_sha:raise ValueError('계승 결과의 원래 계획 신원 변경')
    old=json.loads(prior_plan_path.read_text())
    for key in ('config','preparation','preparation_sha256','source_identity','library_sha256','scope'):
        if old.get(key)!=plan.get(key):raise ValueError('계승 투영의 입력/수송 신원 불일치: '+key)
    if old.get('scope')!='explicit J=0 and incident=0; local selected outputs ONLY; no SE/RE/full field':
        raise ValueError('계승 투영의 시험 J/입사 역할 불일치')
    if (N.sha(old['library'])!=old['library_sha256'] or N.sha(plan['library'])!=plan['library_sha256'] or
            N.sha(old['preparation'])!=old['preparation_sha256'] or
            N.sha(result_path.parent/'scripts/lumina_local_micro.py')!=old['script_sha256']):
        raise ValueError('계승 증거의 바이너리/입력 변경')
    if prior.get('actual_model_F_calls')!=0 or prior.get('SE_RE_updates')!=0 or prior.get('actual_micro_invocations')!=1:
        raise ValueError('계승 결과의 물질/전체 장 실행 범위 불일치')
    accepted={}
    for result in prior.get('results',[]):
        specification=result.get('specification');oracle=result.get('projection_oracle')
        if not oracle or specification not in old['intervals'] or specification not in plan['intervals']:continue
        begin=specification['begin']
        if specification['width']!=512 or oracle.get('columns')!=[begin,begin+256,begin+511]:
            raise ValueError('계승 투영의 출력열 불일치')
        if oracle.get('scope')!='same frozen-midpoint chi and endpoint eta projection ONLY' or oracle.get('criterion')!=3e-12:
            raise ValueError('계승 투영의 수식/수용 기준 불일치')
        a=np.array(oracle.get('C_value'),dtype='f8');b=np.array(oracle.get('independent_value'),dtype='f8')
        if (a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all() or
                np.any(a<0) or np.any(b<0) or b.max()<=0):
            raise ValueError('계승 투영의 원래 비교값 오류')
        relative=float(abs(a-b).max()/b.max())
        if relative>3e-12 or not np.isclose(relative,oracle.get('max_norm_difference'),rtol=1e-12,atol=0):
            raise ValueError('계승 투영의 원래 값이 기준/기록과 불일치')
        index=specification['index']
        if index in accepted:raise ValueError('동일 구간의 계승 결과 중복')
        accepted[index]=dict(oracle=oracle,provenance=dict(status='INHERITED_FIXED_EVIDENCE_NOT_REEXECUTED',
            result_path=str(result_path),result_sha256=result_sha,plan_sha256=prior_plan_sha,
            producer_script_sha256=old['script_sha256'],kernel_sha256=old['library_sha256'],
            current_scalar_formals=0,adaptive_field_inherited=False))
    if not accepted:raise ValueError('현재 구간과 같은 계승 가능한 투영 없음')
    return accepted


def run(plan_path,plan_sha,output,*,reuse_result=None,reuse_sha=None):
    if N.sha(plan_path)!=plan_sha:raise ValueError('국소 시험 계획 변경')
    plan=json.loads(Path(plan_path).read_text());output=Path(output)
    if output.exists():raise ValueError('기존 결과 덮어쓰기 금지')
    if N.sha(plan['library'])!=plan['library_sha256'] or N.sha(Path(__file__))!=plan['script_sha256']:
        raise ValueError('준비 후 소스/바이너리 변경')
    if plan.get('cached_affine',False) and N.sha(Path(__file__).with_name('lumina_nodal_cached.py'))!=plan.get('cached_affine_script_sha256'):
        raise ValueError('준비 후 전파계수 재사용 소스 변경')
    if len(plan['intervals'])!=2 or any(x['width']!=512 for x in plan['intervals']):raise ValueError('고정 국소 범위 오류')
    inherited=load_projection_reuse(plan,reuse_result,reuse_sha)
    marker=output.parent/'actual_micro.started'
    with marker.open('x') as out:out.write(plan_sha+'\n')
    budget=Budget(maximum=plan['limits']['max_scalar_formals']);record=dict(plan_sha256=plan_sha,status='STARTED',phase='opening',results=[],unmet_criteria=[],
        actual_micro_invocations=1,actual_model_F_calls=0,SE_RE_updates=0,final_goal='NOT_MET',accepted_physical_field=False)
    def checkpoint():
        record.update(scalar_formal_work_reserved=budget.used,elapsed_seconds=time.monotonic()-budget.started,
            process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        text=json.dumps(record,indent=2,allow_nan=False)
        if len(text.encode())>1048576:raise ValueError('국소 결과1MiB 초과')
        temporary=output.with_suffix('.partial');temporary.write_text(text);temporary.replace(output)
    checkpoint();code=1
    try:
        with N.open_sealed_native(plan['config'],plan['preparation'],plan['preparation_sha256']) as source:
            if source.identity!=plan['source_identity']:raise ValueError('국소 시험 원본 신원 변경')
            g=open_geometry(plan['preparation']);field=ZeroTrial(source)
            if plan.get('cached_affine',False):
                from lumina_nodal_cached import CachedKernel,CachedComparison
                kernel=CachedKernel(plan['library'])
            else:kernel=NodalKernel(plan['library'])
            with CompiledInput(plan['library'],g,source,field,slab_width=128,cache_rows=128,profiles=1024) as provider:
                provider.bind_zero_trial()
                try:
                    for specification in plan['intervals']:
                        interval=Interval(g,specification['step']);begin=specification['begin'];columns=[begin,begin+256,begin+511]
                        result=dict(specification=specification);record['results'].append(result)
                        record['phase']='projection_oracle';checkpoint()
                        previous=inherited.get(specification['index'])
                        if previous is None:
                            result['projection_oracle']=projected_oracle(kernel,provider,interval,columns,budget)
                            result['projection_oracle_provenance']=dict(status='COMPUTED_THIS_INVOCATION')
                        else:
                            result['projection_oracle']=previous['oracle']
                            result['projection_oracle_provenance']=previous['provenance']
                        record['phase']='adaptive';checkpoint()
                        try:
                            if plan.get('cached_affine',False):
                                comparison=CachedComparison(kernel,provider,begin,512)
                                try:value,detail=adaptive(kernel,provider,interval,begin,512,budget,comparison=comparison)
                                finally:
                                    result['cache']=dict(built_leaves=comparison.built_leaves,
                                        replayed_leaf_columns=comparison.replayed_leaf_columns,
                                        fallback_vectors=comparison.fallback_vectors,
                                        fallback_scalar_formals=comparison.fallback_scalar_formals,
                                        unsafe_leaf_columns_built=comparison.unsafe_leaf_columns_built,
                                        workspace_peak_bytes=comparison.peak_workspace_bytes)
                                    del comparison
                            else:value,detail=adaptive(kernel,provider,interval,begin,512,budget)
                            result.update(adaptive=detail,value=value.tolist(),value_status='ADAPTIVE_MET_NOT_PHYSICAL_ACCEPTANCE')
                        except MicroLimit as error:
                            result['adaptive']=dict(status='ADAPTIVE_NOT_MET',reason=str(error),kind=error.kind,detail=error.detail)
                            record['unmet_criteria'].append(dict(interval=specification['index'],criterion='adaptive_internal_tolerance'))
                            if error.partial is not None:result.update(value=error.partial.tolist(),value_status='PROVISIONAL')
                            checkpoint()
                            if error.kind!='INTERVAL_TIME' or error.partial is None:raise
                            # 구간 미달과 전역 자원 한도는 다르다. 전역 한도는 계속 즉시 종료한다.
                            budget.check();value=error.partial
                        record['phase']='fixed_reference';checkpoint()
                        coarse=fixed_reference(kernel,provider,interval,columns,4096,budget)
                        fine=fixed_reference(kernel,provider,interval,columns,8192,budget)
                        scale=float(abs(fine).max());gap=float(abs(fine-coarse).max());error=float(abs(value[[0,256,511]]-fine).max())
                        refgap=gap/scale if scale else (0. if gap==0 else None);difference=error/scale if scale else (0. if error==0 else None)
                        reference_met=refgap is not None and refgap<=1e-4
                        comparison_met=reference_met and difference is not None and difference<=1e-4
                        result['spatial_reference']=dict(columns=columns,panels=[4096,8192],coarse=coarse.tolist(),fine=fine.tolist(),
                            self_mesh_difference=refgap,adaptive_difference=difference,criterion=1e-4,independent_physics=False,
                            self_mesh_status='MET' if reference_met else 'NOT_MET',
                            candidate_comparison_status=('MET' if comparison_met else 'NOT_MET') if reference_met else 'UNINTERPRETED_REFERENCE_NOT_MET')
                        if not comparison_met:
                            record['unmet_criteria'].append(dict(interval=specification['index'],criterion='spatial_reference_comparison'))
                        checkpoint()
                finally:record['source']=provider.report();checkpoint()
        budget.check();code=1 if record['unmet_criteria'] else 0
        record.update(status='INCOMPLETE_OR_FAILED' if code else 'LOCAL_CRITERIA_MET',phase='finished')
    except Exception as error:record.update(status='INCOMPLETE_OR_FAILED',error_type=type(error).__name__,
        error=str(error),error_detail=getattr(error,'detail',None),error_kind=getattr(error,'kind',None))
    finally:checkpoint()
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('--config');p.add_argument('--preparation')
    p.add_argument('--preparation-sha256');p.add_argument('--library');p.add_argument('--plan');p.add_argument('--plan-sha256');p.add_argument('--output',required=True)
    p.add_argument('--reuse-result');p.add_argument('--reuse-sha256')
    p.add_argument('--cached-affine',action='store_true')
    p.add_argument('--max-scalar-formals',type=int,choices=(2000000,32000000),default=2000000)
    a=p.parse_args()
    if a.mode=='prepare':
        prepare(a.config,a.preparation,a.preparation_sha256,a.library,a.output,
            max_scalar_formals=a.max_scalar_formals,cached_affine=a.cached_affine);return 0
    return run(a.plan,a.plan_sha256,a.output,reuse_result=a.reuse_result,reuse_sha=a.reuse_sha256)


if __name__=='__main__':raise SystemExit(main())
