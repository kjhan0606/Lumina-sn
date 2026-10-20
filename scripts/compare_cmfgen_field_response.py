#!/usr/bin/env python3
"""같은 SCRTEMP59 상태에서 부모 J59와 새 J60의 Si/S 광율·가열 반응을 계산한다.

새 native PRRR이나 SE 폐합 비교가 아니다. 기존 raw PHOT 평가기를 쓰고,
같은 주파수 격자의 양쪽 장 전체를 포함한다. 부분 EDDFACTOR는 거부한다.
"""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path
import sys
import time

import numpy as np

import compare_cmfgen_bf_superlevel_rates as bf
from prepare_cmfgen_mom_capture import sha
from cmfgen_phot_smoothing import read_smoothing_controls


def linear_sigma_weights(nu,terms):
    """각 term=(확률, sigma, 구조ν, 지지시작ν, 실제문턱ν)의 선형 J 함수 가중치."""
    nu=np.asarray(nu,dtype=float)
    if nu.ndim!=1 or len(nu)<2 or not np.isfinite(nu).all() or np.any(nu<=0) or np.any(np.diff(nu)<=0):
        raise ValueError('광율 주파수 격자 오류')
    gamma=np.zeros_like(nu);heat=np.zeros_like(nu)
    nodes,weights=np.polynomial.legendre.leggauss(8)
    for probability,sigma_fn,knots,start,threshold in terms:
        if not math.isfinite(probability) or probability<0 or not math.isfinite(threshold) or threshold<=0:
            raise ValueError('광율 채널 확률/문턱 오류')
        lo=max(float(nu[0]),float(start),float(threshold));hi=float(nu[-1])
        if lo>=hi or probability==0:continue
        knots=np.asarray(knots,dtype=float)
        if knots.ndim!=1 or not np.isfinite(knots).all() or not math.isfinite(start) or start<=0:
            raise ValueError('광율 채널 지지/구조 격자 오류')
        breaks=np.unique(np.concatenate(([lo,hi],nu[(nu>lo)&(nu<hi)],knots[(knots>lo)&(knots<hi)])))
        left,right=breaks[:-1],breaks[1:]
        center=.5*(left+right);half=.5*(right-left)
        q=(center[:,None]+half[:,None]*nodes).ravel()
        qw=(half[:,None]*np.broadcast_to(weights,(len(left),8))).ravel()
        sigma=np.asarray(sigma_fn(q),dtype=float)
        if sigma.shape!=q.shape or not np.isfinite(sigma).all() or np.any(sigma<0):raise ValueError('raw PHOT 평가 오류')
        index=np.searchsorted(nu,q,side='right')-1
        if np.any(index<0) or np.any(index>=len(nu)-1):raise ValueError('광율 보간 범위 오류')
        fraction=(q-nu[index])/(nu[index+1]-nu[index])
        kernel=4*math.pi*probability*qw*sigma/q
        for result,value in ((gamma,kernel/bf.H_CGS),(heat,kernel*(q-threshold))):
            result+=np.bincount(index,weights=value*(1-fraction),minlength=len(nu))
            result+=np.bincount(index+1,weights=value*fraction,minlength=len(nu))
    return gamma,heat


def apply_field_weights(gamma,heat,old,new,absolute):
    """같은 선형 함수를 묶어 계산한다. 작은 값은 버리지 않고 정확한0만 생략한다."""
    nonzero=np.flatnonzero((gamma!=0)|(heat!=0))
    if not len(nonzero):return np.zeros((6,old.shape[1]))
    support=slice(nonzero[0],nonzero[-1]+1)
    weights=np.stack((gamma[support],heat[support]))
    before=weights@old[support];after=weights@new[support];bound=weights@absolute[support]
    return np.stack((before[0],after[0],bound[0],before[1],after[1],bound[1]))


def target_coverage_complete(populations,ion_sums,completed):
    return (completed==sum(model['n'] for model in populations.values()) and
        set(ion_sums)=={str(key) for key in populations} and
        all(ion_sums[str(key)]['levels_completed']==model['n'] for key,model in populations.items()))


def read_complete_pair(seed,run):
    record=json.loads((run/'seq_logs/whole-runtime.json').read_text())
    if not record.get('diagnostic_complete') or record.get('returncode')!=97 or record['coverage']['count']!=197291:
        raise ValueError('새 전주파수 J60 완료 증거 없음; 부분 파일의 부모 record 소비 금지')
    inputs=json.loads((run/'seq_logs/whole-inputs.json').read_text())
    if Path(inputs['seed']).resolve()!=seed.resolve():raise ValueError('J59 부모 신원 불일치')
    if sha(run/'SCRTEMP')!=sha(seed/'SCRTEMP'):raise ValueError('pre-pop 물질 상태 변경')
    old,nu,nd,parent_finish=bf.read_eddfactor(str(seed/'EDDFACTOR'))
    if parent_finish!=1.:raise ValueError('부모 EDDFACTOR 자체가 미완성')
    new,nu_new,nd_new,_=bf.read_eddfactor(str(run/'EDDFACTOR'))
    if nd!=nd_new or len(nu)!=197291 or not np.array_equal(nu,nu_new):raise ValueError('전주파수/깊이 대응 불일치')
    if old.shape!=new.shape or not np.isfinite(old).all() or not np.isfinite(new).all() or np.any(old<0) or np.any(new<0):
        raise ValueError('복사장 값/지지 오류')
    return old,new,nu,nd


def population_inputs(seed,models,links):
    nd,nt=bf.read_dimensions(seed);reader=bf.ScrtempReader(seed/'SCRTEMP',nd,nt)
    if reader.last_it!=59:raise ValueError('부모 물질 상태가59가 아님')
    state=reader.state(59);variables=bf.read_variables(seed,nt-2)
    indices={(v.ion,v.superlevel):v.index-1 for v in variables}
    positive=bf.scrtemp_positive_ion_population(state,variables)
    temperature=state[nt-1];ne=state[nt-2]
    result={}
    for key in sorted(bf.TARGETS):
        spec=models[key];upper=models[(key[0],key[1]+1)]
        lower_osc=bf.parse_osc(links[key]['osc']);upper_osc=bf.parse_osc(links[(key[0],key[1]+1)]['osc'])
        n=int(spec['n_full']);un=int(upper['n_full'])
        low_map,low_names,sequence,_=bf.parse_f_to_s_int_seq(links[key]['f_to_s'],n)
        up_map,_,_,_=bf.parse_f_to_s_int_seq(links[(key[0],key[1]+1)]['f_to_s'],un)
        lower_edge=bf.cmfgen_runtime_edge_nu15(float(lower_osc.ionization_eV)/bf.CM_TO_EV,np.asarray(lower_osc.levels['E_cm'][:n]))
        upper_edge=bf.cmfgen_runtime_edge_nu15(float(upper_osc.ionization_eV)/bf.CM_TO_EV,np.asarray(upper_osc.levels['E_cm'][:un]))
        sl=bf.scrtemp_superlevel_populations(state,str(spec['label']),int(spec['n_super']),indices)
        upper_sl=bf.scrtemp_superlevel_populations(state,str(upper['label']),int(upper['n_super']),indices)
        full=bf.cmfgen_sup_to_full_v3b(sl,lower_edge,np.asarray(lower_osc.levels['g'][:n]),low_map,sequence,
            upper_edge,np.asarray(upper_osc.levels['g'][:un]),up_map,upper_sl[:,0],temperature,ne,positive,float(key[1]-1),
            bf.parse_model_do_lev_dissolution(seed/'MODEL'))
        result[key]=dict(osc=lower_osc,n=n,population=full,mapping=low_map)
    return result,temperature,ne


def terms_for_level(source,channels,index,configuration,energy,ionization,Z,z_screen,smoother):
    terms=[]
    for route in range(int(channels['offsets'][index]),int(channels['offsets'][index+1])):
        one=dict(channels);one['offsets']=channels['offsets'].copy()
        one['offsets'][index:index+2]=(route,route+1)
        threshold=(ionization-energy+float(channels['target_energy_eV'][route]))*bf.atomic.EV_TO_ERG/bf.atomic.H_CGS/(1+1e-14)
        if threshold<=0:continue
        raw=bf.cmfgen_raw_sigma_terms_for_level(source,one,index,configuration,energy,ionization,Z,z_screen,smoother)
        terms.extend((*term,threshold) for term in raw)
    return terms


def compare(seed,run,deck,out,max_seconds=250,max_levels=None):
    if out.exists():raise ValueError('기존 복사장 반응 결과 보존')
    started=time.monotonic()
    old,new,nu,nd=read_complete_pair(seed,run)
    models=bf.atomic.load_cmfgen_model_spec(seed/'MODEL_SPEC',seed/'atomic_links.txt')
    links=bf.atomic.load_cmfgen_links(seed/'atomic_links.txt')
    manifest=json.loads((deck/'cmfgen_bf_channels_manifest.json').read_text())
    channels=bf.parse_cmfgen_bf_channels(deck/'cmfgen_bf_channels.bin',int(manifest['n_levels']),int(manifest['n_freq_bins']),manifest)
    with (deck/'levels.csv').open() as f:levels=list(csv.DictReader(f))
    index={(int(row['atomic_number']),int(row['ion_number'])+1,int(row['level_number'])+1):i for i,row in enumerate(levels)}
    identity={}
    for key in sorted(bf.TARGETS):
        sealed=manifest['ions']['Z%d_stage%d'%key]['sources']
        actual=bf.atomic._phot_route_paths(links[key]['phot'])
        hashes=[sha(p) for p in actual]
        if hashes!=[row['sha256'] for row in sealed]:raise ValueError('PHOT 문턱/경로 신원 불일치: '+str(key))
        identity[str(key)]=[dict(path=str(p),sha256=h) for p,h in zip(actual,hashes)]
    populations,temperature,ne=population_inputs(seed,models,links)
    provenance=json.loads((deck/'DECK_PROVENANCE.json').read_text())['phot_preprocessing']
    smoother=bf.PhotSmoother.from_provenance(provenance)
    if read_smoothing_controls(seed/'MODEL')!=smoother.controls:
        raise ValueError('같은 상태의 PHOT smoothing 제어 불일치')
    out.mkdir();completed=0;all_done=True;ion_sums={}
    delta=new-old;absolute=np.abs(delta)
    with (out/'field_response.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Z','stage','level_1based','depth_1based','population_cm3','gamma59_s-1','gamma60_s-1','gamma_abs_envelope_s-1','heat59_erg_s-1','heat60_erg_s-1','heat_abs_envelope_erg_s-1'])
        for key in sorted(bf.TARGETS):
            model=populations[key];osc=model['osc'];acc=np.zeros((6,nd));ion_completed=0
            for level in range(1,model['n']+1):
                if (max_levels is not None and level>max_levels) or time.monotonic()-started>max_seconds:
                    all_done=False;break
                g=index[(*key,level)];row=levels[g]
                energy=float(osc.levels['E_cm'][level-1])*bf.CM_TO_EV
                if (abs(float(row['energy_eV'])-energy)>1e-9 or float(row['g'])!=float(osc.levels['g'][level-1])
                    or int(row['super_level'])+1!=int(model['mapping'][level-1])
                    or bf.atomic._norm_cfg(row['configuration'])!=bf.atomic._norm_cfg(str(osc.levels['config'][level-1]))):
                    raise ValueError('같은 model-atom 준위 신원 불일치')
                terms=terms_for_level(links[key]['phot'],channels,g,str(osc.levels['config'][level-1]),energy,float(osc.ionization_eV),key[0],float(osc.z_screen),smoother)
                wg,wh=linear_sigma_weights(nu,terms)
                values=apply_field_weights(wg,wh,old,new,absolute)
                pop=model['population'][:,level-1];acc+=values*pop
                for depth in range(nd):writer.writerow([*key,level,depth+1,pop[depth],*values[:,depth]])
                f.flush();completed+=1;ion_completed+=1
            ion_sums[str(key)]=dict(levels_requested=model['n'],levels_completed=ion_completed,partial_photo59=acc[0].tolist(),partial_photo60=acc[1].tolist(),
                partial_abs_envelope=acc[2].tolist(),partial_heat59=acc[3].tolist(),partial_heat60=acc[4].tolist(),partial_heat_abs_envelope=acc[5].tolist())
            if time.monotonic()-started>max_seconds:break
    source_paths=[Path(__file__).resolve(),Path(bf.__file__).resolve(),Path(bf.atomic.__file__).resolve()]
    source_paths.extend(Path(sys.modules[obj.__module__].__file__).resolve() for obj in (
        bf.PhotSmoother,bf.parse_osc,bf.read_eddfactor,bf.ScrtempReader))
    all_done=all_done and target_coverage_complete(populations,ion_sums,completed)
    source_snapshot=out/'source_snapshot';source_snapshot.mkdir()
    for p in set(source_paths):shutil.copy2(p,source_snapshot/p.name)
    physical_paths=set()
    for key in bf.TARGETS:
        for ion in (key,(key[0],key[1]+1)):
            physical_paths.update(Path(links[ion][kind]).resolve() for kind in ('osc','f_to_s'))
    report=dict(scope='같은 SCRTEMP59 인구/T/ne에서 J59→J60 광율·가열 반응; native PRRR/전체SE/새분포 판정 아님',
        material_record_for_rate_evaluation=59,new_field_material_record=59,
        interpretation_warning='두 장의 율을 물질59에서 평가하지만 부모 J59 생성에는 이전 물질상태가 쓰였다. 장/물질 반복 변화와 알고리즘 수리 효과를 단독 인과로 혼동하지 않는다.',
        isolated_algorithm_effect=False,converged_reference=False,
        seed=str(seed),run=str(run),deck=str(deck),completed_full_levels=completed,all_target_levels_complete=all_done,
        expected_full_levels=sum(model['n'] for model in populations.values()),
        temperature_K=(temperature*1e4).tolist(),electron_density_cm3=ne.tolist(),ions=ion_sums,phot_identity=identity,
        absolute_envelope='같은 nodal grid의 |J60−J59| 선형보간을 적분한 상쇄없는 차이 상한; 참해 오차 상한 아님',
        input_sha256={str(p):sha(p) for p in (seed/'SCRTEMP',seed/'EDDFACTOR',seed/'MODEL',seed/'MODEL_SPEC',seed/'atomic_links.txt',
            seed/'LEVEL_SL_STEQ_LINKS',seed/'RVTJ',run/'EDDFACTOR',run/'seq_logs/whole-runtime.json',run/'seq_logs/whole-inputs.json',
            deck/'cmfgen_bf_channels.bin',deck/'cmfgen_bf_channels_manifest.json',deck/'levels.csv',deck/'DECK_PROVENANCE.json',Path(__file__),*sorted(physical_paths))},
        producer_sha256={str(p):sha(p) for p in source_paths},phot_preprocessing=smoother.provenance(),
        source_snapshot_scope='직접 수치 모듈의 사본이며 독립 패키지는 아님. 원래 프로젝트의 import 경로가 필요함.',
        output_sha256=sha(out/'field_response.csv'),elapsed_seconds=time.monotonic()-started,final_goal='NOT_JUDGED')
    with (out/'field-response.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),completed=completed,all_target_levels_complete=all_done,elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','run','deck','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=250);p.add_argument('--max-levels-per-ion',type=int)
    a=p.parse_args();compare(a.seed,a.run,a.deck,a.out,a.max_seconds,a.max_levels_per_ion)
