#!/usr/bin/env python3
"""전 model-atom의 실제 PHOT target과 NLTE 연속 χ/η를 생산한다.

기존 v5/고정σ 입력을 변경하지 않는다. MODEL의 smoothing을 적용한 원자료를
요청 주파수에서 평가하며 흡수·유도 재결합·자발 방출이 같은 route를 쓴다.
복사장 또는 SE/RE를 푸는 프로그램이 아니며 출력 계수만으로 목표를 판정하지 않는다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
import struct
from types import SimpleNamespace

import numpy as np

from analyze_cmfgen_material_inventory import sha
import expand_atomic_data_cmfgen as atomic
from cmfgen_parser import parse_osc, parse_f_to_s_int_seq
from cmfgen_phot_smoothing import PhotSmoother

H=6.62607015e-27
KB=1.380649e-16
ME=9.1093837015e-28
C=2.99792458e10
EV=1.602176634e-12


@dataclass
class Route:
    Z: int
    stage: int                       # spectroscopic stage, II=2
    lower: int                       # full-level row
    target: int                      # anchor FL, -1 only for declared terminal
    target_superlevel: int            # local SL, -1 terminal
    phot_grid: int
    probability: float
    target_g: float
    target_energy_eV: float
    threshold_Hz: float
    models: tuple
    dissolution_primary: bool = False

    def sigma(self, frequency):
        frequency=np.asarray(frequency,dtype='f8')
        if (frequency.ndim!=1 or not np.isfinite(frequency).all() or
                np.any(frequency<=0)):
            raise ValueError('잘못된 단면 평가 주파수')
        result=np.zeros_like(frequency)
        # 실제 주파수 평가다. bin 평균σ를 중심값으로 재해석하지 않는다.
        for function,_nodes,start,_kind in self.models:
            active=frequency>=start
            if np.any(active):
                values=np.asarray(function(frequency[active]),dtype='f8')
                if values.shape!=frequency[active].shape or not np.isfinite(values).all() or np.any(values<0):
                    raise ValueError(f'PHOT 단면 평가 실패: {self.Z}:{self.stage}:{self.lower}')
                result[active]+=values
        if not np.isfinite(result).all():raise ValueError('단면 합 비유한')
        if np.any((frequency<self.threshold_Hz)&(result>0)):
            raise ValueError('문턱 아래의 PHOT 단면을 순연속으로 사용할 수 없음')
        return result


def all_model_policy(model,models,sources):
    """외부 DR 선택을 σ 존재 여부가 아닌 실제 MODEL의 선택에서 얻는다."""
    text=model.read_text(encoding='latin-1')
    records=[]
    for (Z,stage),data in sorted(models.items()):
        tag='DIE_'+data['label']
        flags=re.findall(r'^\s*([TF])\s*,\s*([TF])\s+\['+re.escape(tag)+r'\]',text,re.M)
        if flags!=[('F','F')]:
            raise ValueError(f'미지원/누락/중복 독립 DIE 선택: {tag}={flags}')
        records.append(dict(Z=Z,lower_ion_number=stage-1,upper_ion_number=stage,
            model_key=tag,die_auto=0,die_wi=0,external_dr_fit_allowed=0,
            treatment='PHOT_MILNE_NO_EXTERNAL_FIT',phot_sources=sources[Z,stage]))
    return dict(schema='lumina-all-model-process-selection-v1',
                model=dict(path=str(model),sha256=sha(model)),ions=records)


def terminal_component(phot):
    # GET_NUM_PHOT_ROUTES_V1: 다음 이온 부재 시 각 PHOT 파일의 GION/EION을
    # 보존해 같은 가상 준위1로 보낸다. PHOT 파일 수를1로 제한하지 않는다.
    energy,g=float(phot.final_state_excit_eV),float(phot.final_state_g)
    if (not phot.final_state.strip() or not np.isfinite(energy) or energy<0 or
            not np.isfinite(g) or g<=0):
        raise ValueError('terminal PHOT의 EION/GION 누락/손상')
    return [dict(upper_global=-1,upper_superlevel=-1,energy_eV=energy,g=g,probability=1.)]


def compile_routes(ion_data,level_lookup,terminal_keys,read_phot):
    """준위/SL 모형을 이미 고정한 호출자가 전 이온의 route를 만든다."""
    routes=[];sources={};counts={}
    if any((Z,stage+1) in ion_data for Z,stage in terminal_keys):
        raise ValueError('bound upper와 terminal 위상 중복')
    for key,data in sorted(ion_data.items()):
        Z,stage=key;upper=ion_data.get((Z,stage+1))
        if upper is None and key not in terminal_keys:
            raise ValueError(f'선언되지 않은 terminal: {key}')
        paths=atomic._phot_route_paths(Path(data['provenance']['phot_path']))
        identities=[dict(path=str(p.resolve()),sha256=sha(p.resolve())) for p in paths]
        phot_grids=[read_phot(path) for path in paths]
        if (not phot_grids or any(int(p.n_routes)!=len(paths) for p in phot_grids)):
            raise ValueError(f'PHOT 파일 수/선택 불일치: {key}')
        levels=data['levels'];nf=int(data['n_kept'])
        exact={};terms={}
        for l in range(nf):
            cfg=atomic._norm_cfg(levels['config'][l])
            if cfg in exact:raise ValueError(f'중복 full-level 이름: {key}:{cfg}')
            exact[cfg]=l;terms.setdefault(atomic._term_cfg(cfg),[]).append(l)
        begin=len(routes)
        for grid,(path,phot) in enumerate(zip(paths,phot_grids,strict=True)):
            targets=(terminal_component(phot) if upper is None else
                atomic._phot_target_components(phot,upper,level_lookup,Z,stage+1))
            entries={}
            for entry in phot.entries:
                cfg=atomic._norm_cfg(entry.config)
                lower=([exact[cfg]] if cfg in exact else []) if phot.split_J else terms.get(cfg,[])
                for l in lower:entries.setdefault(l,[]).append(entry)
            absent=set(range(nf))-entries.keys()
            if absent:
                raise ValueError(f'PHOT의 실제 lower 준위 누락: {key} grid={grid} count={len(absent)} first={min(absent)+1}')
            for lower,entry_list in sorted(entries.items()):
                for target_index,target in enumerate(targets):
                    threshold=(float(data['osc'].ionization_eV)-
                        float(levels['E_cm'][lower])*1.239841984e-4+target['energy_eV'])*EV/H
                    if threshold<=0:
                        # native의 닫힌 continuum과 같은 물리적 비활성 route.
                        continue
                    models=tuple(atomic._route_sigma_model(entry,threshold,Z,
                        float(data['osc'].z_screen)) for entry in entry_list)
                    routes.append(Route(Z,stage,level_lookup[Z,stage,lower+1],
                        int(target['upper_global']),int(target['upper_superlevel']),grid,
                        float(target['probability']),float(target['g']),
                        float(target['energy_eV']),float(threshold),models,
                        grid==0 and target_index==0))
        counts[f'Z{Z}_stage{stage}']=len(routes)-begin
        if any(sha(Path(row['path']))!=row['sha256'] for row in identities):
            raise ValueError(f'계수 생산 중 PHOT 원자료 변경: {key}')
        sources[key]=identities
    return routes,sources,counts


def load_model(run,atomic_dir,smoothing_library):
    models=atomic.load_cmfgen_model_spec(run/'MODEL_SPEC',run/'atomic_links.txt')
    links=atomic.load_cmfgen_links(run/'atomic_links.txt')
    actual_sl=atomic.load_cmfgen_level_sl_map(run/'LEVEL_SL_STEQ_LINKS')
    rows=list(csv.DictReader((atomic_dir/'levels.csv').open()))
    lookup={(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
            for i,r in enumerate(rows)}
    expected={(Z,stage,l) for (Z,stage),m in models.items() for l in range(1,int(m['n_full'])+1)}
    if len(lookup)!=len(rows) or set(lookup)!=expected:
        raise ValueError('재시작 FL 목록과 실제 MODEL_SPEC 불일치')
    ions={}
    for key,model in sorted(models.items()):
        osc=parse_osc(links[key]['osc'],include_transitions=False)
        model['ionization_eV']=float(osc.ionization_eV)
        model['z_screen']=float(osc.z_screen)
        nf=int(model['n_full']);levels=osc.levels[:nf]
        if len(levels)!=nf:raise ValueError(f'원자료 준위 누락: {key}')
        sl=[]
        for l,level in enumerate(levels):
            row=rows[lookup[*key,l+1]];native=actual_sl[model['label'],l+1]
            if (int(row['super_level'])!=native['superlevel']-1 or
                row['configuration']!=native['configuration'] or
                level['config']!=native['configuration'] or
                float(row['g'])!=float(level['g']) or
                abs(float(row['energy_eV'])-float(level['E_cm'])*1.239841984e-4)>5.1e-11):
                raise ValueError(f'원래 FL/SL 신원 불일치: {key}:{l+1}')
            sl.append(native['superlevel']-1)
        ions[key]=dict(osc=osc,levels=levels,n_kept=nf,
            ftos=SimpleNamespace(sl_of_fl=np.asarray(sl)),
            provenance=dict(phot_path=links[key]['phot']))
    smoother=PhotSmoother(run/'MODEL',smoothing_library)
    # 실제 물질 사다리의 terminal은 입력 변수에서 다시 확인한다.
    terminals={(Z,stage) for Z,stage in models if (Z,stage+1) not in models}
    routes,sources,counts=compile_routes(ions,lookup,terminals,smoother.read)
    for key,model in models.items():
        mapping,_,sequence,_=parse_f_to_s_int_seq(links[key]['f_to_s'],model['n_full'])
        for identifier in set(sequence)-{0}:
            if len(set(mapping[sequence==identifier]))>1:
                raise ValueError(f'아직 미지원인 실제 SL 간 departure 보간: {key}:{identifier}')
    for route in routes:
        if route.target<0 and route.dissolution_primary:
            models[route.Z,route.stage]['terminal_g']=route.target_g
    policy=all_model_policy(run/'MODEL',models,sources)
    return rows,models,routes,policy,smoother.provenance(),counts


def read_material(path,rows,models,occupation=None):
    with path.open('rb') as stream:
        magic,ns,nc,nl,epoch=struct.unpack('<8sIIId',stream.read(28))
        stride=6 if magic==b'LUMCEL01' else 4 if magic==b'LUMNOD01' else 0
        if not stride or ns<1 or nc<1 or nl!=len(rows) or not np.isfinite(epoch) or epoch<=0:
            raise ValueError('물질 재시작 header 오류')
        columns=np.frombuffer(stream.read(12*nc),dtype='<i4').reshape(nc,3)
        physical=np.frombuffer(stream.read(8*stride*ns),dtype='<f8').reshape(ns,stride)
        pops=np.frombuffer(stream.read(8*ns*nc),dtype='<f8').reshape(ns,nc)
        if stream.read(1):raise ValueError('물질 입력 후행 데이터')
    if not np.isfinite(physical).all() or not np.isfinite(pops).all() or np.any(pops<0):
        raise ValueError('비유한/음수 물질')
    t,ne=(physical[:,2],physical[:,3]) if stride==6 else (physical[:,1],physical[:,2])
    if np.any(t<=0) or np.any(ne<=0) or (stride==6 and not np.all(physical[:,5]==1)):
        raise ValueError('양의 T/ne 및 smooth 물질만 지원')
    index={tuple(map(int,c)):i for i,c in enumerate(columns)}
    expected={(Z,stage-1,sl) for (Z,stage),m in models.items() for sl in range(m['n_super'])}
    terminal={(Z,stage,-1) for Z,stage in models if (Z,stage+1) not in models}
    if len(index)!=nc or set(index)!=expected|terminal:
        raise ValueError('전체 SL/terminal 변수 위상 불일치')
    if occupation is None:occupation=np.ones((len(rows),ns))
    if (occupation.shape!=(len(rows),ns) or not np.isfinite(occupation).all() or
        np.any(occupation<0) or np.any(occupation>1)):
        raise ValueError('점유확률 프로필 손상')
    fl=np.empty((len(rows),ns));groups={}
    for l,row in enumerate(rows):
        key=(int(row['atomic_number']),int(row['ion_number']),int(row['super_level']))
        groups.setdefault(key,[]).append(l)
    for key,members in groups.items():
        energy=np.array([float(rows[l]['energy_eV']) for l in members])
        g=np.array([float(rows[l]['g']) for l in members])
        if not np.isfinite(energy).all() or not np.isfinite(g).all() or np.any(g<=0):
            raise ValueError('잘못된 FL 에너지/통계중량')
        weight=g[:,None]*np.exp(-(energy-energy.min())[:,None]*EV/(KB*t[None,:]))
        weight*=occupation[members]
        if np.any(weight.sum(axis=0)<=0):raise ValueError('SL의 점유 가중 분배함수 0')
        fl[members]=weight/weight.sum(axis=0)*pops[:,index[key]][None,:]
    return dict(columns=columns,physical=physical,temperature=t,ne=ne,full=fl,
                superlevel=pops.T,sl_index=index,epoch_seconds=epoch,occupation=occupation)


def dissolution_flag(model):
    tokens=re.findall(r'^\s*(T|F|\.TRUE\.|\.FALSE\.)\s+\[DO_DIS\]',
                      model.read_text(),re.M)
    if len(tokens)!=1:raise ValueError('MODEL DO_DIS 누락/중복')
    return tokens[0] in ('T','.TRUE.')


def dissolution_vectors(state):
    columns=state['columns'];nion=state['superlevel'][columns[:,1]>0].sum(axis=0)
    t4=state['temperature']*1e-4;ne=state['ne']
    if np.any(nion<=0):raise ValueError('level dissolution의 양이온 총수0')
    # 원본 COMP_LEV_DIS_BLK의 지수0.333333을 그대로 유지한다.
    b=(8.589e14*nion**.333333/ne)**1.5
    a=.0009*ne**.1667/np.sqrt(t4);x=(1+a)**3.15
    return b,a,x


def material_occupation(rows,models,state):
    b,a,x=dissolution_vectors(state)
    result=np.ones_like(state['full'])
    for l,row in enumerate(rows):
        Z,stage=int(row['atomic_number']),int(row['ion_number'])+1
        model=models[Z,stage];z=model['z_screen']
        edge=(model['ionization_eV']-float(row['energy_eV']))*EV/H
        value=3.289395e15*z*z/edge if edge!=0 else 0
        effective=np.sqrt(value) if 0<value<961 else 0
        if effective<=2.01*z:continue
        k=16*effective/(1+effective)**2/3 if effective>3 else 1.
        k=(k/z*(z/effective)**4)**1.5
        beta=k*b;y=1.091*(x+4*(z-1)*a)
        f=y*beta*beta/(7.782+x*beta)
        # 1에 가까운 값을 새로 잘라 올리지 않고 occupation fit 자체를 쓴다.
        result[l]=f/(1+f)
    if not np.isfinite(result).all():raise ValueError('점유확률 비유한')
    return result


def ground_lte_reference(rows,models,state):
    """GENOPAETA의 PHOT_ID=1 문턱 아래 HNST 기준을 구성한다."""
    reference=np.empty_like(state['full']);t=state['temperature']
    thermal=state['ne']*(H*H/(2*np.pi*ME*KB*t))**1.5
    groups={}
    for l,row in enumerate(rows):
        groups.setdefault((int(row['atomic_number']),int(row['ion_number'])+1),[]).append(l)
    for key,members in groups.items():
        Z,stage=key;upper=(Z,stage+1)
        if upper in models:
            gs=[l for l in groups[upper] if int(rows[l]['super_level'])==0]
            energy=np.array([float(rows[l]['energy_eV']) for l in gs])
            g=np.array([float(rows[l]['g']) for l in gs])
            partition=(g[:,None]*np.exp(-energy[:,None]*EV/(KB*t[None,:]))).sum(axis=0)
            population=state['superlevel'][state['sl_index'][Z,stage,0]]
        else:
            partition=models[key]['terminal_g']
            population=state['superlevel'][state['sl_index'][Z,stage,-1]]
        for l in members:
            reference[l]=thermal*population/partition*float(rows[l]['g'])*state['occupation'][l]/2
    return reference


def _dissolution_candidate_start(k,xdis,ydis):
    """원래 fraction>1e-4의 가능한 구간을 보수적으로 찾는다.

    k는 주파수에 대해 감소하고 f(k)=(A+Bk)/(A+Bk+Ck²)는 감소한다.
    양의 유한 연산의 반올림 여유32eps는 탐색 구간을 넓히는 데만 쓴다.
    최종 fraction과 엄격한1e-4 선택은 원래 연산으로 다시 평가한다.
    단조성/유한성이 성립하지 않으면 원래 전체 구간으로 돌아간다.
    """
    if (not len(k) or not np.isfinite(k).all() or np.any(k<0) or
        np.any(np.diff(k)>0) or not np.isfinite(xdis).all() or
        not np.isfinite(ydis).all() or np.any(xdis<0) or np.any(ydis<0)):
        return 0
    with np.errstate(over='ignore',invalid='ignore'):
        largest=7.782+xdis*k[0]
        denominator=largest+ydis*k[0]**2
    if not np.isfinite(denominator).all():return 0
    lo,hi=0,len(k);expansion=1+32*np.finfo(float).eps
    while lo<hi:
        mid=(lo+hi)//2;first=7.782+xdis*k[mid]
        fraction=first/(first+ydis*k[mid]**2)
        upper=np.nextafter(fraction*expansion,np.inf)
        if np.any(upper>1e-4):hi=mid
        else:lo=mid+1
    return lo


def add_dissolved_continuum(result,routes,rows,models,state,nu,*,support_policy='opacity'):
    """native 소광/SE의 서로 다른 pseudo-continuum 지지를 명시한다."""
    if support_policy not in ('opacity','native-SE'):
        raise ValueError('알 수 없는 준위 용해 소비 규약')
    b,a,x=dissolution_vectors(state);reference=ground_lte_reference(rows,models,state)
    t=state['temperature'];ns=len(t)
    absorption=np.zeros((ns,len(nu)));stimulated=np.zeros_like(absorption)
    cells=0;xdis=b*x;ydis_by_charge={}
    for route in routes:
        if not route.dissolution_primary:continue
        if route.target_energy_eV!=0:
            raise ValueError('미지원인 비바닥 PHOT 첫 route의 level dissolution')
        z=models[route.Z,route.stage]['z_screen'];edge=route.threshold_Hz
        active=(nu<edge)&(nu>edge-3.289395e15/4)
        # QUAD_MULTI_V10의 조건. GENOPAETA_V10 소광에는 이 조건이 없다.
        if support_policy=='native-SE':active &= nu>.8*edge
        if not np.any(active):continue
        sigma_edge=float(route.sigma(np.array([edge*(1+1e-14)]))[0])*route.probability
        if sigma_edge==0:continue
        bins=np.flatnonzero(active);frequency=nu[bins]
        effective=np.sqrt(3.289395e15*z*z/(edge-frequency))
        k=np.minimum(1.,16*effective/(1+effective)**2/3)
        k=(k*z**3/effective**4)**1.5
        if z not in ydis_by_charge:ydis_by_charge[z]=1.091*(x+4*(z-1)*a)*b*b
        ydis=ydis_by_charge[z]
        # 원래 지수 overflow가 발생하는 입력을 탐색 생략으로 숨기지 않는다.
        safe_exp=H*(edge-frequency[0])/(KB*float(t.min()))<np.log(np.finfo(float).max)
        start=_dissolution_candidate_start(k,xdis,ydis) if safe_exp else 0
        if start==len(k):continue
        bins=bins[start:];frequency=frequency[start:];k=k[start:]
        first=7.782+xdis[:,None]*k[None,:]
        fraction=first/(first+ydis[:,None]*k[None,:]**2)
        # 원본 MOD_LEV_DIS_BLK/GENOPAETA의 명시된 pseudo-continuum 지지 선택.
        support=fraction>1e-4
        coefficient=np.where(support,sigma_edge*fraction,0.)
        absorption[:,bins]+=coefficient*state['full'][route.lower,:,None]
        induced=reference[route.lower,:,None]*np.exp(H*(edge-frequency[None,:])/(KB*t[:,None]))
        stimulated[:,bins]+=coefficient*induced
        cells+=int(np.count_nonzero(support))
    result['chi_abs']+=absorption;result['chi_stim']+=stimulated
    result['chi_net']=result['chi_abs']-result['chi_stim']
    result['eta_spont']+=stimulated*(2*H*nu**3/(C*C))[None,:]
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('용해 연속 비유한')
    return dict(active_route_sample_terms=cells,
        chi_abs_max=float(absorption.max()),chi_stim_max=float(stimulated.max()),
        eta_spont_max=float((stimulated*(2*H*nu**3/(C*C))[None,:]).max()))


def milne_route(nu,sigma,n_lower,n_target,temperature,ne,g_lower,g_target,
                threshold,probability):
    """같은 target의 순χ와 자발η. n_lower=0이어도 재결합 방출을 남긴다."""
    nu,sigma=np.asarray(nu),np.asarray(sigma)
    t,ne=np.asarray(temperature),np.asarray(ne)
    lower,upper=np.asarray(n_lower),np.asarray(n_target)
    if (nu.ndim!=1 or sigma.shape!=nu.shape or t.ndim!=1 or
        any(a.shape!=t.shape for a in (ne,lower,upper)) or
        not all(np.isfinite(a).all() for a in (nu,sigma,t,ne,lower,upper)) or
        np.any(nu<=0) or np.any(sigma<0) or np.any(t<=0) or np.any(ne<0) or
        np.any(lower<0) or np.any(upper<0) or
        not all(np.isfinite(x) and x>0 for x in (g_lower,g_target,threshold,probability))):
        raise ValueError('잘못된 target Milne 입력')
    active=sigma>0
    if np.any(nu[active]<threshold):raise ValueError('물리 문턱 아래의 양의 PHOT 단면')
    chi_abs=probability*lower[:,None]*sigma[None,:]
    induced=np.zeros_like(chi_abs)
    if np.any(active):
        thermal_volume=(H*H/(2*np.pi*ME*KB*t))**1.5
        nstar=upper*ne*thermal_volume*(g_lower/(2*g_target))
        induced[:,active]=(probability*nstar[:,None]*sigma[None,active]*
            np.exp(-H*(nu[None,active]-threshold)/(KB*t[:,None])))
    spontaneous=induced*(2*H*nu**3/(C*C))[None,:]
    if not all(np.isfinite(a).all() for a in (chi_abs,induced,spontaneous)):
        raise ValueError('비유한 BF 연속 계수')
    # 유도항은 순흡수에서 한 번만 뺀다. 자발η에 다시 더하지 않는다.
    return chi_abs,induced,spontaneous


def continuum_block(routes,rows,state,nu):
    nu=np.asarray(nu,dtype='f8');ns=len(state['temperature'])
    sums=[np.zeros((ns,len(nu))) for _ in range(3)]
    compensation=[np.zeros_like(sums[0]) for _ in range(3)]
    for route in routes:
        if route.target>=0:
            # PHOT sum(g)는 분기확률의 권위일 뿐 SL의 LTE partition이 아니다.
            # n_SL/G_SL(T;E_anchor)=n_FL_anchor/g_anchor를 사용한다. 이 값은
            # C pair/EW의 g_anchor/f_anchor와 같은 실제 SL 재구성이다.
            upper=state['full'][route.target]
            target_g=float(rows[route.target]['g'])
        else:
            target_key=(route.Z,route.stage,-1)
            upper=state['superlevel'][state['sl_index'][target_key]]
            target_g=route.target_g
        occupation=state.get('occupation')
        if occupation is not None:
            target_occupation=occupation[route.target] if route.target>=0 else 1.
            if np.any(np.asarray(target_occupation)<=0):raise ValueError('상위 target 점유확률0')
            upper=upper*occupation[route.lower]/target_occupation
        terms=milne_route(nu,route.sigma(nu),state['full'][route.lower],upper,
            state['temperature'],state['ne'],float(rows[route.lower]['g']),
            target_g,route.threshold_Hz,route.probability)
        for total,correction,value in zip(sums,compensation,terms,strict=True):
            y=value-correction;updated=total+y
            correction[:]=(updated-total)-y;total[:]=updated
    return dict(chi_abs=sums[0],chi_stim=sums[1],chi_net=sums[0]-sums[1],eta_spont=sums[2])


def _continuum_grid_steps(routes,rows,state,nu,block_bins=1024,sigma_provider=None):
    """생산 경로: route×주파수 블록만 만들고 실제 인구와 BLAS로 수축한다.

    exp 기준점은 블록의 최저 주파수다. 더 낮은 edge의 역재결합이 나눠진
    지수 때문에 먼저 underflow하는 것을 피하고, 양의 지수 범위만 블록으로
    제한한다. 주파수·준위·인구를 삭제하거나 물리값을 제한하지 않는다.
    """
    nu=np.asarray(nu,dtype='f8');t=state['temperature'];ns=len(t);nr=len(routes)
    if (nu.ndim!=1 or not len(nu) or not np.isfinite(nu).all() or np.any(nu<=0) or
            np.any(np.diff(nu)<=0) or not 1<=block_bins<=16384 or not nr):
        raise ValueError('생산 주파수/블록 입력 오류')
    lower=np.empty((ns,nr));nstar=np.empty_like(lower)
    threshold=np.array([r.threshold_Hz for r in routes]);prob=np.array([r.probability for r in routes])
    # Route.sigma의 개별 PHOT 지지구간보다 낮으면 정확히0이다. 그런 route까지
    # 매 블록 Python 호출/행렬로 만들지 않는다. 인구나 계수 크기로 자르지 않는다.
    support_start=np.array([min((entry[2] for entry in r.models),default=np.inf) for r in routes])
    thermal=state['ne']*(H*H/(2*np.pi*ME*KB*t))**1.5
    for i,route in enumerate(routes):
        lower[:,i]=state['full'][route.lower]
        if route.target>=0:
            upper=state['full'][route.target];target_g=float(rows[route.target]['g'])
        else:
            upper=state['superlevel'][state['sl_index'][route.Z,route.stage,-1]]
            target_g=route.target_g
        occupation=state.get('occupation')
        if occupation is not None:
            target_occupation=occupation[route.target] if route.target>=0 else 1.
            if np.any(np.asarray(target_occupation)<=0):raise ValueError('상위 target 점유확률0')
            upper=upper*occupation[route.lower]/target_occupation
        nstar[:,i]=upper*thermal*(float(rows[route.lower]['g'])/(2*target_g))
    if not all(np.isfinite(a).all() and np.all(a>=0) for a in (lower,nstar)):
        raise ValueError('생산 route의 비유한/음수 인구')
    absorption=np.zeros((ns,len(nu)));stimulated=np.zeros_like(absorption)
    start=0
    while start<len(nu):
        # 공유 생산기는 각 물질의 다음 블록 시작점만 교차 실행한다. 각자의
        # T 최소값·지수 기준점·행렬 크기는 분리 계산과 동일하게 유지한다.
        yield start
        # 범위32는 exp 인수의 수치 표현을 위한 분할 기준이다. bin은 모두 남는다.
        max_frequency=nu[start]+32*KB*float(t.min())/H
        end=min(len(nu),start+block_bins,int(np.searchsorted(nu,max_frequency,side='right')))
        if end<=start:raise ValueError('수치 블록 구성 실패')
        frequency=nu[start:end]
        possible=np.flatnonzero(support_start<=frequency[-1])
        if not len(possible):
            start=end
            continue
        sigma=(np.stack([routes[i].sigma(frequency) for i in possible])*prob[possible,None]
               if sigma_provider is None else sigma_provider(start,end,possible))
        active=np.any(sigma>0,axis=1)
        if np.any(active):
            # 활성 σ가 없으면 생략하는 것은 물리적 exact zero다. 작은 인구를
            # 문턱으로 골라 버리거나 낮은 edge 전체를 버리지 않는다.
            basis=sigma[active]
            selected=possible[active]
            absorption[:,start:end]=lower[:,selected]@basis
            offset=H*(threshold[None,selected]-frequency[0])/(KB*t[:,None])
            coefficient=nstar[:,selected]*np.exp(offset)
            decay=np.exp(-H*(frequency[None,:]-frequency[0])/(KB*t[:,None]))
            stimulated[:,start:end]=(coefficient@basis)*decay
        start=end
    eta=stimulated*(2*H*nu**3/(C*C))[None,:]
    result=dict(chi_abs=absorption,chi_stim=stimulated,
                chi_net=absorption-stimulated,eta_spont=eta)
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError('생산 연속 계수 비유한')
    return result


def continuum_grid(routes,rows,state,nu,block_bins=1024):
    """한 물질의 기존 블록/합산 규약. 공유 경로의 직접 대조도 이 함수를 쓴다."""
    steps=_continuum_grid_steps(routes,rows,state,nu,block_bins)
    while True:
        try:next(steps)
        except StopIteration as completed:return completed.value


def continuum_grids_shared(routes,rows,states,nu,block_bins=1024,cache_metrics=None):
    """서로 다른 물질의 σ만 최대 두 주파수 블록으로 공유한다.

    각 물질의 exp 분할·기준점·occupation·T/ne·인구·행렬 축은 그대로다.
    전체 route×frequency 캐시는 만들지 않는다. σ의 주파수별 평가가
    입력 배열의 분할과 무관함을 원자료 직접시험으로 확인해야 한다.
    """
    if not states:raise ValueError('공유할 물질 없음')
    nu=np.asarray(nu,dtype='f8');cache={};peak=0;evaluations=0
    support=np.array([min((entry[2] for entry in r.models),default=np.inf) for r in routes])
    def sigma_block(start,end,possible):
        nonlocal peak,evaluations
        pieces=[]
        for block in range(start//block_bins,(end-1)//block_bins+1):
            lo=block*block_bins;hi=min(len(nu),lo+block_bins)
            if block not in cache:
                data=np.zeros((len(routes),hi-lo))
                for i in np.flatnonzero(support<=nu[hi-1]):
                    data[i]=routes[i].sigma(nu[lo:hi])*routes[i].probability
                    evaluations+=1
                cache[block]=data;peak=max(peak,len(cache))
            pieces.append(cache[block][possible,max(start,lo)-lo:min(end,hi)-lo])
        return pieces[0] if len(pieces)==1 else np.concatenate(pieces,axis=1)
    steps=[_continuum_grid_steps(routes,rows,state,nu,block_bins,sigma_block) for state in states]
    positions=[next(step) for step in steps];results=[None]*len(states)
    while any(position<len(nu) for position in positions):
        i=int(np.argmin(positions))
        try:positions[i]=next(steps[i])
        except StopIteration as completed:
            results[i]=completed.value;positions[i]=len(nu)
        earliest=min(positions)
        for block in tuple(cache):
            if min(len(nu),(block+1)*block_bins)<=earliest:
                del cache[block]
    if cache_metrics is not None:
        cache_metrics.update(peak_blocks=peak,route_evaluations=evaluations,
            full_block_bytes=len(routes)*block_bins*8,
            cache_bound_bytes=2*len(routes)*block_bins*8,
            physical_field_executions=0)
    return results


def consume_c_full_population(path,state):
    with path.open('rb') as stream:
        magic,nl,ns=struct.unpack('<8sII',stream.read(16))
        if magic!=b'LMATFL01' or (nl,ns)!=state['full'].shape:
            raise ValueError('C FL 입력 header/형상 불일치')
        data=np.frombuffer(stream.read(8*nl*ns),dtype='<f8').reshape(nl,ns).copy()
        if stream.read(1) or not np.isfinite(data).all() or np.any(data<0):
            raise ValueError('C FL 입력 손상')
    difference=abs(data-state['full']);scale=np.maximum(data,state['full'])
    relative=np.divide(difference,scale,out=np.zeros_like(data),where=scale>0)
    maximum=float(relative.max())
    if maximum>1e-12:
        raise ValueError(f'C FL 인구와 같은 SL/물질이 아님: max_relative={maximum}')
    state['full']=data
    return dict(source=str(path),max_relative_error=maximum,
        max_absolute_error=float(difference.max()),values=int(data.size),
        consumer='exact C imported full populations; no LTE fallback')


def write_continuum_binary(path,frequency,state,result):
    """C source 조립기가 같은 표본의 실제 η를 직접 읽는 데이터 입력."""
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sII',b'LMBFCO01',len(state['temperature']),len(frequency)))
        for values in (frequency,state['temperature'],state['ne'],result['chi_net'],result['eta_spont']):
            stream.write(np.asarray(values,dtype='<f8').tobytes())


def prepare(run,material,atomic_dir,library,out,frequency,c_full=None,block_bins=1024,export_occupation=False):
    if out.exists():raise ValueError('기존 연속 계수 산출물 보존')
    out.mkdir()
    fixed=[run/n for n in ('MODEL','MODEL_SPEC','LEVEL_SL_STEQ_LINKS','atomic_links.txt')]
    fixed += [material,atomic_dir/'levels.csv',library,Path(__file__).resolve()]
    if c_full is not None:fixed.append(c_full)
    fixed += [Path(__file__).with_name(n) for n in ('cmfgen_parser.py',
        'expand_atomic_data_cmfgen.py','cmfgen_phot_smoothing.py')]
    for link in atomic.load_cmfgen_links(run/'atomic_links.txt').values():
        fixed.extend(Path(link[k]) for k in ('osc','f_to_s'))
    fixed.extend(p for p in atomic.HYD_BF_DATA_FILES if p.is_file())
    before={str(p):sha(p) for p in fixed}
    rows,models,routes,policy,smoothing,counts=load_model(run,atomic_dir,library)
    state=read_material(material,rows,models)
    do_dis=dissolution_flag(run/'MODEL')
    if do_dis:
        occupation=material_occupation(rows,models,state)
        state=read_material(material,rows,models,occupation)
    if export_occupation:
        if any(sha(Path(p))!=v for p,v in before.items()):raise ValueError('점유확률 준비 중 입력 변경')
        target=out/'occupation.bin'
        with target.open('xb') as stream:
            stream.write(struct.pack('<8sII',b'LMATOCC1',*state['occupation'].shape))
            stream.write(state['occupation'].astype('<f8').tobytes())
        report=dict(schema='lumina-material-occupation-v1',do_lev_dissolution=do_dis,
            occupation_fit='OCCUPATION_PROB continuous fit; native near-unity quantization not imposed',
            shape=list(state['occupation'].shape),minimum=float(state['occupation'].min()),
            input_sha256=before,output_sha256={'occupation.bin':sha(target)},
            physical_RT_executions=0,final_goal='NOT_JUDGED')
        with (out/'material-occupation.json').open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
        return {k:report[k] for k in ('do_lev_dissolution','shape','minimum','final_goal')}
    comparison=consume_c_full_population(c_full,state) if c_full is not None else None
    result=continuum_grid(routes,rows,state,frequency,block_bins)
    dissolved=add_dissolved_continuum(result,routes,rows,models,state,frequency) if do_dis else None
    if any(sha(Path(p))!=v for p,v in before.items()):raise ValueError('계수 생산 중 고정 입력 변경')
    np.savez(out/'continuum.npz',nu_Hz=frequency,**result)
    write_continuum_binary(out/'continuum.bin',frequency,state,result)
    report=dict(schema='lumina-model-continuum-v1',model_ions=len(models),
        full_levels=len(rows),routes=len(routes),routes_by_ion=counts,
        cells=len(state['temperature']),frequencies=len(frequency),
        coordinate='same material samples; not a radiation-field average',
        coefficient_method='pointwise native PHOT; blocked target contraction; stimulated counted once',
        block_bins_max=block_bins,
        C_full_population_comparison=comparison,
        do_lev_dissolution=do_dis,dissolved_continuum=dissolved,
        occupation_fit='OCCUPATION_PROB continuous fit; no near-unity quantization',
        process_selection=policy,phot_preprocessing=smoothing,input_sha256=before,
        chi_net_min_max=[float(result['chi_net'].min()),float(result['chi_net'].max())],
        eta_spont_min_max=[float(result['eta_spont'].min()),float(result['eta_spont'].max())],
        physical_RT_executions=0,native_executions=0,SE_RE_solved=False,
        final_goal='NOT_JUDGED',output_sha256={n:sha(out/n) for n in ('continuum.npz','continuum.bin')})
    with (out/'continuum.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return {k:report[k] for k in ('model_ions','full_levels','routes','cells','frequencies','chi_net_min_max','eta_spont_min_max','final_goal')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','material','atomic','library','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    grids=parser.add_mutually_exclusive_group(required=True)
    grids.add_argument('--frequency-Hz',nargs='+',type=float)
    grids.add_argument('--bf-grid',action='store_true')
    parser.add_argument('--C-full-population',type=Path)
    parser.add_argument('--block-bins',type=int,default=1024)
    parser.add_argument('--export-occupation',action='store_true')
    args=parser.parse_args()
    frequency=(atomic.BF_NU_MIN*np.exp((np.arange(atomic.BF_N_FREQ_BIN)+.5)*
        np.log(atomic.BF_NU_MAX/atomic.BF_NU_MIN)/atomic.BF_N_FREQ_BIN)
        if args.bf_grid else np.asarray(args.frequency_Hz))
    print(json.dumps(prepare(args.run.resolve(),args.material.resolve(),args.atomic.resolve(),
        args.library.resolve(),args.out.resolve(),frequency,
        args.C_full_population.resolve() if args.C_full_population else None,args.block_bins,
        args.export_occupation),ensure_ascii=False))
