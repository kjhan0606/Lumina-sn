#!/usr/bin/env python3
"""동일 물질의 BF·FF·선 자발방출·전자산란을 요청 주파수에서 생산한다.

실제 수송이나 SE/RE 실행은 아니다. FF 기본은 모든 양이온이며 명시적으로
native upper 선택도 지원한다. 두 계수의 차이는 보존한다. 선은 실제 MODEL의
GENOSC 선택과 FIX_DOP profile을 사용하고 η에 탈출확률을 곱하지 않는다.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
from pathlib import Path
import re
import shutil
import struct

import numpy as np

import cmfgen_model_continuum as B


def control(model,key,default=None):
    matches=re.findall(r'^\s*(\S+)\s+\['+re.escape(key)+r'\]',model.read_text(),re.M)
    if not matches and default is not None:return default
    if len(matches)!=1:raise ValueError(f'MODEL [{key}] 누락/중복')
    return matches[0]


def number(model,key,default=None):
    value=float(control(model,key,default).replace('D','E'))
    if not np.isfinite(value):raise ValueError(f'MODEL [{key}] 비유한')
    return value


def fixed_profile(model):
    # RD_CONTROL_VARIABLES: FIX_DOP의 기본값은 T이며 GLOBAL_PROF보다 우선한다.
    if control(model,'FIX_DOP','T')!='T':raise ValueError('아직 미지원인 가변 폭 선 profile')
    tdop=number(model,'TDOP');mass=number(model,'AMASS_DOP');turb=number(model,'VTURB')
    if tdop<=0 or mass<=0 or turb<0:raise ValueError('고정 Doppler 폭 입력 오류')
    if control(model,'SCL_SL_OPAC','F')!='F':raise ValueError('미지원 SL 선 소광 재척도')
    velocity=12.85*np.sqrt(tdop/mass+(turb/12.85)**2)
    # SET_PROF_LIMITS_V4의 DOP_FIX NUM_DOP=6은 MAX_DOP/DOP_LIM과 별개다.
    # SET_LINE_OPAC의 profile 폭에는 2.998E5, 지지에는 SPEED_OF_LIGHT를 쓴다.
    return dict(kind='DOP_FIX',TDOP_1e4K=tdop,AMASS_DOP_amu=mass,VTURB_kms=turb,
        velocity_kms=float(velocity),width_fraction=float(velocity/2.998e5),
        support_fraction=float(6*velocity/(B.C*1e-5)),support_native_NUM_DOP=6,
        profile_normalization='native Gaussian; no finite-support renormalization',
        MAX_DOP=number(model,'MAX_DOP'),FRAC_DOP=number(model,'FRAC_DOP'))


def gf_selection(lower,upper,f,A,g_lower,Z,model,nlevels):
    """GENOSC_V9의 SET_ZERO 선택. 상위 준위별 최소 전이 수와 동률을 보존한다."""
    cut=number(model,'GF_CUT') if Z>=number(model,'AT_CUT') else 0.
    lev=number(model,'GF_LEV_CUT');minimum=number(model,'MIN_TRANS')
    if lev<0 or lev!=int(lev) or minimum<1 or minimum!=int(minimum) or cut<0:
        raise ValueError('MODEL 선 선택 기준 손상')
    lev,minimum=int(lev),int(minimum)
    keep=np.ones(len(f),dtype=bool)
    if not len(f) or cut==0 or max(lev,minimum)>=nlevels:return keep
    order=np.argsort(upper,kind='stable');groups=upper[order]
    starts=np.r_[0,np.flatnonzero(np.diff(groups))+1,len(order)]
    for begin,end in zip(starts[:-1],starts[1:],strict=True):
        indices=order[begin:end];up=int(upper[indices[0]])
        if up<max(lev,minimum):continue
        # native DOWN에는 누락된 전이의0도 있다. 양의 A가 minimum보다
        # 적으면 순위 문턱은0이며 추가로 지울 수 있는 양의 선은 없다.
        positive=A[indices][A[indices]>0]
        threshold=np.partition(positive,-minimum)[-minimum] if len(positive)>=minimum else 0.
        remove=(g_lower[indices]*f[indices]<cut)&(lower[indices]+1>lev)&(A[indices]<threshold)
        keep[indices[remove]]=False
    return keep


def native_line_domain(run):
    """SET_FREQUENCY_GRID_V2의 최종 복사 선 roster 범위(SN frozen 모델).

    INS_LINE_V6의 격자 삽입용 제외 수와 혼동하지 않는다. VEL_LAW11/12의
    실제 외곽 V(1)은 MODEL의 명목 VINF가 아니라 보존 RVTJ에서 얻는다.
    충돌 전이를 제외하는 규칙이 아니며 부족한 J를 0으로 채우지도 않는다.
    """
    model=run/'MODEL';rvtj=run/'RVTJ'
    before={str(p):B.sha(p) for p in (model,rvtj)}
    if control(model,'RD_CF_FILE')!='F' or number(model,'VEL_LAW') not in (11,12):
        raise ValueError('원 continuum 격자/SN 외곽 속도 선택을 확인해야 하는 입력')
    minimum=number(model,'MIN_CF')*1.e15
    if minimum<=0:raise ValueError('MIN_CF는 양수여야 함')
    lines=rvtj.read_text().splitlines()
    dimensions=re.findall(r'^\s*ND:\s*(\d+)\s*$', '\n'.join(lines),re.M)
    markers=[i for i,s in enumerate(lines) if s.strip()=='Velocity (km/s)']
    if len(dimensions)!=1 or int(dimensions[0])<2 or len(markers)!=1:
        raise ValueError('RVTJ 깊이/속도 헤더 누락·중복')
    nd=int(dimensions[0]);values=[]
    for line in lines[markers[0]+1:]:
        try:values.extend(float(s.replace('D','E')) for s in line.split())
        except ValueError:break
        if len(values)>=nd:break
    velocity=np.asarray(values,dtype='f8')
    if (len(velocity)!=nd or not np.isfinite(velocity).all() or np.any(velocity<=0)
            or np.any(np.diff(velocity)>=0)):
        raise ValueError('RVTJ 실제 외곽 우선 속도 격자 손상')
    vinf=float(velocity[0]);cutoff=minimum*(1.+1.1*vinf/2.998e5)
    if not np.isfinite(cutoff):raise ValueError('복사 선 하한 수치 범위 초과')
    if any(B.sha(Path(p))!=digest for p,digest in before.items()):
        raise ValueError('복사 선 범위 판독 중 입력 변경')
    return dict(policy='SET_FREQUENCY_GRID_V2_FINAL_IR_ROSTER',
        minimum_continuum_Hz=minimum,outer_velocity_kms=vinf,
        outer_velocity_source='RVTJ V(1); text precision, not nominal MODEL VINF',
        minimum_line_Hz_exclusive=cutoff,comparison='nu0 > cutoff',
        collision_selection_changed=False,source_sha256=before)


def native_radiative_selection(nu,domain):
    nu=np.asarray(nu,dtype='f8');cutoff=float(domain['minimum_line_Hz_exclusive'])
    if nu.ndim!=1 or not np.isfinite(nu).all() or np.any(nu<=0) or not np.isfinite(cutoff) or cutoff<=0:
        raise ValueError('복사 선 주파수/선택 범위 오류')
    return nu>cutoff


def load_lines(run,rows,models,*,line_domain=None):
    """반올림 CSV 파장이 아니라 실제 OSC 에너지와 f로 선을 다시 구성한다."""
    links=B.atomic.load_cmfgen_links(run/'atomic_links.txt')
    lookup={(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
            for i,r in enumerate(rows)}
    domain=native_line_domain(run) if line_domain is None else line_domain
    collected={k:[] for k in ('lower','upper','nu0','f','A','g_ratio')};counts={}
    identities=dict(domain['source_sha256'])
    for (Z,stage),model in sorted(models.items()):
        path=Path(links[Z,stage]['osc']);before=B.sha(path)
        osc=B.parse_osc(path);n=int(model['n_full']);trans=osc.transitions
        if len(trans)!=osc.declared_n_transitions:raise ValueError(f'OSC 전이 선언/판독 수 불일치: {path}')
        trans=trans[(trans['i']>=1)&(trans['i']<=n)&(trans['j']>=1)&(trans['j']<=n)]
        low=trans['i'].astype('i4')-1;up=trans['j'].astype('i4')-1;f=abs(trans['f'])
        if np.any(low>=up):raise ValueError('native GENOSC가 거부하는 전이 준위 순서')
        if len(np.unique(low.astype('i8')*n+up))!=len(low):
            raise ValueError('중복 OSC 전이: native 덮어쓰기 규약을 확인해야 함')
        nu=(osc.levels['E_cm'][up]-osc.levels['E_cm'][low])*B.C
        if np.any((nu<=0)&(f>0)):raise ValueError('양의 f를 가진0/음수 전이 주파수')
        gl=osc.levels['g'][low];gu=osc.levels['g'][up]
        A=B.atomic.A_PREFACTOR*f*(gl/gu)*nu**2
        gf_keep=gf_selection(low,up,f,A,gl,Z,run/'MODEL',n)&(f>0)
        # 0인 OSC 항은 주파수0일 수 있으므로 양의 f만 범위를 검사한다.
        in_domain=np.zeros(len(f),dtype=bool)
        in_domain[f>0]=native_radiative_selection(nu[f>0],domain)
        keep=gf_keep&in_domain
        global_rows=np.array([lookup[Z,stage,l+1] for l in range(n)],dtype='i4')
        values=dict(lower=global_rows[low[keep]],upper=global_rows[up[keep]],
            nu0=nu[keep],f=f[keep],A=A[keep],g_ratio=(gl/gu)[keep])
        for key in collected:collected[key].append(values[key])
        counts[f'Z{Z}_stage{stage}']=dict(raw_in_model=int(len(f)),
            raw_nonzero=int(np.count_nonzero(f)),retained_after_GF=int(gf_keep.sum()),
            omitted_by_native_IR=int(np.count_nonzero(gf_keep&~in_domain)),retained=int(keep.sum()),
            omitted_by_native_selection=int(np.count_nonzero((f>0)&~keep)))
        if before!=B.sha(path):raise ValueError('선 생성 중 원자료 변경')
        identities[str(path)]=before
    if any(B.sha(Path(p))!=v for p,v in identities.items()):raise ValueError('선 roster 생산 중 입력 변경')
    return {k:np.ascontiguousarray(np.concatenate(v)) for k,v in collected.items()},counts,identities


class SourceKernels:
    def __init__(self,library):
        self.path=Path(library).resolve(strict=True);self.dll=ct.CDLL(str(self.path))
        dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int32)
        self.gaunt_function=self.dll.lumina_native_gaunt
        self.gaunt_function.argtypes=[ct.c_int,ct.c_int,dp,dp,ct.c_double,dp]
        self.gaunt_function.restype=None
        self.line_function=self.dll.lumina_model_gaussian_sources
        self.line_function.argtypes=[ct.c_size_t]*4+[dp,dp,ip,ip]+[dp]*4+[ct.c_double]*4+[dp,dp]
        self.line_function.restype=ct.c_int

    @staticmethod
    def pointer(a,kind=ct.c_double):return a.ctypes.data_as(ct.POINTER(kind))

    def gaunt(self,nu,temperature,z):
        nu=np.ascontiguousarray(nu,dtype='f8');t=np.ascontiguousarray(temperature,dtype='f8')
        if (nu.ndim!=1 or t.ndim!=1 or not len(nu) or not len(t) or
            not np.isfinite(nu).all() or not np.isfinite(t).all() or np.any(nu<=0) or
            np.any(t<=0) or not np.isfinite(z) or z<=0 or z!=int(z)):
            raise ValueError('Gaunt 입력 손상')
        result=np.empty((len(t),len(nu)),order='F')
        self.gaunt_function(len(t),len(nu),self.pointer(nu),self.pointer(t),z,self.pointer(result))
        if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('Gaunt 출력 손상')
        return result

    def lines(self,nu,population,lines,profile):
        nu=np.ascontiguousarray(nu,dtype='f8');pop=np.ascontiguousarray(population,dtype='f8')
        if pop.ndim!=2:raise ValueError('FL 인구 형상 손상')
        arrays={k:np.ascontiguousarray(v,dtype='i4' if k in ('lower','upper') else 'f8')
                for k,v in lines.items()}
        nt=len(arrays['lower'])
        if set(arrays)!=set(('lower','upper','nu0','f','A','g_ratio')) or any(
            a.shape!=(nt,) for a in arrays.values()):raise ValueError('선 배열 형상 손상')
        chi=np.empty((pop.shape[1],len(nu)));eta=np.empty_like(chi)
        if nt==0:
            chi.fill(0);eta.fill(0)
            return chi,eta
        opacity=np.pi*B.atomic.E_CGS**2/(B.ME*B.C)
        status=self.line_function(pop.shape[0],pop.shape[1],len(nu),nt,
            self.pointer(nu),self.pointer(pop),self.pointer(arrays['lower'],ct.c_int32),
            self.pointer(arrays['upper'],ct.c_int32),
            *[self.pointer(arrays[k]) for k in ('nu0','f','A','g_ratio')],
            profile['width_fraction'],profile['support_fraction'],opacity,B.H/(4*np.pi),
            self.pointer(chi),self.pointer(eta))
        if status:raise ValueError(f'국소 선 계수 생산 실패: status={status}')
        return chi,eta


def free_free(nu,state,models,kernels):
    nu=np.asarray(nu,dtype='f8');t=state['temperature'];ns=len(t)
    x=B.H*nu[None,:]/(B.KB*t[:,None]);decay=np.exp(-x)
    stim=-np.expm1(-x);all_sum=np.zeros((ns,len(nu)));native_sum=np.zeros_like(all_sum)
    by_charge={};native_by_charge={}
    # 양이온 총수와 전하합을 혼동하지 않는다. terminal도 실제 물질 변수다.
    for Z,charge,sl in state['columns']:
        charge=int(charge)
        if charge<=0:continue
        value=state['superlevel'][state['sl_index'][int(Z),charge,int(sl)]]
        by_charge.setdefault(charge,np.zeros(ns));by_charge[charge]+=value
        # COMP_OPAC는 존재하는 lower PHOT 이온의 upper 인구에만 FF를 더한다.
        if (int(Z),charge) in models:
            native_by_charge.setdefault(charge,np.zeros(ns));native_by_charge[charge]+=value
    for charge,total in sorted(by_charge.items()):
        gaunt=kernels.gaunt(nu,t,float(charge))
        all_sum+=charge**2*total[:,None]*gaunt
        if charge in native_by_charge:
            native_sum+=charge**2*native_by_charge[charge][:,None]*gaunt
    # GENOPAETA_V10의 CHIFF=3.69e-29, ν15,T4,χnative→cgs 단위 변환.
    prefactor=3.69e8*state['ne'][:,None]/np.sqrt(t[:,None])
    chi=prefactor*all_sum*stim/nu[None,:]**3
    eta=prefactor*all_sum*decay*(2*B.H/B.C**2)
    native_chi=prefactor*native_sum*stim/nu[None,:]**3
    native_eta=prefactor*native_sum*decay*(2*B.H/B.C**2)
    if not all(np.isfinite(a).all() and np.all(a>=0) for a in (chi,eta,native_chi,native_eta)):
        raise ValueError('FF 인구/계수 비유한 또는 음수')
    return dict(chi=chi,eta=eta,native_chi=native_chi,native_eta=native_eta,
        charges=sorted(by_charge),native_charges=sorted(native_by_charge))


def continuum_frequencies(nu,evaluation=None):
    """명시적 CONT_FREQ 대응만 소비한다. 새 격자로 대응을 외삽하지 않는다."""
    nu=np.asarray(nu,dtype='f8')
    if (nu.ndim!=1 or not len(nu) or not np.isfinite(nu).all() or
            np.any(nu<=0) or np.any(np.diff(nu)<=0)):
        raise ValueError('요청 주파수는 양의 유한 오름차순 벡터여야 함')
    if evaluation is None:return nu,None
    evaluation=np.asarray(evaluation,dtype='f8')
    if (evaluation.shape!=nu.shape or not np.isfinite(evaluation).all() or
            np.any(evaluation<=0)):
        raise ValueError('CONT_FREQ 대응의 형상/주파수 손상')
    return np.unique(evaluation,return_inverse=True)


def rescale_continuum(nu,evaluation,temperature,chi,eta):
    """COMP_OPAC의 진η·유도항 보정. 선/ES에는 이 보정을 적용하지 않는다.

    계수는 이미 각 요청점의 CONT_FREQ 열로 선택된 상태다. 부호 있는 χ를
    보존하며 음수값의 clamp/floor나 적합 상수는 사용하지 않는다.
    """
    nu=np.asarray(nu,dtype='f8');evaluation=np.asarray(evaluation,dtype='f8')
    continuum_frequencies(nu,evaluation)
    temperature=np.asarray(temperature,dtype='f8')
    chi=np.asarray(chi,dtype='f8');eta=np.asarray(eta,dtype='f8')
    if (temperature.ndim!=1 or not len(temperature) or
            not np.isfinite(temperature).all() or np.any(temperature<=0) or
            chi.shape!=(len(temperature),len(nu)) or eta.shape!=chi.shape or
            not np.isfinite(chi).all() or not np.isfinite(eta).all() or np.any(eta<0)):
        raise ValueError('CONT_FREQ 보정의 물질/계수 손상')
    if np.array_equal(nu,evaluation):return chi,eta
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        mapped_eta=eta*(nu/evaluation)[None,:]**3*np.exp(
            -B.H*(nu-evaluation)[None,:]/(B.KB*temperature[:,None]))
        mapped_chi=chi+(eta/(2*B.H*evaluation**3/B.C**2)[None,:]-
                       mapped_eta/(2*B.H*nu**3/B.C**2)[None,:])
    unchanged=nu==evaluation
    mapped_chi[:,unchanged]=chi[:,unchanged];mapped_eta[:,unchanged]=eta[:,unchanged]
    if not np.isfinite(mapped_chi).all() or not np.isfinite(mapped_eta).all():
        raise ValueError('CONT_FREQ 보정 출력 비유한')
    return mapped_chi,mapped_eta


def inactive_processes(run,models,policy):
    model=run/'MODEL'
    if control(model,'INC_XRAYS')!='F' or control(model,'INC_RAY')!='F':
        raise ValueError('현재 생산자가 지원하지 않는 X-ray/Rayleigh 과정 활성')
    if control(model,'ADD_OPAC','F')!='F':raise ValueError('추가 소광 활성')
    if any(p['external_dr_fit_allowed'] or p['die_auto'] or p['die_wi'] for p in policy['ions']):
        raise ValueError('미지원 독립 DIE/FF resonance 활성')
    # 이 파일의 실제 2-photon 목록이 H/He이고 실제 모델에 둘 다 없음을 확인한다.
    two=run/'TWO_PHOT_DATA';text=two.read_text()
    count=re.findall(r'^\s*(\d+)\s+!Number of 2-photon transitions',text,re.M)
    identities=re.findall(r'^\s*(\S+).*?!2-photon identification',text,re.M)
    if count!=[str(len(identities))] or not set(identities)<=set(('HI','HeI','He2')) or any(Z<=2 for Z,_ in models):
        raise ValueError('2-photon 물리적 비활성을 입증할 수 없음')
    return dict(xrays='MODEL F',rayleigh='MODEL F',additional_opacity='MODEL F/default F',
        two_photon=dict(model_flag=control(model,'INC_TWO'),input=str(two),sha256=B.sha(two),
            listed_ions=identities,active_in_model=0),
        ff_resonances='all actual DIE flags F/F; RD_PHOT_DIE not called')


def write_binary(path,nu,state,components):
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sII',b'LMPHYS01',len(state['temperature']),len(nu)))
        for value in (nu,state['temperature'],state['ne'],*[components[k] for k in
            ('bf_chi','bf_eta','ff_chi','ff_eta','bb_chi','bb_eta','es_chi')]):
            stream.write(np.asarray(value,dtype='<f8').tobytes())


def source_memory_inventory(cells,frequencies,routes,block_bins=1024,continuum_frequencies=None):
    """한/두 상태 BF 보존 + 한 상태 후처리의 알려진 동시 할당을 센다.

    후처리는 순차이며 dissolution/FF의 배열·식 임시값/직렬화를 최대12개
    상태×주파수 배열로 잡는다. 캐시2개와 조립/선택2개를 따로 합한다.
    원자자료/line catalogue/인구/BLAS/runtime에는4GiB를 남긴다.
    실제 전체 격자 peak RSS를 측정했다는 뜻은 아니다.
    """
    if not 1<=len(cells)<=2 or min(cells)<1 or frequencies<2 or routes<1:
        raise ValueError('공유 생산 메모리 차원 손상')
    parts=dict(states_BF=4*sum(cells)*frequencies*8,
        one_state_postprocessing=12*max(cells)*frequencies*8,
        sigma_cache_and_assembly=4*routes*block_bins*8,
        atomic_catalogue_populations_runtime=4*1024**3)
    if continuum_frequencies is not None:
        if not 1<=continuum_frequencies<=frequencies:
            raise ValueError('CONT_FREQ 메모리 차원 손상')
        parts['states_BF']=4*sum(cells)*continuum_frequencies*8
        # 선택 열·BF/FF 보정 출력·exp/Planck 임시값을 포함한 보수적 상한.
        parts['one_state_postprocessing']=32*max(cells)*frequencies*8
    return dict(parts=parts,known_allocation_bound_bytes=sum(parts.values()),
        whole_process_peak_RSS_verified=False,sequential_state_postprocessing=True)


def paired_memory_inventory(cells,frequencies,routes,block_bins=1024,continuum_frequencies=None):
    """기존 두 상태 공개 형식과 수치를 보존한다."""
    if len(cells)!=2:raise ValueError('두 상태 계수 생산 재고 필요')
    result=source_memory_inventory(cells,frequencies,routes,block_bins,continuum_frequencies)
    result['parts']['both_states_BF']=result['parts'].pop('states_BF')
    return result


def source_output_reserve(cells,frequencies,free_free_policy='ALL_POSITIVE_IONS'):
    """binary7평면+NPZ9평면·최대2ν벡터, catalogue/헤더 여유를 센다."""
    if cells<1 or frequencies<2 or free_free_policy not in ('ALL_POSITIVE_IONS','NATIVE_UPPER_IONS'):
        raise ValueError('계수 출력 차원/FF 정책 오류')
    # NPZ에는 공통7평면 외에 ff_native_chi/eta도 저장한다. 명시 CONT_FREQ의
    # 추가ν 벡터도 포함하며 catalogue와 zip/header에는 별도1GiB를 남긴다.
    planes=18 if free_free_policy=='NATIVE_UPPER_IONS' else 16
    return 16+8*(3*frequencies+2*cells+planes*cells*frequencies)+1024**3


def prepare(run,material,atomic_dir,phot_library,kernel_library,c_full,nu,out,frequency_grid=None,
            peer=None,continuum_frequency_grid=None,free_free_policy='ALL_POSITIVE_IONS'):
    """peer=(물질, C full 인구, 출력)일 때 σ만 공유하고 각 상태는 따로 쓴다."""
    if free_free_policy not in ('ALL_POSITIVE_IONS','NATIVE_UPPER_IONS'):
        raise ValueError('명시한 지원 FF 계수 정책 필요')
    nu=np.asarray(nu,dtype='f8')
    evaluation=(np.load(continuum_frequency_grid,allow_pickle=False)
                if continuum_frequency_grid is not None else None)
    continuum_nu,continuum_index=continuum_frequencies(nu,evaluation)
    requests=[(material,c_full,out)]
    if peer is not None:requests.append(peer)
    if len({p[2].resolve() for p in requests})!=len(requests) or any(p[2].exists() for p in requests):
        raise ValueError('기존/중복 물리 계수 산출물 보존')
    for _,_,folder in requests:folder.mkdir()
    fixed=[run/n for n in ('MODEL','MODEL_SPEC','LEVEL_SL_STEQ_LINKS','atomic_links.txt','TWO_PHOT_DATA')]
    fixed += [material,atomic_dir/'levels.csv',phot_library,kernel_library,c_full,Path(__file__).resolve()]
    if peer is not None:fixed.extend(peer[:2])
    if frequency_grid is not None:fixed.append(Path(frequency_grid))
    if continuum_frequency_grid is not None:fixed.append(Path(continuum_frequency_grid))
    fixed += [Path(__file__).with_name(n) for n in ('cmfgen_model_continuum.py','cmfgen_parser.py',
        'expand_atomic_data_cmfgen.py','cmfgen_phot_smoothing.py')]
    for link in B.atomic.load_cmfgen_links(run/'atomic_links.txt').values():
        fixed.extend(Path(link[k]) for k in ('osc','f_to_s'))
    fixed.extend(p for p in B.atomic.HYD_BF_DATA_FILES if p.is_file())
    before={str(p):B.sha(p) for p in fixed}
    if frequency_grid is not None and not np.array_equal(np.load(frequency_grid,allow_pickle=False),nu):
        raise ValueError('선택한 주파수 격자 파일과 실제 생산 주파수 불일치')
    if continuum_frequency_grid is not None and not np.array_equal(
            np.load(continuum_frequency_grid,allow_pickle=False),evaluation):
        raise ValueError('CONT_FREQ 파일과 실제 생산 대응 불일치')
    rows,models,routes,policy,smoothing,route_counts=B.load_model(run,atomic_dir,phot_library)
    inactive=inactive_processes(run,models,policy);profile=fixed_profile(run/'MODEL')
    states=[];full_comparisons=[]
    for mat,full,_ in requests:
        state=B.read_material(mat,rows,models)
        if B.dissolution_flag(run/'MODEL'):
            state=B.read_material(mat,rows,models,B.material_occupation(rows,models,state))
        full_comparisons.append(B.consume_c_full_population(full,state));states.append(state)
    inventory=paired_memory_inventory if peer is not None else source_memory_inventory
    memory=inventory([len(s['temperature']) for s in states],len(nu),len(routes),
                     continuum_frequencies=len(continuum_nu) if evaluation is not None else None)
    if memory['known_allocation_bound_bytes']>32*1024**3:
        raise ValueError('계수 생산의 알려진 할당이 봉인32GiB 초과')
    if peer is None and shutil.disk_usage(out).free<source_output_reserve(len(states[0]['temperature']),len(nu),free_free_policy):
        raise OSError('새 단일 상태 계수의 두 원출력과 catalogue 보존 공간 부족')
    cache_metrics={}
    bfs=([B.continuum_grid(routes,rows,states[0],continuum_nu)] if peer is None else
         B.continuum_grids_shared(routes,rows,states,continuum_nu,cache_metrics=cache_metrics))
    summaries=[]
    for i,(_,_,folder) in enumerate(requests):
        summaries.append(_finish_sources(run,nu,folder,states[i],bfs[i],rows,models,routes,
            policy,smoothing,route_counts,full_comparisons[i],inactive,profile,before,
            kernel_library,dict(cache=cache_metrics,memory=memory) if peer is not None else None,
            continuum_nu,continuum_index,continuum_frequency_grid,memory,free_free_policy))
        bfs[i]=None  # 다음 상태의 FF/선 생산 전에 앞 상태의 큰 BF 배열을 해제한다.
    return summaries[0] if peer is None else dict(states=summaries,shared_sigma=cache_metrics,memory=memory)


def _finish_sources(run,nu,out,state,bf,rows,models,routes,policy,smoothing,route_counts,
                    full_comparison,inactive,profile,before,kernel_library,cache_metrics,
                    continuum_nu=None,continuum_index=None,continuum_frequency_grid=None,allocation_inventory=None,
                    free_free_policy='ALL_POSITIVE_IONS'):
    if free_free_policy not in ('ALL_POSITIVE_IONS','NATIVE_UPPER_IONS'):
        raise ValueError('지원하지 않는 FF 게시 정책')
    if continuum_nu is None:continuum_nu=nu
    dissolved=B.add_dissolved_continuum(bf,routes,rows,models,state,continuum_nu) if B.dissolution_flag(run/'MODEL') else None
    kernels=SourceKernels(kernel_library)
    ff=free_free(continuum_nu,state,models,kernels)
    if continuum_index is not None:
        evaluation=continuum_nu[continuum_index]
        bf['chi_net'],bf['eta_spont']=rescale_continuum(nu,evaluation,state['temperature'],
            bf['chi_net'][:,continuum_index],bf['eta_spont'][:,continuum_index])
        for chi,eta in (('chi','eta'),('native_chi','native_eta')):
            ff[chi],ff[eta]=rescale_continuum(nu,evaluation,state['temperature'],
                ff[chi][:,continuum_index],ff[eta][:,continuum_index])
    line_domain=native_line_domain(run)
    lines,line_counts,osc_identities=load_lines(run,rows,models,line_domain=line_domain)
    bb_chi,bb_eta=kernels.lines(nu,state['full'],lines,profile)
    es=np.broadcast_to(state['ne'][:,None]*6.6524587321e-25,bb_chi.shape)
    native_ff=free_free_policy=='NATIVE_UPPER_IONS'
    components=dict(bf_chi=bf['chi_net'],bf_eta=bf['eta_spont'],
        ff_chi=ff['native_chi'] if native_ff else ff['chi'],
        ff_eta=ff['native_eta'] if native_ff else ff['eta'],bb_chi=bb_chi,bb_eta=bb_eta,es_chi=es)
    if any(B.sha(Path(p))!=v for p,v in before.items()):raise ValueError('물리 계수 생산 중 입력 변경')
    np.savez(out/'line_catalog.npz',**lines)
    evaluation_payload=({'continuum_nu_Hz':continuum_nu[continuum_index]}
                        if continuum_index is not None else {})
    if native_ff:
        evaluation_payload.update(ff_all_positive_chi=ff['chi'],ff_all_positive_eta=ff['eta'])
    np.savez(out/'sources.npz',nu_Hz=nu,**evaluation_payload,**components,
        ff_native_chi=ff['native_chi'],ff_native_eta=ff['native_eta'])
    write_binary(out/'sources.bin',nu,state,components)
    total_chi=components['bf_chi']+components['ff_chi']+bb_chi+es
    total_eta=components['bf_eta']+components['ff_eta']+bb_eta
    report=dict(schema='lumina-model-local-sources-v1',model_ions=len(models),
        full_levels=len(rows),routes=len(routes),routes_by_ion=route_counts,
        lines=len(lines['lower']),lines_by_ion=line_counts,cells=len(state['temperature']),frequencies=len(nu),
        source='same actual C full populations; unescaped local eta; stimulated counted once',
        evaluation='requested frequency POINT samples, not frequency-bin averages or a resolved field',
        continuum_evaluation=dict(mode='explicit-CONT_FREQ' if continuum_index is not None else 'direct',
            frequency_grid=str(continuum_frequency_grid) if continuum_frequency_grid is not None else None,
            unique_frequencies=len(continuum_nu),
            correction='native COMP_OPAC thermal eta/stimulated correction; BB and ES at requested nu',
            mapping_inferred=False,continuum_sampling_is_physical_truth=False),
        C_full_population_comparison=full_comparison,phot_preprocessing=smoothing,
        profile=profile,line_domain=line_domain,dissolved_continuum=dissolved,inactive_processes=inactive,
        free_free=dict(policy=free_free_policy,coefficient_cgs=3.69e8,
            native_selection='existing lower PHOT ions -> upper populations only',
            charges=ff['charges'],native_charges=ff['native_charges'],
            gaunt='original GFF_VEC including its explicit finite-table domain treatment',
            missing_native_chi_max=float((ff['chi']-ff['native_chi']).max())),
        signed_line_chi_negative_samples=int(np.count_nonzero(bb_chi<0)),
        chi_total_min_max=[float(total_chi.min()),float(total_chi.max())],
        eta_total_min_max=[float(total_eta.min()),float(total_eta.max())],
        total_negative_samples=int(np.count_nonzero(total_chi<0)),
        native_differences=['CODATA constants vs native historical constants',
            *([] if native_ff else ['all-positive-ion FF vs lower-PHOT-target-only FF']),
            'signed raw line chi; native zero replacement/SRCE_CHK not applied'],
        process_selection=policy,input_sha256=before,osc_sha256=osc_identities,
        shared_continuum_sigma=cache_metrics,
        allocation_inventory=allocation_inventory,
        output_sha256={n:B.sha(out/n) for n in ('sources.bin','sources.npz','line_catalog.npz')},
        physical_RT_executions=0,native_executions=0,SE_RE_solved=False,final_goal='NOT_JUDGED')
    with (out/'sources.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return {k:report[k] for k in ('model_ions','full_levels','routes','lines','cells','frequencies',
        'chi_total_min_max','total_negative_samples','final_goal')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','material','atomic','phot-library','kernel-library','C-full-population','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    grid=parser.add_mutually_exclusive_group(required=True)
    grid.add_argument('--frequency-Hz',nargs='+',type=float);grid.add_argument('--bf-grid',action='store_true')
    grid.add_argument('--frequency-grid',type=Path,help='Hz 단위 오름차순 .npy 전체 격자')
    parser.add_argument('--continuum-frequency-grid',type=Path,
        help='요청 주파수마다 명시한 CONT_FREQ(Hz) .npy; 기본은 직접 평가')
    parser.add_argument('--free-free-policy',choices=('ALL_POSITIVE_IONS','NATIVE_UPPER_IONS'),default='ALL_POSITIVE_IONS')
    for name in ('peer-material','peer-C-full-population','peer-out'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    peer=(args.peer_material,args.peer_C_full_population,args.peer_out)
    if any(p is not None for p in peer) and not all(p is not None for p in peer):
        parser.error('peer 물질/C 인구/출력 세 항목을 함께 지정해야 함')
    nu=(B.atomic.BF_NU_MIN*np.exp((np.arange(B.atomic.BF_N_FREQ_BIN)+.5)*
        np.log(B.atomic.BF_NU_MAX/B.atomic.BF_NU_MIN)/B.atomic.BF_N_FREQ_BIN)
        if args.bf_grid else np.load(args.frequency_grid,allow_pickle=False)
        if args.frequency_grid is not None else np.asarray(args.frequency_Hz,dtype='f8'))
    print(json.dumps(prepare(args.run.resolve(),args.material.resolve(),args.atomic.resolve(),
        args.phot_library.resolve(),args.kernel_library.resolve(),args.C_full_population.resolve(),
        nu,args.out.resolve(),args.frequency_grid.resolve() if args.frequency_grid else None,
        tuple(p.resolve() for p in peer) if peer[0] is not None else None,
        args.continuum_frequency_grid.resolve() if args.continuum_frequency_grid else None,
        args.free_free_policy),ensure_ascii=False))
