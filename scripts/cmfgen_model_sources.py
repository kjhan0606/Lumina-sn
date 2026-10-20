#!/usr/bin/env python3
"""동일 물질의 BF·FF·선 자발방출·전자산란을 요청 주파수에서 생산한다.

실제 수송이나 SE/RE 실행은 아니다. FF는 모든 양이온을 합하며 원본
COMP_OPAC의 PHOT target 선택과의 차이를 별도 보존한다. 선은 실제 MODEL의
GENOSC 선택과 FIX_DOP profile을 사용하고 η에 탈출확률을 곱하지 않는다.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
from pathlib import Path
import re
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


def load_lines(run,rows,models):
    """반올림 CSV 파장이 아니라 실제 OSC 에너지와 f로 선을 다시 구성한다."""
    links=B.atomic.load_cmfgen_links(run/'atomic_links.txt')
    lookup={(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
            for i,r in enumerate(rows)}
    collected={k:[] for k in ('lower','upper','nu0','f','A','g_ratio')};counts={};identities={}
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
        keep=gf_selection(low,up,f,A,gl,Z,run/'MODEL',n)&(f>0)
        global_rows=np.array([lookup[Z,stage,l+1] for l in range(n)],dtype='i4')
        values=dict(lower=global_rows[low[keep]],upper=global_rows[up[keep]],
            nu0=nu[keep],f=f[keep],A=A[keep],g_ratio=(gl/gu)[keep])
        for key in collected:collected[key].append(values[key])
        counts[f'Z{Z}_stage{stage}']=dict(raw_in_model=int(len(f)),
            raw_nonzero=int(np.count_nonzero(f)),retained=int(keep.sum()),
            omitted_by_native_selection=int(np.count_nonzero((f>0)&~keep)))
        if before!=B.sha(path):raise ValueError('선 생성 중 원자료 변경')
        identities[str(path)]=before
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


def prepare(run,material,atomic_dir,phot_library,kernel_library,c_full,nu,out):
    if out.exists():raise ValueError('기존 물리 계수 산출물 보존')
    out.mkdir()
    fixed=[run/n for n in ('MODEL','MODEL_SPEC','LEVEL_SL_STEQ_LINKS','atomic_links.txt','TWO_PHOT_DATA')]
    fixed += [material,atomic_dir/'levels.csv',phot_library,kernel_library,c_full,Path(__file__).resolve()]
    fixed += [Path(__file__).with_name(n) for n in ('cmfgen_model_continuum.py','cmfgen_parser.py',
        'expand_atomic_data_cmfgen.py','cmfgen_phot_smoothing.py')]
    for link in B.atomic.load_cmfgen_links(run/'atomic_links.txt').values():
        fixed.extend(Path(link[k]) for k in ('osc','f_to_s'))
    fixed.extend(p for p in B.atomic.HYD_BF_DATA_FILES if p.is_file())
    before={str(p):B.sha(p) for p in fixed}
    rows,models,routes,policy,smoothing,route_counts=B.load_model(run,atomic_dir,phot_library)
    inactive=inactive_processes(run,models,policy);profile=fixed_profile(run/'MODEL')
    state=B.read_material(material,rows,models)
    if B.dissolution_flag(run/'MODEL'):
        state=B.read_material(material,rows,models,B.material_occupation(rows,models,state))
    full_comparison=B.consume_c_full_population(c_full,state)
    bf=B.continuum_grid(routes,rows,state,nu)
    dissolved=B.add_dissolved_continuum(bf,routes,rows,models,state,nu) if B.dissolution_flag(run/'MODEL') else None
    kernels=SourceKernels(kernel_library)
    ff=free_free(nu,state,models,kernels)
    lines,line_counts,osc_identities=load_lines(run,rows,models)
    bb_chi,bb_eta=kernels.lines(nu,state['full'],lines,profile)
    es=np.broadcast_to(state['ne'][:,None]*6.6524587321e-25,bb_chi.shape)
    components=dict(bf_chi=bf['chi_net'],bf_eta=bf['eta_spont'],ff_chi=ff['chi'],
        ff_eta=ff['eta'],bb_chi=bb_chi,bb_eta=bb_eta,es_chi=es)
    if any(B.sha(Path(p))!=v for p,v in before.items()):raise ValueError('물리 계수 생산 중 입력 변경')
    np.savez(out/'line_catalog.npz',**lines)
    np.savez(out/'sources.npz',nu_Hz=nu,**components,ff_native_chi=ff['native_chi'],ff_native_eta=ff['native_eta'])
    write_binary(out/'sources.bin',nu,state,components)
    total_chi=components['bf_chi']+components['ff_chi']+bb_chi+es
    total_eta=components['bf_eta']+components['ff_eta']+bb_eta
    report=dict(schema='lumina-model-local-sources-v1',model_ions=len(models),
        full_levels=len(rows),routes=len(routes),routes_by_ion=route_counts,
        lines=len(lines['lower']),lines_by_ion=line_counts,cells=len(state['temperature']),frequencies=len(nu),
        source='same actual C full populations; unescaped local eta; stimulated counted once',
        evaluation='requested frequency POINT samples, not frequency-bin averages or a resolved field',
        C_full_population_comparison=full_comparison,phot_preprocessing=smoothing,
        profile=profile,dissolved_continuum=dissolved,inactive_processes=inactive,
        free_free=dict(policy='ALL_POSITIVE_IONS',coefficient_cgs=3.69e8,
            native_selection='existing lower PHOT ions -> upper populations only',
            charges=ff['charges'],native_charges=ff['native_charges'],
            gaunt='original GFF_VEC including its explicit finite-table domain treatment',
            missing_native_chi_max=float((ff['chi']-ff['native_chi']).max())),
        signed_line_chi_negative_samples=int(np.count_nonzero(bb_chi<0)),
        chi_total_min_max=[float(total_chi.min()),float(total_chi.max())],
        eta_total_min_max=[float(total_eta.min()),float(total_eta.max())],
        total_negative_samples=int(np.count_nonzero(total_chi<0)),
        native_differences=['CODATA constants vs native historical constants',
            'all-positive-ion FF vs lower-PHOT-target-only FF',
            'signed raw line chi; native zero replacement/SRCE_CHK not applied'],
        process_selection=policy,input_sha256=before,osc_sha256=osc_identities,
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
    args=parser.parse_args()
    nu=(B.atomic.BF_NU_MIN*np.exp((np.arange(B.atomic.BF_N_FREQ_BIN)+.5)*
        np.log(B.atomic.BF_NU_MAX/B.atomic.BF_NU_MIN)/B.atomic.BF_N_FREQ_BIN)
        if args.bf_grid else np.asarray(args.frequency_Hz,dtype='f8'))
    print(json.dumps(prepare(args.run.resolve(),args.material.resolve(),args.atomic.resolve(),
        args.phot_library.resolve(),args.kernel_library.resolve(),args.C_full_population.resolve(),
        nu,args.out.resolve()),ensure_ascii=False))
