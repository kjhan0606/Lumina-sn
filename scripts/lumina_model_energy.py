"""전 SL/terminal 에너지 소유자. 광자·native 열장부의 에너지 기준을 분리한다.

원자 에너지는 보존한다. fraction은 각 trial의 occupation·온도에서 caller가
재생산한다. 이 모듈은 SE/RE를 풀거나 물질을 게시하지 않는다.
"""
import csv
from pathlib import Path
import re

import numpy as np

import cmfgen_model_continuum as B


class ModelThermalEnergy:
    MODES=('physical_neutral','native_static_raw','native_static_aligned')

    def __init__(self,rows,columns,ion_data,neutral_offsets_erg):
        self.columns=np.array(columns,dtype='i4',copy=True)
        self.index={tuple(map(int,k)):i for i,k in enumerate(self.columns)}
        if len(self.index)!=len(self.columns):raise ValueError('중복 SL 좌표')
        lookup={(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])):i
                for i,r in enumerate(rows)}
        if len(lookup)!=len(rows):raise ValueError('중복 FL 좌표')
        self.groups=[];self.full_to_sl=np.full(len(rows),-1,dtype='i4')
        self.physical_full=np.full(len(rows),np.nan)
        self.static=np.full(len(columns),np.nan);self.alignment=np.zeros(len(columns))
        self.mean_edge=np.full(len(columns),np.nan);self.terminal_physical={}
        expected=set();used=set()
        for z in sorted({key[0] for key in ion_data}):
            stages=sorted(stage for zz,stage in ion_data if zz==z)
            if stages!=list(range(stages[0],stages[-1]+1)):raise ValueError('닫히지 않은 이온 사다리')
            offset=float(neutral_offsets_erg[z])
            if not np.isfinite(offset) or offset<0:raise ValueError('중성 바닥 기준 누락')
            physical_base=offset;native_base=0.;alignment=None
            for stage in stages:
                data=ion_data[z,stage]
                edge=np.asarray(data['edge_Hz'],dtype='f8');exc=np.asarray(data['excitation_Hz'],dtype='f8')
                g=np.asarray(data['g'],dtype='f8');n=len(edge)
                if (edge.shape!=(n,) or exc.shape!=(n,) or g.shape!=(n,) or not n or
                    not np.isfinite([edge,exc,g]).all() or np.any(g<=0) or exc[0]!=0.):
                    raise ValueError('원 에너지/통계중량 오류')
                members=np.array([lookup[z,stage,l] for l in range(n)])
                used.update(map(int,members));sl=np.array([int(rows[i]['super_level']) for i in members])
                if 0 not in sl:raise ValueError('이온 바닥 SL 누락')
                means={int(s):float(np.sum(edge[sl==s]*g[sl==s],dtype=np.longdouble)/
                                    np.sum(g[sl==s],dtype=np.longdouble)) for s in np.unique(sl)}
                ground=means[0]
                if alignment is None:alignment=offset+B.H*(edge[0]-ground)
                self.physical_full[members]=physical_base+B.H*exc
                for s,mean in means.items():
                    key=(z,stage-1,s);expected.add(key);col=self.index[key];group=members[sl==s]
                    self.groups.append((col,group));self.full_to_sl[group]=col
                    self.mean_edge[col]=mean;self.static[col]=native_base+B.H*(ground-mean)
                    self.alignment[col]=alignment
                physical_base+=B.H*edge[0];native_base+=B.H*ground
            key=(z,stages[-1],-1);expected.add(key);col=self.index[key]
            self.terminal_physical[col]=physical_base;self.static[col]=native_base
            self.alignment[col]=alignment
        if expected!=set(self.index) or used!=set(range(len(rows))) or not np.isfinite(self.static).all():
            raise ValueError('전 SL/FL/terminal 에너지 누락')
        for a in (self.columns,self.full_to_sl,self.physical_full,self.static,self.alignment,self.mean_edge):
            a.flags.writeable=False
        self.input_sha256={}

    @classmethod
    def from_model(cls,run,rows,models,columns,reference):
        links=B.atomic.load_cmfgen_links(run/'atomic_links.txt');ions={};identities={}
        for key,model in models.items():
            path=Path(links[key]['osc']);identities[str(path)]=B.sha(path)
            levels=B.parse_osc(path,include_transitions=False).levels[:model['n_full']]
            header=re.findall(r'^\s*(\S+)\s+!Ionization energy\b',path.read_text(),re.M)
            if len(header)!=1 or len(levels)!=model['n_full']:raise ValueError('원 FEDGE 누락')
            ion=float(header[0].replace('D','E'))
            ions[key]=dict(edge_Hz=(ion-levels['E_cm'])*B.C,
                           excitation_Hz=levels['E_cm']*B.C,g=levels['g'])
        identities[str(reference)]=B.sha(reference)
        table={}
        for row in csv.DictReader(reference.open()):
            key=(int(row['atomic_number']),int(row['ion_number']))
            if key in table:raise ValueError('중복 중성 기준 이온화 에너지')
            table[key]=float(row['ionization_energy_eV'])*B.EV
        offsets={z:sum(table[z,charge] for charge in range(min(s for zz,s in models if zz==z)-1))
                 for z in {zz for zz,s in models}}
        result=cls(rows,columns,ions,offsets)
        if any(B.sha(Path(p))!=h for p,h in identities.items()):raise ValueError('원자료 변경')
        result.input_sha256=identities
        return result

    def superlevel_energy(self,fraction,mode):
        if mode not in self.MODES:raise ValueError('열장부를 명시해야 함')
        f=np.asarray(fraction,dtype='f8')
        if (f.ndim!=2 or f.shape[0]!=len(self.physical_full) or not f.shape[1] or
            not np.isfinite(f).all() or np.any(f<0) or np.any(f>1)):
            raise ValueError('trial SL 분율 오류')
        for _,group in self.groups:
            if not np.allclose(f[group].sum(axis=0),1.,rtol=0,atol=1.e-12):
                raise ValueError('trial SL 분율 정규화 오류')
        if mode!='physical_neutral':
            energy=self.static if mode=='native_static_raw' else self.static+self.alignment
            return np.broadcast_to(energy[:,None],(len(self.columns),f.shape[1])).copy()
        result=np.empty((len(self.columns),f.shape[1]))
        for col,group in self.groups:
            result[col]=np.sum(self.physical_full[group,None]*f[group],axis=0)
        for col,value in self.terminal_physical.items():result[col]=value
        return result

    def internal_energy(self,population,fraction,mode):
        energy=self.superlevel_energy(fraction,mode);population=np.asarray(population,dtype='f8')
        if population.shape!=energy.shape or not np.isfinite(population).all() or np.any(population<0):
            raise ValueError('trial SL/terminal 인구 오류')
        nuclei=population.sum(axis=0)
        if np.any(nuclei<=0):raise ValueError('원자 총수0')
        return np.sum(np.asarray(population,dtype=np.longdouble)*energy,axis=0)/nuclei

    def line_scaling(self,lower,upper,frequency,nuclei,*,enabled,ratio_limit,density_limit):
        lo=np.asarray(lower);up=np.asarray(upper);nu=np.asarray(frequency,dtype='f8')
        density=np.asarray(nuclei,dtype='f8')
        if (lo.ndim!=1 or up.shape!=lo.shape or nu.shape!=lo.shape or
            not np.issubdtype(lo.dtype,np.integer) or not np.issubdtype(up.dtype,np.integer) or
            np.any(lo<0) or np.any(up<0) or np.any(lo>=len(self.full_to_sl)) or
            np.any(up>=len(self.full_to_sl)) or density.ndim!=1 or not len(density) or
            not np.isfinite(nu).all() or np.any(nu<=0) or not np.isfinite(density).all() or
            np.any(density<=0) or not np.isfinite([ratio_limit,density_limit]).all() or
            ratio_limit<0 or density_limit<=0):raise ValueError('선 열장부 입력 오류')
        lc=self.full_to_sl[lo];uc=self.full_to_sl[up]
        if not np.array_equal(self.columns[lc,:2],self.columns[uc,:2]):
            raise ValueError('서로 다른 이온의 BB 선')
        ratio=(self.mean_edge[lc]-self.mean_edge[uc])/nu
        factor=np.where(np.abs(ratio-1)<=ratio_limit,ratio,1.) if enabled else np.ones_like(ratio)
        return np.where(density[None,:]>=density_limit,1.,factor[:,None])


def instantaneous_local_decay_heating(isotope_number_density,channels):
    """현재 핵종의 국소 순간출력 Σ n*lambda*E (erg cm^-3 s^-1).

    channels는 (핵종 이름,붕괴상수 s^-1,반응당 에너지 erg)다. isotope 진화나
    gamma 수송은 하지 않는다. 같은 나이의 두 상태라 해도 현재 붕괴출력은
    유한하며, 이전 epoch와의 시간차로 나누거나0으로 만들지 않는다.
    원 입력 재현에서 저장0가열을 선택하는 정책과 이 물리 함수를 혼합하지 않는다.
    """
    result=None;seen=set()
    for name,decay_constant,energy in channels:
        if (name in seen or name not in isotope_number_density or
            not np.isfinite([decay_constant,energy]).all() or decay_constant<=0 or energy<0):
            raise ValueError('핵종/붕괴 채널 입력 오류')
        seen.add(name);density=np.asarray(isotope_number_density[name],dtype='f8')
        if (density.ndim!=1 or not len(density) or not np.isfinite(density).all() or np.any(density<0) or
            (result is not None and density.shape!=result.shape)):
            raise ValueError('핵종 수밀도 입력 오류')
        if result is None:result=np.zeros(density.shape,dtype=np.longdouble)
        result+=np.asarray(density,dtype=np.longdouble)*decay_constant*energy
    if result is None:raise ValueError('빈 붕괴 채널')
    result=np.asarray(result,dtype='f8')
    if not np.isfinite(result).all():raise ValueError('순간 침적열 범위 초과')
    return result
