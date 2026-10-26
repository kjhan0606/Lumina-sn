"""전활성 model-atom의 Ω(T) 원자료 생산자. 이온/주파수/인구 선택을 줄이지 않는다.

원 이온마다 OSC/COL을 한 번 읽고 여러 점의 온도에 전 FL쌍을 블록 생산한다.
선 roster 밖의 쌍도 OMEGA_SET 충돌을 유지한다. 전하/SE/RE 해법은 아니다.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import build_cmfgen_coldata_all as col
import build_cmfgen_fullpair_omega as legacy
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from cmfgen_parser import parse_col,parse_osc


def ex_e1x(x):
    """native EX_E1X_FUN의 같은 다항식. 양수 x만 허용한다."""
    x=np.asarray(x,dtype='f8')
    if not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError('EX_E1X의 비양수/비유한 인수')
    result=np.empty_like(x)
    small=x<=1.;a=x[small]
    p=np.full_like(a,.00107857)
    for c in (-.00976004,.05519968,-.24991055,.99999193,-.57721566):
        p=p*a+c
    result[small]=np.exp(a)*(p-np.log(a))
    a=x[~small];numerator=np.ones_like(a);denominator=np.ones_like(a)
    for n,d in zip((8.5733287401,18.0590169730,8.6347608925,.2677737343),
                   (9.5733223454,25.6329561486,21.0996530827,3.9584969228)):
        numerator=numerator*a+n;denominator=denominator*a+d
    result[~small]=numerator/denominator/a
    return result


def table_values(grid,values,temperature):
    """native log-log 보간/명시된 양끝 동작. 임의의 물리 바닥값은 없다."""
    grid=np.asarray(grid,dtype='f8');values=np.asarray(values,dtype='f8')
    temperature=np.atleast_1d(np.asarray(temperature,dtype='f8'))
    if (grid.ndim!=1 or len(grid)<2 or values.ndim!=2 or values.shape[1]!=len(grid) or
        temperature.ndim!=1 or not len(temperature) or np.any(temperature<=0) or
        not np.isfinite(temperature).all() or not np.isfinite(grid).all() or
        not np.isfinite(values).all() or np.any(grid<=0) or np.any(np.diff(grid)<=0) or
        np.any(values<=0)):
        raise ValueError('충돌 온도/표의 범위·양수성 오류')
    result=np.empty((len(temperature),len(values)))
    for k,t in enumerate(temperature/1.e4):
        if t<=grid[0]:result[k]=values[:,0]
        elif t>=grid[-1]:result[k]=values[:,-1]
        else:
            hi=int(np.searchsorted(grid,t,side='left'));lo=hi-1
            alpha=np.log(values[:,hi]/values[:,lo])/np.log(grid[hi]/grid[lo])
            result[k]=values[:,lo]*(t/grid[lo])**alpha
    return result


def pair_blocks(n,maximum):
    """모든 i<j를 사전식으로 정확히 한 번 생산한다. N² 인덱스를 상주시키지 않는다."""
    if type(n) is not int or n<1 or type(maximum) is not int or not 1<=maximum<=65536:
        raise ValueError('충돌쌍 블록 차원 오류')
    lower=[];upper=[];used=0
    for lo in range(n-1):
        up=lo+1
        while up<n:
            count=min(n-up,maximum-used)
            lower.append(np.full(count,lo,dtype='i4'))
            upper.append(np.arange(up,up+count,dtype='i4'))
            up+=count;used+=count
            if used==maximum:
                yield np.concatenate(lower),np.concatenate(upper)
                lower=[];upper=[];used=0
    if used:yield np.concatenate(lower),np.concatenate(upper)


def locate(sorted_keys,query):
    positions=np.searchsorted(sorted_keys,query)
    present=positions<len(sorted_keys)
    present[present]&=sorted_keys[positions[present]]==query[present]
    return present,positions


@dataclass
class IonCollisionSource:
    Z:int
    stage:int
    offset:int
    n:int
    energy:np.ndarray
    g:np.ndarray
    same_key:np.ndarray
    omega_set:float
    omega_scale:float
    table_keys:np.ndarray
    table_grid:np.ndarray
    table:np.ndarray
    vr_keys:np.ndarray
    vr_f:np.ndarray
    identity:dict
    counts:dict

    def blocks(self,temperature,maximum=16384):
        temperatures=np.atleast_1d(np.asarray(temperature,dtype='f8'))
        if (temperatures.ndim!=1 or not 1<=len(temperatures)<=90 or
            not np.isfinite(temperatures).all() or np.any(temperatures<=0)):
            raise ValueError('실제 trial 온도1~90개 필요')
        tab=(table_values(self.table_grid,self.table,temperatures)
             if len(self.table_keys) else np.empty((len(temperatures),0)))
        for lo,up in pair_blocks(self.n,maximum):
            keys=lo.astype('i8')*self.n+up
            omega=np.full((len(temperatures),len(lo)),self.omega_set)
            tabulated,tp=locate(self.table_keys,keys)
            if np.any(tabulated):omega[:,tabulated]=tab[:,tp[tabulated]]
            vr,vp=locate(self.vr_keys,keys);vr&=~tabulated
            if np.any(vr):
                l,u=lo[vr],up[vr];gap=self.energy[u]-self.energy[l]
                if np.any(gap<=0):raise ValueError('양의 f를 가진 동률 준위')
                x=gap[None,:]/(legacy.KB_EV*temperatures[:,None]);function=ex_e1x(x)
                if self.stage==1:
                    gbar=np.where(x<=14.,.276*function,.066*(1.+1.5/x)/np.sqrt(x))
                else:
                    same=(self.same_key[l]>=0)&(self.same_key[l]==self.same_key[u])
                    gbar=np.maximum(np.where(same,.7,.2)[None,:],.276*function)
                frequency=gap*legacy.EV_ERG/(legacy.H_CGS*1.e15)
                omega[:,vr]=47.972*self.omega_scale*gbar*self.vr_f[vp[vr]][None,:]*self.g[l][None,:]/frequency[None,:]
            if not np.isfinite(omega).all() or np.any(omega<=0):
                raise ValueError('전 FL쌍 Ω 비유한/비양수')
            yield lo+self.offset,up+self.offset,omega,dict(
                TABULATED=int(tabulated.sum()),VR=int(vr.sum()),
                OMEGA_SET=int(len(lo)-tabulated.sum()-vr.sum()))

    def verify_identity(self):
        if any(B.sha(Path(p))!=h for p,h in self.identity.items()):
            raise ValueError('충돌 원자료 생산 중 입력 변경')


class ModelCollisionSource:
    """MODEL_SPEC 전범위를 등록한다. caller는 이온별 여러 점을 배치 처리한다."""
    def __init__(self,run,rows):
        self.run=Path(run);self.rows=rows
        self.links=B.atomic.load_cmfgen_links(self.run/'atomic_links.txt')
        self.models=B.atomic.load_cmfgen_model_spec(self.run/'MODEL_SPEC',self.run/'atomic_links.txt')
        self.identity={str(self.run/name):B.sha(self.run/name)
                       for name in ('MODEL','MODEL_SPEC','atomic_links.txt')}
        self.groups={}
        for i,row in enumerate(rows):
            key=int(row['atomic_number']),int(row['ion_number'])+1
            self.groups.setdefault(key,[]).append(i)
        if set(self.groups)!=set(self.models):raise ValueError('충돌 MODEL_SPEC 활성 이온 누락/추가')
        for key,ids in self.groups.items():
            if (len(ids)!=int(self.models[key]['n_full']) or
                ids!=list(range(ids[0],ids[0]+len(ids))) or
                [int(rows[i]['level_number']) for i in ids]!=list(range(len(ids)))):
                raise ValueError('충돌 FL 모델 순서/개수 오류')

    def load(self,Z,stage):
        key=Z,stage
        if key not in self.models:raise ValueError('활성 모델 밖의 충돌 이온')
        ids=self.groups[key];n=len(ids);paths=self.links[key]
        identity=dict(self.identity)
        for kind in ('osc','col'):
            path=Path(paths[kind]);identity[str(path)]=B.sha(path)
        osc=parse_osc(paths['osc']);g=np.array([float(self.rows[i]['g']) for i in ids])
        energy=np.array([float(self.rows[i]['energy_eV']) for i in ids])
        names=list(map(str,osc.levels['config'][:n]))
        if (len(names)!=n or len(osc.transitions)!=osc.declared_n_transitions or
            not np.isfinite(g).all() or np.any(g<=0) or not np.isfinite(energy).all() or
            np.any(np.diff(energy)<0)):
            raise ValueError('충돌 OSC/준위 범위 오류')
        for j,i in enumerate(ids):
            if (names[j]!=self.rows[i]['configuration'] or g[j]!=osc.levels['g'][j] or
                abs(energy[j]-osc.levels['E_cm'][j]*1.239841984e-4)>5.1e-11):
                raise ValueError('충돌 OSC와 FL 신원 불일치')
        # native는 동일 ordered pair를 마지막 값으로 덮는다. 그 뒤 약선 선택.
        f_by_key={}
        for tr in osc.transitions:
            lo,up=int(tr['i'])-1,int(tr['j'])-1
            if lo>=n or up>=n:continue
            f=abs(float(tr['f']))
            if not 0<=lo<up<n or not np.isfinite(f):raise ValueError('충돌 OSC 전이 주소 오류')
            f_by_key[lo*n+up]=f
        codes=np.array(sorted(f_by_key),dtype='i8')
        f=np.array([f_by_key[k] for k in codes]);lo,up=codes//n,codes%n
        frequency=(energy[up]-energy[lo])*legacy.EV_ERG/legacy.H_CGS
        spontaneous=f*g[lo]/g[up]*frequency**2
        keep=S.gf_selection(lo,up,f,spontaneous,g[lo],Z,self.run/'MODEL',n)
        header=parse_col(paths['col'])
        if not np.isfinite([header.default_omega,header.scale_factor]).all() or min(header.default_omega,header.scale_factor)<=0:
            raise ValueError('원 충돌 OMEGA_SET/척도 오류')
        if header.n_transitions:
            mapping,_,weights,full_n=col.parse_osc(str(paths['osc']))
            table,grid,nt,stats=col.parse_col(str(paths['col']),mapping,weights,full_n)
            if (stats['n_read']!=stats['n_decl'] or stats['n_nonpos'] or stats['n_term_reject'] or len(grid)!=nt):
                raise ValueError('원 충돌 표 coverage/값 오류')
            table={pair:value for pair,value in table.items() if pair[0]<n and pair[1]<n}
        else:table={};grid=[]
        table_codes=np.array(sorted(i*n+j for i,j in table),dtype='i8')
        table_array=np.array([table[int(k)//n,int(k)%n] for k in table_codes],dtype='f8')
        vr=keep&(f>1.e-5)
        native_keys=[legacy.same_n_key(name) for name in names]
        key_ids={};same=np.full(n,-1,dtype='i4')
        for i,k in enumerate(native_keys):
            if k is not None:
                if k not in key_ids:key_ids[k]=len(key_ids)
                same[i]=key_ids[k]
        result=IonCollisionSource(Z,stage,ids[0],n,energy,g,same,
            float(header.default_omega),float(header.scale_factor),table_codes,np.array(grid),
            table_array,codes[vr],f[vr],identity,dict(full_levels=n,pairs=n*(n-1)//2,
                raw_osc_pairs=len(codes),GF_culled_lines=int(np.count_nonzero((f!=0)&~keep)),
                tabulated_pairs=len(table_codes),frequency_basis='SEALED_CSV_EV'))
        result.verify_identity()
        return result
