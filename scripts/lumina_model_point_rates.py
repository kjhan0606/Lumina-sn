"""원래점의 전 rawPHOT 경로를 점값 C 소비율로 만든다.

전역 ν 가중치를 외부에서 명시적으로 받는다. bound PHOT 블록만의 생산기로,
pseudo-continuum/BB/다른 채널이나 전 주파수 완료를 대신하지 않는다.
"""
import ctypes as C

import numpy as np

import cmfgen_model_continuum as B


class Rates(C.Structure):
    _fields_=[(name,C.c_double) for name in ('photo','spontaneous','stimulated',
        'heat_photo','heat_spontaneous','heat_stimulated')]


class ModelPointBoundRates:
    def __init__(self,routes,rows,models,state,node,library):
        if type(node) is not int or not 0<=node<len(state['temperature']):
            raise ValueError('원래점 인덱스 오류')
        self.node=node;self.rows=rows;self.models=models;self.state=state
        self.routes=sorted(routes,key=lambda r:(r.lower,r.phot_grid,r.target_superlevel))
        self.temperature=float(state['temperature'][node]);self.ne=float(state['ne'][node])
        self.library=C.CDLL(str(library));self.function=self.library.bf_rate_point_contract
        ptr=C.POINTER(C.c_double)
        self.function.argtypes=[C.c_size_t]+[ptr]*6+[C.c_double,C.POINTER(Rates)]
        self.function.restype=C.c_int;self.ptr=ptr
        groups={}
        for l,row in enumerate(rows):
            key=(int(row['atomic_number']),int(row['ion_number']),int(row['super_level']))
            groups.setdefault(key,[]).append(l)
        self.fraction=np.zeros(len(rows))
        self.occupation=np.asarray(state['occupation'][:,node])
        for indices in groups.values():
            energy=np.array([float(rows[l]['energy_eV']) for l in indices])
            g=np.array([float(rows[l]['g']) for l in indices])
            weight=g*self.occupation[indices]*np.exp(-(energy-energy.min())*B.EV/(B.KB*self.temperature))
            if not np.isfinite(weight).all() or weight.sum()<=0:raise ValueError('SL 재구성 오류')
            self.fraction[indices]=weight/weight.sum()
        self.terminal={}
        for r in self.routes:
            if r.target>=0:continue
            key=(r.Z,r.stage)
            reference=(r.target_energy_eV,r.target_g)
            if reference[0]!=0 or (key in self.terminal and self.terminal[key]!=reference):
                raise ValueError('단일 terminal ground scalar로 정규화할 수 없는 PHOT EION/GION')
            self.terminal[key]=reference
        self.upper_factor=np.empty(len(self.routes));self.upper_population=np.empty(len(self.routes))
        self.saha=np.empty(len(self.routes));self.sigma_edge=np.empty(len(self.routes))
        counts=np.zeros(len(rows),dtype='i4')
        thermal=self.ne*(B.H*B.H/(2*np.pi*B.ME*B.KB*self.temperature))**1.5
        for i,r in enumerate(self.routes):
            if r.lower<0 or r.lower>=len(rows):raise ValueError('PHOT lower 신원 오류')
            counts[r.lower]+=1
            if r.target>=0:
                if self.occupation[r.target]<=0:raise ValueError('상위 anchor 점유확률0')
                factor=self.fraction[r.target]*self.occupation[r.lower]/self.occupation[r.target]
                gt=float(rows[r.target]['g'])
                upper=state['superlevel'][state['sl_index'][r.Z,r.stage,r.target_superlevel],node]
            else:
                factor=self.occupation[r.lower];gt=r.target_g
                upper=state['superlevel'][state['sl_index'][r.Z,r.stage,-1],node]
            self.upper_factor[i]=factor;self.upper_population[i]=upper
            exponent=B.H*r.threshold_Hz/(B.KB*self.temperature)
            gl=float(rows[r.lower]['g'])
            if factor==0:self.saha[i]=0.
            else:self.saha[i]=np.exp(np.log(thermal*gl/(2*gt))+np.log(factor)+exponent)
            self.sigma_edge[i]=float(r.sigma(np.array([r.threshold_Hz*(1+1e-14)]))[0])
        self.offset=np.r_[0,np.cumsum(counts,dtype='i8')]
        if self.offset[-1]!=len(self.routes) or not all(np.isfinite(x).all() and np.all(x>=0)
            for x in (self.upper_factor,self.upper_population,self.saha,self.sigma_edge)):
            raise ValueError('PHOT roster/역과정 정규화 오류')

    def contract_bound_block(self,frequency,quadrature_Hz,J):
        """전체 격자의 임의 블록/선택 열. 반환값 자체는 완성 snapshot이 아니다."""
        nu,width,j=[np.ascontiguousarray(a,dtype='f8') for a in (frequency,quadrature_Hz,J)]
        if nu.ndim!=1 or not len(nu) or width.shape!=nu.shape or j.shape!=nu.shape:
            raise ValueError('점값 블록 차원 오류')
        if not all(np.isfinite(x).all() for x in (nu,width,j)) or np.any(nu<=0) or np.any(np.diff(nu)<=0) or np.any(width<0) or np.any(j<0):
            raise ValueError('점값 블록 값 오류')
        out=np.empty((len(self.routes),6))
        for i,r in enumerate(self.routes):
            sigma=r.sigma(nu)
            target_g=float(self.rows[r.target]['g']) if r.target>=0 else r.target_g
            coef=B.milne_route(nu,sigma,np.ones(1),np.array([self.upper_factor[i]]),
                np.array([self.temperature]),np.array([self.ne]),float(self.rows[r.lower]['g']),
                target_g,r.threshold_Hz,1.)
            arrays=[nu,width,j,*[np.ascontiguousarray(a[0]) for a in coef]]
            result=Rates()
            if self.function(len(nu),*[a.ctypes.data_as(self.ptr) for a in arrays],r.threshold_Hz,C.byref(result)):
                raise ValueError('전모형 경로의 C 점값 적분 실패')
            out[i]=[getattr(result,key) for key,_ in result._fields_]
        return out

    def physical_ion_rates(self,unit_rates):
        """선택 블록의 실제 물질 소비율. PHOT 확률은 여기서 한 번만 적용."""
        if unit_rates.shape!=(len(self.routes),6) or not np.isfinite(unit_rates).all():
            raise ValueError('rawPHOT 율 테이블 차원 오류')
        out={}
        for i,r in enumerate(self.routes):
            key=(r.Z,r.stage);value=out.setdefault(key,np.zeros(3))
            value[0]+=unit_rates[i,0]*r.probability*self.state['full'][r.lower,self.node]
            value[1]+=unit_rates[i,1]*r.probability*self.upper_population[i]
            value[2]+=unit_rates[i,2]*r.probability*self.upper_population[i]
        return out
