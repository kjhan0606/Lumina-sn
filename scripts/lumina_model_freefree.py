"""공통 nodal 장의 FF 열계수. all-ion/native-upper 선택과 열·냉각을 분리한다.

고정 T/J의 계수는 인구/ne 변화에 재사용한다. T 또는 장이 바뀌면 다시
생산해야 한다. CONT_FREQ 대응은 명시 입력만 허용하며 새 수송을 풀지 않는다.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

import cmfgen_model_continuum as B


@dataclass(frozen=True)
class FreeFreeCoefficients:
    field_signature:str
    temperature:np.ndarray
    heating:np.ndarray
    cooling:np.ndarray


class ModelFreeFree:
    def __init__(self,frequency,weights,J,kernels,models,columns,*,radiation_generation,
                 evaluation=None,kernel_identity=None):
        self.nu=np.array(frequency,dtype='f8',copy=True,order='C')
        self.weights=np.array(weights,dtype='f8',copy=True,order='C')
        self.J=np.array(J,dtype='f8',copy=True,order='C')
        self.evaluation=self.nu.copy() if evaluation is None else np.array(evaluation,dtype='f8',copy=True,order='C')
        self.columns=np.array(columns,dtype='i4',copy=True,order='C')
        if (self.nu.ndim!=1 or len(self.nu)<2 or self.weights.shape!=self.nu.shape or
            self.evaluation.shape!=self.nu.shape or self.J.ndim!=2 or not self.J.shape[0] or
            self.J.shape[1]!=len(self.nu) or not all(np.isfinite(v).all() for v in
                (self.nu,self.weights,self.J,self.evaluation)) or
            np.any(self.nu<=0) or np.any(np.diff(self.nu)<=0) or np.any(self.weights<=0) or
            np.any(self.evaluation<=0) or np.any(self.J<0) or
            self.columns.ndim!=2 or self.columns.shape[1]!=3 or not len(self.columns) or
            len(set(map(tuple,self.columns)))!=len(self.columns) or np.any(self.columns[:,1]<0) or
            type(radiation_generation) is not int or radiation_generation<=0):
            raise ValueError('FF 장/원자 주소 입력 오류')
        self.charges=np.unique(self.columns[self.columns[:,1]>0,1]);self.kernels=kernels
        if not len(self.charges):raise ValueError('양이온 없는 FF 입력')
        self.native=np.array([(int(z),int(charge)) in models for z,charge,sl in self.columns])
        self.radiation_generation=radiation_generation
        self.kernel_identity=B.sha(kernels.path) if kernel_identity is None else str(kernel_identity)
        h=hashlib.sha256(self.kernel_identity.encode())
        for name,v in (('frequency',self.nu),('evaluation',self.evaluation),('weights',self.weights),
                       ('J',self.J),('columns',self.columns),('native_selection',self.native)):
            v.flags.writeable=False;h.update(repr((name,v.shape,v.dtype.str)).encode());h.update(v.tobytes())
        self.signature=h.hexdigest()

    def prepare(self,temperature,*,radiation_generation,frequency_block=2048):
        t=np.array(temperature,dtype='f8',copy=True,order='C')
        if (radiation_generation!=self.radiation_generation or t.shape!=(self.J.shape[0],) or
            not np.isfinite(t).all() or np.any(t<=0) or type(frequency_block) is not int or frequency_block<1):
            raise ValueError('FF 온도/장 세대 오류')
        shape=(len(self.charges),len(t));heating=np.zeros(shape,dtype=np.longdouble);cooling=np.zeros_like(heating)
        for first in range(0,len(self.nu),frequency_block):
            last=min(len(self.nu),first+frequency_block);nu=self.nu[first:last];w=self.weights[first:last]
            evaluation,index=np.unique(self.evaluation[first:last],return_inverse=True)
            x=B.H*nu[None,:]/(B.KB*t[:,None]);decay=np.exp(-x);stim=-np.expm1(-x)
            # COMP_OPAC 보정의 대수적 형태. eta/stim 지수는 실제 nu, Gaunt는 evaluation.
            absorption=stim*self.J[:,first:last]*w[None,:]/self.evaluation[None,first:last]**3
            emission=(2*B.H/B.C**2)*(nu/self.evaluation[first:last])[None,:]**3*decay*w[None,:]
            for i,charge in enumerate(self.charges):
                gaunt=self.kernels.gaunt(evaluation,t,float(charge))[:,index]
                if gaunt.shape!=absorption.shape or not np.isfinite(gaunt).all() or np.any(gaunt<=0):
                    raise ValueError('FF Gaunt 출력 오류')
                factor=4*np.pi*3.69e8*float(charge)**2
                heating[i]+=factor*np.sum(gaunt*absorption,axis=1,dtype=np.longdouble)
                cooling[i]+=factor*np.sum(gaunt*emission,axis=1,dtype=np.longdouble)
        heating=np.array(heating,dtype='f8');cooling=np.array(cooling,dtype='f8')
        if not np.isfinite([heating,cooling]).all() or np.any(heating<0) or np.any(cooling<0):
            raise ValueError('FF 열계수 비유한/음수')
        for a in (t,heating,cooling):a.flags.writeable=False
        return FreeFreeCoefficients(self.signature,t,heating,cooling)

    def contract(self,coefficients,state,*,radiation_generation,material_generation):
        t=np.asarray(state['temperature']);ne=np.asarray(state['ne']);pop=np.asarray(state['superlevel'])
        if (coefficients.field_signature!=self.signature or radiation_generation!=self.radiation_generation or
            type(material_generation) is not int or material_generation<=0 or
            not np.array_equal(state['columns'],self.columns) or
            not np.array_equal(t,coefficients.temperature) or ne.shape!=t.shape or
            pop.shape!=(len(self.columns),len(t)) or not np.isfinite(ne).all() or np.any(ne<=0) or
            not np.isfinite(pop).all() or np.any(pop<0)):
            raise ValueError('FF 다른 장/온도/원자 주소 또는 잘못된 trial')
        all_pop=np.stack([pop[self.columns[:,1]==q].sum(axis=0) for q in self.charges])
        native_pop=np.stack([pop[(self.columns[:,1]==q)&self.native].sum(axis=0) for q in self.charges])
        result={}
        for name,population in (('all_positive_ions',all_pop),('native_upper_ions',native_pop)):
            factor=np.asarray(population*ne[None,:]/np.sqrt(t)[None,:],dtype=np.longdouble)
            heat=np.sum(factor*coefficients.heating,axis=0);cool=np.sum(factor*coefficients.cooling,axis=0)
            if not np.isfinite([heat,cool]).all():raise ValueError('FF trial 열량 비유한')
            result[name]=dict(heating=np.array(heat,dtype='f8'),cooling=np.array(cool,dtype='f8'),
                              net=np.array(heat-cool,dtype='f8'))
        return dict(terms=result,material_generation=material_generation,
                    radiation_generation=radiation_generation,field_signature=self.signature)
