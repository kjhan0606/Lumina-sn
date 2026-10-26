"""전모형 private trial의 다점 BF 율 생산자. 물질 해법/게시/수송은 아니다.

고정된 J·격자를 소유하고 σ는 경로/주파수 블록에서 모든 점에 재사용한다.
trial마다 기존 occupation fit·SL 분율·Saha·Milne 정규화를 다시 만든다.
"""
import ctypes as C
from dataclasses import dataclass
import hashlib

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_point_rates import Rates


DP=C.POINTER(C.c_double)
BP=C.POINTER(C.c_uint8)


class Block(C.Structure):
    _fields_=[(name,C.c_size_t) for name in ('n_frequency','n_routes','n_points')]+[
        (name,DP) for name in ('frequency_Hz','evaluation_Hz','quadrature_Hz','J_nu',
                              'bound_sigma','threshold_Hz','milne_prefactor','inverse_kT_Hz')]+[
        ('dissolved',C.c_int),('dissolution_primary',BP)]+[
        (name,DP) for name in ('sigma_edge','z_screen','material_b','material_a','material_x')]


def owned(values,dtype='f8'):
    result=np.array(values,dtype=dtype,copy=True,order='C')
    result.flags.writeable=False
    return result


@dataclass(frozen=True)
class MaterialTrial:
    owner: object
    material_generation: int
    radiation_generation: int
    material_signature: str
    state: dict
    fraction: np.ndarray
    milne_prefactor: np.ndarray
    saha: np.ndarray
    inverse_kT_Hz: np.ndarray
    material_b: np.ndarray
    material_a: np.ndarray
    material_x: np.ndarray


class ModelMaterialRates:
    def __init__(self,routes,rows,models,library,frequency,quadrature,J,*,
                 radiation_generation,evaluation=None,dissolution_enabled=True):
        if type(radiation_generation) is not int or radiation_generation<=0:
            raise ValueError('등록 복사장 세대 오류')
        self.radiation_generation=radiation_generation
        self.routes=sorted(routes,key=lambda r:(r.lower,r.phot_grid,r.target_superlevel))
        self.rows=rows;self.models=models;self.dissolved=bool(dissolution_enabled)
        self.frequency=owned(frequency);self.width=owned(quadrature);self.J=owned(J)
        self.evaluation=owned(self.frequency if evaluation is None else evaluation)
        nf=len(self.frequency)
        if (self.frequency.ndim!=1 or nf<2 or self.width.shape!=(nf,) or self.evaluation.shape!=(nf,) or
            self.J.ndim!=2 or not self.J.shape[0] or self.J.shape[1]!=nf or not self.routes or
            not all(np.isfinite(x).all() for x in (self.frequency,self.width,self.evaluation,self.J)) or
            np.any(self.frequency<=0) or np.any(np.diff(self.frequency)<=0) or
            np.any(self.width<0) or np.any(self.evaluation<=0) or np.any(self.J<0)):
            raise ValueError('등록 공통장 실제ν/평가ν/가중치/J 오류')
        self.n_points=self.J.shape[0]
        self.groups={}
        for l,row in enumerate(rows):
            key=int(row['atomic_number']),int(row['ion_number']),int(row['super_level'])
            self.groups.setdefault(key,[]).append(l)
        for r in self.routes:
            if not 0<=r.lower<len(rows):
                raise ValueError('PHOT 하위 FL 주소 오류')
            row=rows[r.lower]
            if (int(row['atomic_number']),int(row['ion_number'])+1)!=(r.Z,r.stage):
                raise ValueError('PHOT 하위 원자 주소 오류')
        self.edge=owned([r.threshold_Hz for r in self.routes])
        self.z=owned([models[r.Z,r.stage]['z_screen'] for r in self.routes])
        self.primary=owned([r.dissolution_primary for r in self.routes],'u1')
        self.sigma_edge=owned([float(r.sigma(np.array([r.threshold_Hz*(1+1.e-14)]))[0])
                               for r in self.routes])
        if any(r.target_energy_eV!=0 for r in self.routes if r.dissolution_primary):
            raise ValueError('용해 primary의 비바닥 PHOT 목표')
        self.library=C.CDLL(str(library))
        self.function=self.library.bf_rate_material_block_contract
        self.function.argtypes=[C.POINTER(Block),C.POINTER(Rates),C.POINTER(Rates)]
        self.function.restype=C.c_int
        digest=hashlib.sha256()
        for a in (self.frequency,self.evaluation,self.width,self.J):
            digest.update(a.tobytes())
        self.field_signature=digest.hexdigest()

    def prepare(self,state,*,material_generation):
        if type(material_generation) is not int or material_generation<=0:
            raise ValueError('private trial 세대 오류')
        ns=self.n_points;nl=len(self.rows);nr=len(self.routes)
        trial={k:owned(state[k],dtype='i8' if k=='columns' else 'f8')
               for k in ('columns','superlevel','temperature','ne')}
        columns=trial['columns'];pop=trial['superlevel'];t=trial['temperature'];ne=trial['ne']
        if (columns.ndim!=2 or columns.shape[1]!=3 or pop.shape!=(len(columns),ns) or
            t.shape!=(ns,) or ne.shape!=(ns,) or
            not all(np.isfinite(x).all() for x in (pop,t,ne)) or
            np.any(pop<0) or np.any(t<=0) or np.any(ne<=0)):
            raise ValueError('private trial 물질 형상/값 오류')
        index={tuple(map(int,c)):i for i,c in enumerate(columns)}
        if len(index)!=len(columns) or any(k not in index for k in self.groups):
            raise ValueError('private trial SL/terminal 원자 주소 오류')
        trial['sl_index']=index
        trial['full']=np.empty((nl,ns))
        # SL 인구로 계산하는 양이온 총수가 달라지면 T/ne가 같아도 w를 재생산한다.
        trial['occupation']=owned(B.material_occupation(self.rows,self.models,trial)
                                  if self.dissolved else np.ones((nl,ns)))
        occupation=trial['occupation'];fraction=np.zeros((nl,ns));full=np.empty((nl,ns))
        for key,ids in self.groups.items():
            energy=np.array([float(self.rows[l]['energy_eV']) for l in ids])
            g=np.array([float(self.rows[l]['g']) for l in ids])
            weight=g[:,None]*occupation[ids]*np.exp(-(energy[:,None]-energy.min())*B.EV/(B.KB*t))
            norm=weight.sum(axis=0)
            if not np.isfinite(weight).all() or np.any(norm<=0):
                raise ValueError('private trial LTE FL/SL 분율 오류')
            fraction[ids]=weight/norm
            full[ids]=fraction[ids]*pop[index[key]]
        trial['full']=owned(full)
        thermal=ne*(B.H*B.H/(2*np.pi*B.ME*B.KB*t))**1.5
        pref=np.empty((nr,ns));saha=np.zeros((nr,ns))
        for i,r in enumerate(self.routes):
            if r.target>=0:
                target=r.target
                if not 0<=target<nl or np.any(occupation[target]<=0):
                    raise ValueError('private trial 상위 anchor 오류')
                row=self.rows[target]
                if (int(row['atomic_number']),int(row['ion_number']),int(row['super_level']))!=(r.Z,r.stage,r.target_superlevel):
                    raise ValueError('PHOT 상위 anchor/SL 원자 주소 오류')
                factor=fraction[target]*occupation[r.lower]/occupation[target]
                gt=float(row['g'])
            else:
                if r.target!=-1 or r.target_energy_eV!=0 or (r.Z,r.stage,-1) not in index:
                    raise ValueError('private trial terminal 목표 오류')
                factor=occupation[r.lower];gt=r.target_g
            pref[i]=thermal*float(self.rows[r.lower]['g'])/(2*gt)*factor
            active=pref[i]>0
            with np.errstate(over='raise',invalid='raise'):
                saha[i,active]=np.exp(np.log(pref[i,active])+B.H*r.threshold_Hz/(B.KB*t[active]))
        if not all(np.isfinite(x).all() and np.all(x>=0) for x in (pref,saha)):
            raise ValueError('private trial Milne/Saha 정규화 오류')
        b,a,x=B.dissolution_vectors(trial) if self.dissolved else (np.ones(ns),np.zeros(ns),np.ones(ns))
        digest=hashlib.sha256()
        for value in (columns,pop,t,ne,occupation):
            digest.update(value.tobytes())
        return MaterialTrial(self,material_generation,self.radiation_generation,digest.hexdigest(),trial,
            owned(fraction),owned(pref),owned(saha),owned(B.H/(B.KB*t)),owned(b),owned(a),owned(x))

    def _check(self,trial,material_generation,radiation_generation):
        if (trial.owner is not self or trial.material_generation!=material_generation or
            trial.radiation_generation!=radiation_generation or
            radiation_generation!=self.radiation_generation):
            raise ValueError('다른 모형/복사장 또는 stale private trial 율')

    def frequency_blocks(self,trial,max_bins=1024):
        if type(max_bins) is not int or max_bins<1:
            raise ValueError('주파수 블록 크기 오류')
        start=0;nf=len(self.frequency)
        while start<nf:
            limit=self.frequency[start]+32./float(trial.inverse_kT_Hz.max())
            end=min(nf,start+max_bins,int(np.searchsorted(self.frequency,limit,side='right')))
            if end<=start:
                raise ValueError('주파수 블록 진행 불가')
            yield start,end
            start=end

    def contract_block(self,trial,start,end,route_indices,*,material_generation,radiation_generation):
        self._check(trial,material_generation,radiation_generation)
        ids=np.asarray(route_indices,dtype='i8')
        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(self.frequency) or
            ids.ndim!=1 or not len(ids) or np.any(ids<0) or np.any(ids>=len(self.routes)) or
            np.any(np.diff(ids)<=0) or not np.array_equal(ids,np.asarray(route_indices))):
            raise ValueError('경로/주파수 블록 주소 오류')
        # 공유 σ만 route×주파수다. route×점×주파수 3D 배열은 만들지 않는다.
        evaluation=np.ascontiguousarray(self.evaluation[start:end])
        sigma=np.ascontiguousarray([self.routes[int(i)].sigma(evaluation) for i in ids])
        arrays=[np.ascontiguousarray(a) for a in (self.frequency[start:end],evaluation,self.width[start:end],
                self.J[:,start:end],sigma,self.edge[ids],trial.milne_prefactor[ids],trial.inverse_kT_Hz)]
        prim=np.ascontiguousarray(self.primary[ids])
        dissolved=[np.ascontiguousarray(a) for a in (self.sigma_edge[ids],self.z[ids],trial.material_b,trial.material_a,trial.material_x)]
        block=Block(len(evaluation),len(ids),self.n_points,*[a.ctypes.data_as(DP) for a in arrays],
                    int(self.dissolved),prim.ctypes.data_as(BP),*[a.ctypes.data_as(DP) for a in dissolved])
        shape=(len(ids),self.n_points,6)
        bound=np.empty(shape);total=np.empty(shape)
        if self.function(C.byref(block),bound.ctypes.data_as(C.POINTER(Rates)),total.ctypes.data_as(C.POINTER(Rates))):
            raise ValueError('다점 BF 율 적분 실패: 입력/지수 블록/비유한 값')
        return bound,total

    def integrate(self,trial,*,material_generation,radiation_generation,route_indices=None,
                  frequency_block=1024,route_block=128):
        self._check(trial,material_generation,radiation_generation)
        ids=np.arange(len(self.routes)) if route_indices is None else np.asarray(route_indices,dtype='i8')
        if (ids.ndim!=1 or not len(ids) or np.any(ids<0) or np.any(ids>=len(self.routes)) or
            np.any(np.diff(ids)<=0) or (route_indices is not None and
            not np.array_equal(ids,np.asarray(route_indices)))):
            raise ValueError('완성 적분의 경로 주소 오류')
        if type(route_block) is not int or route_block<1:
            raise ValueError('경로 블록 크기 오류')
        sums=[np.zeros((len(ids),self.n_points,6),dtype=np.longdouble) for _ in range(2)]
        for start,end in self.frequency_blocks(trial,frequency_block):
            for first in range(0,len(ids),route_block):
                part=slice(first,min(len(ids),first+route_block))
                values=self.contract_block(trial,start,end,ids[part],
                    material_generation=material_generation,radiation_generation=radiation_generation)
                for accumulator,value in zip(sums,values):
                    accumulator[part]+=value
        result=tuple(np.asarray(x,dtype='f8') for x in sums)
        if not all(np.isfinite(x).all() and np.all(x[:,:,:3]>=0) for x in result):
            raise ValueError('완성 다점 BF 적분 비유한/음수 율')
        return result
