"""공통 J의 동적 trial → 실제 전원소 C SE → 전하/원자 내부에너지.

고정 T의 bound BF만 검증한 인수분해로 재사용한다. T 변경은 전적분하며
occupation·Saha·Milne·용해·CI/C3·BB·전쌍 충돌은 매 후보마다 갱신한다.
공개 물질을 게시하거나 고정 T 전하를 전체 공동수렴으로 표시하지 않는다.
"""
import time
import json

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_material_rates import FixedTemperatureBoundRates


class ModelResponse:
    def __init__(self,provider,bridge,collision_source,lines,jbar,status,reference_trial,
                 reference_bound,roster,*,deadline,profile_mass=None):
        self.provider=provider;self.bridge=bridge;self.source=collision_source
        self.roster=roster;self.lines=lines;self.jbar=jbar;self.status=status
        self.profile_mass=np.ones(len(jbar)) if profile_mass is None else np.array(profile_mass,dtype='f8',copy=True)
        if (self.profile_mass.shape!=(len(jbar),) or not np.isfinite(self.profile_mass).all() or
                np.any(self.profile_mass<=0)):raise ValueError('BB 자발방출 프로파일 질량 결손')
        self.profile_mass.flags.writeable=False
        self.seed={k:reference_trial.state[k] for k in ('columns','superlevel','temperature','ne')}
        self.bound_cache=FixedTemperatureBoundRates(provider,reference_trial,reference_bound)
        self.species=sorted(set(map(int,self.seed['columns'][:,0])))
        self.z=np.array([int(r['atomic_number']) for r in provider.rows])
        self.ions={};self.generation=10;self.deadline=deadline;self.direct_check=None
        if (len(self.species)!=6 or len(self.source.models)!=27 or provider.n_points!=90 or
            jbar.shape!=(len(lines['lower']),90) or len(status)!=len(jbar) or
            not np.isin(status,[1,2]).all()):raise ValueError('전모형 공통 응답 범위 누락')

    def check_time(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('공통 물질응답 계산 한도; 부분 후보 보존')

    def evaluate(self,temperature,ne,*,save=None):
        self.check_time();started=time.monotonic();self.generation+=1
        trial=self.provider.prepare({**self.seed,'temperature':temperature,'ne':ne},
                                    material_generation=self.generation)
        fixed_T=np.array_equal(trial.state['temperature'],self.bound_cache.temperature)
        print(json.dumps(dict(phase='DYNAMIC_MATERIAL_BF_BEGIN',generation=self.generation,
            fixed_reference_temperature=fixed_T)),flush=True)
        if fixed_T:
            bound,total=self.bound_cache.integrate(trial,material_generation=self.generation,
                                                  radiation_generation=self.provider.radiation_generation)
        else:
            bound,total=self.provider.integrate(trial,material_generation=self.generation,
                radiation_generation=self.provider.radiation_generation,check=self.check_time)
        if self.direct_check is None:
            selected=[]
            for key in sorted(self.source.models):
                ids=[i for i,r in enumerate(self.provider.routes) if (r.Z,r.stage)==key]
                selected.extend((ids[0],ids[len(ids)//2],ids[-1]))
            selected=np.unique(selected)
            direct=self.provider.integrate(trial,material_generation=self.generation,
                radiation_generation=self.provider.radiation_generation,route_indices=selected)
            error=0.
            for value,actual in zip((bound[selected],total[selected]),direct):
                difference=abs(value-actual);scale=np.maximum(abs(value),abs(actual))
                if np.any(difference>5.e-12*scale):raise ValueError('새 ne의 원27이온 직접 BF/캐시 대조 실패')
                error=max(error,float(np.max(np.divide(difference,scale,out=np.zeros_like(difference),where=scale>0))))
            self.direct_check=dict(routes=len(selected),points=self.provider.n_points,maximum_component_relative=error)
        rates_seconds=time.monotonic()-started
        self.check_time();s=trial.state;ns=self.provider.n_points
        self.bridge.prepare(s,trial.fraction,trial.saha,self.roster,total,
                            trial_id=self.generation,radiation_generation=self.provider.radiation_generation)
        output=np.full(s['superlevel'].shape,np.nan);ledger=np.full((len(self.species),ns,7),np.nan)
        pairs=0
        for iz,Z in enumerate(self.species):
            self.check_time();ids=np.flatnonzero(self.z[self.lines['lower']]==Z)
            lines={k:v[ids] for k,v in self.lines.items()};nu=lines['nu0'];A=lines['A']
            bul=A*B.C**2/(2*B.H*nu**3)
            for node in range(ns):
                values=self.jbar[ids,node]
                self.bridge.begin_element(Z,node,lines,bul/lines['g_ratio']*values,
                                          bul*values,A*self.profile_mass[ids],self.status[ids])
            for key in sorted(k for k in self.source.models if k[0]==Z):
                if key not in self.ions:self.ions[key]=self.source.load(*key)
                ion=self.ions[key]
                for lo,up,omega,_ in ion.blocks(s['temperature']):
                    self.check_time();self.bridge.push_points(Z,lo,up,omega);pairs+=len(lo)
                ion.verify_identity()
            ledger[iz]=self.bridge.finish_points(Z,output)
        if pairs!=20323356 or not np.isfinite(output).all() or np.any(output<=0):
            raise ValueError('전 활성쌍/SL/terminal 후보 결손')
        derived=self.bridge.finalize(output)
        if float(ledger[:,:,6].max())>1.e-8:raise ValueError('동적 전원소 정상 SE 잔차 실패')
        result=dict(temperature=trial.state['temperature'],ne=trial.state['ne'],population=output,
                    columns=trial.state['columns'],occupation=trial.state['occupation'],
                    fraction=trial.fraction,ledger=ledger,**derived)
        metrics=dict(generation=self.generation,BF_seconds=rates_seconds,
            wall_seconds=time.monotonic()-started,SE_residual=float(ledger[:,:,6].max()),
            charge_relative=float(np.max(abs(derived['charge_residual']/s['ne']))),
            bound_reused_routes=self.bound_cache.last_reused_routes if fixed_T else 0,
            bound_direct_routes=self.bound_cache.last_direct_routes if fixed_T else len(bound),
            direct_BF_check=self.direct_check)
        if save is not None:save(result,metrics,trial,total)
        return result,metrics,trial,total
