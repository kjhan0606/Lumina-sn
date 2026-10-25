"""같은24×최대32ν 합성 물질의 주파수 이력/공간 적응 수송 연결.

원래 경계·접선·각도 적분을 공유한다. 실제90점 수송 guard를 풀지 않는다.
기하만 세대 사이에 보존하고 χ/총η/자체J/적응 mesh는 매번 새로 평가한다.
"""
from functools import lru_cache
import time

import numpy as np

import lumina_outer_atmosphere as A
from lumina_semi_lagrangian import SmallSemiLagrangianTransport
from lumina_history_spatial import HistoryLimit, propagate


class SmallHistoryTransport(SmallSemiLagrangianTransport):
    def __init__(self,*args,history,panels=8,spatial_rtol=1e-6,**kwargs):
        if not np.isfinite(spatial_rtol) or spatial_rtol<=0:
            raise ValueError('양의 공간 적응 허용치 필요')
        self.history=history;self.spatial_rtol=spatial_rtol
        self._started=0.;self._max_leaves=0;self._max_estimate=0.
        super().__init__(*args,characteristic_segment=None,panels=panels,**kwargs)
        self._point=lru_cache(maxsize=262144)(self._point_uncached)
        self._length=lru_cache(maxsize=262144)(self._length_uncached)

    def _point_uncached(self,key,impact,x):
        z0,z1=self._coordinates[key];r0,r1,_=self._endpoints[key]
        z=z0 if x==0. else z1 if x==1. else z0+(z1-z0)*x
        native=r0 if x==0. else r1 if x==1. else None
        radius=native if native is not None else float(np.hypot(impact,z))
        observer=A.doppler(self.g,impact,z1,r1)
        delta=float(np.log(A.doppler(self.g,impact,z,native)/observer))
        return radius,delta,z,native

    def _length_uncached(self,key,impact,a,b):
        first=self._point(key,impact,a);last=self._point(key,impact,b)
        return float(A.segment_geometry(self.g,impact,first[2],last[2],
            start_radius=first[3],end_radius=last[3])[0]),first[1]-last[1]

    def _sequence(self,steps,incident,own_j,log_chi,log_eta,samples,impact):
        intensity=incident
        for step in steps:
            key=id(step);remaining=120.-(time.monotonic()-self._started)
            if remaining<=0 or self.formal_steps>=2000000:
                raise ValueError('소형 주파수 이력 F 시간/연산 한도 소진; 자동 연장 없음')
            allowed=min(200000,2000000-self.formal_steps)
            try:
                intensity,report=self._propagate(key,impact,intensity,own_j,log_chi,log_eta,allowed,remaining)
            except HistoryLimit as error:
                error.report.update(completed_F_formal_vectors=self.formal_steps,
                    interval_max_calls=allowed,completed_physical_intervals=self.reconstructions,
                    impact_cm=float(impact),first_radius_cm=float(self._endpoints[key][0]),
                    last_radius_cm=float(self._endpoints[key][1]),
                    F_elapsed_seconds=time.monotonic()-self._started)
                raise
            self.formal_steps+=report['formal_vectors'];self.reconstructions+=1
            self.limited_polynomials+=report['limited_polynomials']
            self._max_leaves=max(self._max_leaves,report['spatial_leaves'])
            self._max_estimate=max(self._max_estimate,report['estimated_output_error'])
            for sample in step.observations:
                radius=self.sample_radii[sample];beta=float(A.velocity_at(self.g,radius)/A.C)
                lab_mu=step.z/radius;mu=(lab_mu-beta)/(1-beta*lab_mu)
                samples[sample].append((mu,intensity.copy()))
        return intensity

    def _propagate(self,key,impact,intensity,own_j,log_chi,log_eta,allowed,remaining):
        return propagate(self.history,self.dlog,intensity,lambda a,b:self._length(key,impact,a,b),
            lambda x:self._at(self._point(key,impact,x)[0],own_j,log_chi,log_eta),
            lambda x:self._point(key,impact,x)[1],panels=self.panels,rtol=self.spatial_rtol,
            max_calls=allowed,max_seconds=min(60.,remaining))

    def apply(self,own_j,split_outer=False):
        self._started=time.monotonic();self._max_leaves=0;self._max_estimate=0.
        value,report=super().apply(own_j,split_outer=split_outer)
        elapsed=time.monotonic()-self._started
        if elapsed>120.:raise ValueError('소형 주파수 이력 F 시간 한도 소진; 자동 연장 없음')
        report.update(frequency_scheme='8-point point values; full linear-lognu history; positive rolling moments',
            source_integration='small-shift characteristic endpoint-linear chi/eta; large-shift frozen-chi rolling; adaptive branch-aware estimate',
            coefficient_frequency_rule='all crossed linear-lognu bins; blue endpoint extension',
            spatial_rtol=self.spatial_rtol,max_spatial_leaves=self._max_leaves,
            max_estimated_interval_error=self._max_estimate,
            geometry_point_cache=self._point.cache_info().currsize,
            geometry_length_cache=self._length.cache_info().currsize,elapsed_seconds=elapsed,
            formal_vectors=report['formal_steps'],scalar_formals=report['formal_steps']*self.nf)
        return value,report
