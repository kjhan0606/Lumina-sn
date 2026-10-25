"""C 계수 제공자를 같은 소형 수송에 연결하는 참조 어댑터.

외곽 광선/적응 반복은 아직 Python이다. 이 연결을 생산 C 실행기라고 하지 않는다.
"""
import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_history_nodal import SmallHistoryTransport
from lumina_history_spatial import propagate


class _TrialField:
    def __init__(self,source,values):
        self.ns,self.nf=source.ns,source.nf;self.radius=source.radius;self.frequency=source.frequency
        self.values=np.array(values,dtype='f8',copy=True);self.values.flags.writeable=False;self.closed=False
        self.identity=dict(role='SYNTHETIC_FROZEN_NATIVE_TRIAL_NOT_REFERENCE')

    def read(self,rows,columns):return self.values[np.ix_(rows,columns)]


class SmallCompiledInputTransport(SmallHistoryTransport):
    def __init__(self,*args,input_library,source,**kwargs):
        self.input_library=input_library;self.input_source=source;self._provider=None
        super().__init__(*args,**kwargs)
        if (source.ns,source.nf)!=(24,self.nf) or not np.array_equal(source.frequency,self.nu):
            raise ValueError('같은24점 소형 원본 계수/격자 필요')
        c,s,e=source.read(np.arange(source.ns),np.arange(source.nf))
        if not all(np.array_equal(a,b) for a,b in ((c,self.chi),(s,self.es),(e,self.eta_true))):
            raise ValueError('소형 C 연결과 참조 χ/ES/진η 불일치')

    def _at(self,radius,own_j,log_chi,log_eta):
        if radius not in self._coefficient_cache:
            chi,eta=self._provider.at([radius],np.arange(self.nf))
            self._coefficient_cache[radius]=(chi[0],eta[0])
        return self._coefficient_cache[radius]

    def apply(self,own_j,split_outer=False):
        if self._provider is not None:raise ValueError('중복 C 원래점 연결 실행 금지')
        field=_TrialField(self.input_source,own_j)
        try:
            with CompiledInput(self.input_library,self.g,self.input_source,field,
                    slab_width=min(1024,self.nf),cache_rows=128,profiles=64) as provider:
                self._provider=provider;provider.bind_view()
                value,report=super().apply(own_j,split_outer=split_outer)
                report.update(compiled_source=provider.report(),outer_driver='Python small reference adapter, not production C')
                return value,report
        finally:
            self._provider=None;field.closed=True


class SmallDirectProviderTransport(SmallCompiledInputTransport):
    """전체24점 참조 광선/적응은 공유하되 계수 조회·구간 합성은 C 안에서 한다."""
    def _propagate(self,key,impact,intensity,own_j,log_chi,log_eta,allowed,remaining):
        return propagate(self.history,self.dlog,intensity,lambda a,b:self._length(key,impact,a,b),None,
            lambda x:self._point(key,impact,x)[1],panels=self.panels,rtol=self.spatial_rtol,
            max_calls=allowed,max_seconds=min(60.,remaining),compiled_provider=self._provider,
            radius=lambda x:self._point(key,impact,x)[0])

    def _at(self,*args):
        raise ValueError('직접 제공자 경로에서 Python 계수 표본 조회 금지')
