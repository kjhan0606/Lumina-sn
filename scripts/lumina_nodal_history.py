"""원래점 C 제공자를 공유 수송 kernel의 전역ν 출력 범위와 직접 연결한다.

범위 적분만 제공한다. 공간 적응의 전ν 수용 판단/광선 driver는 아직 별도다.
실제 계수의 부분 formal도 현재 추가 실행 승인 전에는 실행하면 안 된다.
"""
import ctypes as ct
import math
from pathlib import Path

import numpy as np

from lumina_frequency_history import Kernel,Stats


class NodalKernel(Kernel):
    def __init__(self,path):
        super().__init__(path)
        self._binary=Path(path).resolve()
        ptr=np.ctypeslib.ndpointer(dtype='f8',ndim=1,flags='C_CONTIGUOUS')
        self._range=self.library.cmf_nodal_history_range
        self._range.argtypes=[ct.c_void_p]+[ct.c_int]*4+[ct.c_double]*4+[ptr]*4+[ct.POINTER(Stats)]
        self._range.restype=ct.c_int
        self._provider_compare=self.library.cmf_nodal_history_compare
        self._provider_compare.argtypes=[ct.c_void_p]+[ct.c_int]*4+[ct.c_double]+[ptr]*7+[
            ct.POINTER(Stats),ct.POINTER(ct.c_uint64),ct.POINTER(ct.c_uint64)]
        self._provider_compare.restype=ct.c_int

    def _check_provider(self,provider):
        provider._check()
        if provider.identity.get('field',{}).get('role')=='REFERENCE_INPUT_CONNECTION_ONLY_NOT_SELF_SOLUTION':
            raise ValueError('참조 J 연결 객체로 실제/부분 수송을 실행할 수 없다')
        if self._binary!=provider._binary:
            raise ValueError('원래점 context와 kernel은 같은 C 바이너리를 사용해야 한다')

    def range(self,provider,h,length,delta_a,delta_b,radii,characteristic_intensity,*,begin=0,mode='history'):
        self._check_provider(provider)
        values=np.ascontiguousarray(characteristic_intensity,dtype='f8');radii=np.ascontiguousarray(radii,dtype='f8')
        if (provider._bound is None or type(begin) is not int or begin<0 or values.ndim!=1 or
            not 1<=len(values)<=provider.nf or begin+len(values)>provider.nf or radii.shape!=(3,) or
            not np.isfinite(radii).all() or np.any(radii<provider.g.native_radius[-1]) or np.any(radii>provider.g.radius[0]) or
            mode not in ('history','small')):
            raise ValueError('전역ν 구간/공간 표본/동결 J 연결 범위 오류')
        output=np.empty_like(values);response=np.empty_like(values);stats=Stats()
        code=self._range(provider._ctx,provider.nf,begin,len(values),int(mode=='small'),h,length,delta_a,delta_b,
            radii,values,output,response,ct.byref(stats))
        if code:raise ValueError('C 원래점 직접 수송 범위 실패: '+str(code))
        return output,response,dict({key:int(getattr(stats,key)) for key,_ in Stats._fields_},
            global_frequencies=provider.nf,output_begin=begin,output_count=len(values),mode=mode,
            python_coefficient_callbacks=0,interior_intensity_remappings=0,
            provider_coefficient_cache_heap_bound_bytes=min(provider.nf,max(256,min(8192,math.ceil((delta_a-delta_b)/h)+4)))*(4*8+ct.sizeof(ct.c_int)),
            accepted_physical_field=False,final_goal='NOT_JUDGED')

    def compare_provider(self,provider,h,lengths,deltas,radii,initial,*,begin=0):
        self._check_provider(provider)
        lengths,deltas,radii,initial=[np.ascontiguousarray(v,dtype='f8') for v in (lengths,deltas,radii,initial)]
        leaves=len(lengths);width=initial.size
        if (provider._bound is None or type(begin) is not int or begin<0 or initial.shape!=(width,) or
            not 1<=width<=min(65536,provider.nf) or begin+width>provider.nf or not 1<=leaves<=4096 or
            leaves*width>2097152 or lengths.shape!=(leaves,3) or deltas.shape!=(leaves,3) or radii.shape!=(leaves,5) or
            not np.isfinite(radii).all() or np.any(radii<provider.g.native_radius[-1]) or np.any(radii>provider.g.radius[0])):
            raise ValueError('C 제공자 공간/전역ν 비교 범위 오류')
        out=np.empty(width);errors=np.empty((leaves,width));responses=np.empty_like(errors);stats=Stats()
        calls=ct.c_uint64();checks=ct.c_uint64()
        code=self._provider_compare(provider._ctx,provider.nf,begin,width,leaves,h,
            *[v.ravel() for v in (lengths,deltas,radii,initial,out,errors,responses)],
            ct.byref(stats),ct.byref(calls),ct.byref(checks))
        if code:raise ValueError('C 제공자 혼합 공간 비교 실패: '+str(code))
        return out,errors,responses,dict({key:int(getattr(stats,key)) for key,_ in Stats._fields_},
            formal_vectors=int(calls.value),branch_checks=int(checks.value),python_coefficient_callbacks=0,
            global_frequencies=provider.nf,output_begin=begin,output_count=width,
            provider_coefficient_cache_heap_bound_bytes=min(provider.nf,max(256,min(8192,
                math.ceil(float(np.max(deltas[:,0]-deltas[:,2]))/h)+4)))*(4*8+ct.sizeof(ct.c_int)))
