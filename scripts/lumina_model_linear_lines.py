"""고정 프로파일·nodal J의 ν축 선형 표현에 대한 private BB 응답.

기존 native 점표본 연산자와 식별자를 분리한다. Jbar를 프로파일 질량으로
재정규화하거나 외삽하지 않는다. 새 수송/미해상 선 내부 J의 실측이 아니다.
"""
import ctypes as C
import hashlib
from types import MappingProxyType

import numpy as np

import cmfgen_model_continuum as B


class LinearModelLines:
    def __init__(self,lines,profile,library,frequency,J,*,radiation_generation):
        self.lines={name:np.array(value,copy=True,order='C') for name,value in lines.items()}
        if (type(radiation_generation) is not int or radiation_generation<=0 or
            set(self.lines)!=set(('lower','upper','nu0','f','A','g_ratio'))):
            raise ValueError('BB 원자 roster/복사장 세대 오류')
        nt=len(self.lines['nu0'])
        if not nt or any(v.shape!=(nt,) for v in self.lines.values()):
            raise ValueError('BB roster 형상 오류')
        if (not all(np.isfinite(v).all() for v in self.lines.values()) or
            any(np.any(self.lines[k]<0) or np.any(self.lines[k]!=np.floor(self.lines[k]))
                for k in ('lower','upper')) or np.any(self.lines['lower']>=self.lines['upper']) or
            np.any(self.lines['f']<0) or np.any(self.lines['nu0']<=0) or
            np.any(self.lines['A']<0) or np.any(self.lines['g_ratio']<=0)):
            raise ValueError('BB roster 값 오류')
        self.frequency=np.array(frequency,dtype='f8',copy=True,order='C')
        self.J=np.array(J,dtype='f8',copy=True,order='C')
        if (self.frequency.ndim!=1 or len(self.frequency)<2 or self.J.ndim!=2 or
            self.J.shape[0]!=len(self.frequency) or not self.J.shape[1] or
            not np.isfinite(self.frequency).all() or np.any(self.frequency<=0) or
            np.any(np.diff(self.frequency)<=0) or not np.isfinite(self.J).all() or np.any(self.J<0)):
            raise ValueError('ν 선형 BB 장의 형상/값 오류')
        self.profile=dict(profile);self.radiation_generation=radiation_generation
        width=self.profile['width_fraction'];support=self.profile['support_fraction']
        if not np.isfinite([width,support]).all() or width<=0 or not 0<support<1:
            raise ValueError('BB Gaussian 프로파일 오류')
        self.library=C.CDLL(str(library));self.function=self.library.lumina_model_linear_gaussian_jbar
        self.ptr=C.POINTER(C.c_double)
        self.function.argtypes=[C.c_size_t]*3+[self.ptr]*3+[C.c_double]*2+[self.ptr]*2+[C.POINTER(C.c_uint8)]
        self.function.restype=C.c_int
        digest=hashlib.sha256()
        for name,value in [('frequency',self.frequency),('J',self.J),*sorted(self.lines.items())]:
            value.flags.writeable=False
            digest.update(repr((name,value.shape,value.dtype.str)).encode());digest.update(value.tobytes())
        digest.update(repr(sorted(self.profile.items())).encode())
        self.signature=digest.hexdigest()
        self.profile=MappingProxyType(self.profile);self.lines=MappingProxyType(self.lines)

    def contract(self,first,last,*,radiation_generation):
        if radiation_generation!=self.radiation_generation:
            raise ValueError('다른 복사장 세대의 BB cache 소비')
        if type(first) is not int or type(last) is not int or not 0<=first<last<=len(self.lines['nu0']):
            raise ValueError('BB 선 블록 주소 오류')
        center=np.ascontiguousarray(self.lines['nu0'][first:last],dtype='f8')
        jbar=np.empty((len(center),self.J.shape[1]));mass=np.empty(len(center));status=np.empty(len(center),dtype='u1')
        if self.function(len(self.frequency),self.J.shape[1],len(center),
            *[v.ctypes.data_as(self.ptr) for v in (self.frequency,self.J,center)],
            self.profile['width_fraction'],self.profile['support_fraction'],
            jbar.ctypes.data_as(self.ptr),mass.ctypes.data_as(self.ptr),status.ctypes.data_as(C.POINTER(C.c_uint8))):
            raise ValueError('ν 선형 Gaussian BB 적분 실패')
        return dict(jbar=jbar,profile_mass=mass,status=status,first=first,last=last,
                    signature=self.signature,radiation_generation=self.radiation_generation)

    def rates(self,first,last,values):
        if (values.get('signature')!=self.signature or
            values.get('radiation_generation')!=self.radiation_generation or
            values.get('first')!=first or values.get('last')!=last):
            raise ValueError('다른 복사장/선 블록의 BB cache 소비')
        center=self.lines['nu0'][first:last];A=self.lines['A'][first:last]
        bul=A*B.C**2/(2*B.H*center**3)
        return dict(absorption=bul[:,None]/self.lines['g_ratio'][first:last,None]*values['jbar'],
                    stimulated=bul[:,None]*values['jbar'],spontaneous=A.copy())
