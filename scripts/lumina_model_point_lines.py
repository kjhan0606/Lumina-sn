"""수송 source와 같은 raw OSC 선택/profile로 원래점의 BB 복사율을 생산한다.

canonical10km/s cache로 재명명하지 않는다. 지지 밖/미표본 선은 NaN과
상태를 유지하며 자발 A만 남긴 것을 완성된 SE 입력이라고 판정하지 않는다.
"""
import ctypes as C

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S


class ModelPointLines:
    def __init__(self,run,rows,models,library):
        self.profile=S.fixed_profile(run/'MODEL')
        self.domain=S.native_line_domain(run)
        self.lines,self.counts,self.identities=S.load_lines(run,rows,models,line_domain=self.domain)
        self.dll=C.CDLL(str(library));self.function=self.dll.lumina_model_point_gaussian_jbar
        ptr=C.POINTER(C.c_double);self.ptr=ptr
        self.function.argtypes=[C.c_size_t,C.c_size_t]+[ptr]*4+[C.c_double]*2+[ptr]*2+[C.POINTER(C.c_uint8)]
        self.function.restype=C.c_int

    def contract(self,nu,width,j):
        nu,width,j=[np.ascontiguousarray(a,dtype='f8') for a in (nu,width,j)]
        if nu.ndim!=1 or len(nu)<2 or width.shape!=nu.shape or j.shape!=nu.shape:
            raise ValueError('원래점 BB 장/주파수 차원 오류')
        center=np.ascontiguousarray(self.lines['nu0'],dtype='f8')
        nt=len(center);jbar=np.empty(nt);mass=np.empty(nt);status=np.empty(nt,dtype='u1')
        if self.function(len(nu),nt,*[a.ctypes.data_as(self.ptr) for a in (nu,width,j,center)],
            self.profile['width_fraction'],self.profile['support_fraction'],
            jbar.ctypes.data_as(self.ptr),mass.ctypes.data_as(self.ptr),status.ctypes.data_as(C.POINTER(C.c_uint8))):
            raise ValueError('원 source-profile Jbar 생산 실패')
        # 실제 OSC 에너지/A로 Einstein 관계를 구성한다. clipping/탈출확률 없음.
        bul=self.lines['A']*B.C**2/(2*B.H*center**3)
        blu=bul/self.lines['g_ratio']
        absorption=blu*jbar;stimulated=bul*jbar
        valid=(status==1)|(status==2)
        if (not np.isfinite(absorption[valid]).all() or not np.isfinite(stimulated[valid]).all() or
                np.any(absorption[valid]<0) or np.any(stimulated[valid]<0)):
            raise ValueError('BB 복사율 수치 범위 초과')
        return dict(jbar=jbar,profile_mass=mass,status=status,
            absorption=absorption,stimulated=stimulated,spontaneous=self.lines['A'].copy())
