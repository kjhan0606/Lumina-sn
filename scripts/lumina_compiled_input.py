"""원래점 계수의 C 제한 캐시/동결 J slab. 수송/물질 풀이를 실행하지 않는다.

원본 계수는 읽기 전용 fd로 공유하지만 C가 dup를 소유한다. J는 각 bind에서
복사한 같은 전역 열만 사용한다. 참조 장의 입력 연결 역할을 수송해로 바꾸지 않는다.
"""
import ctypes as ct
from pathlib import Path

import numpy as np

import lumina_outer_atmosphere as A
from lumina_native_columns import request


class Stats(ct.Structure):
    _fields_=[(key,ct.c_uint64) for key in ('queries','cache_hits','cache_misses','source_bytes',
        'source_preads','profile_hits','profile_misses','bindings','field_bytes_copied',
        'static_cache_bytes','profile_cache_bytes','field_buffer_bytes','cache_peak_rows')]


class DenseStats(ct.Structure):
    _fields_=[(key,ct.c_uint64) for key in ('canonical_bytes','profile_bytes','native_accesses','profile_accesses')]+[
        ('preparation_seconds',ct.c_double),('ready',ct.c_int)]


class DenseView(ct.Structure):
    _fields_=[(name,ct.c_int) for name in ('ns','nf','ne','jmode')]+[('generation',ct.c_uint64)]+[
        (name,ct.POINTER(ct.c_double)) for name in ('canonical','outer','j','radius','external_radius')]


class CompiledInput:
    def __init__(self,library,geometry,source,field,*,slab_width=1024,cache_rows=128,profiles=2048):
        self._ctx=None;self._source=source;self._field=field;self.g=geometry
        if (source.closed or field.closed or not np.array_equal(geometry.native_radius,source.radius) or
            (field.ns,field.nf)!=(source.ns,source.nf) or not np.array_equal(field.radius,source.radius) or
            (field.frequency is not source.frequency and not np.array_equal(field.frequency,source.frequency)) or
            any(type(v) is not int for v in (slab_width,cache_rows,profiles))):
            raise ValueError('C 원래점 입력 신원/범위 오류')
        self.ns,self.nf=source.ns,source.nf;self._bound=None;self._view=None
        self._binary=Path(library).resolve()
        self._lib=ct.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype='f8',ndim=1,flags='C_CONTIGUOUS')
        iptr=np.ctypeslib.ndpointer(dtype='i4',ndim=1,flags='C_CONTIGUOUS')
        self._create=self._lib.cmf_nodal_input_create
        self._create.argtypes=[ct.c_int]*5+[ptr,ptr,ct.c_double]+[ct.c_int]*3+[ct.POINTER(ct.c_int)]
        self._create.restype=ct.c_void_p
        self._destroy=self._lib.cmf_nodal_input_destroy;self._destroy.argtypes=[ct.c_void_p];self._destroy.restype=None
        self._bind=self._lib.cmf_nodal_input_bind;self._bind.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ptr];self._bind.restype=ct.c_int
        self._bind_view=self._lib.cmf_nodal_input_bind_view;self._bind_view.argtypes=[ct.c_void_p,ptr];self._bind_view.restype=ct.c_int
        self._bind_zero=self._lib.cmf_nodal_input_bind_zero_trial
        self._bind_zero.argtypes=[ct.c_void_p];self._bind_zero.restype=ct.c_int
        self._raw=self._lib.cmf_nodal_input_raw
        self._raw.argtypes=[ct.c_void_p,ct.c_int,iptr,ct.c_int,iptr,ptr,ptr,ptr];self._raw.restype=ct.c_int
        self._eval=self._lib.cmf_nodal_input_eval
        self._eval.argtypes=[ct.c_void_p,ct.c_int,ptr,ct.c_int,ptr,ptr,ptr];self._eval.restype=ct.c_int
        self._stats=self._lib.cmf_nodal_input_stats
        self._stats.argtypes=[ct.c_void_p,ct.POINTER(Stats)];self._stats.restype=ct.c_int
        self._dense=getattr(self._lib,'cmf_nodal_input_prepare_dense',None)
        self._dense_stats=getattr(self._lib,'cmf_nodal_input_dense_stats',None)
        self._dense_view=getattr(self._lib,'cmf_nodal_input_dense_view',None)
        if self._dense is not None:
            self._dense.argtypes=[ct.c_void_p,ct.c_double];self._dense.restype=ct.c_int
            self._dense_stats.argtypes=[ct.c_void_p,ct.POINTER(DenseStats)];self._dense_stats.restype=ct.c_int
        if self._dense_view is not None:
            self._dense_view.argtypes=[ct.c_void_p,ct.POINTER(DenseView)];self._dense_view.restype=ct.c_int
        code=ct.c_int()
        r=np.ascontiguousarray(source.radius,dtype='f8');ext=np.ascontiguousarray(geometry.radius[:geometry.added+1],dtype='f8')
        self._ctx=self._create(source._fd,self.ns,self.nf,geometry.added,geometry.reference_indices[1],
            r,ext,A.LEGACY_ETA_MINIMUM,slab_width,cache_rows,profiles,ct.byref(code))
        if not self._ctx:raise ValueError('C 원래점 cache 생성 실패: '+str(code.value))
        self.identity=dict(source=getattr(source,'identity',{}),field=getattr(field,'identity',{}),
            role='READ_ONLY_COLUMN_CONNECTION_NOT_FORMAL',actual_model_F_calls=0)

    def _record_bound_identity(self):
        # 초기 dict를 붙잡지 않는다. 실제 성공한 J 결합 시점의 표기를 기록한다.
        # 이는 J 본문의 해시 검증을 대신하지 않으며 실행기가 별도로 봉인한다.
        self.identity['field']=dict(getattr(self._field,'identity',{}))

    def _check(self,field=True):
        if self._ctx is None or self._source.closed or (field and self._field.closed):
            raise ValueError('닫힌 C 원래점 연결')

    def bind(self,start,count):
        self._check()
        if type(start) is not int or type(count) is not int or start<0 or count<1 or start+count>self.nf or self.ns*count>131072:
            raise ValueError('동결 J slab 범위 오류')
        values=np.ascontiguousarray(self._field.read(np.arange(self.ns),np.arange(start,start+count)),dtype='f8')
        if values.shape!=(self.ns,count):raise ValueError('동결 J slab 행/열 불일치')
        code=self._bind(self._ctx,start,count,values.ravel())
        if code:raise ValueError('동결 J slab 연결 실패: '+str(code))
        self._bound=(start,count);self._view=None
        self._record_bound_identity()

    def bind_view(self):
        """원래점·같은ν의 읽기 전용 소유 배열/memmap만 복사 없이 연결한다."""
        self._check();values=getattr(self._field,'values',None)
        if (not isinstance(values,np.ndarray) or values.shape!=(self.ns,self.nf) or values.dtype!=np.dtype('f8') or
            not values.flags.c_contiguous or values.flags.writeable or
            (isinstance(values,np.memmap) and values.mode!='r') or
            (not isinstance(values,np.memmap) and not values.flags.owndata)):
            raise ValueError('C-order 읽기 전용 native J 소유 배열 또는 mode=r memmap 필요')
        code=self._bind_view(self._ctx,values.ravel())
        if code:raise ValueError('동결 J view 연결 실패: '+str(code))
        self._view=values;self._bound=('readonly-native-view',self.ns,self.nf)
        self._record_bound_identity()

    def bind_zero_trial(self):
        """명시적 시험 J=0만 연결한다. 기존 자체장/참조장을 몰래 덮어쓰지 않는다."""
        self._check()
        if getattr(self._field,'identity',{}).get('role')!='EXPLICIT_ZERO_J_MICRO_TRIAL_ONLY':
            raise ValueError('명시적 J=0 국소 시험 역할이 필요하다')
        code=self._bind_zero(self._ctx)
        if code:raise ValueError('시험 J=0 연결 실패: '+str(code))
        self._bound=('explicit-zero-trial',0,self.nf);self._view=None
        self._record_bound_identity()

    def raw(self,rows,columns):
        self._check(field=False)
        rows,columns=request(rows,columns,self.ns,self.nf)
        rows=np.ascontiguousarray(rows,dtype='i4');columns=np.ascontiguousarray(columns,dtype='i4')
        arrays=[np.empty((len(rows),len(columns))) for _ in range(3)]
        code=self._raw(self._ctx,len(rows),rows,len(columns),columns,*[v.ravel() for v in arrays])
        if code:raise ValueError('C 원본 계수 조회 실패: '+str(code))
        for value in arrays:value.flags.writeable=False
        return tuple(arrays)

    def prepare_dense(self,*,max_seconds=300.):
        """같은 정적3평면과 현재 J 세대의 외부profile을 명시적으로 준비한다."""
        self._check()
        if self._dense is None or self._bound is None or not np.isfinite(max_seconds) or not 0<max_seconds<=300:
            raise ValueError('직접 배열 경로의 연결/준비 한도 오류')
        code=self._dense(self._ctx,max_seconds)
        if code:raise ValueError('직접 배열 준비 실패: '+str(code))
        return self.report()['dense']

    def borrow_dense_view(self):
        """현재 세대의 host 포인터다. 동기 업로드 중에만 빌린다.

        close/J 재결합 전까지만 유효하며 소유권은 이전되지 않는다.
        새 결합은 prepare_dense 뒤 다시 요청한다. 반환값을 장치 주소로 착각하면 안 된다.
        """
        self._check()
        if self._dense_view is None or self._bound is None:raise ValueError('직접 배열 뷰 준비가 필요하다')
        view=DenseView()
        if self._dense_view(self._ctx,ct.byref(view)):raise ValueError('현재 J 세대의 직접 배열 준비가 필요하다')
        return view

    def at(self,radii,columns):
        self._check()
        radii=np.ascontiguousarray(np.atleast_1d(radii),dtype='f8');columns=np.ascontiguousarray(columns,dtype='f8')
        if radii.ndim!=1 or columns.ndim!=1 or radii.size<1 or columns.size<1 or radii.size*columns.size>131072:
            raise ValueError('C 반경/이동ν 요청 범위 오류')
        chi=np.empty((len(radii),len(columns)));eta=np.empty_like(chi)
        code=self._eval(self._ctx,len(radii),radii,len(columns),columns,chi.ravel(),eta.ravel())
        if code:raise ValueError('C 반경/이동ν 보간 실패: '+str(code))
        return chi,eta

    def report(self):
        self._check(field=False);stats=Stats();code=self._stats(self._ctx,ct.byref(stats))
        if code:raise ValueError('C cache 명세 조회 실패')
        dense={}
        if self._dense_stats is not None:
            state=DenseStats()
            if self._dense_stats(self._ctx,ct.byref(state)):raise ValueError('직접 배열 명세 조회 실패')
            dense={key:getattr(state,key) for key,_ in DenseStats._fields_}
        return dict({key:int(getattr(stats,key)) for key,_ in Stats._fields_},dense=dense,
            bound_field_columns=self._bound,field_view_address_bytes=0 if self._view is None else int(self._view.nbytes),
            memory_scope='cache/scratch/J copy buffers; view is address space, not measured RSS; Python/context/radii separate',
            source_read_scope='coefficient body; C header validation separately reads 16 bytes',
            actual_model_F_calls=0,accepted_physical_field=False,
            full_source_hash_verification=getattr(self._source,'identity',{}).get('full_source_hash_verification','NOT_PERFORMED'))

    def close(self):
        if self._ctx is not None:self._destroy(self._ctx);self._ctx=None;self._bound=None;self._view=None

    def __enter__(self):return self

    def __exit__(self,*args):self.close()
