"""순서 보존 GPU 수송 외곽. import만으로 장치 문맥/계산을 시작하지 않는다.

실행 위치·입력/바이너리·예산/출력 봉인은 상위 실행 계약의 책임이다.
이 모듈의 소형/국소 통과를 native90 물리 합격으로 표시하지 않는다.
"""
import ctypes as ct
from dataclasses import dataclass
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

from lumina_linear_native import DenseView

P=ct.c_void_p
I=ct.c_int
D=ct.c_double
U=ct.c_uint64
S=ct.c_size_t
MAP_BYTES=88
RESULT_BYTES=48
CACHE_LIMIT=1024**3


class Memory(ct.Structure):
    _fields_=[(key,U) for key in ('live','peak','reserved','observed_peak','baseline_free','total','kernel_local','stack','threads')]


class Location(ct.Structure):
    _fields_=[('mode',I),('first',I),('last',I),('weight',D)]


def check(code):
    if code:raise RuntimeError('GPU 호출/장치 상태 실패: '+str(code))


def check_host(extra=0):
    # CUDA의 거대한 가상주소 예약을 RAM으로 오인해 RLIMIT_AS로 막지 않는다.
    # 실제 peak RSS와 알려진 다음 host 출력 배열을16GiB 안에서 제한한다.
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+extra>16*1024**3:
        raise MemoryError('GPU 실행의 host RSS16GiB 한도')


class Buffer:
    def __init__(self,owner,address,size,base=None):
        self.owner,self.pointer,self.size,self.base=owner,int(address),int(size),base
    @property
    def address(self):
        if self.owner.closed or not self.pointer or (self.base is not None and not self.base.address):
            raise ValueError('해제된 GPU 버퍼')
        return self.pointer
    def part(self,offset,size):
        if type(offset) is not int or type(size) is not int or offset<0 or size<1 or offset>self.size-size:
            raise ValueError('GPU 버퍼 범위 오류')
        return Buffer(self.owner,self.address+offset,size,self)
    def close(self):
        if self.base is not None:raise ValueError('부분 뷰의 소유 메모리는 해제하지 않는다')
        if self.pointer and not self.owner.closed:
            check(self.owner.lib.cmf_gpu_release(self.owner.ctx,self.pointer));self.pointer=0


class Runtime:
    def __init__(self,library,baseline_free,*,_functions=None,_abis=None):
        self.ctx=None
        if os.environ.get('CUDA_MODULE_LOADING')!='EAGER':raise ValueError('봉인 실행 환경 CUDA_MODULE_LOADING=EAGER 필요')
        self.library=Path(library).resolve();self.lib=ct.CDLL(str(self.library));lib=self.lib
        common={
            'cmf_gpu_create':([I,U,U,ct.POINTER(I)],P),'cmf_gpu_destroy':([P],None),
            'cmf_gpu_allocate':([P,U,ct.POINTER(P)],I),'cmf_gpu_release':([P,P],I),
            'cmf_gpu_upload':([P,P,P,U],I),'cmf_gpu_download':([P,P,P,U],I),
            'cmf_gpu_zero':([P,P,U],I),'cmf_gpu_sync':([P],I),'cmf_gpu_status':([P],P),
            'cmf_gpu_memory':([P,ct.POINTER(Memory)],I)}
        history={
            'lumina_history_cuda_map_bytes':([],S),'lumina_history_cuda_result_bytes':([],S),
            'lumina_history_cuda_build':([P,P]+[D]*4+[I]*3+[P]*2,I),
            'lumina_history_cuda_query':([P,P]+[D]*4+[I]*4+[P]+[I]*2+[P]*4,I),
            'lumina_history_cuda_replay':([I]*3+[P]*9,I),
            'lumina_history_cuda_initial':([I,I,P,P,ct.c_uint]+[P]*5,I),
            'lumina_history_cuda_reconstruct':([I,D]+[P]*4,I),
            'lumina_history_cuda_boundary':([I,P,P,D,P,P],I),
            'lumina_history_cuda_accumulate':([I]*3+[P]*5,I)}
        for name,(args,result) in dict(common,**(history if _functions is None else _functions)).items():
            fun=getattr(lib,name);fun.argtypes=args;fun.restype=result
        abis=(('lumina_history_cuda_map_bytes',MAP_BYTES),('lumina_history_cuda_result_bytes',RESULT_BYTES)) if _abis is None else _abis
        if any(getattr(lib,name)()!=size for name,size in abis):
            raise ValueError('GPU 공통 코어 ABI 변경')
        code=I();self.ctx=lib.cmf_gpu_create(0,16*1024**3,int(baseline_free),ct.byref(code))
        if not self.ctx:check(code.value or 1)
        self.status=lib.cmf_gpu_status(self.ctx)
    @property
    def closed(self):return self.ctx is None
    def allocate(self,size):
        if self.closed or type(size) is not int or size<1:raise ValueError('GPU 할당 범위 오류')
        pointer=P();check(self.lib.cmf_gpu_allocate(self.ctx,size,ct.byref(pointer)))
        return Buffer(self,pointer.value,size)
    def upload_pointer(self,target,source,size):
        if target.owner is not self or size<1 or size>target.size:raise ValueError('GPU 업로드 범위 오류')
        check(self.lib.cmf_gpu_upload(self.ctx,target.address,source,size))
    def upload(self,target,array):
        values=np.asarray(array)
        if not values.flags.c_contiguous:raise ValueError('연속 host 입력 필요')
        self.upload_pointer(target,values.ctypes.data,int(values.nbytes))
    def download(self,source,shape,dtype='f8'):
        check_host(int(np.prod(shape))*np.dtype(dtype).itemsize)
        out=np.empty(shape,dtype=dtype)
        if source.owner is not self or out.nbytes>source.size:raise ValueError('GPU 다운로드 범위 오류')
        check(self.lib.cmf_gpu_download(self.ctx,out.ctypes.data,source.address,out.nbytes));return out
    def zero(self,target):
        if target.owner is not self:raise ValueError('다른 GPU 문맥')
        check(self.lib.cmf_gpu_zero(self.ctx,target.address,target.size))
    def sync(self):check(self.lib.cmf_gpu_sync(self.ctx))
    def report(self):
        out=Memory();check(self.lib.cmf_gpu_memory(self.ctx,ct.byref(out)))
        return {key:int(getattr(out,key)) for key,_ in out._fields_}
    def close(self):
        if not self.closed:self.lib.cmf_gpu_destroy(self.ctx);self.ctx=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class Arena:
    """캐시 총량1GiB 안의 정렬된 가변 블록. 자식은 cudaMalloc을 하지 않는다."""
    def __init__(self,buffer):
        if buffer.size>CACHE_LIMIT:raise ValueError('활성 캐시 한도 초과')
        self.buffer=buffer;self.free=[(0,buffer.size)];self.used=0;self.peak=0;self.live={}
    def take(self,size):
        aligned=(size+255)//256*256
        for i,(start,length) in enumerate(self.free):
            if length>=aligned:
                self.free[i:i+1]=[] if length==aligned else [(start+aligned,length-aligned)]
                self.live[start]=aligned;self.used+=aligned;self.peak=max(self.peak,self.used)
                return start,self.buffer.part(start,size)
        return None
    def release(self,start):
        if start not in self.live:raise ValueError('중복/알 수 없는 캐시 해제')
        size=self.live.pop(start);self.used-=size;self.free.append((start,size));self.free.sort()
        merged=[]
        for a,n in self.free:
            if merged and merged[-1][0]+merged[-1][1]==a:merged[-1]=(merged[-1][0],merged[-1][1]+n)
            else:merged.append((a,n))
        self.free=merged


@dataclass(slots=True)
class Leaf:
    a:float
    b:float
    depth:int=0
    score:float=0.
    cache:object=None


class Work:
    def __init__(self,maximum,deadline):
        self.maximum=int(maximum);self.deadline=float(deadline);self.used=0
        self.counts=dict(map_queries=0,bin_maps=0,tree_compositions=0,padding_leaves=0,replay_leaf_columns=0,
            reconstructed_columns=0,boundary_columns=0,moment_updates=0)
    def check(self):
        check_host()
        if time.monotonic()>=self.deadline:raise TimeoutError('GPU 실행 시간 한도')
    def take(self,kind,count):
        self.check()
        if type(count) is not int or count<0 or count>self.maximum-self.used:raise RuntimeError('GPU 총 병렬 작업량 한도')
        self.used+=count;self.counts[kind]+=count


class GpuDriver:
    def __init__(self,runtime,provider,cpu_library,*,block_width=256,_layout=None):
        if type(block_width) is not int or not 1<=block_width<=65536:raise ValueError('GPU slab 폭 오류')
        if (Path(cpu_library).resolve()!=provider._binary or
            provider.identity.get('field',{}).get('role','').startswith('REFERENCE')):
            raise ValueError('같은 C context 바이너리와 자체 입력만 사용한다')
        self.rt=runtime;self.provider=provider;self.nf=provider.nf;self.ns=provider.ns;self.width=block_width
        self.lib=runtime.lib;self.blocks=[];self.view=None;self.generation=None
        self.cpu=ct.CDLL(str(Path(cpu_library).resolve()));self.geom=self.cpu.cmf_nodal_driver_geometry
        self.geom.argtypes=[P,I]+[P]*5+[D,I]+[P]*6;self.geom.restype=I
        # 768MiB 맵 +96MiB replay 작업공간과 작은 보조 배열을 합쳐1GiB 이하다.
        layout={} if _layout is None else _layout
        self.arena=Arena(self.alloc(layout.get('arena',768*1024**2)));power=1<<(max(1,self.nf-1)-1).bit_length()
        self.tree=self.alloc(2*power*MAP_BYTES)
        self.ray,self.characteristic,self.candidate=[self.alloc(self.nf*8) for _ in range(3)]
        self.limited=self.alloc(8);self.summaries=self.alloc(layout.get('summaries',2*8))
        self.scores=self.alloc(4096*8);self.results=self.alloc(layout.get('results',2097152*RESULT_BYTES))
        self.pointers,self.alternatives=[self.alloc(4096*8) for _ in range(2)];self.masks=self.alloc(4096*4)
        n=(self.nf+self.width-1)//self.width
        self.errors,self.maxima=[self.alloc(n*8) for _ in range(2)]
        self.canonical=self.outer=self.j=self.radii=self.external=None
        self.moments=self.planck=self.gradient=self.event_nodes=self.event_weights=None;self.event_plan=None
        self.upload_generation()
    def alloc(self,size):
        value=self.rt.allocate(int(size));self.blocks.append(value);return value
    def close(self):
        self.rt.sync()
        for value in reversed(self.blocks):value.close()
        self.blocks=[]
    def upload_generation(self):
        view=self.provider.borrow_dense_view()
        if self.view is not None and self.generation==int(view.generation):return
        n=self.ns*self.nf
        if self.canonical is None:
            self.canonical=self.alloc(3*n*8);self.radii=self.alloc(self.ns*8);self.external=self.alloc(view.ne*8)
            self.rt.upload_pointer(self.canonical,view.canonical,3*n*8)
            self.rt.upload_pointer(self.radii,view.radius,self.ns*8);self.rt.upload_pointer(self.external,view.external_radius,view.ne*8)
            self.outer=self.alloc(self.nf*(2*view.ne+1)*8)
        if view.jmode==1 and self.j is None:self.j=self.alloc(n*8)
        if view.jmode==1:self.rt.upload_pointer(self.j,view.j,n*8)
        self.rt.upload_pointer(self.outer,view.outer,self.outer.size)
        self.rt.sync()
        def pointer(block):return ct.cast(block.address,ct.POINTER(D)) if block is not None else ct.POINTER(D)()
        self.view=DenseView(self.ns,self.nf,view.ne,view.jmode,view.generation,pointer(self.canonical),pointer(self.outer),
            pointer(self.j) if view.jmode else ct.POINTER(D)(),pointer(self.radii),pointer(self.external))
        self.generation=int(view.generation);self.input_identity=dict(self.provider.identity)
    def bound(self):
        view=self.provider.borrow_dense_view()
        if self.generation!=view.generation:raise ValueError('GPU J 세대 재업로드 필요')
    def geometry(self,plan,index,leaves):
        a=plan.arrays;edges=np.array([(x.a,x.b) for x in leaves],dtype='f8');n=len(leaves)
        lengths=np.empty((n,3));deltas=np.empty((n,3));radii=np.empty((n,5));locations=(Location*(5*n))();shift=D()
        check(self.geom(self.provider._ctx,len(a['radius']),*[a[key].ctypes.data for key in ('radius','velocity','gauss_x','gauss_w')],
            a['steps'][index].ctypes.data,self.provider._source.dlognu,n,edges.ctypes.data,lengths.ctypes.data,
            deltas.ctypes.data,radii.ctypes.data,ct.addressof(locations),ct.byref(shift)))
        return lengths,deltas,locations,shift.value
    def piece(self,lengths,deltas,locations,row,k):
        points=((0,2,4),(0,1,2),(2,3,4))[k];ends=((0,2),(0,1),(1,2))[k]
        loc=(Location*3)(*[locations[5*row+p] for p in points])
        return loc,float(lengths[row,k]),float(deltas[row,ends[0]]),float(deltas[row,ends[1]])
    def query(self,geometry,row,k,begin,width,target,small,work):
        lengths,deltas,locations,_=geometry;loc,L,da,db=self.piece(lengths,deltas,locations,row,k)
        h=self.provider._source.dlognu;span=(da-db)/h;low=count=leaves=0;nodes=None
        if not small and L>0 and span>0 and begin+db/h<self.nf-1:
            low=max(0,min(self.nf-2,math.floor(begin+db/h)))
            high=min(self.nf-2,math.floor(begin+width-1+db/h+span));count=high-low+1
            leaves=1<<(count-1).bit_length()
            work.take('bin_maps',count);work.take('tree_compositions',leaves-1);work.take('padding_leaves',leaves-count)
            check(self.lib.lumina_history_cuda_build(ct.byref(self.view),loc,h,L,da,db,low,count,leaves,self.tree.address,self.rt.status))
            nodes=self.tree.address
        work.take('map_queries',width)
        check(self.lib.lumina_history_cuda_query(ct.byref(self.view),loc,h,L,da,db,int(small),low,count,leaves,nodes,
            begin,width,None,None,target.address,self.rt.status))
    def group(self,geometry,row,begin,width,primary,alternative,work):
        d=geometry[1][row];h=self.provider._source.dlognu
        spans=[(d[0]-d[2])/h,(d[0]-d[1])/h,(d[1]-d[2])/h];mask=0
        for k,q in enumerate(spans):
            self.query(geometry,row,k,begin,width,primary.part(k*width*MAP_BYTES,width*MAP_BYTES),q<=1.,work)
            if .75<=q<=1.25:
                mask|=1<<k
                self.query(geometry,row,k,begin,width,alternative.part(k*width*MAP_BYTES,width*MAP_BYTES),q>1.,work)
        return mask
    def interval(self,plan,index,work,*,max_seconds=60.):
        """ray 입사→candidate 완결 때만 ray로 교체한다. 실패 부분장은 반환하지 않는다."""
        self.bound();started=time.monotonic();old_deadline=work.deadline;old_maximum=work.maximum
        work.deadline=min(old_deadline,started+max_seconds);work.maximum=min(old_maximum,work.used+200000*self.nf)
        self.active=[];self.cached=[];self.evictions=0
        n=(self.nf+self.width-1)//self.width;leaves=[[Leaf(0.,1.)] for _ in range(n)]
        try:
            work.check()
            geometry=self.geometry(plan,index,[Leaf(0.,1.)]);shift=geometry[3]
            work.take('reconstructed_columns',self.nf)
            check(self.lib.lumina_history_cuda_reconstruct(self.nf,shift,self.ray.address,self.characteristic.address,self.limited.address,self.rt.status))
            initial_maps=self.arena.take(6*self.nf*MAP_BYTES)
            if initial_maps is None:raise RuntimeError('초기 전체ν 맵의 활성 캐시 한도')
            offset,full=initial_maps;primary=full.part(0,3*self.nf*MAP_BYTES);alternative=full.part(3*self.nf*MAP_BYTES,3*self.nf*MAP_BYTES)
            try:
                mask=self.group(geometry,0,0,self.nf,primary,alternative,work);work.take('replay_leaf_columns',self.nf)
                check(self.lib.lumina_history_cuda_initial(self.nf,self.width,primary.address,alternative.address,mask,
                    self.characteristic.address,self.candidate.address,self.errors.address,self.maxima.address,self.rt.status))
                errors=self.rt.download(self.errors,n);maxima=self.rt.download(self.maxima,n)
            finally:self.arena.release(offset)
            for i in range(n):leaves[i][0].score=float(errors[i])
            rounds=1;max_leaves=1
            while True:
                work.check();tolerance=1e-6*float(maxima.max());active=int(errors.argmax())
                if errors[active]<=tolerance:break
                current=leaves[active];score=max(x.score for x in current)
                selected=[x for x in current if x.score>=.5*score]
                if len(current)+len(selected)>4096:raise RuntimeError('GPU leaf4096 한도')
                next_leaves=[]
                for leaf in current:
                    if leaf.score<.5*score:next_leaves.append(leaf);continue
                    middle=.5*(leaf.a+leaf.b)
                    if leaf.depth>=32 or middle==leaf.a or middle==leaf.b:raise RuntimeError('GPU 깊이32/표현 한도')
                    if leaf.cache is not None:self.arena.release(leaf.cache[0]);leaf.cache=None
                    next_leaves.extend((Leaf(leaf.a,middle,leaf.depth+1),Leaf(middle,leaf.b,leaf.depth+1)))
                leaves[active]=current=next_leaves;self.active=current
                begin=active*self.width;width=min(self.width,self.nf-begin)
                missing=[x for x in current if x.cache is None]
                geometry=self.geometry(plan,index,missing) if missing else None
                for row,leaf in enumerate(missing):
                    size=6*width*MAP_BYTES
                    allocation=self.arena.take(size)
                    while allocation is None and self.cached:
                        other=self.cached.pop(0)
                        if other==active:continue
                        for cached_leaf in leaves[other]:
                            if cached_leaf.cache is not None:self.arena.release(cached_leaf.cache[0]);cached_leaf.cache=None
                        self.evictions+=1;allocation=self.arena.take(size)
                    if allocation is None:raise RuntimeError('GPU 활성 캐시1GiB 한도')
                    offset,buffer=allocation;p=buffer.part(0,3*width*MAP_BYTES);a=buffer.part(3*width*MAP_BYTES,3*width*MAP_BYTES)
                    try:mask=self.group(geometry,row,begin,width,p,a,work)
                    except Exception:self.arena.release(offset);raise
                    leaf.cache=(offset,p,a,mask)
                if active in self.cached:self.cached.remove(active)
                self.cached.append(active)
                pointers=np.array([x.cache[1].address for x in current],dtype='u8')
                alternates=np.array([x.cache[2].address for x in current],dtype='u8')
                masks=np.array([x.cache[3] for x in current],dtype='u4')
                self.rt.upload(self.pointers,pointers);self.rt.upload(self.alternatives,alternates);self.rt.upload(self.masks,masks)
                work.take('replay_leaf_columns',len(current)*width)
                if len(current)*width>2097152:raise RuntimeError('GPU replay 작업공간 한도')
                check(self.lib.lumina_history_cuda_replay(len(current),width,width,self.pointers.address,self.alternatives.address,
                    self.masks.address,self.characteristic.part(begin*8,width*8).address,self.candidate.part(begin*8,width*8).address,
                    self.results.address,self.scores.address,self.summaries.address,self.rt.status))
                scores=self.rt.download(self.scores,len(current));summary=self.rt.download(self.summaries,2)
                for leaf,value in zip(current,scores):leaf.score=float(value)
                errors[active],maxima[active]=summary;rounds+=1;max_leaves=max(max_leaves,len(current))
                # 다음 회차에서 현재 최대값을 다시 구한다. 감소 시 이전 통과 slab도 다시 선택된다.
            self.rt.sync();work.check();self.ray,self.candidate=self.candidate,self.ray
            return dict(elapsed_seconds=time.monotonic()-started,global_error=float(errors.max()),
                global_tolerance=tolerance,max_mesh_leaves=max_leaves,rounds=rounds,cache_evictions=self.evictions,
                limited_polynomials=int(self.rt.download(self.limited,1,'u8')[0]),cache_peak_bytes=self.arena.peak)
        finally:
            work.deadline=old_deadline
            work.maximum=old_maximum
            for slab in leaves:
                for leaf in slab:
                    if leaf.cache is not None:self.arena.release(leaf.cache[0]);leaf.cache=None
            self.cached=[];self.active=[]

    def prepare_events(self,plan):
        if self.event_plan is plan:return
        if self.event_plan is not None:raise ValueError('GPU 문맥의 기하 계획은 고정한다')
        a=plan.arrays;npath=len(a['paths'])
        nodes=np.concatenate((a['event_node'],np.full(npath,self.ns-1,dtype='i4'))).astype('i4',copy=False)
        weights=np.concatenate((a['event_weights'],a['core_weights']),axis=0).astype('f8',copy=False)
        if len(weights)!=len(nodes):raise ValueError('관측 사건/가중치 길이 오류')
        self.event_nodes=self.alloc(nodes.nbytes);self.event_weights=self.alloc(weights.nbytes)
        self.rt.upload(self.event_nodes,nodes);self.rt.upload(self.event_weights,weights);self.event_plan=plan

    def accumulate(self,start,count):
        if count:
            check(self.lib.lumina_history_cuda_accumulate(self.ns,self.nf,count,
                self.event_nodes.part(start*4,count*4).address,self.event_weights.part(start*24,count*24).address,
                self.ray.address,self.moments.address,self.rt.status))

    def field(self,plan,planck,gradient,*,seconds=5400.,max_full_vectors=2000000,progress=None):
        """완전한 물리 광선/JHK다. 중간 구간 성공을 완성장으로 반환하지 않는다."""
        self.bound()
        if (plan.source is not self.provider._source or plan.panels!=1 or plan.driver_block_width!=self.width or
            not 0<seconds<=5400. or type(max_full_vectors) is not int or not 1<=max_full_vectors<=2000000):
            raise ValueError('GPU 장 계획/실행 한도 오류')
        started=time.monotonic();work=Work(max_full_vectors*self.nf,started+seconds);a=plan.arrays
        self.prepare_events(plan)
        if self.moments is None:
            self.moments=self.alloc(3*self.ns*self.nf*8);self.planck=self.alloc(self.nf*8);self.gradient=self.alloc(self.nf*8)
        for buffer,array in ((self.planck,planck),(self.gradient,gradient)):
            array=np.ascontiguousarray(array,dtype='f8')
            if array.shape!=(self.nf,):raise ValueError('같은 전체ν 경계 필요')
            self.rt.upload(buffer,array)
        self.rt.zero(self.moments);intervals=0;limited=0;worst=0.;max_mesh=0;rounds=0
        for path,p in enumerate(a['paths']):
            self.rt.zero(self.ray)
            for leg in (0,1):
                if leg==1 and p[4]:
                    work.take('boundary_columns',self.nf);work.take('moment_updates',3*self.nf)
                    check(self.lib.lumina_history_cuda_boundary(self.nf,self.planck.address,self.gradient.address,
                        float(a['core_mu'][path]),self.ray.address,self.rt.status))
                    self.accumulate(len(a['event_node'])+path,1)
                begin,count=map(int,p[2*leg:2*leg+2])
                for index in range(begin,begin+count):
                    report=self.interval(plan,index,work,max_seconds=min(60.,work.deadline-time.monotonic()))
                    intervals+=1;limited+=report['limited_polynomials'];rounds+=report['rounds']
                    max_mesh=max(max_mesh,report['max_mesh_leaves'])
                    if report['global_tolerance']:
                        worst=max(worst,report['global_error']/report['global_tolerance'])
                    start,end=map(int,a['event_ptr'][index:index+2])
                    work.take('moment_updates',3*(end-start)*self.nf);self.accumulate(start,end-start)
            if progress is not None:
                self.rt.sync();progress(dict(paths_completed=path+1,physical_intervals=intervals,
                    parallel_work=work.used,elapsed_seconds=time.monotonic()-started))
        self.rt.sync();work.check()
        if intervals!=len(a['steps']):raise RuntimeError('GPU 물리구간 누락')
        moments=self.rt.download(self.moments,(3,self.ns,self.nf));work.check()
        report=dict(physical_intervals=intervals,paths=len(a['paths']),elapsed_seconds=time.monotonic()-started,
            parallel_work=work.used,work_counters=work.counts,max_mesh_leaves=max_mesh,rounds=rounds,
            limited_polynomials=limited,maximum_estimated_error_ratio=worst,memory=self.rt.report(),
            source_generation=self.generation,input_identity=self.input_identity,
            accepted_physical_field=False,final_goal='NOT_MET')
        return moments,report
