"""검증된 원래점 계획을 C 광선/적응 driver에 연결한다.

계획 생성은 기하/가중치 직렬화뿐이다. 현재 execute_small은24점·최대512ν
합성 시험만 허용한다. 실제90점 F는 새 실행 계약/자원 승인 전에는 실행하지 않는다.
"""
import ctypes as ct
import resource

import numpy as np

import lumina_outer_atmosphere as A
from lumina_semi_native import NativeStreamingPlan
from lumina_nodal_history import NodalKernel
from lumina_native_columns import sha
from pathlib import Path


class DriverStats(ct.Structure):
    _fields_=[(key,ct.c_uint64) for key in ('scalar_formals','kernel_entries','branch_scalar_formals',
        'physical_intervals','reconstructions','limited_polynomials','slab_evaluations','max_mesh_leaves',
        'max_rounds','driver_heap_peak_bytes','kernel_heap_peak_bound_bytes','failed_interval','failed_path')]+[
        (key,ct.c_double) for key in ('elapsed_seconds','last_global_error','last_global_tolerance')]+[('status',ct.c_int)]


class CachedDriverStats(ct.Structure):
    _fields_=[(key,ct.c_uint64) for key in ('cache_build_scalar_formals','cache_replay_leaf_columns',
            'cache_fallback_scalar_formals','cache_workspace_peak_bytes','cache_leaf_builds',
            'cache_evictions','global_rechecks','cache_peak_active_slabs')]


class DriverFailure(ValueError):
    def __init__(self,code,report):
        super().__init__('C 원래점 driver 실패: '+str(code));self.code,self.report=code,report


class CompiledDriverPlan(NativeStreamingPlan):
    def __init__(self,geometry,source,*,panels=16,block_width=256):
        if type(block_width) is not int or not 1<=block_width<=65536:
            raise ValueError('C driver slab 폭 오류')
        super().__init__(geometry,source,panels=panels,block_width=min(32,block_width))
        self.driver_block_width=block_width
        paths=[];steps=[];event_ptr=[0];nodes=[];weights=[];core_mu=[];core_weights=[]
        for path,(impact,is_core,minus,plus) in enumerate(self.paths):
            entry=[]
            for leg,segments in enumerate((minus,plus)):
                entry.extend((len(steps),len(segments)))
                for number,step in enumerate(segments):
                    steps.append((impact,*self._endpoints[id(step)]))
                    for node in step.observations:
                        nodes.append(node);weights.append(self._weights[path,leg,number,node])
                    event_ptr.append(len(nodes))
            paths.append((*entry,int(is_core)))
            if is_core:
                core=float(self.boundaries[0]);z=float(np.sqrt((core-impact)*(core+impact)))
                core_mu.append(self._mu(impact,z,source.ns-1));core_weights.append(self._weights[path,2,0,source.ns-1])
            else:core_mu.append(0.);core_weights.append(np.zeros(3))
        gx,gw=A.gauss_rule(8)
        self.arrays={
            'radius':np.array(geometry.radius,dtype='f8',copy=True),
            'velocity':np.array(geometry.velocity,dtype='f8',copy=True),
            'gauss_x':gx.copy(),'gauss_w':gw.copy(),
            'paths':np.array(paths,dtype='i4'),'steps':np.array(steps,dtype='f8'),
            'core_mu':np.array(core_mu,dtype='f8'),'core_weights':np.array(core_weights,dtype='f8'),
            'event_ptr':np.array(event_ptr,dtype='i4'),'event_node':np.array(nodes,dtype='i4'),
            'event_weights':np.array(weights,dtype='f8')}
        for values in self.arrays.values():values.flags.writeable=False
        if len(steps)!=self.half_segments or len(nodes)+sum(path[4] for path in paths)!=len(self._weights):
            raise ValueError('C 직렬화의 물리구간/관측 사건 누락')

    def layout(self):
        nf,ns=self.source.nf,self.source.ns
        return dict(native_nodes=ns,frequencies=nf,paths=len(self.arrays['paths']),physical_intervals=len(self.arrays['steps']),
            step_observations=len(self.arrays['event_node']),boundary_observations=int(self.arrays['paths'][:,4].sum()),
            panels=self.panels,block_width=self.driver_block_width,
            serialized_geometry_and_events_bytes=sum(a.nbytes for a in self.arrays.values()),
            output_moments_bytes=3*ns*nf*8,readonly_J_view_address_bytes=ns*nf*8,two_ray_bytes=2*nf*8,
            angular_intensity_cube_bytes=0,actual_model_F_calls=0,
            scope='기하/명시 배열 명세; 적응/source cache/동적 kernel/실제RSS는 실행 보고에서 별도')


class DriverKernel(NodalKernel):
    def __init__(self,path):
        super().__init__(path)
        p=np.ctypeslib.ndpointer(dtype='f8',ndim=1,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype='i4',ndim=1,flags='C_CONTIGUOUS')
        self._driver=self.library.cmf_nodal_driver
        self._driver.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,p,p,p,p,
            ct.c_int,ip,p,p,ct.c_int,p,ct.c_int,ip,ip,p,p,p,ct.c_double,ct.c_int,ct.c_int,
            ct.c_double,ct.c_double,ct.c_double,ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,p,ct.POINTER(DriverStats)]
        self._driver.restype=ct.c_int
        self._cached_driver=getattr(self.library,'cmf_nodal_driver_cached',None)
        if self._cached_driver is not None:
            self._cached_driver.argtypes=self._driver.argtypes+[ct.POINTER(CachedDriverStats)];self._cached_driver.restype=ct.c_int
        self._native_driver=getattr(self.library,'cmf_nodal_driver_native90',None)
        if self._native_driver is not None:
            self._native_driver.argtypes=self._driver.argtypes+[ct.POINTER(CachedDriverStats)];self._native_driver.restype=ct.c_int
        self._interval=getattr(self.library,'cmf_nodal_driver_interval',None)
        if self._interval is not None:
            self._interval.argtypes=[ct.c_void_p,ct.c_int]+[p]*5+[ct.c_double,ct.c_int,ct.c_int,
                ct.c_double,ct.c_uint64,p,p,ct.POINTER(DriverStats),ct.POINTER(CachedDriverStats)]
            self._interval.restype=ct.c_int

    def execute_interval(self,plan,provider,index,initial,*,max_seconds=60.,max_full_vectors=2048):
        """전체ν의 국소 비용/동등성 대조. 실패한 부분 벡터는 반환하지 않는다."""
        self._check_provider(provider)
        if (self._interval is None or type(index) is not int or not 0<=index<len(plan.arrays['steps']) or
                plan.source is not provider._source or provider._bound is None or
                not np.array_equal(plan.arrays['radius'],provider.g.radius) or
                not np.array_equal(plan.arrays['velocity'],provider.g.velocity) or
                not np.isfinite(max_seconds) or not 0<max_seconds<=60 or
                type(max_full_vectors) is not int or not 1<=max_full_vectors<=2048):
            raise ValueError('한 구간 전ν 대조의 입력/한도 오류')
        initial=np.ascontiguousarray(initial,dtype='f8')
        if initial.shape!=(provider.nf,):raise ValueError('같은 전역ν 입사 벡터 필요')
        out=np.empty_like(initial);stats=DriverStats();cache=CachedDriverStats();a=plan.arrays
        code=self._interval(provider._ctx,len(a['radius']),a['radius'],a['velocity'],a['gauss_x'],a['gauss_w'],
            np.ascontiguousarray(a['steps'][index]),provider._source.dlognu,plan.panels,plan.driver_block_width,
            max_seconds,max_full_vectors,initial,out,ct.byref(stats),ct.byref(cache))
        report={key:getattr(stats,key) for key,_ in DriverStats._fields_}
        report.update({key:getattr(cache,key) for key,_ in CachedDriverStats._fields_})
        report.update(interval_index=index,panels=plan.panels,global_frequencies=provider.nf,
            source=provider.report(),actual_model_F_calls=0,accepted_physical_field=False,
            scope='한 구간 전체ν; 전체광선/장/SE/RE 아님',final_goal='NOT_MET')
        if code:raise DriverFailure(code,report)
        return out,report

    def execute_small(self,plan,provider,planck,gradient,*,rtol=1e-6,atol=0.,max_seconds=120.,
                      max_full_vectors=2000000,max_interval_vectors=200000,max_leaves=4096,max_depth=32,reuse_maps=False):
        return self._execute(plan,provider,planck,gradient,rtol=rtol,atol=atol,max_seconds=max_seconds,
            max_full_vectors=max_full_vectors,max_interval_vectors=max_interval_vectors,
            max_leaves=max_leaves,max_depth=max_depth,reuse_maps=reuse_maps)

    def execute_native90(self,plan,provider,planck,gradient,contract,*,max_seconds=1500.):
        return self._execute(plan,provider,planck,gradient,rtol=1e-6,atol=0.,max_seconds=max_seconds,
            max_full_vectors=2000000,max_interval_vectors=200000,max_leaves=4096,max_depth=32,
            reuse_maps=True,native_contract=contract)

    def _execute(self,plan,provider,planck,gradient,*,rtol,atol,max_seconds,max_full_vectors,
                 max_interval_vectors,max_leaves,max_depth,reuse_maps,native_contract=None):
        self._check_provider(provider)
        native=native_contract is not None
        if not native:
            if provider.ns!=24 or not 8<=provider.nf<=512:
                raise ValueError('현재 실행은24점·최대512ν 합성만 허용; 실제F는 별도 봉인native90 API 필요')
        else:
            c=native_contract
            if (not isinstance(c,dict) or c.get('schema')!='lumina-native90-first-pair-v1' or
                    c.get('shape')!=[90,1115020] or (provider.ns,provider.nf)!=(90,1115020) or
                    c.get('source_identity')!=provider._source.identity or c.get('plan_layout')!=plan.layout() or
                    c.get('kernel_sha256')!=sha(self._binary) or
                    c.get('driver_wrapper_sha256')!=sha(Path(__file__)) or
                    c.get('accepted_physical_field') is not False or self._native_driver is None or
                    c.get('limits',{}).get('max_F_calls')!=2 or c.get('limits',{}).get('F_seconds')!=1500 or
                    provider._source.identity.get('material_record')!=70):
                raise ValueError('봉인native90 전격자 계약/입력/바이너리 불일치')
        if not np.isfinite(max_seconds) or not 0<max_seconds<=(1500. if native else 120.):
            raise ValueError('선택 경로의 F 시간 상한 오류')
        if type(reuse_maps) is not bool or (reuse_maps and self._cached_driver is None):
            raise ValueError('선택 전파계수 재사용 API가 없는 바이너리 또는 잘못된 선택')
        if plan.source is not provider._source or provider._view is None:
            raise ValueError('같은 source와 동결 전체 J view가 필요하다')
        if not all(np.array_equal(a,b) for a,b in ((plan.arrays['radius'],provider.g.radius),(plan.arrays['velocity'],provider.g.velocity))):
            raise ValueError('C driver 계획/계수 제공자의 기하 불일치')
        integers=(max_full_vectors,max_interval_vectors,max_leaves,max_depth)
        if any(type(value) is not int for value in integers):raise ValueError('정수 작업량/공간 한도 필요')
        if (not 1<=max_full_vectors<=2000000 or not 1<=max_interval_vectors<=200000 or
                not plan.panels<=max_leaves<=4096 or not 0<=max_depth<=32):
            raise ValueError('기존 작업량/공간 한도 범위를 벗어남')
        planck,gradient=[np.ascontiguousarray(value,dtype='f8') for value in (planck,gradient)]
        if planck.shape!=(provider.nf,) or gradient.shape!=planck.shape:
            raise ValueError('C 확산 경계의 전역ν 차원 오류')
        output=np.zeros((3,provider.ns,provider.nf));stats=DriverStats();cache_stats=CachedDriverStats();a=plan.arrays
        arguments=(provider._ctx,provider.ns,provider.nf,len(a['radius']),
            a['radius'],a['velocity'],a['gauss_x'],a['gauss_w'],len(a['paths']),a['paths'].ravel(),
            a['core_mu'],a['core_weights'].ravel(),len(a['steps']),a['steps'].ravel(),len(a['event_node']),
            a['event_ptr'],a['event_node'],a['event_weights'].ravel(),planck,gradient,provider._source.dlognu,
            plan.panels,plan.driver_block_width,rtol,atol,max_seconds,max_full_vectors,max_interval_vectors,
            max_leaves,max_depth,output.ravel(),ct.byref(stats))
        if native:code=self._native_driver(*arguments,ct.byref(cache_stats))
        elif reuse_maps:code=self._cached_driver(*arguments,ct.byref(cache_stats))
        else:code=self._driver(*arguments)
        report={key:getattr(stats,key) for key,_ in DriverStats._fields_}
        report.update({key:getattr(cache_stats,key) for key,_ in CachedDriverStats._fields_})
        report.update(equivalent_full_vectors=report['scalar_formals']/provider.nf,
            propagation_map_reuse=reuse_maps,active_cache_slabs_limit=1 if reuse_maps else 0,
            formal_counter_contract='integer scalar work <= full-vector limit * global_nf; retries/branch comparisons included',
            plan=plan.layout(),source=provider.report(),python_callbacks=0,
            kernel_entries_scope='formal subpiece kernel calls; compare batch calls are slab_evaluations',
            memory_scope='driver owned heap includes two rays; kernel bound/output/source cache/J view/serialized metadata are separate; RSS reported as process high-water',
            process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_scope='Linux process high-water RSS, including earlier tests; not an incremental F allocation',
            accepted_physical_field=False,componentwise_verified=False,actual_model_F_calls=int(native),
            completed_actual_F_calls=int(native and code==0),final_goal='NOT_MET' if native else 'NOT_JUDGED')
        if code:raise DriverFailure(code,report)
        report['moments']=output
        return output[0],report
