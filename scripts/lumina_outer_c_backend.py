"""공통 외부대기 계획을 C 연산자에 연결한다. 자동 생산/제출 경로는 없다."""
import ctypes as ct
from pathlib import Path

import numpy as np

import lumina_outer_atmosphere as A

PTR = ct.POINTER(ct.c_double)


class Step(ct.Structure):
    _fields_ = [(name,ct.c_double) for name in ('length','drift','weight')]+[
        (name,ct.c_int32) for name in ('exterior','index','observation_begin','observation_count')]


class Observation(ct.Structure):
    _fields_ = [('sample',ct.c_int32),('mu',ct.c_double)]


class Ray(ct.Structure):
    _fields_ = [(name,ct.c_int32) for name in ('minus_begin','minus_count','plus_begin',
        'plus_count','core','turn_begin','turn_count')]+[('inner_mu',ct.c_double)]


class Problem(ct.Structure):
    _fields_ = [(name,ct.c_int32) for name in ('cells','frequencies','rays','steps',
        'observations','exterior_nodes')]+[
        ('dlognu',ct.c_double),('log_reference_radius_ratio',ct.c_double),
        ('step',ct.POINTER(Step)),('ray',ct.POINTER(Ray)),
        ('observation',ct.POINTER(Observation))]+[
        (name,PTR) for name in ('turning_mu','exterior_ratio','cell_chi','cell_es',
            'cell_eta','node_chi','node_es','node_eta','inner_planck','inner_gradient')]+[
        ('work_budget_bytes',ct.c_size_t)]


class Report(ct.Structure):
    _fields_ = [(name,ct.c_int) for name in ('status','iterations','formal_calls')]+[
        (name,ct.c_double) for name in ('change_max_norm_ratio','residual_max',
                                      'residual_max_norm_ratio')]+[
        ('work_bytes',ct.c_size_t),('componentwise_error_verified',ct.c_int)]


def pointer(array):
    return array.ctypes.data_as(PTR)


def array(value, shape, name):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype != np.dtype('f8') or
            not result.flags.c_contiguous):
        raise ValueError(f'{name}: 명시한 차원의 연속 native float64 배열 필요')
    return result


def pack_geometry(plan):
    observations,steps,rays = [],[],[]
    turning = -A.velocity_at(plan.g,plan.sample_radii)/A.C

    def sequence(seq, impact):
        begin = len(steps)
        for step in seq:
            first = len(observations)
            for sample in step.observations:
                r = plan.sample_radii[sample]
                beta = float(A.velocity_at(plan.g,r)/A.C)
                lab = step.z/r
                observations.append(Observation(sample,(lab-beta)/(1-beta*lab)))
            steps.append(Step(step.length,step.drift,step.weight,int(step.exterior),
                              step.index,first,len(observations)-first))
        return begin,len(steps)-begin

    for impact,core,minus,plus in plan.paths:
        mb,mc = sequence(minus,impact)
        pb,pc = sequence(plus,impact)
        first = len(observations)
        if not core:
            for sample in np.flatnonzero(plan.sample_radii == impact):
                observations.append(Observation(int(sample),float(turning[sample])))
        if core:
            r = plan.boundaries[0]
            lab = np.sqrt((r-impact)*(r+impact))/r
            beta = float(A.velocity_at(plan.g,r)/A.C)
            inner_mu = (lab-beta)/(1-beta*lab)
        else:
            inner_mu = 0.
        rays.append(Ray(mb,mc,pb,pc,int(core),first,len(observations)-first,inner_mu))
    if max(len(steps),len(rays),len(observations)) > np.iinfo(np.int32).max:
        raise ValueError('C 광선 계획 인덱스 표현 범위 초과')
    return ((Step*len(steps))(*steps),(Ray*len(rays))(*rays),
            (Observation*len(observations))(*observations),np.ascontiguousarray(turning),
            np.ascontiguousarray(plan.g.native_radius[0]/plan.g.radius[:plan.g.added+1][::-1]))


def worker_options(workers,moments,cpus):
    if (type(workers) is not int or not 1 <= workers <= 16 or
            type(moments) is not int or moments not in (1,3) or
            type(cpus) is not int or not 1 <= cpus <= 16):
        raise ValueError('논리 worker/모멘트/승인 CPU 상한 오류')


def estimate_work(library,plan,frequencies,dlognu,weighted_workers=None,moment_count=3,approved_cpu_limit=1):
    """큰 계수/J 배열 없이 실제 C 할당식을 질의한다. 실행 승인/전체 RSS 보증 아님."""
    if type(frequencies) is not int or not 2 <= frequencies <= np.iinfo(np.int32).max-2:
        raise ValueError('표현 가능한 주파수 차원 필요')
    packed = pack_geometry(plan)
    steps,rays,observations,turning,ratio = packed
    problem = Problem()
    problem.cells,problem.frequencies = plan.n,frequencies
    problem.rays,problem.steps,problem.observations = len(rays),len(steps),len(observations)
    problem.exterior_nodes,problem.dlognu = len(ratio),dlognu
    a,b = plan.g.reference_indices
    problem.log_reference_radius_ratio = float(np.log(plan.g.native_radius[a]/plan.g.native_radius[b]))
    problem.step = steps
    dll = ct.CDLL(str(Path(library)))
    if weighted_workers is None:
        if moment_count != 3 or approved_cpu_limit != 1:
            raise ValueError('가중 worker 없이 모멘트/CPU 설정을 바꾸지 않음')
        f = dll.cmf_outer_workspace_bytes
        f.argtypes = [ct.POINTER(Problem),ct.POINTER(ct.c_size_t)]
        extra = []
    else:
        worker_options(weighted_workers,moment_count,approved_cpu_limit)
        f = dll.cmf_outer_weighted_workspace_bytes
        f.argtypes = [ct.POINTER(Problem)]+[ct.c_int]*3+[ct.POINTER(ct.c_size_t)]
        extra = [weighted_workers,moment_count,approved_cpu_limit]
    f.restype = ct.c_int
    work = ct.c_size_t()
    status = f(ct.byref(problem),*extra,ct.byref(work))
    if status:
        raise CandidateFailure(status,None,dict(work_bytes=work.value))
    return dict(work_bytes=work.value,scope='C owned arrays/plans + maximum temporary window + 1MiB reserve',
        excludes=['borrowed coefficient inputs','caller field outputs','Python geometry/arrays','other runtime libraries'],
        physical_RT_executions=0)


class CandidateFailure(ValueError):
    def __init__(self,status,partial,report):
        super().__init__(f'외부대기 C 후보 실패: status={status}')
        self.status,self.partial,self.report = status,partial,report


def diffusion_boundary(library,frequency,temperature,luminosity,radius,chi):
    """현재 full-SR의 원래 최내곽 Planck 확산 경계. 자체 J 항을 더하지 않는다."""
    nu = np.asarray(frequency)
    if (nu.dtype != np.dtype('f8') or nu.ndim != 1 or len(nu) < 2 or
            not nu.flags.c_contiguous or not np.isfinite(nu).all() or
            np.any(nu <= 0) or not np.all(np.diff(nu) > 0)):
        raise ValueError('확산 경계 주파수/차원 오류')
    dlog = float(np.log(nu[1]/nu[0]))
    if np.max(abs(np.log(nu[1:]/nu[:-1])-dlog)) > 64*np.finfo(float).eps:
        raise ValueError('확산 경계 log-uniform 격자 오류')
    opacity = array(chi,nu.shape,'inner chi')
    planck,gradient = np.empty_like(nu),np.empty_like(nu)
    f = ct.CDLL(str(Path(library))).cmf_exact_planck_diffusion_boundary
    f.argtypes = [ct.c_int,ct.c_double,PTR]+[ct.c_double]*3+[PTR]*3
    f.restype = ct.c_int
    status = f(len(nu),dlog,pointer(nu),temperature,luminosity,radius,
               pointer(opacity),pointer(planck),pointer(gradient))
    if status:
        raise CandidateFailure(status,None,dict(boundary_prepared=False))
    # C와 동일한 폭이며 연산자 경계로 사용한 gradient의 광도만 판독한다.
    width = 2*nu*np.sinh(.5*dlog)
    integral = np.sum(gradient.astype(np.longdouble)*width,dtype=np.longdouble)
    measured = np.longdouble(16)*np.arccos(np.longdouble(-1))**2*radius**2/3*integral
    return planck,gradient,dict(kind='CMF_EXACT_INNER_PLANCK_DIFFUSION',
        intensity='B_nu(T_inner)+mu_cmf*gradient_over_chi',
        temperature_K=float(temperature),luminosity_erg_s=float(luminosity),radius_cm=float(radius),
        reconstructed_luminosity_erg_s=float(measured),
        luminosity_relative_error=float(abs(measured-luminosity)/luminosity) if luminosity else float(abs(measured)),
        J_dependent_boundary_scattering=False,physical_RT_executions=0)


class COuterTransport:
    """계수는 불변으로 빌려 쓴다. 반환 후보는 물리 합격/전방 오차 인증이 아니다."""
    def __init__(self, library, plan, frequency, cell_chi, cell_es, cell_eta,
                 node_chi, node_es, node_eta, inner_planck, inner_gradient,
                 work_budget_bytes,weighted_workers=None,moment_count=3,approved_cpu_limit=1):
        self.context = ct.c_void_p()
        self.dll = ct.CDLL(str(Path(library)))
        self.weighted_workers,self.moment_count,self.approved_cpu_limit = weighted_workers,moment_count,approved_cpu_limit
        self.plan,self.n = plan,plan.n
        self.nu = np.asarray(frequency)
        self.nf = len(self.nu)
        if (self.nu.dtype != np.dtype('f8') or self.nu.ndim != 1 or self.nf < 2 or
                not np.isfinite(self.nu).all() or np.any(self.nu <= 0) or
                not np.all(np.diff(self.nu) > 0)):
            raise ValueError('C 외부대기의 실제 주파수 배열 오류')
        dlog = float(np.log(self.nu[1]/self.nu[0]))
        if np.max(abs(np.log(self.nu[1:]/self.nu[:-1])-dlog)) > 64*np.finfo(float).eps:
            raise ValueError('log-uniform 주파수만 지원')
        if type(work_budget_bytes) is not int or not 0 < work_budget_bytes <= 64*1024**3:
            raise ValueError('명시한 양의64GiB 이내 C workspace 예산 필요')
        self.arrays = [array(a,(self.n,self.nf),name) for a,name in
            zip((cell_chi,cell_es,cell_eta),('cell_chi','cell_es','cell_eta'))]+[
            array(a,(2,self.nf),name) for a,name in
            zip((node_chi,node_es,node_eta),('node_chi','node_es','node_eta'))]+[
            array(inner_planck,(self.nf,),'inner_planck'),
            array(inner_gradient,(self.nf,),'inner_gradient')]
        self.packed = pack_geometry(plan)
        steps,rays,observations,turning,ratio = self.packed
        a,b = plan.g.reference_indices
        self.problem = Problem(self.n,self.nf,len(rays),len(steps),len(observations),
            len(ratio),dlog,float(np.log(plan.g.native_radius[a]/plan.g.native_radius[b])),
            steps,rays,observations,pointer(turning),pointer(ratio),
            *[pointer(a) for a in self.arrays],work_budget_bytes)
        if weighted_workers is None:
            if moment_count != 3 or approved_cpu_limit != 1:
                raise ValueError('가중 worker 없이 모멘트/CPU 설정을 바꾸지 않음')
            create = self.dll.cmf_outer_create
            create.argtypes = [ct.POINTER(Problem),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)]
            extra = []
        else:
            worker_options(weighted_workers,moment_count,approved_cpu_limit)
            create = self.dll.cmf_outer_create_weighted
            create.argtypes = [ct.POINTER(Problem)]+[ct.c_int]*3+[
                ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)]
            extra = [weighted_workers,moment_count,approved_cpu_limit]
        create.restype = ct.c_int
        self.dll.cmf_outer_destroy.argtypes = [ct.c_void_p]
        self.dll.cmf_outer_destroy.restype = None
        self.dll.cmf_outer_apply.argtypes = [ct.c_void_p]+[PTR]*4
        self.dll.cmf_outer_apply.restype = ct.c_int
        self.dll.cmf_outer_solve.argtypes = [ct.c_void_p,PTR,ct.c_int,ct.c_double,ct.POINTER(Report)]
        self.dll.cmf_outer_solve.restype = ct.c_int
        if weighted_workers is not None:
            self.dll.cmf_outer_parallel_threads.argtypes = [ct.c_void_p]
            self.dll.cmf_outer_parallel_threads.restype = ct.c_int
        required = ct.c_size_t()
        status = create(ct.byref(self.problem),*extra,ct.byref(self.context),ct.byref(required))
        self.work_bytes = required.value
        if status:
            raise CandidateFailure(status,None,dict(work_bytes=required.value,
                work_budget_bytes=work_budget_bytes,accepted_physical_field=False))

    @classmethod
    def from_small(cls,library,problem,work_budget_bytes=64*1024**2,**worker_settings):
        indices = list(problem.g.reference_indices)
        nodes = [np.ascontiguousarray(a[indices]) for a in
                 (problem.node_chi,problem.node_es,problem.node_eta)]
        return cls(library,problem,problem.nu,problem.cell_chi,problem.cell_es,problem.cell_eta,
                   *nodes,problem.planck,problem.gradient,work_budget_bytes,**worker_settings)

    def close(self):
        if getattr(self,'context',None):
            self.dll.cmf_outer_destroy(self.context)
            self.context = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()

    def __del__(self):
        self.close()

    def apply(self,own_j,split_outer=True):
        if not self.context:
            raise ValueError('닫힌 외부대기 C context')
        if self.moment_count != 3:
            raise ValueError('J-only context는 apply_j를 사용해야 함')
        own_j = array(own_j,(self.n+2,self.nf),'own_j')
        values = np.empty((3,self.n+2,self.nf))
        status = self.dll.cmf_outer_apply(self.context,pointer(own_j),
            *[pointer(a) for a in values])
        if status:
            raise CandidateFailure(status,own_j.copy(),dict(accepted_physical_field=False))
        info = dict(moments=values.transpose(1,0,2),work_bytes=self.work_bytes,
            affine_error_certificate=False,coefficient_rule='same P0 cells and log-linear external profiles')
        if self.weighted_workers is not None:
            info['actual_ray_threads'] = self.dll.cmf_outer_parallel_threads(self.context)
        return values[0],info

    def apply_j(self,own_j):
        if not self.context:
            raise ValueError('닫힌 외부대기 C context')
        own_j = array(own_j,(self.n+2,self.nf),'own_j')
        result = np.empty_like(own_j)
        status = self.dll.cmf_outer_apply(self.context,pointer(own_j),pointer(result),None,None)
        if status:
            raise CandidateFailure(status,own_j.copy(),dict(accepted_physical_field=False))
        info = dict(work_bytes=self.work_bytes,affine_error_certificate=False)
        if self.weighted_workers is not None:
            info['actual_ray_threads'] = self.dll.cmf_outer_parallel_threads(self.context)
        return result,info

    def solve(self,initial,tolerance=1e-10,cap=100):
        if not self.context:
            raise ValueError('닫힌 외부대기 C context')
        own_j = np.array(initial,dtype='f8',order='C',copy=True)
        array(own_j,(self.n+2,self.nf),'initial own_j')
        if type(cap) is not int or not 1 <= cap <= np.iinfo(np.int32).max//2:
            raise ValueError('표현 가능한 명시 반복 상한 필요')
        report = Report()
        status = self.dll.cmf_outer_solve(self.context,pointer(own_j),cap,tolerance,ct.byref(report))
        def number(x): return float(x) if np.isfinite(x) else 'Infinity'
        result = dict(status=status,iterations=report.iterations,formal_calls=report.formal_calls,
            change_max_norm_ratio=number(report.change_max_norm_ratio),
            residual_max=number(report.residual_max),residual_max_norm_ratio=number(report.residual_max_norm_ratio),
            work_bytes=report.work_bytes,componentwise_verified=bool(report.componentwise_error_verified),
            error_upper='NOT_PROVIDED_FOR_NONLINEAR_OPERATOR',accepted_physical_field=False,
            physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
        if self.weighted_workers is not None:
            result.update(weighted_workers=self.weighted_workers,moment_count=self.moment_count,
                          approved_cpu_limit=self.approved_cpu_limit,
                          actual_ray_threads=self.dll.cmf_outer_parallel_threads(self.context))
        if status:
            raise CandidateFailure(status,own_j,result)
        return own_j,result
