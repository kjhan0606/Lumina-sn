"""동결 P0 수송에서 셀 내부만 native 점 계수로 바꾸는 고정 J 진단.

기존 소스·입력은 수정하지 않는다. 생성된 C 변형은 고정 J 한 번의 F에만
사용하며 SE/RE 또는 자기일관 해를 산출하지 않는다.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess

import numpy as np

import lumina_outer_atmosphere as A
from lumina_outer_c_backend import PTR, array, pointer
from lumina_outer_transport import CoupledGeometry, Step


FROZEN = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_weighted_20261021-B0Fepu/threadcap-snapshot/src')
SOURCE_SHA = {
    'cmf_outer_transport.c': '090baed443088a13d1b64567f95fb75485bb162eb82f03192f73bbd4cb1ca8b2',
    'cmf_exact_sliding.c': '7b0290dd7ccf55401ff21c92b76170c10491aff14846c9236719f0310d8404dc',
    'cmf_error_envelope.c': '2bc7daa1817e263674e2e1de02cf34e809924bff5623ca9345c51a076b20613f',
    'cmf_relativistic_characteristic.c': '79dd1b8e4326686cede46f98b56cd6f7051fbf4ba4e5a3c03d4ab17d45d8ac47',
}


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('동결 C 소스의 예상 위치가 유일하지 않음')
    return source.replace(old, new)


def variant_source(source):
    """ABI를 바꾸지 않고 빌린 native 배열을 고정 J 연산에만 연결한다."""
    source = _replace_once(source, '    int *seen_m,*seen_p;\n',
        '    int *seen_m,*seen_p;\n'
        '    const double *native_chi,*native_eta,*native_fixed_j;\n')
    source = _replace_once(source,
        '        }else{\n'
        '            chi=p->cell_chi+(size_t)s->index*nf;\n'
        '            eta=c->cell_eta+(size_t)s->index*nf;\n'
        '        }\n'
        '        CMFExactStatus status=cmf_exact_positive_raw_segment',
        '        }else if(c->native_chi){\n'
        '            size_t inner=(size_t)(p->cells-s->index)*nf,outer=inner-nf;\n'
        '            double w=s->weight;\n'
        '            for(size_t k=0;k<nf;k++){\n'
        '                c->chi[k]=w==0 ? c->native_chi[inner+k] :\n'
        '                    w==1 ? c->native_chi[outer+k] :\n'
        '                    (1-w)*c->native_chi[inner+k]+w*c->native_chi[outer+k];\n'
        '                c->eta[k]=w==0 ? c->native_eta[inner+k] :\n'
        '                    w==1 ? c->native_eta[outer+k] :\n'
        '                    (1-w)*c->native_eta[inner+k]+w*c->native_eta[outer+k];\n'
        '            }\n'
        '            chi=c->chi;eta=c->eta;\n'
        '        }else{\n'
        '            chi=p->cell_chi+(size_t)s->index*nf;\n'
        '            eta=c->cell_eta+(size_t)s->index*nf;\n'
        '        }\n'
        '        CMFExactStatus status=cmf_exact_positive_raw_segment')
    source = _replace_once(source,
        'static CMFExactStatus outer_compute(CMFOuterContext *c,const double *own_j)\n{\n',
        'static CMFExactStatus outer_compute(CMFOuterContext *c,const double *own_j)\n{\n'
        '    if(c->native_chi&&own_j!=c->native_fixed_j)return CMF_EXACT_INVALID_INPUT;\n')
    source = _replace_once(source,
        'int cmf_outer_parallel_threads(const CMFOuterContext *c)\n',
        '/* native 0..cells는 바깥→안쪽. 포인터 수명은 호출자가 보장한다. */\n'
        'CMFExactStatus cmf_outer_set_native_fixed_source(CMFOuterContext *c,\n'
        '    const double *chi,const double *eta,const double *fixed_j)\n'
        '{\n'
        '    if(!c||!chi||!eta||!fixed_j||c->native_chi)return CMF_EXACT_INVALID_INPUT;\n'
        '    size_t count=((size_t)c->p.cells+1)*(size_t)c->p.frequencies;\n'
        '    for(size_t i=0;i<count;i++)\n'
        '        if(!isfinite(chi[i])||!isfinite(eta[i])||eta[i]<0)\n'
        '            return CMF_EXACT_INVALID_INPUT;\n'
        '    c->native_chi=chi;c->native_eta=eta;c->native_fixed_j=fixed_j;\n'
        '    return CMF_EXACT_OK;\n'
        '}\n\n'
        'int cmf_outer_parallel_threads(const CMFOuterContext *c)\n')
    source = _replace_once(source,
        '    if(!c||!own_j||iteration_cap<1||iteration_cap>INT_MAX/2||\n',
        '    if(!c||!own_j||c->native_chi||iteration_cap<1||iteration_cap>INT_MAX/2||\n')
    return source


def build_variant(out, compiler='cc'):
    """새 빈 위치의 동결 소스 복사본만 생성·컴파일한다."""
    out = Path(out)
    if out.exists():
        raise FileExistsError('기존 변형 산출물은 보존')
    for name, expected in SOURCE_SHA.items():
        if hashlib.sha256((FROZEN/name).read_bytes()).hexdigest() != expected:
            raise ValueError('동결 C 입력 신원 불일치: ' + name)
    out.mkdir(parents=True)
    for name in SOURCE_SHA:
        contents = (FROZEN/name).read_text()
        (out/name).write_text(variant_source(contents) if name == 'cmf_outer_transport.c' else contents)
    for name in ('cmf_outer_transport.h','cmf_exact_sliding.h','cmf_error_envelope.h',
                 'cmf_relativistic_characteristic.h'):
        (out/name).write_bytes((FROZEN/name).read_bytes())
    library = out/'outer-native-fixed.so'
    command = [compiler,'-O2','-Wall','-Wextra','-Werror','-std=c11','-fPIC',
               '-shared','-fopenmp','-I'+str(out)] + [str(out/name) for name in SOURCE_SHA] + [
                   '-lm','-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
    return library


class NativeSpatialGeometry(CoupledGeometry):
    """외부 경로는 동일, 내부 셀만 지정한 수만큼 나누고 위치 가중치를 준다."""
    def _path(self, impact, coordinates, radii, subdivisions):
        result = []
        for position,(start,end) in enumerate(zip(coordinates[:-1],coordinates[1:])):
            mid_radius = float(np.hypot(impact,.5*(start+end)))
            pieces = 1 if mid_radius > self.g.native_radius[0] else subdivisions
            parts = np.linspace(start,end,pieces+1)
            for number,(a,b) in enumerate(zip(parts[:-1],parts[1:])):
                radius = float(np.hypot(impact,.5*(a+b)))
                exterior = radius > self.g.native_radius[0]
                if exterior:
                    ascending = self.g.radius[:self.g.added+1][::-1]
                    index = int(np.searchsorted(ascending,radius))-1
                    if not 0 <= index < len(ascending)-1:
                        raise ValueError('외부 반경 보간 범위 오류')
                    weight = float(np.log(radius/ascending[index])/
                                   np.log(ascending[index+1]/ascending[index]))
                else:
                    index = int(np.searchsorted(self.boundaries,radius))-1
                    if not 0 <= index < self.n:
                        raise ValueError('native 셀 반경 범위 오류')
                    weight = float((radius-self.boundaries[index])/
                                   (self.boundaries[index+1]-self.boundaries[index]))
                observations = []
                if number == pieces-1:
                    for i,r in enumerate(self.sample_radii):
                        if impact <= r and abs(end) == np.sqrt((r-impact)*(r+impact)):
                            observations.append(i)
                ds,drift = A.segment_geometry(self.g,float(impact),float(a),float(b),
                    start_radius=float(radii[position]) if number == 0 else None,
                    end_radius=float(radii[position+1]) if number == pieces-1 else None)
                result.append(Step(ds,drift,exterior,index,weight,tuple(observations),float(b)))
        return tuple(result)


def set_native_fixed(operator, native_chi, native_eta, fixed_j):
    """C context에서 고정 J와 native 계수를 한 번만 빌린다."""
    shape = (operator.n+1,operator.nf)
    chi = array(native_chi,shape,'native_chi')
    eta = array(native_eta,shape,'native_total_eta')
    j = array(fixed_j,(operator.n+2,operator.nf),'fixed_j')
    if np.any(eta < 0) or not np.isfinite(eta).all() or not np.isfinite(chi).all():
        raise ValueError('native 계수 비유한/음의 방출')
    setter = operator.dll.cmf_outer_set_native_fixed_source
    setter.argtypes = [ct.c_void_p,PTR,PTR,PTR]
    setter.restype = ct.c_int
    status = setter(operator.context,pointer(chi),pointer(eta),pointer(j))
    if status:
        raise ValueError(f'native 고정 J 계수 연결 실패: {status}')
    operator.native_fixed_arrays = (chi,eta,j)
