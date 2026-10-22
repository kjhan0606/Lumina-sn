"""동결 native 고정-J C 경로에 P0 주파수 상풍만 갈아 끼운 격리 비교기.

원 CMFGEN의 3점 공간 SC와 같은 연산자라는 뜻은 아니다.
"""
from pathlib import Path
import shutil
import subprocess

import lumina_outer_native_spatial as N


CALL=('CMFExactStatus status=cmf_exact_positive_raw_segment(p->frequencies,p->dlognu,\n'
      '            s->length,s->drift,chi,eta,*intensity,*alternate,c->scratch,4*nf);')
REPLACEMENT=('CMFExactStatus status=(CMFExactStatus)lumina_upwind_p0_segment(\n'
             '            p->frequencies,p->dlognu,s->length,s->drift,chi,eta,\n'
             '            *intensity,*alternate);')


def upwind_source(source):
    """고정-J native 변형 뒤 정확한 구간 호출 하나만 바꾼다."""
    if source.count(CALL)!=1:
        raise ValueError('동결 C의 원 구간 호출이 유일하지 않음')
    return ('int lumina_upwind_p0_segment(int,double,double,double,const double *,\n'
            '    const double *,const double *,double *);\n'+source.replace(CALL,REPLACEMENT))


def build_variant(out,compiler='cc'):
    out=Path(out)
    if out.exists():
        raise FileExistsError('기존 비교기 빌드 보존')
    for name,expected in N.SOURCE_SHA.items():
        if N.hashlib.sha256((N.FROZEN/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('동결 C 입력 신원 불일치: '+name)
    out.mkdir(parents=True)
    for name in N.SOURCE_SHA:
        source=(N.FROZEN/name).read_text()
        if name=='cmf_outer_transport.c':
            source=upwind_source(N.variant_source(source))
        (out/name).write_text(source)
    for name in ('cmf_outer_transport.h','cmf_exact_sliding.h','cmf_error_envelope.h',
                 'cmf_relativistic_characteristic.h'):
        (out/name).write_bytes((N.FROZEN/name).read_bytes())
    upwind=Path(__file__).resolve().parents[1]/'src/lumina_upwind_p0.c'
    shutil.copy2(upwind,out/upwind.name)
    library=out/'outer-native-upwind.so'
    command=[compiler,'-O2','-Wall','-Wextra','-Werror','-std=c11','-fPIC',
             '-shared','-fopenmp','-I'+str(out)]+[str(out/name) for name in N.SOURCE_SHA]+[
             str(out/upwind.name),'-lm','-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
    return library
