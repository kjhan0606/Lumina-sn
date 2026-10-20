#!/usr/bin/env python3
"""물리 설정을 보존하고 private 수치 dispatch만 formal 공통장으로 분리한다."""
import argparse
import json
import re
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS,command,once,sha
from prepare_cmfgen_formal_capture import ARGS


def primary_caller(source):
    source=once(source,'\tUSE SET_KIND_MODULE\n','\tUSE SET_KIND_MODULE\n\tUSE LUMINA_FORMAL_PRIMARY,ONLY:FP_ENABLED,FP_START,FP_RECORD,FP_END_ITERATION,FP_NEXT_SOURCE,FP_NODE_K=>KNU_STORE\n')
    opening='\tELSE IF(CONT_VEL)THEN\n';closing='\tELSE IF(EDDINGTON)THEN\n'
    prefix,section=source.split(opening)
    source,suffix=section.split(closing)
    marker='\t  J_IT_COUNTER=0\n\t  DO WHILE(INACCURATE)\n'
    source=once(source,marker,marker+'''\t     CALL FP_START(FL,J_IT_COUNTER,ND,RJ,ETA_CLUMP,CHI_SCAT_CLUMP,COHERENT_ES,USE_J_REL,USE_FORMAL_REL)
''')
    marker="\t     IF(PLANE_PARALLEL_NO_V)THEN\n\t       IF(FIRST_FREQ .AND. J_IT_COUNTER .EQ. 0)WRITE(LUER,*)'Calling MOM_J_PP_V1'"
    source=once(source,marker,"""\t     IF(FP_ENABLED())THEN
\t       CALL FP_RECORD(ND,NP,TC,FEDD,THK_CONT)
\t       RJ(1:ND)=TC(1:ND)
\t       CALL GET_RSQH_REL(RSQHNU,R,V,FL,ND)
\t     ELSE IF(PLANE_PARALLEL_NO_V)THEN
\t       IF(FIRST_FREQ .AND. J_IT_COUNTER .EQ. 0)WRITE(LUER,*)'Calling MOM_J_PP_V1'""")
    marker="\t    END DO\nC\n\t    IF(COHERENT_ES)THEN\n\t      SOURCE(1:ND)=ZETA(1:ND)+THETA(1:ND)*RJ(1:ND)"
    source=once(source,marker,"\t      CALL FP_END_ITERATION(J_IT_COUNTER,T1,ACC_EDD_FAC,INACCURATE)\n\t      IF(FP_ENABLED().AND.INACCURATE)CALL FP_NEXT_SOURCE(ND,TC,RJ)\n"+marker)
    source=once(source,'\t      K_MOM(I)=RJ(I)*FEDD(I)\n\t    END DO\n',
        '\t      K_MOM(I)=RJ(I)*FEDD(I)\n\t    END DO\n\t    IF(FP_ENABLED())K_MOM(1:ND)=FP_NODE_K(1:ND)\n')
    return prefix+opening+source+closing+suffix


def primary_formal(source):
    source=once(source,'\tUSE LUMINA_FORMAL_DIAG\n','\tUSE LUMINA_FORMAL_DIAG\n\tUSE LUMINA_FORMAL_PRIMARY,ONLY:FP_ENABLED,FP_ANGULAR_F,FP_CHECK_EMISSIVITY,FP_LAMBDA_BEGIN,FP_LAMBDA_RAY\n')
    source=once(source,'\tUSE LUMINA_FORMAL_DIAG\n','\tUSE LUMINA_FORMAL_DIAG\n\tUSE LUMINA_FORMAL_PRIMARY_WHOLE,ONLY:PW_REMEMBER\n')
    names=','.join(x[0] for x in ARGS)
    source=once(source,'\tCALL WM_REMEMBER_FORMAL_INPUT(', '\tCALL PW_REMEMBER('+names+')\n\tCALL FP_CHECK_EMISSIVITY(ND,FREQ,ETA,ESEC)\n\tCALL WM_REMEMBER_FORMAL_INPUT(')
    source=once(source,'\tNP_LIMIT=NP-1\n','\tCALL FP_LAMBDA_BEGIN(ND,NP)\n\tNP_LIMIT=NP-1\n')
    marker='            CALL SOLVE_CMF_FORMAL_V2(CHI_RAY,ETA_RAY,IP,FREQ,NU_ON_dNU,INNER_BND_METH,b_planck,dBdTAU,NRAY,NP,NC)'
    if source.count(marker)!=2:raise ValueError('정확한 두 native ray 호출이 아님')
    source=source.replace(marker,'            CALL FP_LAMBDA_RAY(IP,NRAY,ND,NP,NC,R,CHI_RAY,ESEC,dLOG_NU,INITIALIZE)\n'+marker)
    source=once(source,'\tFEDD=KNU_STORE/JNU_STORE\n',"""\tIF(FP_ENABLED())THEN
\t CALL FP_ANGULAR_F(ND,JNU_STORE,HNU_STORE,KNU_STORE,NNU_STORE,FEDD)
\tELSE
\t FEDD=KNU_STORE/JNU_STORE
\tEND IF
""")
    source=once(source,'\tCALL LUMINA_AFFINE_FROM_FORMAL(FREQ,INITIALIZE,NP_LIMIT,ND)\n',"""\tIF(.NOT.FP_ENABLED())THEN
\t CALL LUMINA_AFFINE_FROM_FORMAL(FREQ,INITIALIZE,NP_LIMIT,ND)
\tELSE
\t ! 이 경로는 실제 H/N을 소비한다. 사용하지 않는 MOM 경계비를 만들지 않는다.
\t HBC=0;NBC=0
\tEND IF
""")
    return source


def whole_source():
    declarations=[]
    for name,kind,dim in ARGS:
        if kind=='I':continue
        declarations.append(('CHARACTER(LEN=*)' if kind=='C' else 'LOGICAL' if kind=='L' else 'REAL(KIND=LDP)')+' '+name+('('+dim+')' if dim else ''))
    return Path(__file__).with_name('cmfgen_formal_primary_whole.f90').read_text().replace('__NAMES__',','.join(x[0] for x in ARGS)).replace('__DECLARATIONS__','\n'.join(declarations))


def primary_main(source):
    source=once(source,'\tUSE SET_KIND_MODULE\n',
        '\tUSE SET_KIND_MODULE\n\tUSE LUMINA_FORMAL_PRIMARY,ONLY:FP_ENABLED,FP_NODE_H=>HNU_STORE\n')
    source=once(source,'\t   IF(ACCURATE)THEN\n\t     CALL REGRID_H(',
        '\t   IF(FP_ENABLED())THEN\n\t     SOB(1:ND)=R(1:ND)*R(1:ND)*FP_NODE_H(1:ND)\n\t   ELSE IF(ACCURATE)THEN\n\t     CALL REGRID_H(')
    source=once(source,'\t  H_MOM(1:ND)=SOB(1:ND)/R(1:ND)/R(1:ND)\n\tEND IF\n',
        '\t  H_MOM(1:ND)=SOB(1:ND)/R(1:ND)/R(1:ND)\n\tEND IF\n\tIF(FP_ENABLED())H_MOM(1:ND)=FP_NODE_H(1:ND)\n')
    source=once(source,'\tCALL LUMINA_WHOLE_FREQUENCY_END(ML,NCF)\n',
        '\tCALL FP_WHOLE_FREQUENCY_END(ML,NCF,FL,FQW(ML),R,V,RJ,H_MOM,K_MOM,SOB,ND)\n\tCALL LUMINA_WHOLE_FREQUENCY_END(ML,NCF)\n')
    return once(source,'\tCALL LUMINA_MOM_STOP_BEFORE_POP()\n','\tCALL FP_WHOLE_FINISH()\n\tCALL LUMINA_MOM_STOP_BEFORE_POP()\n')


def primary_kernel(source):
    source=once(source,'      USE MOD_SPACE_GRID_V2\n','      USE MOD_SPACE_GRID_V2\n      USE LUMINA_FORMAL_PRIMARY,ONLY:FP_ENABLED\n      USE LUMINA_POSITIVE_SC,ONLY:SC_BEZIER_SOURCE\n')
    pattern=re.compile(r'^[ \t]*t1=alpha\*source_prime\(iz-1\)\+beta\*source_prime\(iz\)\+gamma\*source_prime\(iz\+1\)\n[ \t]*if\(t1 \.lt\. 0\.0_LDP\)then\n.*?^[ \t]*end if\n',re.M|re.S)
    blocks=pattern.findall(source)
    if len(blocks)!=2:raise ValueError('정확한 양방향 SC source 블록이 아님')
    calls=["source_prime(iz-1),source_prime(iz),source_prime(iz+1),dtau_loc(iz-1),dtau_loc(iz)",
           "source_prime(iz+1),source_prime(iz),source_prime(iz-1),dtau_loc(iz),dtau_loc(iz-1)"]
    for block,args in zip(blocks,calls):
        source=once(source,block,'        if(fp_enabled())then\n          t1=sc_bezier_source('+args+')\n        else\n'+block+'        end if\n')
    return source


def prepare(parent,out):
    if out.exists():raise ValueError('기존 formal 공통장 build 보존')
    record=json.loads((parent/'sweep-build.json').read_text())
    for name,h in record['output_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('수송 부모 변경')
    out.mkdir();native=Path(record['native']);formal=Path(record['formal_build']);mom=Path(record['mom_build'])
    kernel=native/'plane/solve_cmf_formal_v2.f'
    expected=json.loads((formal/'formal-build.json').read_text())['input_sha256'][str(kernel)]
    if sha(kernel)!=expected:raise ValueError('기준 native SC source가 바뀜')
    sources={'cmfgen_formal_primary.f90':Path(__file__).with_name('cmfgen_formal_primary.f90').read_text(),
        'cmfgen_positive_sc.f90':Path(__file__).with_name('cmfgen_positive_sc.f90').read_text(),
        'cmfgen_formal_primary_whole.f90':whole_source(),
        'solve_cmf_formal_v2.f':primary_kernel(kernel.read_text()),
        'comp_j_blank.f':primary_caller((parent/'comp_j_blank.f').read_text()),
        'cmf_formal_rel_v4.f':primary_formal((parent/'cmf_formal_rel_v4.f').read_text()),
        'cmfgen_sub.f':primary_main((parent/'cmfgen_sub.f').read_text())}
    includes=['-I'+str(parent),'-I'+str(formal),'-I'+str(mom),'-I'+str(native/'mod')]
    for name,body in sources.items():
        (out/name).write_text(body)
        flags=FLAGS if name.endswith('.f') else ['-O','-fopenmp','-ffree-line-length-none']
        command(['gfortran',*flags,*includes,'-I.','-c',name],out)
    shutil.copy2(parent/'libsubs.a',out/'libsubs.a')
    command(['ar','r','libsubs.a','comp_j_blank.o','cmf_formal_rel_v4.o','solve_cmf_formal_v2.o'],out)
    common=[out/'cmfgen_formal_primary.o',out/'cmfgen_positive_sc.o',out/'cmfgen_formal_primary_whole.o']+[parent/(k+'.o') for k in ('cmfgen_mom_acceleration','lumina_affine_boundary','lumina_mom_diag','lumina_formal_diag','lumina_whole_monitor')]
    common += [formal/'ext_rel_grid_v4.o',mom/'mod_jrel_v9.o']
    command(['gfortran','-fopenmp','-m64',native/'new_main/cmfgen.o',out/'cmfgen_sub.o',*common,'-L.','-L'+str(native/'lib'),
        '-lsubs','-lpgplt','-lmod',native/'lib/libmy_lapack.a',native/'lib/libmy_blas.a','-o','cmfgen_formal_primary.exe'],out)
    paths=[out/k for k in sources]+[out/'libsubs.a',out/'cmfgen_formal_primary.exe']
    report=dict(scope='uniform formal angular source 고정점 수치 후보; 전체/물리 합격 아님',parent=str(parent),
        parent_manifest_sha256=sha(parent/'sweep-build.json'),output_sha256={str(p):sha(p) for p in paths},
        generator_sha256=sha(Path(__file__)),whole_executable_released=True,
        source_interpolation='positive_continuous_quadratic_Bezier',native_kernel_sha256=expected,
        release_scope='전주파수 pre-pop diagnostic만; 생산/released-T 앵커 미승격',
        final_goal='NOT_JUDGED')
    with (out/'formal-primary-build.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(build=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.out)
