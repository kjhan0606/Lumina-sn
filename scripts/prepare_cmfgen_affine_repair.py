#!/usr/bin/env python3
"""원본을 보존한 private CMFGEN에서 실제 입사 모멘트의 경계항을 복원한다."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS,ARGUMENTS,command,once,sha


def boundary_module():
    return """MODULE LUMINA_AFFINE_BOUNDARY
USE SET_KIND_MODULE
USE,INTRINSIC::IEEE_ARITHMETIC
USE MOD_SPACE_GRID_V2,ONLY:RAY,JQW_P,JQW_M,HQW_P,HQW_M,NQW_P,NQW_M
USE MOD_RAY_MOM_STORE,ONLY:HBC,NBC
IMPLICIT NONE
REAL(KIND=LDP),SAVE::AFFINE_HINC=0.0_LDP,AFFINE_NINC=0.0_LDP
REAL(KIND=LDP),SAVE::AFFINE_HINC_PREV=0.0_LDP,AFFINE_NINC_PREV=0.0_LDP
REAL(KIND=LDP),SAVE::AFFINE_FREQ_SAVE=0.0_LDP
LOGICAL,SAVE::AFFINE_READY=.FALSE.
CONTAINS
SUBROUTINE LUMINA_AFFINE_FROM_FORMAL(FREQ,INIT,NP,ND)
REAL(KIND=LDP) FREQ,JP,HP,NNP,JM,HM,NNM,IP,IM
INTEGER NP,ND,K,L
LOGICAL INIT
IF(INIT)THEN
 AFFINE_HINC_PREV=0.0_LDP;AFFINE_NINC_PREV=0.0_LDP
ELSE
 IF(.NOT.AFFINE_READY.OR.FREQ>AFFINE_FREQ_SAVE)STOP 89
 IF(FREQ/=AFFINE_FREQ_SAVE)THEN
  AFFINE_HINC_PREV=AFFINE_HINC;AFFINE_NINC_PREV=AFFINE_NINC
 END IF
END IF
JP=0.0_LDP;HP=0.0_LDP;NNP=0.0_LDP
JM=0.0_LDP;HM=0.0_LDP;NNM=0.0_LDP
DO K=1,NP
 L=RAY(K)%LNK(1)
 IP=RAY(K)%I_P(L);IM=RAY(K)%I_M(L)
 JP=JP+IP*JQW_P(1,K);HP=HP+IP*HQW_P(1,K);NNP=NNP+IP*NQW_P(1,K)
 JM=JM+IM*JQW_M(1,K);HM=HM+IM*HQW_M(1,K);NNM=NNM+IM*NQW_M(1,K)
END DO
IF(JP<=0.0_LDP.OR.JM<0.0_LDP)STOP 89
IF(.NOT.ALL(IEEE_IS_FINITE([JP,HP,NNP,JM,HM,NNM])))STOP 89
HBC=HP/JP;NBC=NNP/JP
AFFINE_HINC=HBC*JM-HM;AFFINE_NINC=NBC*JM-NNM
AFFINE_FREQ_SAVE=FREQ;AFFINE_READY=.TRUE.
END SUBROUTINE
END MODULE
"""


def repair_moment(source,spatial=False):
    source=once(source,"\tUSE LUMINA_MOM_DIAG\n","\tUSE LUMINA_MOM_DIAG\n\tUSE LUMINA_AFFINE_BOUNDARY,ONLY:AFFINE_HINC,AFFINE_NINC,AFFINE_HINC_PREV,AFFINE_NINC_PREV\n")
    old="\t    XM(1)= PSIPREV(1)*GAM_RSQJNU_PREV(1)\n"
    new=old+"""\t    XM(1)=XM(1)+GAM_REL(1)*R(1)*R(1)*(P_H(1)*AFFINE_HINC+DELTA(1)*(
\t1       AFFINE_HINC-AFFINE_HINC_PREV+(GAM_REL_SQ(1)*(SIGMA(1)+1.0_LDP)-1.0_LDP)*
\t1       (AFFINE_NINC-AFFINE_NINC_PREV)))
"""
    source=once(source,old,new)
    source=once(source,"\t  HFLUX_AT_OB=HBC*JNU_SM(1)","\t  HFLUX_AT_OB=HBC*JNU_SM(1)-AFFINE_HINC")
    if spatial:
        source=once(source,"\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n",
                    "\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n\tREAL(KIND=LDP) LUM_AFFINE_GRAD_SOURCE,LUM_AFFINE_FLUX_SOURCE\n")
        start="\tDO I=1,ND-1\n\t  HU(I)="
        correction="""\tLUM_AFFINE_GRAD_SOURCE=0.0_LDP
\tLUM_AFFINE_FLUX_SOURCE=0.0_LDP
\tIF(INCL_ADVEC_TERMS)THEN
\t VdHdR_TERM(1)=BETA(1)*HBC
\t LUM_AFFINE_GRAD_SOURCE=BETA(1)*GAM_REL(1)*R(1)*R(1)*AFFINE_HINC*Q(1)/DTAU_H(1)
\t LUM_AFFINE_FLUX_SOURCE=LUM_AFFINE_GRAD_SOURCE/(P_H(1)+W(1))
\tEND IF
"""
        source=once(source,start,correction+start)
        source=once(source,"\t  IF(XM_CHK_OPTION .EQ. 'SET_POS')THEN",
                    "\t  XM(2)=XM(2)-LUM_AFFINE_FLUX_SOURCE\n\t  IF(XM_CHK_OPTION .EQ. 'SET_POS')THEN")
        source=once(source,"\t1       (AFFINE_NINC-AFFINE_NINC_PREV)))\n",
                    "\t1       (AFFINE_NINC-AFFINE_NINC_PREV)))\n\t    XM(1)=XM(1)+LUM_AFFINE_GRAD_SOURCE\n")
        event="\t  IF(LUM_SELECTED .AND. (H_CHK_OPTION=='MAX_VAL' .OR. H_CHK_OPTION=='AV_VAL'))THEN"
        source=once(source,event,"\t  GAM_RSQHNU(1)=GAM_RSQHNU(1)+LUM_AFFINE_FLUX_SOURCE\n"+event)
    return source


def repair_formal(source):
    source=once(source,"\tUSE LUMINA_FORMAL_DIAG\n","\tUSE LUMINA_FORMAL_DIAG\n\tUSE LUMINA_AFFINE_BOUNDARY,ONLY:LUMINA_AFFINE_FROM_FORMAL\n")
    old="\tHBC=HNU_AT_OB/JNU_STORE(1)\n\tNBC=NNU_AT_OB/JNU_STORE(1)\n\tIF(HBC .LT. 0.0_LDP)HBC=0.0_LDP\n\tIF(NBC .LT. 0.0_LDP)NBC=0.0_LDP"
    return once(source,old,"\tCALL LUMINA_AFFINE_FROM_FORMAL(FREQ,INITIALIZE,NP_LIMIT,ND)")


def replay_source():
    args=",".join("A%"+name for name,_,_ in ARGUMENTS)
    return """PROGRAM AFFINE_MOM_REPLAY
USE LUMINA_MOM_DIAG
USE MOD_JREL_V9
USE MOD_RAY_MOM_STORE
USE LUMINA_AFFINE_BOUNDARY
IMPLICIT NONE
TYPE(LUMINA_ARG_STATE) A
TYPE(LUMINA_JREL_STATE) J
TYPE(LUMINA_RAY_STATE) S
CHARACTER(LEN=4096) INPUT,BOUNDARY
INTEGER LU,N
REAL(KIND=LDP) HNOW,NNOW,HPREV,NPREV
REAL(KIND=LDP),ALLOCATABLE::JP(:),HP(:),SLOPE(:)
CALL GET_COMMAND_ARGUMENT(1,INPUT);CALL GET_COMMAND_ARGUMENT(2,BOUNDARY)
IF(INPUT==' '.OR.BOUNDARY==' ')STOP 85
CALL LUMINA_SNAPSHOT_IO(TRIM(INPUT),A,J,S,.TRUE.)
CALL LUMINA_JREL_RESTORE(J);CALL LUMINA_RAY_RESTORE(S)
OPEN(NEWUNIT=LU,FILE=BOUNDARY,STATUS='OLD',ACTION='READ')
READ(LU,*) N
IF(N/=ND)STOP 86
READ(LU,*) HNOW,NNOW,AFFINE_HINC,AFFINE_NINC,HPREV,NPREV,AFFINE_HINC_PREV,AFFINE_NINC_PREV
ALLOCATE(JP(N),HP(N),SLOPE(N))
READ(LU,*) JP
READ(LU,*) HP
CLOSE(LU)
IF(ANY(JP<=0.0_LDP))STOP 89
GAM_RSQJNU_PREV=JP;GAM_RSQHNU_PREV=HP
HBC=HNOW;NBC=NNOW;HBC_PREV=HPREV;NBC_PREV=NPREV
IF(A%FREQ/=FREQ_SAVE)THEN
 GAM_RSQJNU_SAVE=JP;GAM_RSQHNU_SAVE=HP
 HBC_SAVE=HPREV;NBC_SAVE=NPREV
ELSE
 HBC_SAVE=HNOW;NBC_SAVE=NNOW
END IF
CALL DERIVCHI(SLOPE,JP,R,ND,'LINMON')
dlnGRSQJdlnR=R*SLOPE/JP
AFFINE_READY=.TRUE.;AFFINE_FREQ_SAVE=A%FREQ
IF(LU_MOM>0)OPEN(UNIT=LU_MOM,FILE='MOM_J_ERRORS',STATUS='NEW',ACTION='WRITE')
CALL MOM_JREL_V9(__ARGS__)
CALL LUMINA_JREL_TAKE(J);CALL LUMINA_RAY_TAKE(S)
CALL LUMINA_SNAPSHOT_IO('AFFINE_RESULT.bin',A,J,S,.FALSE.)
WRITE(6,'(A)')'LUMINA_AFFINE_MOM_RETURNED'
END PROGRAM
""".replace("__ARGS__",args)


def prepare(formal_build,out,spatial=False):
    if out.exists():raise ValueError("기존 affine build 보존")
    manifest=json.loads((formal_build/"formal-build.json").read_text())
    for name,expected in manifest["output_sha256"].items():
        if sha(Path(name))!=expected:raise ValueError("부모 formal build 변경")
    mom=Path(manifest["mom_build"]);native=Path(manifest["native"])
    if sha(formal_build/"baseline/cmfgen_formal_unmodified.exe")!=manifest["rebuilt_unmodified_sha256"]:
        raise ValueError("원래 부모 실행파일 불일치")
    out.mkdir()
    sources={"lumina_affine_boundary.f90":boundary_module(),
             "mom_jrel_v9.f":repair_moment((mom/"mom_jrel_v9.f").read_text(),spatial),
             "cmf_formal_rel_v4.f":repair_formal((formal_build/"cmf_formal_rel_v4.f").read_text()),
             "affine_mom_replay.f90":replay_source()}
    for name,text in sources.items():(out/name).write_text(text)
    for name in sources:
        flags=FLAGS if name.endswith(".f") else ["-fopenmp","-m64","-O","-ffree-line-length-none","-fbacktrace"]
        command(["gfortran",*flags,"-I.","-I"+str(formal_build),"-I"+str(mom),"-I"+str(native/"mod"),"-c",name],out)
    shutil.copy2(formal_build/"libsubs.a",out/"libsubs.a")
    command(["ar","r","libsubs.a","mom_jrel_v9.o","cmf_formal_rel_v4.o"],out)
    common=["lumina_affine_boundary.o",formal_build/"ext_rel_grid_v4.o",formal_build/"lumina_formal_diag.o",
            mom/"mod_jrel_v9.o",mom/"lumina_mom_diag.o"]
    libs=["-L.","-L"+str(native/"lib"),"-lsubs","-lpgplt","-lmod",native/"lib/libmy_lapack.a",native/"lib/libmy_blas.a"]
    command(["gfortran","-fopenmp","-m64","affine_mom_replay.o",*common,*libs,"-o","affine_mom_replay.exe"],out)
    command(["gfortran","-fopenmp","-m64",native/"new_main/cmfgen.o",formal_build/"cmfgen_sub.o",*common,*libs,"-o","cmfgen_affine.exe"],out)
    outputs=[out/name for name in sources]+[out/"libsubs.a",out/"affine_mom_replay.exe",out/"cmfgen_affine.exe"]
    record=dict(scope="원본 보존 private 입사 경계 수리; 전 주파수·물리 목표 미검증",formal_build=str(formal_build),
        mom_build=str(mom),native=str(native),parent_manifest_sha256=sha(formal_build/"formal-build.json"),
        rebuilt_unmodified_sha256=manifest["rebuilt_unmodified_sha256"],generator_sha256=sha(Path(__file__)),
        output_sha256={str(p):sha(p) for p in outputs},spatial_affine=spatial,final_goal="NOT_JUDGED")
    with (out/"affine-build.json").open("x") as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(build=str(out),rebuilt_unmodified_sha256=record["rebuilt_unmodified_sha256"])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formal-build",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--spatial-affine",action="store_true")
    a=p.parse_args();prepare(a.formal_build,a.out,a.spatial_affine)
