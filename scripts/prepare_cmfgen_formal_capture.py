#!/usr/bin/env python3
"""두 실제 주파수의 formal 광선 입력을 보존한다. 원래 산술과 물리는 유지한다."""
import argparse
import json
from pathlib import Path
import re
import shutil

from prepare_cmfgen_mom_capture import FLAGS,command,once,sha,rebuild_unmodified_sources


ARGS=[("ETA","R","ND"),("CHI","R","ND"),("ESEC","R","ND"),
      ("V","R","ND"),("SIGMA","R","ND"),("R","R","ND"),("P","R","NP"),
      ("JNU","R","ND"),("FEDD","R","ND"),("RET_HNU_AT_IB","R",None),
      ("RET_HNU_AT_OB","R",None),("IPLUS","R","NP"),("FREQ","R",None),
      ("dLOG_NU","R",None),("B_PLANCK","R",None),("DBB","R",None),
      ("INNER_BND_METH","C",None),("THICK_OB","L",None),("VDOP_VEC","R","ND"),
      ("VDOP_FRAC","R",None),("REXT_FAC","R",None),("METHOD","C",None),
      ("INITIALIZE","L",None),("NEW_FREQ","L",None),("NC","I",None),
      ("NP","I",None),("ND","I",None)]
OUTPUTS={"JNU","FEDD","RET_HNU_AT_IB","RET_HNU_AT_OB","IPLUS"}
TYPES={"R":"REAL(KIND=LDP)","I":"INTEGER","L":"LOGICAL"}


def argument_code():
    names=",".join(x[0] for x in ARGS)
    lines=["TYPE LUMINA_FORMAL_ARGS"]
    for name,kind,dim in ARGS:
        decl="CHARACTER(LEN=:),ALLOCATABLE" if kind=="C" else TYPES[kind]+(",ALLOCATABLE" if dim else "")
        lines.append(decl+" :: "+name+("(:)" if dim else ""))
    lines += ["END TYPE","CONTAINS","SUBROUTINE FORMAL_ARGS_TAKE(A,IS_OUTPUT,"+names+")",
              "TYPE(LUMINA_FORMAL_ARGS),INTENT(OUT)::A","LOGICAL IS_OUTPUT","INTEGER NC,NP,ND"]
    for name,kind,dim in ARGS:
        if kind=="I":continue
        lines.append(("CHARACTER(LEN=*)" if kind=="C" else TYPES[kind])+" :: "+name+("("+dim+")" if dim else ""))
    for name,_,dim in ARGS:
        if name in OUTPUTS:
            if dim:lines.append("ALLOCATE(A%"+name+"("+dim+"))")
            lines += ["IF(IS_OUTPUT)THEN"," A%"+name+"="+name,"ELSE"," A%"+name+"=0.0_LDP","END IF"]
        else:lines.append("A%"+name+"="+name)
    lines += ["END SUBROUTINE","SUBROUTINE FORMAL_ARGS_IO(LU,A,READING)",
              "INTEGER LU,NCHAR","LOGICAL READING","TYPE(LUMINA_FORMAL_ARGS) A",
              "IF(READING)THEN"," READ(LU) A%NC,A%NP,A%ND",
              " IF(A%ND<3.OR.A%ND>10000.OR.A%NP<A%ND.OR.A%NP>500.OR.A%NC<1.OR.A%NC>A%NP)STOP 86",
              "ELSE"," WRITE(LU) A%NC,A%NP,A%ND","END IF"]
    for name,kind,dim in ARGS:
        if kind=="I":continue
        item="A%"+name
        lines += ["IF(READING)THEN"]
        if kind=="C":
            lines += [" READ(LU) NCHAR"," IF(NCHAR<1.OR.NCHAR>200)STOP 86",
                      " IF(ALLOCATED("+item+"))DEALLOCATE("+item+")",
                      " ALLOCATE(CHARACTER(LEN=NCHAR)::"+item+")"]
        elif dim:
            lines += [" IF(ALLOCATED("+item+"))DEALLOCATE("+item+")",
                      " ALLOCATE("+item+"(A%"+dim+"))"]
        lines += [" READ(LU) "+item,"ELSE"]
        if kind=="C":lines.append(" WRITE(LU) LEN("+item+")")
        lines += [" WRITE(LU) "+item,"END IF"]
    lines += ["IF(A%INITIALIZE.OR.A%INNER_BND_METH/='DIFFUSION')STOP 89","END SUBROUTINE"]
    return "\n".join(lines)


def module_source():
    raw=("ND_STORE HN_DEF_ON_NODES HBC NBC HNU_AT_OB HNU_AT_IB NNU_AT_OB NNU_AT_IB "
         "HNU_AT_OB_PREV HNU_AT_IB_PREV NNU_AT_OB_PREV NNU_AT_IB_PREV").split()
    arrays="R_STORE RMID_STORE EXT_RMID_STORE GAM_REL_STORE JNU_STORE HNU_STORE KNU_STORE NNU_STORE".split()
    ray_take=["SUBROUTINE FORMAL_RAY_TAKE(S)","TYPE(LUMINA_RAY_STATE),INTENT(OUT)::S","CALL LUMINA_RAY_EMPTY(S)"]
    ray_take += ["S%"+name+"="+name for name in raw]
    for name in arrays:
        ray_take += ["IF(.NOT.ALLOCATED("+name+"))STOP 89",
                     "ALLOCATE(S%"+name+"(LBOUND("+name+",1):UBOUND("+name+",1)))"]
        if name=="RMID_STORE":
            ray_take += ["S%RMID_STORE=0.0_LDP","S%RMID_STORE(1:ND_STORE-1)=RMID_STORE(1:ND_STORE-1)"]
        else:ray_take.append("S%"+name+"="+name)
    ray_take += ["END SUBROUTINE"]
    source="""MODULE LUMINA_FORMAL_DIAG
USE SET_KIND_MODULE
USE MOD_SPACE_GRID_V2
USE EXT_REL_GRID_V4
USE MOD_RAY_MOM_STORE
USE LUMINA_MOM_DIAG,ONLY:LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW
IMPLICIT NONE
INTEGER(KIND=8),SAVE :: TOTAL_BYTES=0
LOGICAL,SAVE :: FORMAL_SPLIT_ACTIVE=.FALSE.
REAL(KIND=LDP),SAVE :: FORMAL_SPLIT_INCOMING(500)=0.0_LDP
__ARGUMENT_CODE__
LOGICAL FUNCTION FORMAL_SELECTED()
CHARACTER(LEN=24) MODE
CALL GET_ENVIRONMENT_VARIABLE('LUMINA_FORMAL_CAPTURE_MODE',MODE)
FORMAL_SELECTED=MODE=='CAPTURE_FORMAL'.AND.LUMINA_OUTER==60.AND.LUMINA_FREQ_INDEX>=23731.AND.LUMINA_FREQ_INDEX<=23732
END FUNCTION
__RAY_TAKE__
SUBROUTINE FORMAL_STATE_IO(PATH,A,S,READING)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_FORMAL_ARGS) A
TYPE(LUMINA_RAY_STATE) S
LOGICAL READING
INTEGER LU,META(5),KINDS(4),IOS
INTEGER(KIND=8) POSITION_NOW,FILE_BYTES
CHARACTER(LEN=16) TAG
IF(READING)THEN
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='OLD',ACCESS='STREAM',FORM='UNFORMATTED')
 INQUIRE(UNIT=LU,SIZE=FILE_BYTES)
 IF(FILE_BYTES>67108864_8)STOP 86
 READ(LU) TAG,KINDS,META
 IF(TAG/='LUMINAFORMALV1'.OR.ANY(KINDS/=[LDP,64,32,32]))STOP 86
 LUMINA_OUTER=META(1);CALL LUMINA_MOM_SET_PHASE(META(2),META(3),META(4)/=0,META(5)/=0)
ELSE
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED')
 TAG='LUMINAFORMALV1';META=[LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,MERGE(1,0,LUMINA_FIRST),MERGE(1,0,LUMINA_NEW)]
 WRITE(LU) TAG,[LDP,STORAGE_SIZE(0.0_LDP),STORAGE_SIZE(0),STORAGE_SIZE(.TRUE.)],META
END IF
CALL FORMAL_ARGS_IO(LU,A,READING)
CALL LUMINA_RAY_IO(LU,S,READING)
IF(READING)THEN
 READ(LU) TAG
 IF(TAG/='LUMINAFORMALEND1')STOP 86
 INQUIRE(UNIT=LU,POS=POSITION_NOW)
 IF(POSITION_NOW/=FILE_BYTES+1_8)STOP 86
ELSE
 TAG='LUMINAFORMALEND1';WRITE(LU) TAG
END IF
CLOSE(LU)
IF(.NOT.READING)CALL FORMAL_BUDGET(PATH)
END SUBROUTINE
SUBROUTINE FORMAL_BUDGET(PATH)
CHARACTER(LEN=*) PATH
INTEGER(KIND=8) BYTES
INQUIRE(FILE=PATH,SIZE=BYTES)
TOTAL_BYTES=TOTAL_BYTES+BYTES
IF(BYTES>67108864_8.OR.TOTAL_BYTES>268435456_8)STOP 88
END SUBROUTINE
SUBROUTINE FORMAL_RAYS_IO(PATH,A,READING,IS_OUTPUT)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_FORMAL_ARGS) A
LOGICAL READING,IS_OUTPUT
INTEGER LU,IP,NZ,NP,ND,NC,IOS
INTEGER(KIND=8) ESTIMATE,POSITION_NOW,FILE_BYTES
CHARACTER(LEN=16) TAG
ESTIMATE=64_8
DO IP=1,A%NP
 ESTIMATE=ESTIMATE+16_8*RAY(IP)%NZ+4_8
END DO
IF(ESTIMATE>67108864_8.OR.TOTAL_BYTES+ESTIMATE>268435456_8)STOP 88
IF(READING)THEN
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='OLD',ACCESS='STREAM',FORM='UNFORMATTED')
 INQUIRE(UNIT=LU,SIZE=FILE_BYTES)
 IF(FILE_BYTES>67108864_8)STOP 86
 READ(LU) TAG,NC,NP,ND
 IF(TAG/='LUMINARAYS1'.OR.NP/=A%NP.OR.ND/=A%ND.OR.NC/=A%NC)STOP 86
ELSE
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED')
 TAG='LUMINARAYS1';WRITE(LU) TAG,A%NC,A%NP,A%ND
END IF
DO IP=1,A%NP
 IF(READING)THEN
  READ(LU) NZ
  IF(NZ/=RAY(IP)%NZ)STOP 86
  READ(LU) RAY(IP)%I_P_PREV,RAY(IP)%I_M_PREV
  RAY(IP)%I_P_SAVE=RAY(IP)%I_P_PREV;RAY(IP)%I_M_SAVE=RAY(IP)%I_M_PREV
 ELSE
  WRITE(LU) RAY(IP)%NZ
  IF(IS_OUTPUT)THEN
   WRITE(LU) RAY(IP)%I_P,RAY(IP)%I_M
  ELSE IF(A%NEW_FREQ)THEN
   WRITE(LU) RAY(IP)%I_P_SAVE,RAY(IP)%I_M_SAVE
  ELSE
   WRITE(LU) RAY(IP)%I_P_PREV,RAY(IP)%I_M_PREV
  END IF
 END IF
END DO
IF(READING)THEN
 READ(LU) TAG
 IF(TAG/='LUMINARAYSEND1')STOP 86
 INQUIRE(UNIT=LU,POS=POSITION_NOW)
 IF(POSITION_NOW/=FILE_BYTES+1_8)STOP 86
ELSE
 TAG='LUMINARAYSEND1';WRITE(LU) TAG
END IF
CLOSE(LU)
IF(.NOT.READING)CALL FORMAL_BUDGET(PATH)
END SUBROUTINE
SUBROUTINE FORMAL_GEOMETRY(PATH,A)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_FORMAL_ARGS) A
INTEGER LU,IP,ID,NZ,J
INTEGER(KIND=8) ESTIMATE
CHARACTER(LEN=16) TAG
ESTIMATE=128_8+32_8*ND_EXT+68_8*A%ND*A%NP
DO IP=1,A%NP
 ESTIMATE=ESTIMATE+56_8*RAY(IP)%NZ+8_8
END DO
IF(ESTIMATE>67108864_8.OR.TOTAL_BYTES+ESTIMATE>268435456_8)STOP 88
OPEN(NEWUNIT=LU,FILE=PATH,STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED')
TAG='LUMINAGEOMETRY1'
WRITE(LU) TAG,A%NC,A%NP,A%ND,ND_EXT,ND_ADD,IDMIN,IDMAX,RAY_POINTS_INSERTED
WRITE(LU) R_EXT,V_EXT,VDOP_VEC_EXT,SIGMA_EXT
DO IP=1,A%NP
 NZ=RAY(IP)%NZ
 WRITE(LU) NZ,RAY(IP)%R_RAY(1:NZ),RAY(IP)%S_P(1:NZ),RAY(IP)%S_M(1:NZ),RAY(IP)%MU_P(1:NZ),RAY(IP)%MU_M(1:NZ),RAY(IP)%B_P(1:NZ),RAY(IP)%B_M(1:NZ),RAY(IP)%LNK(1:A%ND)
END DO
DO ID=1,A%ND
 J=A%NP+1-ID
 WRITE(LU) JQW_P(ID,1:J),HQW_P(ID,1:J),KQW_P(ID,1:J),NQW_P(ID,1:J),JQW_M(ID,1:J),HQW_M(ID,1:J),KQW_M(ID,1:J),NQW_M(ID,1:J)
END DO
TAG='LUMINAGEOMEND1';WRITE(LU) TAG
CLOSE(LU)
CALL FORMAL_BUDGET(PATH)
END SUBROUTINE
END MODULE
SUBROUTINE LUMINA_FORMAL_STOP_BEFORE_POP()
CHARACTER(LEN=24) MODE
CALL GET_ENVIRONMENT_VARIABLE('LUMINA_FORMAL_CAPTURE_MODE',MODE)
IF(MODE=='CAPTURE_FORMAL')STOP 94
END SUBROUTINE
"""
    return source.replace("__ARGUMENT_CODE__",argument_code()).replace("__RAY_TAKE__","\n".join(ray_take))


def instrument_formal(text):
    head=text.index("\tSUBROUTINE CMF_FORMAL_REL_V4")
    module,sub=text[:head],text[head:]
    sub=once(sub,"\tUSE MOD_RAY_MOM_STORE\n","\tUSE MOD_RAY_MOM_STORE\n\tUSE LUMINA_FORMAL_DIAG\n")
    sub=once(sub,"\tLOGICAL REALLOCATE_GRID\n","\tLOGICAL REALLOCATE_GRID\n\tLOGICAL LUM_CAPTURE\n\tTYPE(LUMINA_FORMAL_ARGS) LUM_A\n\tTYPE(LUMINA_RAY_STATE) LUM_S\n\tCHARACTER(LEN=80) LUM_PREFIX\n")
    names=",".join(x[0] for x in ARGS)
    begin="""\tLUM_CAPTURE=FORMAL_SELECTED()
\tIF(LUM_CAPTURE)THEN
\t WRITE(LUM_PREFIX,'(A,I5.5,A,I2.2)')'FORMAL_',LUMINA_FREQ_INDEX,'_',LUMINA_JIT
\t CALL FORMAL_ARGS_TAKE(LUM_A,.FALSE.,__ARGS__)
\t CALL FORMAL_RAY_TAKE(LUM_S)
\t CALL FORMAL_STATE_IO(TRIM(LUM_PREFIX)//'_in.bin',LUM_A,LUM_S,.FALSE.)
\t CALL FORMAL_RAYS_IO(TRIM(LUM_PREFIX)//'_rays_in.bin',LUM_A,.FALSE.,.FALSE.)
\t CALL FORMAL_GEOMETRY(TRIM(LUM_PREFIX)//'_geometry.bin',LUM_A)
\tEND IF
""".replace("__ARGS__",names)
    start="\tIF(INITIALIZE)THEN\n\t  NU_ON_dNU=0.0_LDP"
    sub=once(sub,start,begin+start)
    tail="""\tIF(LUM_CAPTURE)THEN
\t CALL FORMAL_ARGS_TAKE(LUM_A,.TRUE.,__ARGS__)
\t CALL FORMAL_RAY_TAKE(LUM_S)
\t CALL FORMAL_STATE_IO(TRIM(LUM_PREFIX)//'_out.bin',LUM_A,LUM_S,.FALSE.)
\t CALL FORMAL_RAYS_IO(TRIM(LUM_PREFIX)//'_rays_out.bin',LUM_A,.FALSE.,.TRUE.)
\t IF(LUMINA_FREQ_INDEX==23732)THEN
\t  WRITE(6,'(A,3I8)')'LUMINA_FORMAL_CAPTURED ',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT
\t  STOP 96
\t END IF
\tEND IF
""".replace("__ARGS__",names)
    sub=once(sub,"\tFIRST_TIME=.FALSE.\n\tRETURN", "\tFIRST_TIME=.FALSE.\n"+tail+"\tRETURN")
    return module,sub


def replay_source(split_replay=False):
    actual=",".join("A%"+x[0] for x in ARGS)
    warm=",".join(".TRUE." if x[0] in ("INITIALIZE","NEW_FREQ") else "B%"+x[0] for x in ARGS)
    return """PROGRAM FORMAL_REPLAY
USE LUMINA_FORMAL_DIAG
IMPLICIT NONE
TYPE(LUMINA_FORMAL_ARGS) A,B
TYPE(LUMINA_RAY_STATE) S
CHARACTER(LEN=4096) PREFIX
CHARACTER(LEN=32) SWITCH
INTEGER IP
CALL GET_COMMAND_ARGUMENT(1,PREFIX)
CALL GET_COMMAND_ARGUMENT(2,SWITCH)
IF(PREFIX==' ')STOP 85
IF(SWITCH/=' '.AND.SWITCH/='--split')STOP 87
IF(SWITCH=='--split'.AND..NOT.__SPLIT_SUPPORTED__)STOP 87
CALL FORMAL_STATE_IO(TRIM(PREFIX)//'_in.bin',A,S,.TRUE.)
B=A
CALL CMF_FORMAL_REL_V4(__WARM__)
CALL FORMAL_GEOMETRY('replay_geometry.bin',A)
IF(SWITCH=='--split')THEN
 CALL FORMAL_RAYS_IO(TRIM(PREFIX)//'_rays_out.bin',A,.TRUE.,.FALSE.)
 DO IP=1,A%NP
  FORMAL_SPLIT_INCOMING(IP)=RAY(IP)%I_M_PREV(RAY(IP)%LNK(1))
 END DO
 FORMAL_SPLIT_ACTIVE=.TRUE.
END IF
CALL FORMAL_RAYS_IO(TRIM(PREFIX)//'_rays_in.bin',A,.TRUE.,.FALSE.)
CALL LUMINA_RAY_RESTORE(S)
CALL CMF_FORMAL_REL_V4(__ACTUAL__)
CALL FORMAL_RAY_TAKE(S)
CALL FORMAL_STATE_IO('replay_out.bin',A,S,.FALSE.)
CALL FORMAL_RAYS_IO('replay_rays_out.bin',A,.FALSE.,.TRUE.)
WRITE(6,'(A)')'LUMINA_FORMAL_REPLAY_COMPLETE'
END PROGRAM
""".replace("__WARM__",warm).replace("__ACTUAL__",actual).replace("__SPLIT_SUPPORTED__",".TRUE." if split_replay else ".FALSE.")


def split_solver(source):
    """모델 표면의 실제 입사 강도에서 안쪽 적분을 재시작하는 동등성 시험 전용."""
    source=once(source,"      USE MOD_SPACE_GRID_V2\n",
                "      USE MOD_SPACE_GRID_V2\n      USE LUMINA_FORMAL_DIAG,ONLY:FORMAL_SPLIT_ACTIVE,FORMAL_SPLIT_INCOMING\n")
    source=once(source,"      integer :: iz\n","      integer :: iz\n      integer :: lum_inward_start\n")
    source=once(source,"      RAY(IP)%I_m(1)=0.0_LDP",
        "      lum_inward_start=1\n      if(formal_split_active)lum_inward_start=ray(ip)%lnk(1)\n"
        "      RAY(IP)%I_m(lum_inward_start)=0.0_LDP\n"
        "      if(formal_split_active)RAY(IP)%I_m(lum_inward_start)=formal_split_incoming(ip)")
    source=once(source,"      do iz=2,ray(ip)%nz-1","      do iz=lum_inward_start+1,ray(ip)%nz-1")
    source=once(source,"      t1=dtau_loc(nzz-1)\n","      if(lum_inward_start<nzz)then\n      t1=dtau_loc(nzz-1)\n")
    source=once(source,"     *     alpha*source_prime(nzz-1)+beta*source_prime(nzz)\n",
                "     *     alpha*source_prime(nzz-1)+beta*source_prime(nzz)\n      end if\n")
    return source


def prepare(parent,mom_build,native,out,split_replay=False):
    if out.exists():raise ValueError("기존 formal build 보존")
    record=json.loads((mom_build/"capture-build.json").read_text())
    for name,expected in record["output_sha256"].items():
        if sha(Path(name))!=expected:raise ValueError("기존 MOM build 신원 변경")
    if sha(parent/"cmfgen_parent_rebuilt.exe")!=record["rebuilt_unmodified_sha256"]:raise ValueError("부모 불일치")
    out.mkdir()
    baseline=rebuild_unmodified_sources(parent,native,out).parent
    source=(native/"plane/cmf_formal_rel_v4.f").read_text()
    (baseline/"cmf_formal_rel_v4.f").write_text(source)
    command(["gfortran",*FLAGS,"-I.","-I"+str(native/"mod"),"-c","cmf_formal_rel_v4.f"],baseline)
    command(["ar","r","libsubs.a","cmf_formal_rel_v4.o"],baseline)
    if split_replay:
        (baseline/"solve_cmf_formal_v2.f").write_text((native/"plane/solve_cmf_formal_v2.f").read_text())
        command(["gfortran",*FLAGS,"-I.","-I"+str(native/"mod"),"-c","solve_cmf_formal_v2.f"],baseline)
        command(["ar","r","libsubs.a","solve_cmf_formal_v2.o"],baseline)
    libs=["-L.","-L"+str(native/"lib"),"-lsubs","-lpgplt","-lmod",native/"lib/libmy_lapack.a",native/"lib/libmy_blas.a"]
    command(["gfortran","-fopenmp","-m64",native/"new_main/cmfgen.o","cmfgen_sub.o",*libs,"-o","cmfgen_formal_unmodified.exe"],baseline)
    if sha(baseline/"cmfgen_formal_unmodified.exe")!=record["rebuilt_unmodified_sha256"]:
        raise ValueError("formal source를 재빌드한 원래 실행파일이 부모와 다름")
    ext,sub=instrument_formal(source)
    caller=(mom_build/"comp_j_blank.f").read_text()
    caller=once(caller,"\t       CALL CMF_FORMAL_REL_V4\n",
                "\t       CALL LUMINA_MOM_SET_PHASE(FREQ_INDX,J_IT_COUNTER,FIRST_FREQ,NEW_FREQ)\n\t       CALL CMF_FORMAL_REL_V4\n")
    main=once((mom_build/"cmfgen_sub.f").read_text(),"\tCALL LUMINA_MOM_STOP_BEFORE_POP()\n",
              "\tCALL LUMINA_MOM_STOP_BEFORE_POP()\n\tCALL LUMINA_FORMAL_STOP_BEFORE_POP()\n")
    files={"ext_rel_grid_v4.f":ext,"cmf_formal_rel_v4.f":sub,"lumina_formal_diag.f90":module_source(),
           "comp_j_blank.f":caller,"cmfgen_sub.f":main,"formal_replay.f90":replay_source(split_replay)}
    if split_replay:files["solve_cmf_formal_v2.f"]=split_solver((native/"plane/solve_cmf_formal_v2.f").read_text())
    for name,text in files.items():(out/name).write_text(text)
    for name in ("ext_rel_grid_v4.f","lumina_formal_diag.f90","cmf_formal_rel_v4.f","comp_j_blank.f","cmfgen_sub.f","formal_replay.f90"):
        flags=FLAGS if name.endswith(".f") else ["-fopenmp","-m64","-O","-ffree-line-length-none","-fbacktrace"]
        command(["gfortran",*flags,"-I.","-I"+str(mom_build),"-I"+str(native/"mod"),"-c",name],out)
    if split_replay:
        command(["gfortran",*FLAGS,"-I.","-I"+str(mom_build),"-I"+str(native/"mod"),"-c","solve_cmf_formal_v2.f"],out)
    shutil.copy2(mom_build/"libsubs.a",out/"libsubs.a")
    command(["ar","r","libsubs.a","cmf_formal_rel_v4.o","comp_j_blank.o"],out)
    if split_replay:command(["ar","r","libsubs.a","solve_cmf_formal_v2.o"],out)
    common=["ext_rel_grid_v4.o","lumina_formal_diag.o",mom_build/"mod_jrel_v9.o",mom_build/"lumina_mom_diag.o"]
    command(["gfortran","-fopenmp","-m64",native/"new_main/cmfgen.o","cmfgen_sub.o",*common,*libs,"-o","cmfgen_formal_capture.exe"],out)
    command(["gfortran","-fopenmp","-m64","formal_replay.o",*common,*libs,"-o","formal_replay.exe"],out)
    outputs=[out/name for name in files]+[out/"cmfgen_formal_capture.exe",out/"formal_replay.exe",out/"libsubs.a"]
    inputs=[parent/"parent-build-identity.json",mom_build/"capture-build.json",native/"plane/cmf_formal_rel_v4.f",
            native/"plane/mod_space_grid_v2.f",native/"plane/define_grid_v2.f",native/"plane/solve_cmf_formal_v2.f"]
    result=dict(scope="두 실제 주파수의 formal 광선 입력/출력; 원래 산술 불변; 수렴 아님",
        parent=str(parent),mom_build=str(mom_build),native=str(native),
        rebuilt_unmodified_sha256=record["rebuilt_unmodified_sha256"],generator_sha256=sha(Path(__file__)),
        output_sha256={str(p):sha(p) for p in outputs},input_sha256={str(p):sha(p) for p in inputs},
        args=ARGS,ray_inventory=json.loads((mom_build/"state_inventory.json").read_text())["ray"],split_replay=split_replay,
        final_goal="NOT_JUDGED")
    with (out/"formal-build.json").open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps(dict(build=str(out),rebuilt_unmodified_sha256=result["rebuilt_unmodified_sha256"])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","mom-build","native","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--split-replay",action="store_true")
    a=p.parse_args();prepare(a.parent,a.mom_build,a.native,a.out,a.split_replay)
