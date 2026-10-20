#!/usr/bin/env python3
"""동일 부모 빌드의 사본에 MOM 한 호출의 상태 포획/replay만 추가한다.

원래 수송 산술/경계/보정식은 바꾸지 않는다. native 원본은 읽기만 하고,
새 출력 디렉터리에서 모듈 상태 직렬화와 선택 주파수 trace를 생성한다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


FLAGS = ["-fopenmp", "-m64", "-O", "-fallow-argument-mismatch", "-fallow-invalid-boz",
         "-std=legacy", "-fno-range-check", "-ffixed-line-length-0", "-fno-backslash",
         "-fd-lines-as-comments", "-fbacktrace"]
ARGUMENTS = [
    ("ETA_SM", "R", True), ("CHI_SM", "R", True), ("ESEC_SM", "R", True),
    ("V_SM", "R", True), ("SIGMA_SM", "R", True), ("R_SM", "R", True),
    ("JNU_SM", "R", True), ("RSQHNU_SM", "R", True),
    ("HFLUX_AT_IB", "R", False), ("HFLUX_AT_OB", "R", False),
    ("VDOP_VEC", "R", True), ("VDOP_FRAC", "R", False), ("FREQ", "R", False),
    ("dLOG_NU", "R", False), ("DBB", "R", False),
    ("XM_CHK_OPTION", "C", False), ("J_CHK_OPTION", "C", False), ("H_CHK_OPTION", "C", False),
    ("IB_STAB_FACTOR", "R", False), ("INNER_BND_METH", "C", False),
    ("OUTER_BND_METH", "C", False), ("METHOD", "C", False), ("COHERENT", "L", False),
    ("N_TYPE", "C", False), ("INCL_ADVEC_TERMS", "L", False),
    ("INCL_REL_TERMS", "L", False), ("INIT", "L", False), ("ND_SM", "I", False)]
FTYPE = {"R": "REAL(KIND=LDP)", "I": "INTEGER", "L": "LOGICAL"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("native 포획 삽입점 불일치: "+old[:80])
    return text.replace(old, new, 1)


def fields_of(module):
    """현재 두 작은 native 모듈의 선언을 전부 읽고 미지원 선언은 거부한다."""
    fields = []
    for raw in module.splitlines():
        if not raw.strip() or raw.lstrip().startswith("!"):
            continue
        line = raw.split("!", 1)[0].strip()
        if re.match(r"(?i)(MODULE |END MODULE|USE |IMPLICIT |COMMON )", line):
            continue
        match = re.fullmatch(r"(REAL\(KIND=LDP\)|INTEGER|LOGICAL)(.*)", line, re.I)
        if not match:
            raise ValueError("미지원 module 선언: "+line)
        kind, body = match.groups()
        if "PARAMETER" in body.upper():
            continue
        attributes, names = body.split("::", 1) if "::" in body else ("", body)
        alloc = "ALLOCATABLE" in attributes.upper()
        if attributes.strip() and attributes.strip().upper() not in (", ALLOCATABLE", ", SAVE"):
            raise ValueError("미지원 module 속성: "+attributes)
        for item in names.split(","):
            item = item.split("=", 1)[0].strip()
            part = re.fullmatch(r"(\w+)(\(:\))?", item)
            if not part or bool(part[2]) != alloc:
                raise ValueError("미지원 module 변수: "+item)
            name = part[1]
            fixed = "N_ERR_MAX" if name.upper() == "MOM_ERR_ON_FREQ" else None
            fields.append(dict(name=name, type=kind.upper(), alloc=alloc, fixed=fixed))
    if len({f["name"].upper() for f in fields}) != len(fields):
        raise ValueError("module 변수 중복")
    return fields


def io_lines(fields, target="S"):
    result = []
    for f in fields:
        item = target+"%"+f["name"]
        if f["alloc"]:
            result += ["IF(READING)THEN", " READ(LU) PRESENT_ARRAY", "ELSE",
                       f" PRESENT_ARRAY=ALLOCATED({item})", " WRITE(LU) PRESENT_ARRAY", "END IF",
                       "IF(PRESENT_ARRAY)THEN", " IF(READING)THEN", "  READ(LU) BOUNDS",
                       "  IF(BOUNDS(2)-BOUNDS(1)+1 < 0 .OR. BOUNDS(2)-BOUNDS(1)+1 > 2000000)STOP 86",
                       f"  IF(ALLOCATED({item}))DEALLOCATE({item})",
                       f"  ALLOCATE({item}(BOUNDS(1):BOUNDS(2)))", f"  READ(LU) {item}", " ELSE",
                       f"  BOUNDS=[LBOUND({item},1),UBOUND({item},1)]", "  WRITE(LU) BOUNDS",
                       f"  WRITE(LU) {item}", " END IF", "ELSE IF(READING)THEN",
                       f" IF(ALLOCATED({item}))DEALLOCATE({item})", "END IF"]
        else:
            result += ["IF(READING)THEN", f" READ(LU) {item}", "ELSE", f" WRITE(LU) {item}", "END IF"]
    return result


def instrument_module(text, module_name, prefix):
    end = re.search(r"(?im)^\s*END MODULE\s+"+module_name+r"\s*$", text)
    if not end:
        raise ValueError("module 끝 없음")
    fields = fields_of(text[:end.end()])
    declaration = [f"TYPE {prefix}_STATE"]
    for f in fields:
        suffix = "(:)" if f["alloc"] else "("+f["fixed"]+")" if f["fixed"] else ""
        declaration.append(f["type"]+(", ALLOCATABLE" if f["alloc"] else "")+" :: "+f["name"]+suffix)
    declaration += ["END TYPE", "CONTAINS", f"SUBROUTINE {prefix}_TAKE(S)",
                    f"TYPE({prefix}_STATE),INTENT(OUT)::S"]
    for f in fields:
        name = f["name"]
        if f["alloc"]:
            declaration += [f"IF(ALLOCATED({name}))THEN",
                            f" ALLOCATE(S%{name}(LBOUND({name},1):UBOUND({name},1)))",
                            f" S%{name}={name}", "END IF"]
        else:
            declaration.append(f"S%{name}={name}")
    declaration += ["END SUBROUTINE", f"SUBROUTINE {prefix}_RESTORE(S)",
                    f"TYPE({prefix}_STATE),INTENT(IN)::S"]
    for f in fields:
        name = f["name"]
        if f["alloc"]:
            declaration += [f"IF(ALLOCATED({name}))DEALLOCATE({name})", f"IF(ALLOCATED(S%{name}))THEN",
                            f" ALLOCATE({name}(LBOUND(S%{name},1):UBOUND(S%{name},1)))",
                            f" {name}=S%{name}", "END IF"]
        else:
            declaration.append(f"{name}=S%{name}")
    declaration += ["END SUBROUTINE", f"SUBROUTINE {prefix}_IO(LU,S,READING)",
                    "INTEGER,INTENT(IN)::LU", "LOGICAL,INTENT(IN)::READING",
                    f"TYPE({prefix}_STATE),INTENT(INOUT)::S", "LOGICAL PRESENT_ARRAY", "INTEGER BOUNDS(2)"]
    declaration += io_lines(fields)+["END SUBROUTINE",f"INTEGER(KIND=8) FUNCTION {prefix}_BYTES()",f"{prefix}_BYTES=0_8"]
    for f in fields:
        name=f["name"]
        count=f"INT(SIZE({name}),8)" if f["alloc"] or f["fixed"] else "1_8"
        add=f"{prefix}_BYTES={prefix}_BYTES+{count}*INT(STORAGE_SIZE({name})/8,8)"
        declaration += ([f"IF(ALLOCATED({name})){add}"] if f["alloc"] else [add])
    declaration += ["END FUNCTION"]
    declaration += [f"SUBROUTINE {prefix}_EMPTY(S)",f"TYPE({prefix}_STATE),INTENT(OUT)::S"]
    for f in fields:
        if not f["alloc"]:
            zero=".FALSE." if f["type"]=="LOGICAL" else "0.0_LDP" if f["type"].startswith("REAL") else "0"
            declaration.append("S%"+f["name"]+"="+zero)
    declaration += ["END SUBROUTINE"]
    if module_name=="MOD_RAY_MOM_STORE":
        raw_names={"ND_STORE","R_STORE","RMID_STORE","EXT_RMID_STORE","JNU_STORE","HNU_STORE","KNU_STORE","NNU_STORE","GAM_REL_STORE","HBC","NBC","HN_DEF_ON_NODES"}
        raw_names.update(f["name"] for f in fields if "PLUS_" in f["name"] or "MIN_" in f["name"] or "NU_AT_" in f["name"])
        declaration += ["SUBROUTINE LUMINA_RAY_RAW_TAKE(S)","TYPE(LUMINA_RAY_STATE),INTENT(OUT)::S",
                        "CALL LUMINA_RAY_EMPTY(S)"]
        for f in fields:
            name=f["name"]
            if name not in raw_names:continue
            if f["alloc"]:
                declaration += [f"IF(ALLOCATED({name}))THEN",f" ALLOCATE(S%{name}(LBOUND({name},1):UBOUND({name},1)))"]
                if name=="RMID_STORE":
                    declaration += [" S%RMID_STORE=0.0_LDP"," S%RMID_STORE(1:ND_STORE-1)=RMID_STORE(1:ND_STORE-1)"]
                else:declaration.append(f" S%{name}={name}")
                declaration.append("END IF")
            else:declaration.append(f"S%{name}={name}")
        declaration.append("END SUBROUTINE")
    return text[:end.start()]+"\n"+"\n".join("\t"+s for s in declaration)+"\n"+text[end.start():], fields


def arg_type():
    lines = ["TYPE LUMINA_ARG_STATE"]
    for name, kind, array in ARGUMENTS:
        decl = "CHARACTER(LEN=:),ALLOCATABLE" if kind == "C" else FTYPE[kind]+(",ALLOCATABLE" if array else "")
        lines.append(decl+" :: "+name+("(:)" if array else ""))
    return "\n".join(lines+["END TYPE"])


def diagnostic_module(first_repair=False):
    names = ",".join(name for name,_,_ in ARGUMENTS)
    copy = [f"SUBROUTINE LUMINA_TAKE_ARGS(S,{names})", "TYPE(LUMINA_ARG_STATE),INTENT(OUT)::S",
            "INTEGER ND_SM"]
    for name, kind, array in ARGUMENTS:
        if name != "ND_SM":
            copy.append(("CHARACTER(LEN=*)" if kind == "C" else FTYPE[kind])+" :: "+name+("(ND_SM)" if array else ""))
    copy += [f"S%{name}={name}" for name,_,_ in ARGUMENTS]+["END SUBROUTINE"]
    io = ["SUBROUTINE LUMINA_ARGS_IO(LU,S,READING)", "INTEGER LU,LENGTH", "LOGICAL READING",
          "TYPE(LUMINA_ARG_STATE) S", "IF(READING)THEN", " READ(LU) S%ND_SM",
          " IF(S%ND_SM<2 .OR. S%ND_SM>2000000)STOP 86", "ELSE", " WRITE(LU) S%ND_SM", "END IF"]
    for name, kind, array in ARGUMENTS:
        if name == "ND_SM":
            continue
        item = "S%"+name
        io += ["IF(READING)THEN"]
        if kind == "C":
            io += [" READ(LU) LENGTH", " IF(LENGTH<1 .OR. LENGTH>200)STOP 86",
                   f" IF(ALLOCATED({item}))DEALLOCATE({item})", f" ALLOCATE(CHARACTER(LEN=LENGTH)::{item})"]
        elif array:
            io += [f" IF(ALLOCATED({item}))DEALLOCATE({item})", f" ALLOCATE({item}(S%ND_SM))"]
        io += [f" READ(LU) {item}", "ELSE"]
        if kind == "C":
            io.append(f" WRITE(LU) LEN({item})")
        io += [f" WRITE(LU) {item}", "END IF"]
    io += ["END SUBROUTINE"]
    source = """MODULE LUMINA_MOM_DIAG
USE SET_KIND_MODULE
USE MOD_JREL_V9, ONLY: LUMINA_JREL_STATE,LUMINA_JREL_IO
USE MOD_RAY_MOM_STORE, ONLY: LUMINA_RAY_STATE,LUMINA_RAY_IO
IMPLICIT NONE
INTEGER :: LUMINA_OUTER=-1,LUMINA_FREQ_INDEX=-1,LUMINA_JIT=-1
LOGICAL :: LUMINA_FIRST=.FALSE.,LUMINA_NEW=.FALSE.
__ARG_TYPE__
CONTAINS
LOGICAL FUNCTION LUMINA_SELECTED(FREQ)
REAL(KIND=LDP) FREQ
CHARACTER(LEN=16) MODE
CALL GET_ENVIRONMENT_VARIABLE('LUMINA_MOM_CAPTURE_MODE',MODE)
IF(MODE/=' ' .AND. MODE/='CAPTURE' .AND. MODE/='REPLAY')STOP 87
LUMINA_SELECTED=(MODE=='CAPTURE' .OR. MODE=='REPLAY') .AND. ABS(FREQ-6.40196372_LDP)<5.0E-9_LDP
END FUNCTION
SUBROUTINE LUMINA_TRACE_BUDGET(LU,N)
INTEGER LU,N
INTEGER(KIND=8) POSITION_NOW
INQUIRE(UNIT=LU,POS=POSITION_NOW)
IF(POSITION_NOW+64_8*N+1024_8>33554432_8)STOP 88
END SUBROUTINE
__ARGS_COPY__
__ARGS_IO__
SUBROUTINE LUMINA_SNAPSHOT_IO(PATH,A,J,R,READING)
CHARACTER(LEN=*) PATH
TYPE(LUMINA_ARG_STATE) A
TYPE(LUMINA_JREL_STATE) J
TYPE(LUMINA_RAY_STATE) R
LOGICAL READING
CHARACTER(LEN=16) HEADER
INTEGER LU,INFO(4),IOS
INTEGER(KIND=8) FILE_BYTES,POSITION_NOW
IF(READING)THEN
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='OLD',FORM='UNFORMATTED',ACCESS='STREAM',ACTION='READ')
 INQUIRE(UNIT=LU,SIZE=FILE_BYTES)
 IF(FILE_BYTES>268435456_8)STOP 86
 READ(LU) HEADER,INFO
 IF(HEADER/='LUMINAMOMCAPV1' .OR. ANY(INFO/=[LDP,STORAGE_SIZE(0.0_LDP),STORAGE_SIZE(0),STORAGE_SIZE(.TRUE.)]))STOP 86
 READ(LU) LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW
ELSE
 OPEN(NEWUNIT=LU,FILE=PATH,STATUS='NEW',FORM='UNFORMATTED',ACCESS='STREAM',ACTION='WRITE')
 HEADER='LUMINAMOMCAPV1'
 WRITE(LU) HEADER,[LDP,STORAGE_SIZE(0.0_LDP),STORAGE_SIZE(0),STORAGE_SIZE(.TRUE.)]
 WRITE(LU) LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW
END IF
CALL LUMINA_ARGS_IO(LU,A,READING)
CALL LUMINA_JREL_IO(LU,J,READING)
CALL LUMINA_RAY_IO(LU,R,READING)
IF(READING)THEN
 READ(LU) HEADER
 IF(HEADER/='LUMINAMOMENDV1')STOP 86
 INQUIRE(UNIT=LU,POS=POSITION_NOW)
 IF(POSITION_NOW/=FILE_BYTES+1_8)STOP 86
ELSE
 HEADER='LUMINAMOMENDV1'
 WRITE(LU) HEADER
 INQUIRE(UNIT=LU,SIZE=FILE_BYTES)
 IF(FILE_BYTES>234881024_8)STOP 88
END IF
CLOSE(LU)
END SUBROUTINE
END MODULE
SUBROUTINE LUMINA_MOM_SET_PHASE(INDEX,JIT,FIRST,NEW)
USE LUMINA_MOM_DIAG
INTEGER INDEX,JIT
LOGICAL FIRST,NEW
LUMINA_FREQ_INDEX=INDEX;LUMINA_JIT=JIT;LUMINA_FIRST=FIRST;LUMINA_NEW=NEW
END SUBROUTINE
SUBROUTINE LUMINA_MOM_SET_ITERATION(ITERATION)
USE LUMINA_MOM_DIAG
INTEGER ITERATION
LUMINA_OUTER=ITERATION
END SUBROUTINE
SUBROUTINE LUMINA_MOM_STOP_BEFORE_POP()
USE LUMINA_MOM_DIAG
CHARACTER(LEN=16) MODE
CALL GET_ENVIRONMENT_VARIABLE('LUMINA_MOM_CAPTURE_MODE',MODE)
IF(MODE=='CAPTURE')THEN
 WRITE(6,'(A,I8)')'LUMINA_NO_CAPTURE_BEFORE_POP ',LUMINA_OUTER
 STOP 94
END IF
END SUBROUTINE
"""
    if first_repair:
        source=source.replace("IF(MODE/=' ' .AND. MODE/='CAPTURE' .AND. MODE/='REPLAY')STOP 87",
            "IF(MODE/=' ' .AND. MODE/='CAPTURE_FIRST' .AND. MODE/='REPLAY_FIRST')STOP 87")
        source=source.replace("LUMINA_SELECTED=(MODE=='CAPTURE' .OR. MODE=='REPLAY') .AND. ABS(FREQ-6.40196372_LDP)<5.0E-9_LDP",
            "LUMINA_SELECTED=(MODE=='CAPTURE_FIRST' .OR. MODE=='REPLAY_FIRST')")
        source=source.replace("IF(MODE=='CAPTURE')THEN","IF(MODE=='CAPTURE_FIRST')THEN")
        source=source.replace("IF(FILE_BYTES>234881024_8)STOP 88","IF(FILE_BYTES>67108864_8)STOP 88")
        source=once(source,"CONTAINS\nLOGICAL FUNCTION LUMINA_SELECTED", """TYPE(LUMINA_ARG_STATE),SAVE :: LUM_LAST_A,LUM_PREV_A
TYPE(LUMINA_JREL_STATE),SAVE :: LUM_LAST_J,LUM_PREV_J
TYPE(LUMINA_RAY_STATE),SAVE :: LUM_LAST_R,LUM_PREV_R
INTEGER,SAVE :: LUM_LAST_META(5)=0,LUM_PREV_META(5)=0
REAL(KIND=LDP),SAVE :: LUM_LAST_FREQ=0.0_LDP
LOGICAL,SAVE :: LUM_LAST_VALID=.FALSE.,LUM_PREV_VALID=.FALSE.
CONTAINS
LOGICAL FUNCTION LUMINA_SELECTED""")
        source=once(source,"END MODULE\n",first_event_procedure()+"\nEND MODULE\n")
    return source.replace("__ARG_TYPE__",arg_type()).replace("__ARGS_COPY__","\n".join(copy)).replace("__ARGS_IO__","\n".join(io))


def first_event_procedure():
    return """SUBROUTINE LUMINA_FIRST_BEGIN(FREQ)
REAL(KIND=LDP) FREQ
IF(LUM_LAST_VALID .AND. FREQ/=LUM_LAST_FREQ)THEN
 LUM_PREV_A=LUM_LAST_A;LUM_PREV_J=LUM_LAST_J;LUM_PREV_R=LUM_LAST_R
 LUM_PREV_META=LUM_LAST_META;LUM_PREV_VALID=.TRUE.
END IF
END SUBROUTINE
SUBROUTINE LUMINA_FIRST_COMPLETE(A,J,R)
TYPE(LUMINA_ARG_STATE) A
TYPE(LUMINA_JREL_STATE) J
TYPE(LUMINA_RAY_STATE) R
LUM_LAST_A=A;LUM_LAST_J=J;LUM_LAST_R=R
LUM_LAST_META=[LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,MERGE(1,0,LUMINA_FIRST),MERGE(1,0,LUMINA_NEW)]
LUM_LAST_FREQ=A%FREQ;LUM_LAST_VALID=.TRUE.
END SUBROUTINE
SUBROUTINE LUMINA_FIRST_EVENT(KIND_EVENT,ITERATION,BAD,N,FREQ,TRI,RAW,HRAW,SLOPE,INITIAL_PENDING,A,J,R,JK,RK)
INTEGER KIND_EVENT,ITERATION,BAD,N,LU,META(5)
LOGICAL INITIAL_PENDING
REAL(KIND=LDP) FREQ,TRI(4,N),RAW(N),HRAW(N),SLOPE(N)
TYPE(LUMINA_ARG_STATE) A
TYPE(LUMINA_JREL_STATE) J,JK
TYPE(LUMINA_RAY_STATE) R,RK
CHARACTER(LEN=16) HEADER
IF(56_8*N+128_8>33554432_8)STOP 88
CALL LUMINA_SNAPSHOT_IO('MOM_CAPTURE.bin',A,J,R,.FALSE.)
CALL LUMINA_SNAPSHOT_IO('MOM_KERNEL.bin',A,JK,RK,.FALSE.)
IF(LUM_PREV_VALID)THEN
 META=[LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,MERGE(1,0,LUMINA_FIRST),MERGE(1,0,LUMINA_NEW)]
 LUMINA_OUTER=LUM_PREV_META(1);LUMINA_FREQ_INDEX=LUM_PREV_META(2);LUMINA_JIT=LUM_PREV_META(3)
 LUMINA_FIRST=LUM_PREV_META(4)/=0;LUMINA_NEW=LUM_PREV_META(5)/=0
 CALL LUMINA_SNAPSHOT_IO('MOM_PREVIOUS.bin',LUM_PREV_A,LUM_PREV_J,LUM_PREV_R,.FALSE.)
 LUMINA_OUTER=META(1);LUMINA_FREQ_INDEX=META(2);LUMINA_JIT=META(3)
 LUMINA_FIRST=META(4)/=0;LUMINA_NEW=META(5)/=0
END IF
OPEN(NEWUNIT=LU,FILE='MOM_FAULT.bin',STATUS='NEW',FORM='UNFORMATTED',ACCESS='STREAM',ACTION='WRITE')
HEADER='LUMINAMOMFAULT1'
WRITE(LU) HEADER,[LDP,STORAGE_SIZE(0.0_LDP),STORAGE_SIZE(0),STORAGE_SIZE(.TRUE.)]
WRITE(LU) LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW
WRITE(LU) KIND_EVENT,ITERATION,BAD,N,MERGE(1,0,INITIAL_PENDING),FREQ,TRI,RAW,HRAW,SLOPE
HEADER='LUMINAMOMFEND1'
WRITE(LU) HEADER
CLOSE(LU)
WRITE(6,'(A,6I8,ES24.16)')'LUMINA_MOM_FIRST_REPAIR ',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,KIND_EVENT,ITERATION,BAD,FREQ
STOP 95
END SUBROUTINE
"""


def instrument_mom(source):
    source = once(source,"\tUSE MOD_RAY_MOM_STORE\n", "\tUSE MOD_RAY_MOM_STORE\n\tUSE LUMINA_MOM_DIAG\n")
    declarations = """
\tTYPE(LUMINA_JREL_STATE) LUM_J
\tTYPE(LUMINA_RAY_STATE) LUM_R
\tTYPE(LUMINA_ARG_STATE) LUM_A
\tLOGICAL LUM_SELECTED
\tINTEGER LUM_TRACE
"""
    start = "\tC_KMS=1.0E-05_LDP*SPEED_OF_LIGHT()"
    source = once(source,"\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n",
                  "\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n"+declarations)
    take = """\tLUM_SELECTED=LUMINA_SELECTED(FREQ)
\tIF(LUM_SELECTED)THEN
\t IF(LUMINA_JREL_BYTES()+LUMINA_RAY_BYTES()+72_8*ND_SM+32768_8>234881024_8)STOP 88
\t CALL LUMINA_JREL_TAKE(LUM_J)
\t CALL LUMINA_RAY_TAKE(LUM_R)
\t CALL LUMINA_TAKE_ARGS(LUM_A,__NAMES__)
\t OPEN(NEWUNIT=LUM_TRACE,FILE='MOM_TRACE.bin',FORM='UNFORMATTED',ACCESS='STREAM',STATUS='UNKNOWN',POSITION='APPEND')
\t WRITE(LUM_TRACE) 'CALL',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW,FREQ,FREQ_SAVE,ND_SM
\tEND IF
""".replace("__NAMES__",",".join(name for name,_,_ in ARGUMENTS))
    source = once(source,start,take+start)
    source = once(source,"\t  CALL THOMAS(TA,TB,TC,XM,ND,1)",
                  "\t  IF(LUM_SELECTED)CALL LUMINA_TRACE_BUDGET(LUM_TRACE,ND)\n\t  IF(LUM_SELECTED)WRITE(LUM_TRACE) 'TRI0',ICOUNT,ND,TA,TB,TC,XM\n\t  CALL THOMAS(TA,TB,TC,XM,ND,1)")
    source = once(source,"\t  RECORDED_ERROR=.FALSE.",
                  "\t  IF(LUM_SELECTED)WRITE(LUM_TRACE) 'RAW0',ICOUNT,ND,XM\n\t  RECORDED_ERROR=.FALSE.")
    source = once(source,"\t    IF(MAX_ER .LT. 1.0E-08_LDP)ACCURATE=.TRUE.",
                  "\t    IF(LUM_SELECTED)WRITE(LUM_TRACE) 'ERR0',ICOUNT,ND,MAX_ER,GAM_RSQJOLD,GAM_RSQJNU,dlnGRSQJdlnR\n\t    IF(MAX_ER .LT. 1.0E-08_LDP)ACCURATE=.TRUE.")
    failure = """\t      IF(LUM_SELECTED .AND. MAX_ER >= 1.0E-08_LDP)THEN
\t       CALL LUMINA_SNAPSHOT_IO('MOM_CAPTURE.bin',LUM_A,LUM_J,LUM_R,.FALSE.)
\t       CLOSE(LUM_TRACE)
\t       WRITE(6,'(A,3I8,ES24.16)')'LUMINA_MOM_CAPTURED ',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,MAX_ER
\t       STOP 93
\t      END IF
"""
    source = once(source,"\t    IF(ICOUNT .EQ.  100)THEN\n", "\t    IF(ICOUNT .EQ.  100)THEN\n"+failure)
    tail = "\tRETURN\n\tEND"
    if not source.rstrip().endswith(tail):
        raise ValueError("MOM 종료 위치 불일치")
    source = source.rstrip()[:-len(tail)]+"\tIF(LUM_SELECTED)CLOSE(LUM_TRACE)\n"+tail+"\n"
    return source


def instrument_first_mom(source):
    """원래 J/H 대체 또는100회 실패 전에만 포획. 정상 호출은 디스크를 쓰지 않는다."""
    source=once(source,"\tUSE MOD_RAY_MOM_STORE\n","\tUSE MOD_RAY_MOM_STORE\n\tUSE LUMINA_MOM_DIAG\n")
    declarations="""
\tTYPE(LUMINA_JREL_STATE) LUM_J,LUM_JK
\tTYPE(LUMINA_RAY_STATE) LUM_R,LUM_RK
\tTYPE(LUMINA_ARG_STATE) LUM_A
\tLOGICAL LUM_SELECTED,LUM_TRACE_ACTIVE,LUM_INITIAL_PENDING
\tINTEGER LUM_TRACE,LUM_BAD,LUM_I
\tCHARACTER(LEN=16) LUM_MODE
\tREAL(KIND=LDP),ALLOCATABLE :: LUM_TRI(:,:),LUM_RAW(:),LUM_SLOPE(:)
\tREAL(KIND=LDP) LUM_LIMIT
"""
    source=once(source,"\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n",
                "\tEXTERNAL SPEED_OF_LIGHT,ERROR_LU,GET_RSQJ_FROM_J\n"+declarations)
    take="""\tLUM_SELECTED=LUMINA_SELECTED(FREQ)
\tLUM_TRACE_ACTIVE=.FALSE.
\tCALL GET_ENVIRONMENT_VARIABLE('LUMINA_MOM_CAPTURE_MODE',LUM_MODE)
\tIF(LUM_SELECTED)THEN
\t CALL LUMINA_FIRST_BEGIN(FREQ)
\t IF(LUMINA_JREL_BYTES()+LUMINA_RAY_BYTES()+72_8*ND_SM+32768_8>67108864_8)STOP 88
\t LUM_INITIAL_PENDING=INIT .AND. .NOT.ALLOCATED(AV_SIGMA)
\t IF(LUM_INITIAL_PENDING)THEN
\t  IF(ALLOCATED(K_ON_J))STOP 89
\t  CALL LUMINA_JREL_EMPTY(LUM_J)
\t  CALL LUMINA_RAY_RAW_TAKE(LUM_R)
\t ELSE
\t  CALL LUMINA_JREL_TAKE(LUM_J)
\t  CALL LUMINA_RAY_TAKE(LUM_R)
\t END IF
\t CALL LUMINA_TAKE_ARGS(LUM_A,__NAMES__)
\t LUM_TRACE_ACTIVE=LUM_MODE=='REPLAY_FIRST'
\t IF(LUM_TRACE_ACTIVE)THEN
\t  OPEN(NEWUNIT=LUM_TRACE,FILE='MOM_TRACE.bin',FORM='UNFORMATTED',ACCESS='STREAM',STATUS='NEW')
\t  WRITE(LUM_TRACE) 'CALL',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,LUMINA_FIRST,LUMINA_NEW,FREQ,FREQ_SAVE,ND_SM
\t END IF
\tEND IF
""".replace("__NAMES__",",".join(name for name,_,_ in ARGUMENTS))
    source=once(source,"\tC_KMS=1.0E-05_LDP*SPEED_OF_LIGHT()",take+"\tC_KMS=1.0E-05_LDP*SPEED_OF_LIGHT()")
    kernel="""\tIF(LUM_SELECTED)THEN
\t IF(LUMINA_JREL_BYTES()+LUMINA_RAY_BYTES()+72_8*ND_SM+32768_8>67108864_8)STOP 88
\t CALL LUMINA_JREL_TAKE(LUM_JK)
\t CALL LUMINA_RAY_TAKE(LUM_RK)
\t ALLOCATE(LUM_TRI(4,ND),LUM_RAW(ND),LUM_SLOPE(ND))
\tEND IF
"""
    source=once(source,"\tICOUNT=0\n","\tICOUNT=0\n"+kernel)
    before="""\t  IF(LUM_SELECTED)THEN
\t   LUM_TRI(1,:)=TA;LUM_TRI(2,:)=TB;LUM_TRI(3,:)=TC;LUM_TRI(4,:)=XM
\t   LUM_SLOPE=dlnGRSQJdlnR
\t  END IF
\t  IF(LUM_TRACE_ACTIVE)THEN
\t   CALL LUMINA_TRACE_BUDGET(LUM_TRACE,ND)
\t   WRITE(LUM_TRACE) 'TRI0',ICOUNT,ND,TA,TB,TC,XM
\t  END IF
"""
    source=once(source,"\t  CALL THOMAS(TA,TB,TC,XM,ND,1)",before+"\t  CALL THOMAS(TA,TB,TC,XM,ND,1)")
    event_args="ND,FREQ,LUM_TRI,LUM_RAW,GAM_RSQHNU,LUM_SLOPE,LUM_INITIAL_PENDING,LUM_A,LUM_J,LUM_R,LUM_JK,LUM_RK"
    negative="""\t  IF(LUM_SELECTED)THEN
\t   LUM_RAW=XM
\t   IF(LUM_TRACE_ACTIVE)WRITE(LUM_TRACE) 'RAW0',ICOUNT,ND,XM
\t   IF(J_CHK_OPTION=='FORM_VAL' .OR. J_CHK_OPTION=='ABS_VAL')THEN
\t    DO LUM_I=1,ND
\t     IF(XM(LUM_I)<0.0_LDP)THEN
\t      IF(LUM_TRACE_ACTIVE)CLOSE(LUM_TRACE)
\t      CALL LUMINA_FIRST_EVENT(1,ICOUNT,LUM_I,__EVENT_ARGS__)
\t     END IF
\t    END DO
\t   END IF
\t  END IF
""".replace("__EVENT_ARGS__",event_args)
    source=once(source,"\t  RECORDED_ERROR=.FALSE.",negative+"\t  RECORDED_ERROR=.FALSE.")
    h_event="""\t  IF(LUM_SELECTED .AND. (H_CHK_OPTION=='MAX_VAL' .OR. H_CHK_OPTION=='AV_VAL'))THEN
\t   DO LUM_I=1,ND-1
\t    IF(H_CHK_OPTION=='MAX_VAL')THEN
\t     LUM_LIMIT=MAX(XM(LUM_I),XM(LUM_I+1))
\t    ELSE
\t     LUM_LIMIT=(XM(LUM_I)+XM(LUM_I+1))/2.0_LDP
\t    END IF
\t    IF(GAM_RSQHNU(LUM_I)>LUM_LIMIT .OR. GAM_RSQHNU(LUM_I)<-LUM_LIMIT)THEN
\t     IF(LUM_TRACE_ACTIVE)CLOSE(LUM_TRACE)
\t     CALL LUMINA_FIRST_EVENT(2,ICOUNT,LUM_I,__EVENT_ARGS__)
\t    END IF
\t   END DO
\t  END IF
""".replace("__EVENT_ARGS__",event_args)
    source=once(source,"\t  IF(H_CHK_OPTION .EQ. 'AV_VAL')THEN",h_event+"\t  IF(H_CHK_OPTION .EQ. 'AV_VAL')THEN")
    source=once(source,"\t    IF(MAX_ER .LT. 1.0E-08_LDP)ACCURATE=.TRUE.",
        "\t    IF(LUM_TRACE_ACTIVE)WRITE(LUM_TRACE) 'ERR0',ICOUNT,ND,MAX_ER,GAM_RSQJOLD,GAM_RSQJNU,dlnGRSQJdlnR\n\t    IF(MAX_ER .LT. 1.0E-08_LDP)ACCURATE=.TRUE.")
    failure="""\t      IF(LUM_SELECTED .AND. MAX_ER>=1.0E-08_LDP)THEN
\t       IF(LUM_TRACE_ACTIVE)CLOSE(LUM_TRACE)
\t       CALL LUMINA_FIRST_EVENT(3,ICOUNT,0,__EVENT_ARGS__)
\t      END IF
""".replace("__EVENT_ARGS__",event_args)
    source=once(source,"\t    IF(ICOUNT .EQ.  100)THEN\n","\t    IF(ICOUNT .EQ.  100)THEN\n"+failure)
    tail="\tRETURN\n\tEND"
    if not source.rstrip().endswith(tail):raise ValueError("MOM 종료 위치 불일치")
    complete="""\tIF(LUM_SELECTED)THEN
\t CALL LUMINA_JREL_TAKE(LUM_JK)
\t CALL LUMINA_RAY_TAKE(LUM_RK)
\t CALL LUMINA_FIRST_COMPLETE(LUM_A,LUM_JK,LUM_RK)
\tEND IF
\tIF(LUM_TRACE_ACTIVE)CLOSE(LUM_TRACE)
"""
    return source.rstrip()[:-len(tail)]+complete+tail+"\n"


def replay_source():
    actuals = ",".join("A%"+name for name,_,_ in ARGUMENTS)
    return """PROGRAM LUMINA_MOM_REPLAY
USE SET_KIND_MODULE
USE MOD_JREL_V9
USE MOD_RAY_MOM_STORE
USE LUMINA_MOM_DIAG
IMPLICIT NONE
TYPE(LUMINA_ARG_STATE) A
TYPE(LUMINA_JREL_STATE) J
TYPE(LUMINA_RAY_STATE) RSTATE
CHARACTER(LEN=4096) PATH
CALL GET_COMMAND_ARGUMENT(1,PATH)
IF(PATH==' ')STOP 85
CALL LUMINA_SNAPSHOT_IO(TRIM(PATH),A,J,RSTATE,.TRUE.)
CALL LUMINA_JREL_RESTORE(J)
CALL LUMINA_RAY_RESTORE(RSTATE)
! 원래 출력 unit만 열고, 저장된 물리 상태/INIT/이전 주파수는 변경하지 않는다.
IF(LU_MOM>0)OPEN(UNIT=LU_MOM,FILE='MOM_J_ERRORS',STATUS='NEW',ACTION='WRITE')
WRITE(6,'(A,3I8,2ES24.16)')'LUMINA_REPLAY_INPUT ',LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT,A%FREQ,FREQ_SAVE
CALL MOM_JREL_V9(__ACTUALS__)
WRITE(6,'(A)')'LUMINA_MOM_RETURNED_WITHOUT_CAPTURE'
END PROGRAM
""".replace("__ACTUALS__",actuals)


def command(args, cwd):
    with (cwd/"build.log").open("a") as log:
        log.write("COMMAND "+" ".join(str(a) for a in args)+"\n");log.flush()
        subprocess.run([str(a) for a in args],cwd=cwd,stdout=log,stderr=log,check=True,timeout=120)


def rebuild_unmodified_sources(parent,native,out):
    """수정할 네 translation unit의 원본도 부모와 byte 동일하게 재빌드한다."""
    base=out/"baseline"
    base.mkdir()
    paths={"mod_ray_mom_store.f":native/"plane/mod_ray_mom_store.f",
           "mom_jrel_v9.f":native/"plane/mom_jrel_v9.f",
           "comp_j_blank.f":native/"new_main/mod_subs/comp_j_blank.f",
           "cmfgen_sub.f":native/"new_main/cmfgen_sub.f",
           "GET_J_CHI.INC":native/"new_main/GET_J_CHI.INC"}
    for name,path in paths.items():
        shutil.copy2(path,base/name)
    for name in paths:
        if name.endswith(".f"):
            command(["gfortran",*FLAGS,"-I.","-I"+str(native/"mod"),"-c",name],base)
    shutil.copy2(parent/"libsubs.a",base/"libsubs.a")
    command(["ar","r","libsubs.a","mom_jrel_v9.o","mod_ray_mom_store.o","comp_j_blank.o"],base)
    libs=["-L.","-L"+str(native/"lib"),"-lsubs","-lpgplt","-lmod",native/"lib/libmy_lapack.a",native/"lib/libmy_blas.a"]
    command(["gfortran","-fopenmp","-m64",native/"new_main/cmfgen.o","cmfgen_sub.o",*libs,"-o","cmfgen_unmodified.exe"],base)
    if sha(base/"cmfgen_unmodified.exe")!=sha(parent/"cmfgen_parent_rebuilt.exe"):
        raise ValueError("포획 대상 source를 재빌드한 baseline이 실제 부모와 다름")
    return base/"cmfgen_unmodified.exe"


def prepare(parent, native, out, first_repair=False):
    identity = json.loads((parent/"parent-build-identity.json").read_text())
    inputs = [parent/"libsubs.a",parent/"cmfgen_parent_rebuilt.exe",native/"new_main/cmfgen.o",
              native/"new_main/cmfgen_sub.o",native/"plane/mom_jrel_v9.f",native/"plane/mod_ray_mom_store.f",
              native/"new_main/mod_subs/comp_j_blank.f",native/"tools/set_kind_module.f",
              native/"new_main/cmfgen_sub.f",native/"new_main/GET_J_CHI.INC",
              native/"lib/libmod.a",native/"lib/libpgplt.a",native/"lib/libmy_lapack.a",native/"lib/libmy_blas.a"]
    for path in inputs:
        expected=identity["evidence_sha256"].get(str(path))
        if expected is None and path in (native/"new_main/cmfgen_sub.f",native/"new_main/GET_J_CHI.INC"):
            continue  # 아래 원본 전체 실행파일의 바이트 일치로 신원을 확인한다.
        if sha(path) != expected:
            raise ValueError("부모 source/object/library 신원 변경: "+str(path))
    mom = (native/"plane/mom_jrel_v9.f").read_text()
    split = re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",mom)
    if not split:
        raise ValueError("MOM module 분리 불가")
    jrel, jfields = instrument_module(mom[:split.end()]+"\n","MOD_JREL_V9","LUMINA_JREL")
    ray, rfields = instrument_module((native/"plane/mod_ray_mom_store.f").read_text(),"MOD_RAY_MOM_STORE","LUMINA_RAY")
    sub = (native/"new_main/cmfgen_sub.f").read_text()
    # 첫 외부 반복60 직전의 기존 단일 증분 위치. 나머지 산술/분기는 불변이다.
    sub = once(sub,"\tMAIN_COUNTER=MAIN_COUNTER+1\n",
               "\tMAIN_COUNTER=MAIN_COUNTER+1\n\tCALL LUMINA_MOM_SET_ITERATION(MAIN_COUNTER)\n")
    sub = once(sub,"\tCALL SOLVE_FOR_POPS(",
               "\tCALL LUMINA_MOM_STOP_BEFORE_POP()\n\tCALL SOLVE_FOR_POPS(")
    caller = (native/"new_main/mod_subs/comp_j_blank.f").read_text()
    caller = once(caller,"\t       CALL MOM_JREL_V9", "\t       CALL LUMINA_MOM_SET_PHASE(FREQ_INDX,J_IT_COUNTER,FIRST_FREQ,NEW_FREQ)\n\t       CALL MOM_JREL_V9")
    mom_builder=instrument_first_mom if first_repair else instrument_mom
    sources = {"mod_jrel_v9.f":jrel,"mod_ray_mom_store.f":ray,
               "lumina_mom_diag.f90":diagnostic_module(first_repair),"mom_jrel_v9.f":mom_builder(mom[split.end():]),
               "comp_j_blank.f":caller,"cmfgen_sub.f":sub,"replay_mom.f90":replay_source()}
    if out.exists():
        raise ValueError("기존 build 보존")
    out.mkdir()
    unmodified=rebuild_unmodified_sources(parent,native,out)
    for name,text in sources.items():
        (out/name).write_text(text)
    shutil.copy2(native/"new_main/GET_J_CHI.INC",out/"GET_J_CHI.INC")
    (out/"state_inventory.json").write_text(json.dumps(dict(jrel=jfields,ray=rfields),indent=2)+"\n")
    for name in ("mod_jrel_v9.f","mod_ray_mom_store.f","lumina_mom_diag.f90","mom_jrel_v9.f","comp_j_blank.f","cmfgen_sub.f","replay_mom.f90"):
        opts = FLAGS if name.endswith(".f") else ["-fopenmp","-m64","-O","-ffree-line-length-none","-fbacktrace"]
        command(["gfortran",*opts,"-I.","-I"+str(native/"mod"),"-c",name,"-o",Path(name).stem+".o"],out)
    shutil.copy2(parent/"libsubs.a",out/"libsubs.a")
    command(["ar","r","libsubs.a","mom_jrel_v9.o","mod_ray_mom_store.o","comp_j_blank.o"],out)
    libs = ["-L.","-L"+str(native/"lib"),"-lsubs","-lpgplt","-lmod",native/"lib/libmy_lapack.a",native/"lib/libmy_blas.a"]
    common = ["mod_jrel_v9.o","lumina_mom_diag.o"]
    command(["gfortran","-fopenmp","-m64",native/"new_main/cmfgen.o","cmfgen_sub.o",*common,*libs,"-o","cmfgen_capture.exe"],out)
    command(["gfortran","-fopenmp","-m64","replay_mom.o",*common,*libs,"-o","replay_mom.exe"],out)
    output_files = [out/name for name in sources]+[out/"cmfgen_capture.exe",out/"replay_mom.exe",out/"libsubs.a",out/"state_inventory.json"]
    record = dict(scope="부모 동일 산술의 선택 MOM 호출 상태 포획; 실제 실패 포획/물리 수렴은 별도",
                  parent=str(parent),native=str(native),generator_sha256=sha(Path(__file__)),
                  parent_manifest_sha256=sha(parent/"parent-build-identity.json"),
                  rebuilt_unmodified_sha256=sha(unmodified),
                  input_sha256={str(p):sha(p) for p in inputs},output_sha256={str(p):sha(p) for p in output_files},
                  JREL_fields=len(jfields),RAY_fields=len(rfields),first_repair=first_repair,final_goal="NOT_JUDGED")
    (out/"capture-build.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if not k.endswith("sha256")},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent","native","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--first-repair",action="store_true")
    a=parser.parse_args()
    prepare(a.parent.resolve(),a.native.resolve(),a.out.resolve(),a.first_repair)
