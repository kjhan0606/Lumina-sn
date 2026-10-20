#!/usr/bin/env python3
"""StageA59의 불변 복제본에서 고정 T·LAMBDA 4회만 실행한다.

Slurm 내부 자식 프로세스의 중단 제어다. 분당 최대 한 번 새 로그 바이트만
읽으며 Slurm 조회나 상주·반복 제출을 하지 않는다. 원본은 수정하지 않는다.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from configure_cmfgen_stageA import set_control
from ptf_gated_metrics import ScrtempReader, read_dimensions, validate_rvtj

INPUTS = ("VADAT", "IN_ITS", "MODEL_SPEC", "SN_HYDRO_DATA", "MODEL",
          "SPECIES_MASSES", "LEVEL_SL_STEQ_LINKS", "model_spec_isf.txt",
          "atomic_links.txt", "setup_links.sh", "snia_toy06_19.48d.dat")
RESTART = ("SCRTEMP", "POINT1", "POINT2", "EDDFACTOR", "EDDFACTOR_INFO",
           "CONT_FREQ", "CUR_MODEL_DATA", "STEQ_VALS", "RVTJ", "BAMATPNT", "BAMAT", "JEW")
BAD_LOG = re.compile(
    r"Error in MOM_J_REL|MOM_JREL[^\n]*(?:error|fail)|(?:error|fail)[^\n]*moment|"
    r"grey[^\n]*(?:error|fail)|(?:error|fail)[^\n]*grey|\b(?:NaN|[+-]?Infinity)\b", re.I)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def control(text,key):
    values=re.findall(r"^(?!\s*[!#])\s*(\S+)\s+\["+re.escape(key)+r"\]",text,re.M)
    if len(values)!=1: raise ValueError(f"제어 키 누락/중복: {key}")
    return values[0]


def configure(vadat,in_its):
    for key,value in (("FIX_T","T"),("FIX_T_AUTO","F"),("DO_DDT","F"),
                      ("TRT_NON_TE","F"),("DO_NG","F"),("USE_FIXED_J","F"),
                      ("FLUX_CAL_ONLY","F"),("WRITE_RATES","T"),("WRITE_JH","T")):
        if control(vadat,key)!=value:raise ValueError(f"봉인 제어 불일치 {key}")
    if control(in_its,"DO_LAM_IT")!="F":raise ValueError("원래 LAMBDA 제어가 예상과 다름")
    if control(in_its,"DO_LAM_AUTO")!="F" or control(in_its,"DO_T_AUTO")!="F":
        raise ValueError("원래 자동 전환 제어 불일치")
    return set_control(set_control(in_its,"NUM_ITS","4"),"DO_LAM_IT","T")


def scan_logs(run,cursors):
    reason=None
    for name in ("OUTGEN","seq_logs/cmfgen.stdout","seq_logs/cmfgen.stderr"):
        path=run/name
        if not path.exists():continue
        offset,tail=cursors.get(name,(0,""))
        with path.open("r",errors="replace") as f:
            f.seek(offset);new=f.read();offset=f.tell()
        text=tail+new
        match=BAD_LOG.search(text)
        if match:reason=f"solver_error:{name}:{match.group(0)}"
        if re.search(r"Maximum[^\n]*\(BA computed\)[^\n]*iteration\s+(?:60|61|62|63)\b",text):
            reason="unexpected_full_BA_in_lambda_probe"
        cursors[name]=(offset,text[-512:])
    return reason


def stop_child(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)


def stage(seed,run,executable,source,*,write_record=True):
    if run.exists():raise ValueError(f"기존 계산 보존: {run}")
    for name in INPUTS+RESTART:
        p=seed/name
        if not p.is_file() or (name in RESTART and p.is_symlink()):
            raise ValueError(f"restart 입력 누락/외부 링크: {p}")
    for name in ("POINT1","POINT2"):
        if (seed/name).read_text().splitlines()[1].split()!=["59","59","1","-1000","F"]:
            raise ValueError("StageA59 포인터가 아님")
    nd,nt=read_dimensions(seed);reader=ScrtempReader(seed/"SCRTEMP",nd,nt)
    if reader.last_it!=59:raise ValueError("SCRTEMP 최종 상태가59가 아님")
    validation=validate_rvtj(reader,seed/"RVTJ")
    if not validation["complete"]:raise ValueError("RVTJ/SCRTEMP 연결 불완전")
    next_in_its=configure((seed/"VADAT").read_text(),(seed/"IN_ITS").read_text())
    # Verify the same executable used by the parent calculation.
    entries=(seed/"seq_logs/stageA-pre-run.sha256").read_text().splitlines()
    expected=[line.split()[0] for line in entries if line.split(maxsplit=1)[1].strip()==str(executable)]
    if len(expected)!=1 or sha(executable)!=expected[0]:raise ValueError("부모 바이너리 신원 불일치")
    run.mkdir();(run/"seq_logs").mkdir()
    seed_hashes={str((seed/name).resolve()):sha(seed/name) for name in INPUTS+RESTART}
    for name in INPUTS+RESTART:shutil.copy2(seed/name,run/name)
    local=("atomic_local/SUL/III/19apr23/phot_data_A", "atomic_local/COB/II/19apr23/phot_data_A")
    for name in local:
        p=run/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/name,p)
        seed_hashes[str((seed/name).resolve())]=sha(seed/name)
    for name in ("setup_links.sh","atomic_links.txt"):
        p=run/name;text=p.read_text()
        if str(seed) not in text:raise ValueError(f"원래 계산 경로가 없음: {name}")
        p.write_text(text.replace(str(seed),str(run)))
    (run/"IN_ITS").write_text(next_in_its)
    subprocess.run(["bash",str(run/"setup_links.sh")],cwd=run,check=True)
    links={}
    for line in (run/"setup_links.sh").read_text().splitlines():
        fields=line.split()
        if len(fields)>=4 and fields[:2]==["ln","-sf"]:
            p=run/fields[3]
            if not p.is_file():
                if p.name in ("HI_IS_LINE_LIST","H2_IS_LINE_LIST"):continue
                raise ValueError(f"필수 입력 링크 없음: {p}")
            links[p.name]=dict(target=str(p.resolve()),sha256=sha(p))
    if (run/"VADAT").read_bytes()!=(seed/"VADAT").read_bytes():raise ValueError("VADAT 변경됨")
    if (run/"XzV_IN").exists():raise ValueError("외부 인구 벡터 금지")
    record=dict(scope="StageA59 fixed-T LAMBDA-only four-iteration probe; not oracle",
        seed=str(seed),run=str(run),seed_validation=validation,seed_sha256=seed_hashes,
        binary=dict(path=str(executable),sha256=sha(executable)),linked_inputs=links,
        source_sha256={str(source/name):sha(source/name) for name in
            ("new_main/cmfgen_sub.f","new_main/mod_subs/solve_for_pops.f","new_main/mod_subs/rd_control_variables.f")},
        changed_controls=dict(NUM_ITS="19 -> 4",DO_LAM_IT="F -> T"),
        preserved_controls="VADAT byte-identical; DO_LAM_AUTO=F; DO_T_AUTO=F",
        first_iteration=60,last_iteration_cap=63,max_seconds=13800,
        resources="CPU16 RAM200GiB GPU0 Slurm wall4h, no node pin",
        useful_probe_criteria=dict(no_solver_errors=True,subset_p95_percent_max=39.6438,
            delta_ln_ne_max=.018887720378449203,
            Si_III_max_abs_fraction_step=.07292085643231966,
            S_III_max_abs_fraction_step=.020017557204485947),
        final_goal="NOT_JUDGED",runner_sha256=sha(Path(__file__)))
    if write_record:
        (run/"seq_logs/lambda-probe-inputs.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",type=Path,required=True);ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--executable",type=Path,required=True);ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--stage-only",action="store_true")
    a=ap.parse_args();seed,run,exe,source=(p.resolve() for p in (a.seed,a.run,a.executable,a.source))
    stage(seed,run,exe,source)
    if a.stage_only:return
    cursors={};reason=None;started=time.monotonic()
    with (run/"seq_logs/cmfgen.stdout").open("xb") as out,(run/"seq_logs/cmfgen.stderr").open("xb") as err:
        proc=subprocess.Popen(["srun","--cpu-bind=cores",str(exe)],cwd=run,stdin=subprocess.DEVNULL,
                              stdout=out,stderr=err,start_new_session=True)
        while proc.poll() is None:
            reason=scan_logs(run,cursors)
            if reason or time.monotonic()-started>=13800:
                reason=reason or "runtime_cap_13800s";stop_child(proc);break
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:pass
        reason=reason or scan_logs(run,cursors)
    result=dict(returncode=proc.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-started,
                final_goal="NOT_JUDGED",run=str(run))
    if not reason and proc.returncode==0:
        fields=(run/"POINT1").read_text().splitlines()[1].split()
        result["last_iteration"]=int(fields[0])
        result["four_iterations_complete"]=fields[:2]==["63","63"]
    (run/"seq_logs/lambda-probe-runtime.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    return 0 if proc.returncode==0 and not reason else 30


if __name__=="__main__":raise SystemExit(main())
