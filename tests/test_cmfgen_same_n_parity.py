#!/usr/bin/env python3
"""Compare every Si/S FL-pair SAME_N classification with native Fortran."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_cmfgen_fullpair_omega import IONS, same_n_key
from cmfgen_parser import parse_osc
from expand_atomic_data_cmfgen import load_cmfgen_links, load_cmfgen_model_spec
from cmfgen_model_continuum import sha


def run(args):
    if args.out.exists():
        raise FileExistsError("기존 SAME_N 대조 보존")
    links=load_cmfgen_links(args.material_run/"atomic_links.txt")
    models=load_cmfgen_model_spec(args.material_run/"MODEL_SPEC",
                                  args.material_run/"atomic_links.txt")
    source_paths=(args.native_source,Path(__file__),Path(__file__).with_name("cmfgen_same_n_parity.f90"),
                  Path(__file__).resolve().parents[1]/"scripts/build_cmfgen_fullpair_omega.py",
                  *(links[Z,ion0+1]["osc"] for Z,ion0,_,_ in IONS))
    before={str(path):sha(path) for path in source_paths}
    pairs=[]
    for Z,ion0,name,_ in IONS:
        osc=parse_osc(links[Z,ion0+1]["osc"])
        n=int(models[Z,ion0+1]["n_full"])
        names=[str(label) for label in osc.levels["config"][:n]]
        if len(names)!=n or any(not label or any(c.isspace() for c in label) for label in names):
            raise ValueError(f"{name} SAME_N 이름 형상 오류")
        for i in range(n):
            for j in range(i+1,n):
                pairs.append((name,names[i],names[j]))
    binary=args.out.with_suffix(".exe")
    subprocess.run(["gfortran","-O0","-ffixed-line-length-none",
                    str(Path(__file__).with_name("cmfgen_same_n_parity.f90")),
                    str(args.native_source),"-o",str(binary)],check=True,timeout=60)
    lines="".join(f"{a} {b}\n" for _,a,b in pairs)
    completed=subprocess.run([str(binary)],input=lines,text=True,capture_output=True,
                             check=True,timeout=120)
    result=completed.stdout.splitlines()
    if len(result)!=len(pairs) or any(value not in ("0","1") for value in result):
        raise ValueError("native SAME_N 출력 누락")
    mismatch=[];native_true=python_true=0
    for (name,a,b),native in zip(pairs,result,strict=True):
        wanted=same_n_key(a) is not None and same_n_key(a)==same_n_key(b)
        native_true+=int(native=="1");python_true+=int(wanted)
        if (native=="1")!=wanted and len(mismatch)<12:
            mismatch.append(dict(ion=name,lower=a,upper=b,native=native,python=int(wanted)))
    if any(sha(Path(path))!=digest for path,digest in before.items()):
        raise ValueError("SAME_N source 시험 중 변경")
    report=dict(state="PASS" if not mismatch and native_true==python_true else "MISMATCH",
                pairs=len(pairs),native_true=native_true,python_true=python_true,
                mismatches=mismatch,source_sha256=before,binary_sha256=sha(binary))
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False))
    if report['state']!='PASS':
        raise ValueError("native SAME_N 분기 불일치")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-run",type=Path,required=True)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args())
