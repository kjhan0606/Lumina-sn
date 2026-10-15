#!/usr/bin/env python3
"""수렴 toy 시간해의 절대 스펙트럼/Si/S를 같은 공개 시각과 비교한다.

측정만 수행하며 최종 동등성 PASS를 자동 부여하지 않는다. 자료를 외삽하거나
스펙트럼 진폭을 맞추지 않는다. 공개값은 계산 입력으로 되돌려주지 않는다.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from read_cmf_common_material import public_block, interpolate


def linear(points, x):
    if abs(x-points[0][0]) < 1e-10*points[0][0]:
        return points[0][1]
    if abs(x-points[-1][0]) < 1e-10*points[-1][0]:
        return points[-1][1]
    if not points[0][0] <= x <= points[-1][0]:
        raise ValueError("파장 외삽은 허용하지 않는다")
    i = bisect.bisect_left(points, (x, -math.inf))
    if points[i][0] == x:
        return points[i][1]
    a,b = points[i-1],points[i]
    return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])


def integrate(x, y):
    return math.fsum((b-a)*(u+v)/2 for a,b,u,v in zip(x,x[1:],y,y[1:]))


def absolute_integral(x, y):
    # 선형 보간 차이가 구간 안에서0을 지나는 경우 두 삼각형을 정확히 적분한다.
    return math.fsum((b-a)*(u*u+v*v)/(2*(abs(u)+abs(v))) if u*v<0
        else (b-a)*(abs(u)+abs(v))/2 for a,b,u,v in zip(x,x[1:],y,y[1:]))


def spectrum_metrics(model, reference, lo=2500., hi=25500.):
    for points in (model, reference):
        if len(points)<2 or any(not all(math.isfinite(v) for v in p) or p[0]<=0 or p[1]<0 for p in points):
            raise ValueError("스펙트럼의 비유한/음수/빈 자료")
        if any(a[0]>=b[0] for a,b in zip(points,points[1:])):
            raise ValueError("스펙트럼 파장 순서/중복 오류")
    x=sorted({lo,hi,*[p[0] for p in model+reference if lo<p[0]<hi]})
    m=[linear(model,v) for v in x];r=[linear(reference,v) for v in x]
    ml,rl=integrate(x,m),integrate(x,r)
    if not ml>0 or not rl>0:
        raise ValueError("적분 광도는 양수여야 한다")
    bands=[]
    for a,b in ((2500,3500),(3500,5000),(5000,6500),(6500,9000),(9000,25500)):
        if a<lo or b>hi:
            continue
        bx=sorted({a,b,*[v for v in x if a<v<b]})
        bm=integrate(bx,[linear(model,v) for v in bx])
        br=integrate(bx,[linear(reference,v) for v in bx])
        bands.append(dict(wavelength_A=[a,b],model_erg_s=bm,reference_erg_s=br,ratio=bm/br if br else None))
    return dict(status="MEASURED",range_A=[lo,hi],model_erg_s=ml,reference_erg_s=rl,
        luminosity_ratio=ml/rl,absolute_L1_relative=absolute_integral(x,[a-b for a,b in zip(m,r)])/rl,
        unit_area_shape_L1_diagnostic=absolute_integral(x,[a/ml-b/rl for a,b in zip(m,r)]),
        amplitude_fitting=False,bolometric=False,bands=bands)


def public_spectrum(path, day):
    result=[];column=None
    with path.open() as handle:
        for line in handle:
            if line.startswith("#TIMES[d]:"):
                times=[float(x) for x in line.split()[1:]]
                indices=[i for i,t in enumerate(times) if abs(t-day)<1e-8]
                if len(indices)!=1:
                    return None
                column=indices[0]+1
            elif line.strip() and not line.startswith("#"):
                if column is None:
                    raise ValueError("공개 스펙트럼 시각 헤더 없음")
                row=line.split();result.append((float(row[0]),float(row[column])))
    return result


def run_nonthermal(lines, marker):
    records=[line for line in lines if line.startswith(marker)]
    if len(records)!=1:
        raise ValueError("수렴 성공 기록 없음 또는 중복")
    modes=[item.split("=",1)[1] for item in records[0].split() if item.startswith("nonthermal=")]
    if len(modes)!=1 or modes[0] not in ("ALL_THERMAL_CONTROL", "RESOLVED_NONTHERMAL_SUBSET"):
        raise ValueError("비열 모형 정체성 없음 또는 알 수 없는 처방")
    return modes[0]


def compare(run, public, out):
    work=run/"work"
    kind="step" if (work/"step_state.bin").exists() else "initial"
    state=work/f"{kind}_state.bin"
    with state.open("rb") as handle:
        head=handle.read(96)
    if len(head)!=96 or head[:8]!=b"CMFINIT1":
        raise ValueError("실제 과거 파일 헤더 없음")
    version,ns,nb,ni,nl,pg,tg=struct.unpack("<7Q",head[8:64])
    epoch,nu,dlog,decay=struct.unpack("<4d",head[64:96]);day=epoch/86400
    if version!=1 or decay!=0 or not epoch>0 or not math.isfinite(epoch):
        raise ValueError("시점 정체성 오류")
    marker=f"TOY_{kind.upper()}_CONVERGED "
    with (run/f"{kind}.log").open() as handle:
        nonthermal=run_nonthermal(handle,marker)
    obs=json.loads((work/f"{kind}_spectrum.json").read_text())
    if obs["units"]!="erg/s/angstrom" or obs["material_generation"]!=pg:
        raise ValueError("스펙트럼 단위/물질 세대 불일치")
    material=work/("step_material.csv" if kind=="step" else "initial_converged_material.csv")
    with material.open() as handle:
        status=handle.readline()
        if not status.startswith("# CONVERGED_") or status.rstrip().split("; ")[-1]!=nonthermal:
            raise ValueError("수렴 물질표가 아님")
        rows=list(csv.DictReader(handle))
    physhead,phys=public_block(public/"phys_toy06_cmfgen.txt",day)
    results=[];metrics=[];seen=set();shells={};missing={}
    for z,name,mass in ((14,"si",28.0855),(16,"s",32.065)):
        header,ref=public_block(public/f"ionfrac_{name}_toy06_cmfgen.txt",day)
        columns={int(key[len(name):]):i for i,key in enumerate(header) if key.startswith(name)}
        for row in rows:
            if int(row["Z"])!=z:
                continue
            s,q=int(row["shell"]),int(row["stage"]);key=(s,z,q)
            if key in seen:
                raise ValueError("이온 행 중복")
            seen.add(key)
            velocity=(float(row["r_inner_cm"])+float(row["r_outer_cm"]))/2/epoch/1e5
            density=float(row["rho_g_cm3"])*float(row["mass_fraction"])/(mass*1.660539066e-24)
            population=float(row["n_ion_cm3"])
            if not math.isfinite(density) or density<0 or not math.isfinite(population) or population<0:
                raise ValueError("비유한/음수 이온 밀도")
            if density==0 and population!=0:
                raise ValueError("원소 핵수0에 양의 이온 인구")
            fraction=population/density if density>0 else None
            reference=interpolate(ref,velocity,columns[q]) if q in columns else None
            results.append(dict(shell=s,Z=z,stage=q,velocity_km_s=velocity,nuclei_cm3=density,
                fraction=fraction,reference=reference,difference=fraction-reference if fraction is not None and reference is not None else None))
            shells[s]=dict(shell=s,velocity_km_s=velocity,T_K=float(row["T_K"]),ne_cm3=float(row["n_e_cm3"]),
                T_reference_K=interpolate(phys,velocity,physhead.index("temp[K]")),
                ne_reference_cm3=interpolate(phys,velocity,physhead.index("ne[/cm^3]")))
        present={r["stage"] for r in results if r["Z"]==z}
        missing[str(z)]=sorted(set(columns)-present)
        for q in sorted(present):
            errors=[abs(r["difference"]) for r in results if r["Z"]==z and r["stage"]==q and r["difference"] is not None]
            metrics.append(dict(Z=z,stage=q,shells=len(errors),mean_absolute_fraction_difference=math.fsum(errors)/len(errors) if errors else None,
                max_absolute_fraction_difference=max(errors) if errors else None))
    if len(shells)!=ns:
        raise ValueError("물질 셸 수 불일치")
    with (work/f"{kind}_spectrum.csv").open() as handle:
        spectrum=[(float(r["wavelength_angstrom"]),float(r["flux"])) for r in csv.DictReader(handle)]
    reference=public_spectrum(public/"spectra_toy06_cmfgen.txt",day)
    spectral=spectrum_metrics(spectrum,reference) if reference is not None else dict(status="NO_EXACT_PUBLIC_EPOCH",day=day)
    summary=dict(day=day,material_generation=pg,temperature_generation=tg,goal_verdict="NOT_JUDGED",
        denominator="rho*mass_fraction/(mass_amu*AMU); no renormalization",velocity_interpolation="linear; no extrapolation",
        missing_atomic_stages=missing,missing_atomic_stages_are_not_zero=True,
        nonthermal=nonthermal,spectrum=spectral,ionization=metrics)
    out.mkdir(parents=True,exist_ok=False)
    for name,values in (("ionization.csv",results),("plasma.csv",[shells[s] for s in sorted(shells)])):
        with (out/name).open("x",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    inputs=[state,material,work/f"{kind}_spectrum.csv",work/f"{kind}_spectrum.json",run/f"{kind}.log",
        public/"spectra_toy06_cmfgen.txt",public/"phys_toy06_cmfgen.txt",public/"ionfrac_si_toy06_cmfgen.txt",public/"ionfrac_s_toy06_cmfgen.txt",Path(__file__),Path(__file__).with_name("read_cmf_common_material.py")]
    hashes={}
    for path in inputs:
        with path.open("rb") as handle:
            hashes[str(path)]=hashlib.file_digest(handle,"sha256").hexdigest()
    summary["input_sha256"]=hashes
    with (out/"summary.json").open("x") as handle:
        json.dump(summary,handle,indent=2,allow_nan=False);handle.write("\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","public","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("비교 계산은 Slurm에서 실행한다")
    print(json.dumps(compare(args.run,args.public,args.out),ensure_ascii=False,allow_nan=False))
