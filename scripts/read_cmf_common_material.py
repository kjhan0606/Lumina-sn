#!/usr/bin/env python3
"""같은 최종 물질의 Si/S 분포를 공개 CMFGEN과 비교한다. 최종 통과 판정은 아니다.

SAHA_STAGE와 NLTE_LEVEL은 별도 열이다. 분모는 덱의 rho*abundance/(mass*AMU)다.
누락 준위·범위 밖 기준은 결측으로 남기며 강제 정규화하거나 외삽하지 않는다.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

AMU = 1.660539066e-24
ELEMENTS = {14: "si", 16: "s"}


def rows(path):
    with Path(path).open(newline="") as handle:
        yield from csv.DictReader(handle)


def public_block(path, day):
    active = False
    hits = 0
    header, data = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("#TIME:"):
                active = abs(float(line.split()[1])-day) < 1e-8
                hits += int(active)
            elif active and line.startswith("#vel_mid"):
                header = line.lstrip("#").split()
            elif active and line.strip() and not line.startswith("#"):
                data.append([float(x) for x in line.split()])
    if hits != 1 or not header or not data:
        raise ValueError("기준의 정확한 시각/헤더가 없거나 중복됨")
    if any(len(r) != len(header) or not all(math.isfinite(x) for x in r) for r in data):
        raise ValueError("기준 행/유한값 오류")
    if any(a[0] >= b[0] for a,b in zip(data,data[1:])):
        raise ValueError("기준 속도 순서 오류")
    return header, data


def interpolate(data, velocity, column):
    if not data[0][0] <= velocity <= data[-1][0]:
        return None
    k = bisect.bisect_left([r[0] for r in data], velocity)
    if data[k][0] == velocity:
        return data[k][column]
    low, high = data[k-1], data[k]
    w = (velocity-low[0])/(high[0]-low[0])
    return (1-w)*low[column]+w*high[column]


def nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("밀도/개체수는 유한한 비음수여야 함")
    return result


def write_csv(path, data):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def analyze(run, model, public, out, day=19.48):
    work = run/"work"
    final = json.loads((work/"cmfgen_final_publication.json").read_text())
    obs = json.loads((work/"lumina_spectrum_common_obs.json").read_text())
    pg = final["population_generation"]
    if not final["complete"] or pg != final["required_population_generation"] or pg != obs["material_generation"]:
        raise ValueError("최종 물질/관측 출력 세대 불일치")
    geo = {int(r["shell_id"]): r for r in rows(model/"geometry.csv")}
    rho = {int(r["shell_id"]): nonnegative(r["rho"]) for r in rows(model/"density.csv")}
    abundance = {int(r["atomic_number"]): r for r in rows(model/"abundances.csv")}
    mass = {int(r["atomic_number"]): float(r["mass_amu"]) for r in rows(model/"atom_masses.csv")}
    expected = defaultdict(set)
    for index,r in enumerate(rows(model/"levels.csv")):
        z, ion = int(r["atomic_number"]), int(r["ion_number"])
        if z in ELEMENTS:
            expected[z,ion].add(index)
    stage, levels, identities = {}, defaultdict(list), defaultdict(set)
    for r in rows(work/"lumina_ion_pops.csv"):
        z = int(r["Z"])
        if z not in ELEMENTS:
            continue
        key = int(r["shell_id"]), z, int(r["stage"])
        if int(r["population_generation"]) != pg or key in stage:
            raise ValueError("단계별 밀도 세대/중복 오류")
        stage[key] = nonnegative(r["n_ion"])
    for r in rows(work/"nlte_levels_final.csv"):
        z = int(r["Z"])
        if z not in ELEMENTS:
            continue
        shell, ion, index = int(r["shell"]), int(r["ion"]), int(r["global_idx"])
        key = shell,z,ion
        if (int(r["population_generation"]) != pg or
                index not in expected[z,ion] or index in identities[key]):
            raise ValueError("NLTE 준위 세대/원자 매핑/중복 오류")
        identities[key].add(index)
        levels[key].append(nonnegative(r["n_pop"]))
    result, totals, metrics = [], [], []
    for z,name in ELEMENTS.items():
        header, reference = public_block(public/f"ionfrac_{name}_toy06_cmfgen.txt", day)
        columns = {int(h[len(name):]):i for i,h in enumerate(header[1:],1) if h.startswith(name)}
        stages = sorted(set(columns) | {i for s,e,i in stage if e == z} | {i for e,i in expected if e == z})
        for shell,g in sorted(geo.items()):
            velocity = (float(g["v_inner"])+float(g["v_outer"]))/2e5
            nuclei = nonnegative(abundance[z][str(shell)])*rho[shell]/mass[z]/AMU
            sums = {"SAHA_STAGE":[], "NLTE_LEVEL":[]}
            missing = []
            for ion in stages:
                key = shell,z,ion
                ref = interpolate(reference,velocity,columns[ion]) if ion in columns else None
                nstage = stage.get(key)
                complete = bool(expected[z,ion]) and identities[key] == expected[z,ion]
                # 일부 준위의 합을 완전한 이온 밀도로 사용하지 않는다.
                nnlte = math.fsum(levels[key]) if complete else None
                if not complete:
                    missing.append(ion)
                fractions = {}
                for authority,number in (("SAHA_STAGE",nstage),("NLTE_LEVEL",nnlte)):
                    fraction = number/nuclei if number is not None and nuclei > 0 else None
                    fractions[authority] = fraction
                    if fraction is not None:
                        sums[authority].append(fraction)
                result.append(dict(shell=shell,Z=z,stage=ion,velocity_km_s=velocity,n_nuclei=nuclei,
                    population_generation=pg,cmfgen_fraction=ref,
                    stage_fraction=fractions["SAHA_STAGE"],nlte_fraction=fractions["NLTE_LEVEL"],
                    nlte_levels_present=len(identities[key]),nlte_levels_expected=len(expected[z,ion]),
                    stage_minus_cmfgen=fractions["SAHA_STAGE"]-ref if ref is not None and fractions["SAHA_STAGE"] is not None else None,
                    nlte_minus_cmfgen=fractions["NLTE_LEVEL"]-ref if ref is not None and fractions["NLTE_LEVEL"] is not None else None))
            totals.append(dict(shell=shell,Z=z,n_nuclei=nuclei,
                stage_fraction_sum=math.fsum(sums["SAHA_STAGE"]),
                available_nlte_fraction_sum=math.fsum(sums["NLTE_LEVEL"]),
                missing_nlte_stages=":".join(map(str,missing))))
        for ion in stages:
            for authority,column in (("SAHA_STAGE","stage_minus_cmfgen"),("NLTE_LEVEL","nlte_minus_cmfgen")):
                errors = [abs(r[column]) for r in result if r["Z"]==z and r["stage"]==ion and r[column] is not None]
                metrics.append(dict(Z=z,stage=ion,authority=authority,compared_shells=len(errors),
                    mean_absolute_fraction_difference=math.fsum(errors)/len(errors) if errors else None,
                    max_absolute_fraction_difference=max(errors) if errors else None))
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/"ion_fractions.csv",result)
    write_csv(out/"element_totals.csv",totals)
    inputs = [work/name for name in ("cmfgen_final_publication.json","lumina_spectrum_common_obs.json",
                                    "lumina_ion_pops.csv","nlte_levels_final.csv")]
    inputs += [model/name for name in ("geometry.csv","density.csv","abundances.csv","atom_masses.csv","levels.csv")]
    inputs += [public/f"ionfrac_{name}_toy06_cmfgen.txt" for name in ELEMENTS.values()]
    hashes = {}
    for path in inputs+[Path(__file__)]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1<<20),b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    summary = dict(day=day,population_generation=pg,radiation_generation=obs["radiation_generation"],
        outer_convergence_verified=obs["outer_convergence_verified"],goal_verdict="NOT_JUDGED",
        denominator="rho*abundance/(mass_amu*AMU); no renormalization",interpolation="linear velocity; no extrapolation",
        metrics=metrics,input_sha256=hashes)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("run","model","public","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("판독 계산은 Slurm에서 실행한다")
    summary = analyze(args.run,args.model,args.public,args.out)
    print(json.dumps(summary["metrics"],ensure_ascii=False))
