#!/usr/bin/env python3
"""Build CMFGEN-style full-level BB collision strengths for Si/S at one material point.

This is an uncommitted physical input to the private model SE path, not a new
CMFGEN field, SE solution, or production material update.  The table reader,
GENOSC weak-line cut, SAME_N and OMEGA_GEN_V3 branches are kept explicit so
their separate contributions can be compared before using the coefficients.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

import build_cmfgen_coldata_all as col
import cmfgen_model_continuum as material
from cmfgen_parser import parse_col as parse_col_header, parse_osc
from expand_atomic_data_cmfgen import load_cmfgen_links, load_cmfgen_model_spec
from l1a_collision import _ex_e1x

IONS = ((14, 1, "Si II", 406), (14, 2, "Si III", 178),
        (14, 3, "Si IV", None), (14, 4, "Si V", 0),
        (16, 1, "S II", 2415), (16, 2, "S III", 227),
        (16, 3, "S IV", None), (16, 4, "S V", None))
KB_EV = 8.617333262145e-5
EV_ERG = 1.602176634e-12
H_CGS = 6.62607015e-27
GF_CUT = 0.003
GF_LEV_CUT = 9
MIN_TRANS = 10


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def same_n_key(name):
    """CMFGEN subs/same_n.f POSN_SAME, retaining its core-prefix comparison."""
    def char(k):  # Fortran one-based substring index
        return name[k-1] if 1 <= k <= len(name) else " "

    special = name.find("__")
    if special < 0:
        special = max(name.find("SNG"), name.find("TRP"))
    if special >= 0:
        end = special  # one-based index of the character before marker
        start = end
        for _ in range(2):
            if start > 1 and "0" < char(start-1) < "9":
                start -= 1
        return name[start-1:end], name[:start]

    bracket = name.find("[")
    last = bracket if bracket >= 0 else len(name.rstrip())
    if char(last) in "eo":
        last -= 1
    j = last-3
    if not "d" <= char(j+1) <= "s":
        j -= 1
    while j > 0:
        if char(j) == ")":
            while j > 0 and char(j) != "(":
                j -= 1
            j -= 1
        if "0" <= char(j) <= "9" and "d" <= char(j+1) <= "s":
            end = start = j
            for _ in range(2):
                if start > 1 and "0" <= char(start-1) <= "9":
                    start -= 1
            return name[start-1:end], name[:start]
        j -= 1
    return None


def selected_controls(path):
    text = path.read_text(encoding="latin-1")
    result = {}
    for tag in ("GF_CUT", "AT_CUT", "GF_LEV_CUT", "MIN_TRANS"):
        matches = re.findall(r"^\s*(\S+)\s+\["+tag+r"\]", text, re.M)
        require(len(matches) == 1, f"MODEL {tag} 신원 누락/중복")
        result[tag] = float(matches[0].replace("D", "E"))
    require(result == {"GF_CUT": GF_CUT, "AT_CUT": 10., "GF_LEV_CUT": GF_LEV_CUT,
                       "MIN_TRANS": MIN_TRANS}, "다른 GENOSC 절단 제어")
    return result


def weak_line_cut(f, energy, g):
    """GENOSC_V9 lower I>LEV_CUT, upper J>max(LEV_CUT,MIN_TRANS)."""
    n = len(g)
    modified = 0
    for up in range(max(GF_LEV_CUT, MIN_TRANS), n):
        dE = (energy[up]-energy[:up])*EV_ERG
        nu = dE/H_CGS
        spontaneous = np.where(dE > 0., f[:up, up]*g[:up]/g[up]*nu*nu, 0.)
        tenth = np.partition(spontaneous, len(spontaneous)-MIN_TRANS)[len(spontaneous)-MIN_TRANS]
        for lo in range(GF_LEV_CUT, up):
            if g[lo]*f[lo, up] < GF_CUT and spontaneous[lo] < tenth:
                modified += int(f[lo, up] != 0.)
                f[lo, up] = 0.
    return modified


def table_at_temperature(table, tgrid, temperature_K):
    require(len(table) == len(tgrid) and len(tgrid) >= 2 and
            min(table) > 0. and all(tgrid[i] < tgrid[i+1]
                                     for i in range(len(tgrid)-1)),
            "CMFGEN Ω 표/온도 격자 오류")
    t4 = temperature_K/10000.
    if t4 <= tgrid[0]:
        return table[0]
    if t4 >= tgrid[-1]:
        return table[-1]
    upper = int(np.searchsorted(tgrid, t4, side="left"))
    lower = upper-1
    alpha = math.log(table[upper]/table[lower])/math.log(tgrid[upper]/tgrid[lower])
    return table[lower]*(t4/tgrid[lower])**alpha


def ion_strengths(Z, ion0, name, expected_tab, ids, rows, models, links, temperature):
    n = len(ids)
    require(n == int(models[Z, ion0+1]["n_full"]) and
            [int(rows[k]["level_number"]) for k in ids] == list(range(n)),
            f"{name} FL 신원/순서 불일치")
    osc_path, col_path = links[Z, ion0+1]["osc"], links[Z, ion0+1]["col"]
    osc = parse_osc(osc_path)
    names = [str(s) for s in osc.levels["config"][:n]]
    energy = np.array([float(rows[k]["energy_eV"]) for k in ids])
    g = np.array([float(rows[k]["g"]) for k in ids])
    require(np.all(g > 0.) and np.all(np.diff(energy) >= 0.), f"{name} 에너지/g 오류")
    for l,k in enumerate(ids):
        require(rows[k]["configuration"] == names[l] and
                g[l] == float(osc.levels["g"][l]) and
                abs(energy[l]-float(osc.levels["E_cm"][l])*1.239841984e-4) < 5.1e-11,
                f"{name} OSC/FL 이름·에너지·g 불일치")
    f = np.zeros((n,n), dtype="f8")
    raw_pairs = set()
    for tr in osc.transitions:
        lo, up = int(tr["i"])-1, int(tr["j"])-1
        if lo < n and up < n:
            require(0 <= lo < up and math.isfinite(float(tr["f"])),
                    f"{name} OSC 전이 인덱스/강도 오류")
            f[lo,up] = abs(float(tr["f"]))  # GENOSC: repeated ordered pair last wins
            raw_pairs.add((lo,up))
    cut = weak_line_cut(f, energy, g)
    header = parse_col_header(col_path)
    if header.n_transitions:
        name_to_index, _, weights, full_n = col.parse_osc(str(osc_path))
        table, grid, nt, stats = col.parse_col(str(col_path), name_to_index, weights, full_n)
        table = {(i,j):value for (i,j),value in table.items() if i < n and j < n}
    else:
        table,grid,nt,stats = {},[],0,{"n_read":0,"n_decl":0,"n_nonpos":0,"n_term_reject":0}
    require((expected_tab is None or len(table) == expected_tab) and
            stats["n_read"] == stats["n_decl"] and
            stats["n_nonpos"] == 0 and stats["n_term_reject"] == 0 and
            len(grid) == nt and header.default_omega > 0 and
            header.scale_factor > 0 and math.isfinite(header.scale_factor),
            f"{name} native 표/기본값 오류")
    lo, up = np.triu_indices(n,1)
    omega = np.empty(len(lo), dtype="<f8")
    branch = {"TABULATED":0,"VR":0,"OMEGA_SET":0}
    same_n_vr_pairs = 0
    keys = [same_n_key(label) for label in names]
    for p,(i,j) in enumerate(zip(lo,up,strict=True)):
        i,j=int(i),int(j)
        pair=(i,j)
        if pair in table:
            value=table_at_temperature(table[pair],grid,temperature)
            kind="TABULATED"
        elif f[i,j] > 1.e-5:
            delta=energy[j]-energy[i]
            require(delta > 0., f"{name} f>0 동률 에너지 전이")
            x=delta/(KB_EV*temperature)
            if ion0 == 0:
                gbar=0.276*_ex_e1x(x) if x <= 14. else 0.066*(1.+1.5/x)/math.sqrt(x)
            else:
                same = keys[i] is not None and keys[i] == keys[j]
                same_n_vr_pairs += int(same)
                gbar=max(0.7 if same else 0.2,0.276*_ex_e1x(x))
            fl=delta*EV_ERG/(H_CGS*1.e15)
            value=47.972*header.scale_factor*gbar*f[i,j]*g[i]/fl
            kind="VR"
        else:
            value=header.default_omega
            kind="OMEGA_SET"
        require(math.isfinite(value) and value > 0., f"{name} Ω 비유한/음수")
        omega[p]=value
        branch[kind]+=1
    require(sum(branch.values()) == n*(n-1)//2,
            f"{name} 전쌍 coverage 실패")
    return lo.astype("<i4"),up.astype("<i4"),omega,dict(
        ion=name,full_levels=n,pairs=len(omega),raw_osc_pairs=len(raw_pairs),
        GF_culled_lines=cut,tabulated_pairs=len(table),branches=branch,
        same_n_vr_pairs=same_n_vr_pairs,
        omega_set=header.default_omega,omega_scale=header.scale_factor,
        osc_sha256=material.sha(osc_path),col_sha256=material.sha(col_path))


def run(args):
    out=Path(args.out)
    require(not out.exists(), "기존 전쌍 출력 보존")
    atomic=Path(args.atomic)
    seed=json.loads((atomic.parent/"material-seed.json").read_text())
    require(seed["source_record"] == 70 and seed["material_points"] == 90 and
            material.sha(args.material_nodes) == seed["output_sha256"]["native_material_nodes.bin"],
            "다른 원 물질70/90점")
    rows=list(csv.DictReader((atomic/"levels.csv").open(newline="")))
    with args.material_nodes.open("rb") as stream:
        magic,ns,_,nl,_=struct.unpack("<8sIIId",stream.read(28))
    require(magic in (b"LUMNOD01",b"LUMCEL01") and ns == 90 and nl == len(rows) and
            0 <= args.node < ns,"물질 형상 오류")
    links=load_cmfgen_links(args.material_run/"atomic_links.txt")
    models=load_cmfgen_model_spec(args.material_run/"MODEL_SPEC",
                                  args.material_run/"atomic_links.txt")
    controls=selected_controls(args.material_run/"MODEL")
    source_paths=(args.material_nodes,atomic/"levels.csv",
                  atomic.parent/"material-seed.json",args.material_run/"MODEL",
                  args.material_run/"MODEL_SPEC",args.material_run/"atomic_links.txt",
                  Path(__file__),
                  *(links[Z,ion0+1][kind] for Z,ion0,_,_ in IONS
                    for kind in ("osc","col")))
    input_sha256={str(path):material.sha(path) for path in source_paths}
    # Uses the same read-only native material parser as the established point calculation.
    occupation=np.ones((nl,ns),dtype="f8")
    state=material.read_material(args.material_nodes,rows,models,occupation)
    temperature=float(state["temperature"][args.node])
    ne=float(state["ne"][args.node])
    require(math.isfinite(temperature) and temperature > 0 and
            math.isfinite(ne) and ne > 0,"물질 T/ne 오류")
    payload=[];summary=[]
    for Z,ion0,name,expected in IONS:
        ids=[i for i,r in enumerate(rows) if int(r["atomic_number"]) == Z and
             int(r["ion_number"]) == ion0]
        lo,up,omega,info=ion_strengths(Z,ion0,name,expected,ids,rows,models,links,temperature)
        offset=ids[0]
        require(ids == list(range(offset,offset+len(ids))), f"{name} 비연속 global FL")
        payload.append((lo+offset,up+offset,omega))
        summary.append(info)
    ordered=[item for trio in payload for item in zip(*trio,strict=True)]
    require(all(ordered[i][:2] < ordered[i+1][:2] for i in range(len(ordered)-1)),
            "원소/이온별 FL 쌍 순서 오류")
    require(all(material.sha(Path(path)) == digest for path,digest in input_sha256.items()),
            "전쌍 계산 중 입력/소스 변경")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as stream:
        stream.write(struct.pack("<8sIIII2d",b"LMCBPTS1",nl,ns,args.node,len(ordered),temperature,ne))
        for lower,upper,omega in ordered:
            stream.write(struct.pack("<iid",int(lower),int(upper),float(omega)))
    result=dict(schema="lumina-cmfgen-fullpair-omega-v1",state="PRIVATE_INPUT_ONLY",
                material_record=70,native_node=args.node,temperature_K=temperature,
                electron_density_cm3=ne,levels=nl,pairs=len(ordered),
                controls=controls,ions=summary,output_sha256=material.sha(out),
                input_sha256=input_sha256,
                exclusions=["용해 이온화/CI/C3","새 SE/RE", "새 복사장/수송"],
                final_goal="NOT_MET")
    meta=out.with_suffix(out.suffix+".json")
    require(not meta.exists(),"기존 manifest 보존")
    meta.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic",type=Path,required=True)
    parser.add_argument("--material-run",type=Path,required=True)
    parser.add_argument("--material-nodes",type=Path,required=True)
    parser.add_argument("--node",type=int,default=49)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args())
