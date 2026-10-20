#!/usr/bin/env python3
"""Si DR 총량과 도착 준위를 분리하는 고정 율계 실험.

CMFGEN auto_data/osc_data의 자발복사 cascade를 계산한다. 저장된 NLTE 행렬에서
DR만 교체하며, 이 단순 cascade를 실제 SN의 재결합 처방으로 채택하지 않는다.
생산 실행/입력은 읽기 전용이다.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import cmfgen_parser as P
import det_si_excitation_budget as EX
import det_si_trace_readout as TR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    tr = a.run/"si_trace"
    read = lambda name: TR.read_trace(tr/name)
    call, why = EX.det_si_exc_adopt(read("si10_ne.csv"),read("si10_calls.csv"),
                                  read("si10_cevec.csv"),read("si10_commit.csv"),EX.ADOPTED)
    if call is None or why:
        raise ValueError(why)
    cid, n, gh = call["call"],int(call["N"]),int(call["ground_hi"])
    sm = sorted((r for r in read("si10_slmap.csv") if r["call"] == cid),key=lambda r:int(r["i_fl"]))
    ops = [r for r in read("si10_ops.csv") if r["call"] == cid]
    matrix_path = tr/f"si10_mat_{int(cid):06d}_gen.bin"
    matrix,rhs,_ = TR.read_matrix(matrix_path)
    replay,rb = TR.replay(ops,n)
    if not np.array_equal(matrix,replay) or not np.array_equal(rhs,rb):
        raise ValueError("trace operator not reproduced")
    dr = [r for r in ops if r["site"] == "8"]
    if len(dr) != 1 or int(dr[0]["src"]) != gh or int(dr[0]["dst"]) != 0:
        raise ValueError("unexpected DR statement")
    raw = Path(__file__).resolve().parents[1]/"data/atomic/cmfgen/SIL/II/19apr23"
    osc = P.parse_osc(raw/"osc_data")
    if len(osc.levels) != 157 or len(osc.transitions) != 4232:
        raise ValueError("raw atom changed")
    lower = [r for r in sm if r["upper_ion"] == "0"]
    if len(lower) != 125:
        raise ValueError("retained Si model changed")
    fl = {int(r["level_num"]):int(r["i_fl"]) for r in lower}
    if sorted(fl) != list(range(125)):
        raise ValueError("retained level identity")
    for r in lower:
        raw_level = osc.levels[int(r["level_num"])]
        if abs(float(r["E_eV"])-raw_level["E_cm"]*P._CM2EV) > 1e-7 or float(r["g"]) != raw_level["g"]:
            raise ValueError("raw/trace identity")
    upper_ground = next(int(r["i_fl"]) for r in sm if r["upper_ion"] == "1" and r["level_num"] == "0")
    ai = np.zeros(157)
    auto_ids = []
    for line in (raw/"auto_data").read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in set(osc.levels["config"]):
            continue
        k = list(osc.levels["config"]).index(fields[0])
        auto_ids.append(k)
        rate = float(fields[2].replace("D","E"))
        if osc.levels[k]["E_cm"]*P._CM2EV > osc.ionization_eV:
            ai[k] = rate  # below-threshold entries cannot autoionize to this parent
    if len(auto_ids) != 14:
        raise ValueError("auto inventory changed")
    # Absorption probability at the first retained bound level. Above-threshold
    # autoionization is a loss channel; collisions and incident photons are absent.
    cascade = np.zeros((157,125))
    cascade[:125] = np.eye(125)
    arad = np.zeros(157)
    for k in range(125,157):
        trans = osc.transitions[osc.transitions["j"] == k+1]
        down = [t for t in trans if t["A"] > 0 and t["i"] <= k]
        arad[k] = sum(float(t["A"]) for t in down)
        denom = arad[k]+ai[k]
        if denom <= 0:
            raise ValueError(("omitted level without exit",k))
        for t in down:
            cascade[k] += float(t["A"])/denom*cascade[int(t["i"])-1]
    T,ne = float(call["T_e"]),float(call["n_e"])
    lam3 = (EX.H_PLANCK**2/(2*math.pi*EX.M_ELECTRON*EX.K_BOLTZMANN*T))**1.5
    energies = osc.levels["E_cm"]*P._CM2EV-osc.ionization_eV
    capture = lam3*osc.levels["g"]/2*np.exp(-energies*EX.EV_TO_ERG/(EX.K_BOLTZMANN*T))*ai
    landing = capture @ cascade
    # The 2Fo[7/2] state is the above-threshold component of the term which
    # the photo-A raw header says is missing (its '3Fo' spelling is inconsistent
    # with osc_data/auto_data). Report the spelling issue, do not silently infer
    # that all other auto_data states are absent from the photo cross sections.
    k2f = list(osc.levels["config"]).index("3s_3p(3Po)3d_2Fo[7/2]")
    landing2f = capture[k2f]*cascade[k2f]
    if landing.sum() <= 0 or landing2f.sum() <= 0:
        raise ValueError("no stabilization")
    total = float(next(r for r in read("si10_final.csv") if r["call"] == cid)["n_total"])
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    refs = json.loads(EX.PREVIOUS.read_text())["ratios"]
    old_rate = float(dr[0]["v1"])
    no_dr = [r for r in ops if r["site"] != "8"]
    base,_,bad,base_channels,_ = TR.det_si_exc_build_fl(ops,sm,n,gh)
    if bad:
        raise ValueError(bad)
    dist = {
        "F0":None,
        "DR_removed":np.zeros(125),
        "same_total_raw_cascade":old_rate*landing/landing.sum(),
        "same_total_2Fo_cascade":old_rate*landing2f/landing2f.sum(),
        "raw_auto_partial_cascade":ne*landing,
        "raw_auto_partial_ground":np.eye(1,125)[0]*ne*landing.sum(),
        "2Fo_partial_cascade":ne*landing2f,
    }
    results = {}
    for name,rates in dist.items():
        extra = [] if rates is None else [(fl[k],upper_ground,float(v),"DR landing experiment") for k,v in enumerate(rates) if v > 0]
        mat,_,bad,channels,_ = TR.det_si_exc_build_fl(ops if rates is None else no_dr,sm,n,gh,extra=extra)
        if bad:
            raise ValueError(bad)
        for key,v in base_channels.items():
            if key != "DR" and not np.array_equal(v,channels[key]):
                raise ValueError("non-DR channel changed")
        pop,bad,error = EX.det_si_exc_solve(mat,total)
        if bad:
            raise ValueError(bad)
        ratio = TR.ratio(pop,upper)
        results[name] = dict(III_II=ratio,delta_F0_dex=TR.dex(ratio,refs["R_F"]),
                             delta_public_dex=TR.dex(ratio,refs["public"]),backward_error=error)
    if abs(results["F0"]["delta_F0_dex"]) > 1e-10:
        raise ValueError("F0 mismatch")
    raw_rows = [dict(level=k,config=str(osc.levels[k]["config"]),energy_above_threshold_eV=float(energies[k]),
                     auto_rate=float(ai[k]),Arad_to_lower=float(arad[k]),capture_coefficient=float(capture[k]),
                     stabilization_probability=float(cascade[k].sum()),alpha=float(capture[k]*cascade[k].sum())) for k in auto_ids]
    inputs = [raw/"osc_data",raw/"auto_data",raw/"phot_data_A",matrix_path,tr/"si10_ops.csv",tr/"si10_slmap.csv",EX.PREVIOUS]
    report = dict(scope="fixed Si shell10 material13/radiation12; spontaneous-cascade sensitivity, not a production DR replacement",
                  total_raw_partial_alpha=float(landing.sum()),twoFo_partial_alpha=float(landing2f.sum()),
                  old_alpha=old_rate/ne,states=results,raw_states=raw_rows,
                  landing=[dict(level=k,alpha=float(v),fraction=float(v/landing.sum())) for k,v in enumerate(landing) if v > 0],
                  source_sha256={str(p):TR.sha(p) for p in inputs},
                  code_sha256={str(Path(m.__file__)):TR.sha(m.__file__) for m in (EX,TR,P)},script_sha256=TR.sha(__file__))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
