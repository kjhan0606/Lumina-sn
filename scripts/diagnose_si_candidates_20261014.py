#!/usr/bin/env python3
"""저장된 Si 셸10 상태에서 인구 가중 차이와 전체 C1 처방을 확인한다.

생산 상태는 쓰지 않는다. stdout JSON은 고정 J/T/ne의 조건부 결과다.
CMFGEN 원자자료가 없는 18준위에는 봉인 Lumina의 기존 Kramers 처방을
사용한다. 이 보충을 CMFGEN 계수 확보나 자기일관 물리 검증으로 부르지 않는다.
"""
import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np

import det_si_excitation_budget as EX
import det_si_trace_readout as TR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    tr = args.run / "si_trace"
    read = lambda name: TR.read_trace(tr / name)
    call, problems = EX.det_si_exc_adopt(
        read("si10_ne.csv"), read("si10_calls.csv"), read("si10_cevec.csv"),
        read("si10_commit.csv"), EX.ADOPTED)
    if problems or call is None:
        raise ValueError(problems)
    cid = call["call"]
    sm = sorted((r for r in read("si10_slmap.csv") if r["call"] == cid),
                key=lambda r: int(r["i_fl"]))
    ops = [r for r in read("si10_ops.csv") if r["call"] == cid]
    n, gh = int(call["N"]), int(call["ground_hi"])
    original, rhs, _ = TR.read_matrix(tr / f"si10_mat_{int(cid):06d}_gen.bin")
    replay, replay_rhs = TR.replay(ops, n)
    if not (np.array_equal(original, replay) and np.array_equal(rhs, replay_rhs)):
        raise ValueError("recorded operator not reproduced")
    a0, _, unknown, _, maps = TR.det_si_exc_build_fl(ops, sm, n, gh)
    if unknown or maps["gh_fl"] is None or any(r["site"] == "6" for r in ops):
        raise ValueError((unknown, "C1 must be absent at baseline"))
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    final = {int(r["i"]): float(r["value"]) for r in read("si10_vec.csv")
             if r["call"] == cid and r["phase"] == "FINAL_FL"}
    if set(final) != set(range(len(sm))):
        raise ValueError("incomplete FINAL_FL")
    population = np.array([final[i] for i in range(len(sm))])
    total = float(next(r for r in read("si10_final.csv") if r["call"] == cid)["n_total"])
    bf = {}
    for r in ops:
        if r["site"] != "5":
            continue
        fl = maps["fl_of_global"][int(r["lev_b"])]
        if fl in bf:
            raise ValueError("duplicate BF level")
        bf[fl] = r
    if set(bf) != set(np.flatnonzero(~upper)) or len(bf) != 125:
        raise ValueError("BF coverage")
    ladder = [r for r in read("si10_ladder.csv")
              if r["bundle"] == call["bundle"] and r["ne_iter"] == call["ne_iter"]]
    adopted_ladder = ([r for r in ladder if r["ch_armed"] == "1"] or ladder)[-1]
    levels = {int(r["global_level"]): r for r in read("si10_ladder_levels.csv")
              if r["lseq"] == adopted_ladder["lseq"]}
    if set(levels) != {int(r["lev_a"]) for r in bf.values()}:
        raise ValueError("ladder/pair level identities")
    coefficient_equal = all(float(r["x1"]) == float(levels[int(r["lev_a"])]["gamma_level"])
                            for r in bf.values())
    gamma_lte = math.fsum(float(r["f_lev"]) * float(r["gamma_level"]) for r in levels.values())
    gamma_nlte = math.fsum(float(r["x1"]) * final[i] for i, r in bf.items()) / population[~upper].sum()

    sigma_path = args.run / "input/model/cmfgen_sigma_bf.bin"
    edges = EX.det_si_exc_sigma_edges(sigma_path, sorted(levels))
    with sigma_path.open("rb") as f:
        _, _, nlev, nfreq = struct.unpack("<IIii", f.read(16))
        nu_min, nu_max = struct.unpack("<dd", f.read(16))
        flags = f.read(nlev)
    dlog = math.log(nu_max / nu_min) / nfreq
    centers = [math.exp(math.log(nu_min) + b * dlog + 0.5 * dlog) for b in range(nfreq)]
    T, ne = float(call["T_e"]), float(call["n_e"])
    gh_fl = maps["gh_fl"]
    partial, complete, fallback = [], [], []
    saha = {}
    for fl, r in bf.items():
        gl, nu, nstar = int(r["lev_a"]), float(r["x6"]), float(r["x5"])
        sig = edges[gl]
        saha[fl] = EX.det_si_exc_saha(T, ne, int(sm[fl]["g"]), int(sm[gh_fl]["g"]), nu)
        if not math.isclose(saha[fl], nstar, rel_tol=1e-9):
            raise ValueError("recorded inverse/Saha mismatch")
        if sig is not None:
            partial.append((fl, sig, nu, nstar))
        else:
            if flags[gl] != 0:
                raise ValueError("nonpositive populated CMFGEN row cannot become Kramers")
            first = next(v for v in centers if v >= nu)
            # get_bf_sigma0(14,1)=1e-18; pair uses first eligible bin, not threshold.
            sig = 1.0e-18 * math.pow(nu / first, 3.0)
            fallback.append({"level_num": int(sm[fl]["level_num"]), "global_level": gl,
                             "nu_threshold": nu, "nu_first_bin": first, "sigma_edge": sig})
        complete.append((fl, sig, nu, nstar))
    if len(partial) != 107 or len(fallback) != 18:
        raise ValueError("unexpected sigma coverage")
    states = {}
    for name, lv, strength in (("F0", [], 1), ("C1_107", partial, 1),
                               ("C1_125", complete, 1), ("C1_125_zero", complete, 0)):
        extra, undefined = EX.det_si_exc_c1(lv, T, ne, gh_fl, strength)
        if undefined or EX.det_si_exc_detailed_balance(extra, saha):
            raise ValueError((name, undefined, "detailed balance"))
        a, _, undefined, _, _ = TR.det_si_exc_build_fl(ops, sm, n, gh, extra=extra)
        if undefined:
            raise ValueError(undefined)
        if name == "C1_125_zero" and not np.array_equal(a, a0):
            raise ValueError("zero intervention changes operator")
        pop, why, error = EX.det_si_exc_solve(a, total)
        if why:
            raise ValueError((name, why))
        states[name] = {"III_II": TR.ratio(pop, upper), "max_backward_error": error,
                        "included_levels": len(lv)}
    reference = json.loads(EX.PREVIOUS.read_text())
    if abs(TR.dex(states["F0"]["III_II"], reference["ratios"]["R_F"])) > 1e-10:
        raise ValueError("baseline not reproduced")
    previous_partial = 3.559773862125424  # saved excitation table F1; verification only
    if abs(TR.dex(states["C1_107"]["III_II"], previous_partial)) > 1e-10:
        raise ValueError("107-level result not reproduced")
    for state in states.values():
        state["delta_F0_dex"] = TR.dex(state["III_II"], states["F0"]["III_II"])
        state["delta_public_dex"] = TR.dex(state["III_II"], reference["ratios"]["public"])
    report = {"scope": "Si shell10 generation13 call36, fixed J/T/ne; no production mutation",
              "job": os.environ.get("SLURM_JOB_ID"), "run": str(args.run), "T_e": T, "n_e": ne,
              "adopted": EX.ADOPTED, "per_level_coefficients_equal": coefficient_equal,
              "levels_compared": len(bf), "gamma_LTE": gamma_lte, "gamma_NLTE": float(gamma_nlte),
              "gamma_ratio": float(gamma_lte / gamma_nlte), "final_III_II": TR.ratio(population, upper),
              "ladder_III_II": float(adopted_ladder["r_final"]),
              "public_III_II": reference["ratios"]["public"], "C1_states": states,
              "fallback_prescription": "existing Lumina Si II sigma0=1e-18; not measured CMFGEN data",
              "fallback_levels": fallback,
              "reader_sha256": {Path(m.__file__).name: TR.sha(m.__file__) for m in (TR, EX)},
              "script_sha256": TR.sha(__file__)}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
