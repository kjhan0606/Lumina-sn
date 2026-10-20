#!/usr/bin/env python3
"""공개 toy06의 Si 모델 크기 차이를 고정 율계에서 분리한다.

논문 표6의 62/61준위·마지막 항에 대응하는 현행 원자의 부분집합이다.
공개 실행의 원자자료 판본, superlevel 묶음, gf cut을 재현한 것은 아니다.
유지 준위 사이의 율만 남기고 제거 준위로 가는 손실도 제거한다.
입력/생산 실행은 읽기 전용이며 기존 자료에만 쓰는 민감도 시험이다.
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


def restrict_generator(matrix, keep):
    if np.all(keep):
        return matrix.copy()
    out = matrix[np.ix_(keep, keep)].copy()
    np.fill_diagonal(out, 0.0)
    # An absorbing sink into discarded states would be a different experiment.
    np.fill_diagonal(out, -np.sum(out, axis=0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    tr = args.run / "si_trace"
    inputs = {}

    def read(name):
        p = tr / name
        inputs[str(p)] = TR.sha(p)
        return TR.read_trace(p)

    call, why = EX.det_si_exc_adopt(read("si10_ne.csv"), read("si10_calls.csv"),
                                  read("si10_cevec.csv"), read("si10_commit.csv"), EX.ADOPTED)
    if call is None or why:
        raise ValueError(why)
    cid, n, gh = call["call"], int(call["N"]), int(call["ground_hi"])
    sm = sorted((r for r in read("si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    ops = [r for r in read("si10_ops.csv") if r["call"] == cid]
    path = tr / f"si10_mat_{int(cid):06d}_gen.bin"
    inputs[str(path)] = TR.sha(path)
    saved, rhs, _ = TR.read_matrix(path)
    replay, rb = TR.replay(ops, n)
    if not np.array_equal(saved, replay) or not np.array_equal(rhs, rb):
        raise ValueError("trace replay failed")
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    lev = np.array([int(r["level_num"]) for r in sm])
    if (sum(~upper), sum(upper)) != (125, 147):
        raise ValueError("unexpected full atom")
    raw = Path(__file__).resolve().parents[1] / "data/atomic/cmfgen/SIL"
    last = {}
    for ion, cap, isupper in (("II", 62, False), ("III", 61, True)):
        p = raw / ion / "19apr23/osc_data"
        inputs[str(p)] = TR.sha(p)
        atom = P.parse_osc(p)
        for r in (r for r in sm if (r["upper_ion"] == "1") == isupper):
            level = atom.levels[int(r["level_num"])]
            if abs(float(r["E_eV"]) - level["E_cm"] * P._CM2EV) > 1e-7 or float(r["g"]) != level["g"]:
                raise ValueError("raw/trace level identity")
        last[ion] = str(atom.levels[cap - 1]["config"])
    if last != {"II": "3s2_7g_2Ge[7/2]", "III": "3s5g3Ge[5]"}:
        raise ValueError(("published last term mismatch", last))
    total = float(next(r for r in read("si10_final.csv") if r["call"] == cid)["n_total"])
    refs = json.loads(EX.PREVIOUS.read_text())["ratios"]
    inputs[str(EX.PREVIOUS)] = TR.sha(EX.PREVIOUS)
    # Analytic three-state test: dropping state2 must remove both directions,
    # leave the 0->1 rate2 and 1->0 rate3, and preserve both retained diagonals.
    toy = np.array([[-7., 3., 7.], [2., -14., 13.], [5., 11., -20.]])
    if not np.array_equal(restrict_generator(toy, np.array([True, True, False])),
                          np.array([[-2., 3.], [2., -3.]])):
        raise ValueError("restriction test failed")
    states = {}
    for use_dr in (True, False):
        mat, _, bad, _, _ = TR.det_si_exc_build_fl(
            ops if use_dr else [r for r in ops if r["site"] != "8"], sm, n, gh)
        if bad:
            raise ValueError(bad)
        for nlow, nhi in ((125, 147), (62, 147), (125, 61), (62, 61)):
            keep = ((~upper) & (lev < nlow)) | (upper & (lev < nhi))
            sub = restrict_generator(mat, keep)
            pop, bad, error = EX.det_si_exc_solve(sub, total)
            if bad:
                raise ValueError((nlow, nhi, use_dr, bad))
            up = upper[keep]
            ratio = TR.ratio(pop, up)
            gamma = float(np.sum(sub[np.ix_(up, ~up)] @ pop[~up]) / pop[~up].sum())
            recomb = float(np.sum(sub[np.ix_(~up, up)] @ pop[up]) / pop[up].sum())
            states[f"II{nlow}_III{nhi}_DR{int(use_dr)}"] = dict(
                III_II=ratio, delta_F0_dex=TR.dex(ratio, refs["R_F"]),
                delta_public_dex=TR.dex(ratio, refs["public"]),
                ionization_per_II_s=gamma, recombination_per_III_s=recomb,
                backward_error=error)
    if not math.isclose(states["II125_III147_DR1"]["III_II"], refs["R_F"], rel_tol=1e-12):
        raise ValueError("F0 not reproduced")
    print(json.dumps(dict(job=os.environ.get("SLURM_JOB_ID"),
                         scope="fixed material13/radiation12; model-size sensitivity, not public atom reproduction",
                         public_model_source="https://arxiv.org/html/2209.11671v2#S0.T6",
                         retained_last_config=last, states=states, input_sha256=inputs,
                         code_sha256={str(Path(m.__file__)): TR.sha(m.__file__) for m in (P, EX, TR)},
                         script_sha256=TR.sha(__file__)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
