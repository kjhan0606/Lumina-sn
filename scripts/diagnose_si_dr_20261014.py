#!/usr/bin/env python3
"""고정된 Si 셸10 율계에서 별도 DR 항 유무의 민감도만 확인한다.

DR 제거는 생산 처방이나 물리적 경계값이 아니다. 복사장·T·ne와
DR 이외의 기록 율을 고정하며, 원본과 생산 코드는 변경하지 않는다.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import det_si_excitation_budget as EX
import det_si_trace_readout as TR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    tr = args.run / "si_trace"
    used = {}

    def read(name):
        path = tr / name
        used[str(path)] = TR.sha(path)
        return TR.read_trace(path)

    call, why = EX.det_si_exc_adopt(
        read("si10_ne.csv"), read("si10_calls.csv"), read("si10_cevec.csv"),
        read("si10_commit.csv"), EX.ADOPTED)
    if why or call is None:
        raise ValueError(why)
    cid = call["call"]
    sm = sorted((r for r in read("si10_slmap.csv") if r["call"] == cid),
                key=lambda r: int(r["i_fl"]))
    ops = [r for r in read("si10_ops.csv") if r["call"] == cid]
    n, gh = int(call["N"]), int(call["ground_hi"])
    path = tr / f"si10_mat_{int(cid):06d}_gen.bin"
    used[str(path)] = TR.sha(path)
    original, rhs, _ = TR.read_matrix(path)
    replay, replay_rhs = TR.replay(ops, n)
    if not (np.array_equal(original, replay) and np.array_equal(rhs, replay_rhs)):
        raise ValueError("recorded operator not reproduced")
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    total = float(next(r for r in read("si10_final.csv") if r["call"] == cid)["n_total"])
    dr = [r for r in ops if r["site"] == "8"]
    if len(dr) != 1 or int(dr[0]["src"]) != gh or int(dr[0]["dst"]) != 0:
        raise ValueError("unexpected DR routing")
    reference = json.loads(EX.PREVIOUS.read_text())
    used[str(EX.PREVIOUS)] = TR.sha(EX.PREVIOUS)
    states = {}
    baseline_channels = None
    baseline_operator = None
    for name, statements in (("F0", ops), ("DR_removed", [r for r in ops if r["site"] != "8"])):
        a, _, undefined, channels, _ = TR.det_si_exc_build_fl(statements, sm, n, gh)
        if undefined:
            raise ValueError(undefined)
        if name == "F0":
            baseline_channels, baseline_operator = channels, a
        else:
            if set(channels) != set(baseline_channels) - {"DR"}:
                raise ValueError("non-DR channel missing")
            if not all(np.array_equal(v, baseline_channels[k]) for k, v in channels.items()):
                raise ValueError("non-DR channel changed")
            # Assembling without DR avoids subtractive loss from a large diagonal.
            if not np.allclose(a + baseline_channels["DR"], baseline_operator,
                               rtol=1e-14, atol=0.0):
                raise ValueError("operator difference not DR")
        population, why, error = EX.det_si_exc_solve(a, total)
        if why:
            raise ValueError((name, why))
        ratio = TR.ratio(population, upper)
        states[name] = {"III_II": ratio, "delta_F0_dex": TR.dex(ratio, reference["ratios"]["R_F"]),
                        "delta_public_dex": TR.dex(ratio, reference["ratios"]["public"]),
                        "max_backward_error": error}
    if not math.isclose(states["F0"]["III_II"], reference["ratios"]["R_F"], rel_tol=1e-12):
        raise ValueError("F0 not reproduced")
    print(json.dumps({"scope": "Si shell10 generation13 call36 fixed-rate sensitivity; not a physical repair or bound",
                      "job": os.environ.get("SLURM_JOB_ID"), "run": str(args.run),
                      "public_III_II": reference["ratios"]["public"], "DR_statement": dr[0],
                      "states": states, "input_sha256": used,
                      "code_sha256": {Path(m.__file__).name: TR.sha(m.__file__) for m in (EX, TR)},
                      "script_sha256": TR.sha(__file__)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
