#!/usr/bin/env python3
"""고정 Si 율계에서 명시적 영 A 경로와 누락 B 경로를 분리한다.

기록 율을 재현한 경우에만 같은 저장 J로 B의 정·역율을 추가한다.
공개 이온비로 계수를 맞추지 않으며 생산 입력과 코드는 쓰지 않는다.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import pandas as pd

import bake_cmfgen_sigma_bf_carsus as BA
import cmfgen_parser as PARSER
import expand_atomic_data_cmfgen as AVG
import det_si_excitation_budget as EX
import det_si_trace_readout as TR

H, KB, C = 6.62607015e-27, 1.380649e-16, 2.99792458e10


def entries_by_level(phot, configs):
    exact, terms = defaultdict(list), defaultdict(list)
    for entry in phot.entries:
        exact[AVG._norm_cfg(entry.config)].append(entry)
        terms[AVG._term_cfg(entry.config)].append(entry)
    return {gl: exact.get(AVG._norm_cfg(cfg)) or terms.get(AVG._term_cfg(cfg), [])
            for gl, cfg in configs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    tr = args.run / "si_trace"
    read = lambda name: TR.read_trace(tr / name)
    call, why = EX.det_si_exc_adopt(read("si10_ne.csv"), read("si10_calls.csv"),
                                  read("si10_cevec.csv"), read("si10_commit.csv"), EX.ADOPTED)
    if why or call is None:
        raise ValueError(why)
    cid = call["call"]
    sm = sorted((r for r in read("si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    ops = [r for r in read("si10_ops.csv") if r["call"] == cid]
    n, gh = int(call["N"]), int(call["ground_hi"])
    matrix_path = tr / f"si10_mat_{int(cid):06d}_gen.bin"
    original, rhs, _ = TR.read_matrix(matrix_path)
    replay, rb = TR.replay(ops, n)
    if not (np.array_equal(original, replay) and np.array_equal(rhs, rb)):
        raise ValueError("original operator mismatch")
    a0, _, bad, channels0, maps = TR.det_si_exc_build_fl(ops, sm, n, gh)
    if bad:
        raise ValueError(bad)
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    total = float(next(r for r in read("si10_final.csv") if r["call"] == cid)["n_total"])
    ref = json.loads(EX.PREVIOUS.read_text())["ratios"]
    T, ne = float(call["T_e"]), float(call["n_e"])
    raw = BA.CMFGEN_ROOT / "SIL/II/19apr23"
    osc = BA.parse_osc(raw / "osc_data")
    pa, pb = BA.parse_phot(raw / "phot_data_A"), BA.parse_phot(raw / "phot_data_B")
    df = pd.DataFrame([{"global_idx": int(r["global_level"]), "energy_eV": float(r["E_eV"]), "g": int(r["g"])}
                       for r in sm if r["upper_ion"] == "0"])
    configs = BA.match_levels_to_cmfgen(df, osc.levels)
    if len(configs) != 125:
        raise ValueError("lower-level identity incomplete")
    ea, eb = entries_by_level(pa, configs), entries_by_level(pb, configs)
    explicit_zero = {gl for gl, entries in ea.items() if entries and all(
        e.cs_type == 1 and len(e.sigma_Mb) > 0 and np.all(e.sigma_Mb == 0.0) for e in entries)}
    bf = [r for r in ops if r["site"] == "5"]
    if len(explicit_zero) != 18 or len(bf) != 125:
        raise ValueError("unexpected A coverage")
    sigma_path = args.run / "input/model/cmfgen_sigma_bf.bin"
    with sigma_path.open("rb") as f:
        magic, ver, nl, nf = struct.unpack("<IIii", f.read(16))
        numin, numax = struct.unpack("<dd", f.read(16))
        flags = f.read(nl)
    if magic != 0x434D4644 or ver != 1 or any(flags[gl] for gl in explicit_zero):
        raise ValueError("explicit zero/flag mismatch")
    sigma = np.memmap(sigma_path, mode="r", dtype="<f8", offset=32 + nl + (-nl) % 8, shape=(nl, nf))
    dlog = math.log(numax / numin) / nf
    edges = np.exp(math.log(numin) + np.arange(nf + 1) * dlog)
    nu = np.exp(math.log(numin) + (np.arange(nf) + .5) * dlog)
    width = np.diff(edges)
    spectral = args.run / "work/physics_comparison/physics_DET_iter0010.spectral.csv"
    sr = sorted((r for r in TR.read_trace(spectral) if r["shell_id"] == "10"), key=lambda r: int(r["bin_id"]))
    if len(sr) != nf or not np.allclose([float(r["nu_lo_Hz"]) for r in sr], edges[:-1], rtol=1e-12, atol=0):
        raise ValueError("stored J grid mismatch")
    J = np.array([float(r["J_nu"]) for r in sr])

    def rates(sig, threshold, nstar):
        active = edges[1:] > threshold
        lo = np.maximum(edges[:-1][active], threshold)
        hi = edges[1:][active]
        step = sig[active] * width[active] / (hi - lo)
        gamma = 4 * math.pi / H * np.sum(J[active] * step * np.log(hi / lo))
        milne_nu = np.where((nu < threshold) & (edges[1:] > threshold), (threshold + edges[1:]) / 2, nu)
        spontaneous = 2 * H * milne_nu ** 3 / C ** 2
        inv_kernel = 4 * math.pi * sig / (H * nu) * width * np.exp(-H * milne_nu / (KB * T))
        rec = nstar * np.sum(inv_kernel * (spontaneous + J))
        return float(gamma), float(rec)

    replays = []
    for r in bf:
        gl = int(r["lev_a"])
        if not flags[gl]:
            continue
        gamma, rec = rates(sigma[gl], float(r["x6"]), float(r["x5"]))
        replays.append({"global_level": gl,
                        "gamma_rel": abs(gamma - float(r["x1"])) / max(gamma, float(r["x1"])),
                        "rec_rel": abs(rec - float(r["x3"])) / max(rec, float(r["x3"]))})
    # Stored field is a valid replacement only if it reproduces both directions.
    field_ok = max(max(r["gamma_rel"], r["rec_rel"]) for r in replays) <= 1e-6
    extra, route_rows = [], []
    if field_ok:
        hi_osc = BA.parse_osc(BA.CMFGEN_ROOT / "SIL/III/19apr23/osc_data")
        # Keep an independent raw-header cross-check of the repaired parser.
        # An empty target must never silently select the ground state.
        final_state = next(line.split("!", 1)[0].strip() for line in (raw / "phot_data_B").read_text().splitlines()
                           if "!Final state in ion" in line)
        if pb.final_state != final_state:
            raise ValueError("parser lost or changed the B target identity")
        aliases = {AVG._norm_cfg(s).replace("_", "") for s in final_state.split("/")}
        target = [i for i, lev in enumerate(hi_osc.levels)
                  if AVG._term_cfg(lev["config"]).replace("_", "") in aliases]
        hi_map = {int(r["level_num"]): int(r["i_fl"]) for r in sm if r["upper_ion"] == "1"}
        if target != [1, 2, 3] or any(i not in hi_map for i in target):
            raise ValueError(("B target must be the Si III triplet fine-structure states",
                              target, sorted(aliases), [i for i in target if i not in hi_map]))
        gsum = sum(int(hi_osc.levels[i]["g"]) for i in target)
        if gsum != pb.final_state_g:
            raise ValueError("B statistical weight mismatch")
        raw_configs = {v["config"]: v for v in osc.levels}
        for r in bf:
            gl = int(r["lev_a"])
            fl = maps["fl_of_global"][int(r["lev_b"])]
            lower = raw_configs[configs[gl]]
            for hi in target:
                gu = int(hi_osc.levels[hi]["g"])
                exc = float(hi_osc.levels[hi]["E_cm"]) * 1.239841984e-4
                chi = osc.ionization_eV - float(lower["E_cm"]) * 1.239841984e-4 + exc
                threshold = chi * BA.EV_TO_ERG / H
                sig = np.zeros(nf)
                for entry in eb[gl]:
                    model = AVG._sigma_model(entry.cs_type, entry.energy, entry.sigma_Mb, threshold,
                                             zion=osc.z_screen, nef=AVG._cmfgen_nef(14, osc.z_screen, threshold))
                    if model is not None:
                        fn, knots, start, _ = model
                        sig += AVG._bin_average_sigma(fn, knots, edges, width, start)
                nstar = EX.det_si_exc_saha(T, ne, int(lower["g"]), gu, threshold)
                gamma, rec = rates(sig * gu / gsum, threshold, nstar)
                if gamma == 0 and rec == 0:
                    continue
                extra.extend(((hi_map[hi], fl, gamma, "B ionization"), (fl, hi_map[hi], rec, "B recombination")))
                route_rows.append({"lower": int(sm[fl]["level_num"]), "upper": hi, "p": gu / gsum,
                                   "threshold_Hz": threshold, "gamma": gamma, "rec": rec})
    states = {}
    for remove_dr in (False, True):
        for correct_a in (False, True):
            for add_b in (False, True):
                if add_b and not field_ok:
                    continue
                selected = [r for r in ops if not (remove_dr and r["site"] == "8") and not (
                    correct_a and r["site"] == "5" and int(r["lev_a"]) in explicit_zero)]
                a, _, bad, channels, _ = TR.det_si_exc_build_fl(selected, sm, n, gh, extra=extra if add_b else ())
                if bad:
                    raise ValueError(bad)
                for k, v in channels0.items():
                    if k not in ("DR", "BF ionization", "BF recombination") and not np.array_equal(v, channels[k]):
                        raise ValueError("non-target channel changed")
                pop, bad, error = EX.det_si_exc_solve(a, total)
                if bad:
                    raise ValueError(bad)
                name = f"A{'_zero' if correct_a else '_legacy'}_B{int(add_b)}_DR{int(not remove_dr)}"
                ratio = TR.ratio(pop, upper)
                states[name] = {"III_II": ratio, "delta_F0_dex": TR.dex(ratio, ref["R_F"]),
                                "delta_public_dex": TR.dex(ratio, ref["public"]), "backward_error": error}
    if abs(states["A_legacy_B0_DR1"]["delta_F0_dex"]) > 1e-10:
        raise ValueError("F0 not reproduced")
    report = {"job": os.environ.get("SLURM_JOB_ID"), "scope": "Si shell10 generation13 call36, fixed J/T/ne, no production mutation",
              "explicit_zero_A_levels": sorted(explicit_zero), "field_reproduces_recorded_A": field_ok,
              "generic_parser_B_final_state": pb.final_state,
              "A_rate_replay": replays, "B_routes": route_rows, "states": states,
              "target_rule": "CMFGEN get_num_phot_routes_v1: resolve actual upper levels; p=g_target/sum_g, target-specific edge and inverse weight",
              "helper_sha256": {str(Path(m.__file__)): TR.sha(m.__file__)
                                for m in (BA, PARSER, AVG, EX, TR)},
              "script_sha256": TR.sha(__file__), "input_sha256": {str(p): TR.sha(p) for p in (
                  raw / "osc_data", raw / "phot_data_A", raw / "phot_data_B", spectral, matrix_path,
                  BA.CMFGEN_ROOT / "SIL/III/19apr23/osc_data",
                  tr / "si10_ops.csv", tr / "si10_slmap.csv", sigma_path)}}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
