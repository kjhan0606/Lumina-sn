#!/usr/bin/env python3
"""S II 누락 7경로 복원의 고정 상태 단계율 효과. NLTE 해/스펙트럼 예측 아님.

같은 J/T/ne에서 기존 단계 광전리와 RR을 재현한 경우에만 반사실을 보고한다.
생산 입력은 읽기 전용이며 stdout JSON 이외의 산출은 없다.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import pandas as pd

import cmfgen_parser as P
import expand_atomic_data_cmfgen as B
import det_si_trace_readout as TR
from diagnose_si_routes_20261014 import entries_by_level

H, KB, C, ME, EV = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 9.1093837015e-28, 1.602176634e-12


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    model = args.run / "input/model"
    work = args.run / "work"
    prefix = work / "physics_comparison/physics_DET_iter0010"
    manifest = json.loads(Path(str(prefix) + ".manifest.json").read_text())
    if (manifest["population_generation"], manifest["radiation_generation"]) != (13, 12):
        raise ValueError("unexpected saved generations")
    levels = pd.read_csv(model / "levels.csv")
    lo = levels[(levels.atomic_number == 16) & (levels.ion_number == 1)]
    hi = levels[(levels.atomic_number == 16) & (levels.ion_number == 2)]
    vintage = pd.read_csv(model / "atomic_vintage_manifest.csv")
    src = vintage[(vintage.atomic_number == 16) & (vintage.ion_number == 1)].iloc[0]
    osc, phot = P.parse_osc(Path(src.osc_path)), P.parse_phot(Path(src.phot_path))
    np.testing.assert_array_equal(lo.configuration, osc.levels["config"][:len(lo)])
    sigma_path = model / "cmfgen_sigma_bf.bin"
    with sigma_path.open("rb") as f:
        magic, version, nl, nf = struct.unpack("<IIii", f.read(16))
        nmin, nmax = struct.unpack("<dd", f.read(16))
        flags = np.frombuffer(f.read(nl), dtype="u1")[lo.index]
    if (magic, version, nl) != (0x434D4644, 1, len(levels)):
        raise ValueError("unknown sigma identity")
    saved = np.memmap(sigma_path, mode="r", dtype="<f8", offset=32 + nl + (-nl) % 8, shape=(nl, nf))[lo.index]
    dlog = math.log(nmax / nmin) / nf
    edges = np.exp(math.log(nmin) + np.arange(nf + 1) * dlog)
    nu = np.exp(math.log(nmin) + (np.arange(nf) + .5) * dlog)
    width = np.diff(edges)
    # Runtime uses deck level/threshold rounding, while the baker uses raw E_cm.
    ion_e = pd.read_csv(model / "ionization_energies.csv")
    ion_e = ion_e[(ion_e.atomic_number == 16) & (ion_e.ion_number == 1)].iloc[0]
    chi = (float(ion_e.iloc[-1]) - lo.energy_eV.to_numpy()) * EV
    threshold = chi / H
    missing = np.flatnonzero(flags == 0)
    if list(lo.level_number.iloc[missing]) != [117, 118, 123, 148, 152, 173, 179]:
        raise ValueError("unexpected missing S routes")
    rebuilt = saved.copy()
    entries = entries_by_level(phot, {int(i): str(cfg) for i, cfg in zip(lo.index, lo.configuration)})
    for k in missing:
        raw_edge = (osc.ionization_eV - float(osc.levels[k]["E_cm"]) * 1.239841984e-4) * EV / H
        curve = np.zeros(nf)
        for e in entries[int(lo.index[k])]:
            fn, knots, start, _ = B._sigma_model(e.cs_type, e.energy, e.sigma_Mb, raw_edge)
            curve += B._bin_average_sigma(fn, knots, edges, width, start)
        rebuilt[k] = curve
    lower_edge = np.maximum(edges[:-1][None, :], threshold[:, None])
    active = edges[1:][None, :] > lower_edge
    step_factor = np.divide(width[None, :], edges[1:][None, :] - lower_edge,
                            out=np.zeros((len(lo), nf)), where=active)
    log_weight = np.log(np.divide(edges[1:][None, :], lower_edge,
                                 out=np.ones((len(lo), nf)), where=active))
    kernel1 = rebuilt * step_factor * log_weight * (4 * math.pi / H)
    kernel0 = saved * step_factor * log_weight * (4 * math.pi / H)
    for k in missing:
        step = np.where(edges[:-1] >= threshold[k], 2e-18 * (threshold[k] / nu) ** 3, 0.0)
        part = (edges[:-1] < threshold[k]) & (edges[1:] > threshold[k])
        step[part] = 2e-18 * (1 - (threshold[k] / edges[1:][part]) ** 3) / (3 * log_weight[k, part])
        kernel0[k] = step * log_weight[k] * (4 * math.pi / H)
    shells = pd.read_csv(str(prefix) + ".shell.csv").set_index("shell_id")
    spectrum = pd.read_csv(str(prefix) + ".spectral.csv")
    channels = pd.read_csv(work / "lumina_ion_channels.csv")
    # The dump's 'gen' is a sweep serial (incremented on returning to shell 0),
    # NOT population_generation. Match state and both reconstructed rates.
    channels = channels[(channels.Z == 16) & (channels.stage == 1)]
    ions = pd.read_csv(str(prefix) + ".ion.csv").set_index("shell_id")
    records = []
    alpha_nu = np.where((nu[None, :] < threshold[:, None]) & active,
                        (threshold[:, None] + edges[1:][None, :]) / 2, nu[None, :])
    for s, state in shells.iterrows():
        T, ne = float(state.T_e_K), float(state.n_e_cm3)
        sc = channels[channels.shell == s]
        match = sc[(abs(sc.T_e / T - 1) < 1e-8) & (abs(sc.n_e / ne - 1) < 1e-8)]
        if len(match) == 0:
            raise ValueError(("no matching T/ne channel", s, T, ne))
        spec = spectrum[spectrum.shell_id == s].sort_values("bin_id")
        np.testing.assert_allclose(spec.nu_lo_Hz, edges[:-1], rtol=1e-12, atol=0)
        J = spec.J_nu.to_numpy()
        weights = lo.g.to_numpy() * np.exp(-lo.energy_eV.to_numpy() * EV / (KB * T))
        weights /= weights.sum()
        gam0, gam1 = kernel0 @ J, kernel1 @ J
        gamma0, gamma1 = float(weights @ gam0), float(weights @ gam1)
        upper_x = hi.energy_eV.to_numpy() * EV / (KB * T)
        u_upper = float(np.sum(hi.g.to_numpy()[upper_x < 50] * np.exp(-upper_x[upper_x < 50])))
        lam3 = (H * H / (2 * math.pi * ME * KB * T)) ** 1.5
        x = H * alpha_nu / (KB * T)
        y = (H * alpha_nu - chi[:, None]) / (KB * T)
        milne = (8 * math.pi / C ** 2) * alpha_nu ** 2 * np.exp(-y) / (-np.expm1(-x)) * width
        alpha0 = float(np.sum(milne * saved, axis=1) @ lo.g.to_numpy()) * lam3 / (2 * u_upper)
        alpha1 = float(np.sum(milne * rebuilt, axis=1) @ lo.g.to_numpy()) * lam3 / (2 * u_upper)
        match = match[(abs(gamma0 / match.gamma_phot - 1) < 2e-6) &
                      (abs(ne * alpha0 / match.ne_alpha_rr - 1) < 2e-6)]
        if len(match) == 0:
            raise ValueError(("no saved channel reproduces J/T/ne rates", s, gamma0, ne * alpha0))
        row = match.iloc[-1]
        gamma_rel = abs(gamma0 / float(row.gamma_phot) - 1)
        rr_rel = abs(ne * alpha0 / float(row.ne_alpha_rr) - 1)
        if max(gamma_rel, rr_rel) > 2e-6:
            raise ValueError(("baseline rate not reproduced", s, gamma_rel, rr_rel))
        fixed_up = float(row.gamma_nt + row.c_ion)
        fixed_down = float(row.ne_alpha_dr + row.c_rec)
        r0 = (gamma0 + fixed_up) / (ne * alpha0 + fixed_down)
        r1 = (gamma1 + fixed_up) / (ne * alpha1 + fixed_down)
        records.append({"shell": int(s), "T": T, "ne": ne, "channel_sweep_serial": int(row.gen),
                        "S_total": float(ions.loc[s, "Z16_element_total_cm3"]),
                        "gamma0": gamma0, "gamma1": gamma1, "ne_alpha0": ne * alpha0, "ne_alpha1": ne * alpha1,
                        "gamma_replay_rel": gamma_rel, "rr_replay_rel": rr_rel,
                        "old7_gamma_share": float(weights[missing] @ gam0[missing]) / gamma0,
                        "new7_gamma_share": float(weights[missing] @ gam1[missing]) / gamma1,
                        "ladder0": r0, "ladder1": r1, "delta_ladder_dex": math.log10(r1 / r0),
                        "saved_final_NLTE_III_II": (float(ions.loc[s, "Z16_nlte_q2_cm3"] / ions.loc[s, "Z16_nlte_q1_cm3"])
                                                    if ions.loc[s, "Z16_nlte_q1_cm3"] > 0 else None)})
    occupied = [r for r in records if r["S_total"] > 0]
    paths = [Path(__file__), Path(P.__file__), Path(B.__file__), Path(src.osc_path), Path(src.phot_path),
             model / "levels.csv", model / "ionization_energies.csv", sigma_path,
             work / "lumina_ion_channels.csv", Path(str(prefix) + ".spectral.csv"),
             Path(str(prefix) + ".shell.csv"), Path(str(prefix) + ".ion.csv")]
    print(json.dumps({"job": os.environ.get("SLURM_JOB_ID"), "scope": "S II fixed J/T/ne stage-rate response, not a re-solved NLTE pair",
                      "generation": manifest["population_generation"], "records": records,
                      "summary": {"shells": len(records), "occupied_shells": len(occupied),
                                  "max_abs_delta_ladder_dex_occupied": max(abs(r["delta_ladder_dex"]) for r in occupied),
                                  "max_gamma_replay_rel": max(r["gamma_replay_rel"] for r in records),
                                  "max_rr_replay_rel": max(r["rr_replay_rel"] for r in records)},
                      "sha256": {str(p): TR.sha(p) for p in paths}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
