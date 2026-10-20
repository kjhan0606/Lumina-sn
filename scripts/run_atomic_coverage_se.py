#!/usr/bin/env python3
"""Run one isolated full/bound CMFGEN Si/S fixed-field SE solve.

This is an operator diagnostic.  It reuses the sealed native BB/BF/GTH
library, but reads the selected atomic deck and sidecars from a new output
root.  It never publishes a production parent or modifies the existing model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import resource
import signal
import struct
import time
from pathlib import Path

import numpy as np


ROOT = Path("/gpfs/kjhan/lumina/cmfgen_map/si_s_cmfgen_parity_20260915")
BASE = ROOT / "operator_si_s_element_se_v2" / "run.py"
CURRENT_MODEL = Path("/gpfs/kjhan/lumina/cmfgen_map/new_radiation_ionization_20260914/physical_run/input/model")
BF_HEADER = "<IIiiiidd"
IGC_HEADER = "<IIiiiii"
NB = 1234
EV_TO_ERG = 1.602176634e-12
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def read_igc(path: Path, expected_z: int, expected_ion: int,
             expected_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as fp:
        magic, version, z, ion, nr, nt, nref = struct.unpack(IGC_HEADER, fp.read(28))
        require((magic, version, z, ion, nref) ==
                (0x49474331, 1, expected_z, expected_ion, expected_n),
                f"IGC1 header mismatch: {path}")
        temperatures = np.fromfile(fp, "<f8", nt)
        pairs = np.empty((nr, 2), dtype="<i4")
        omega = np.empty((nr, nt), dtype="<f8")
        for i in range(nr):
            pairs[i] = struct.unpack("<ii", fp.read(8))
            omega[i] = np.fromfile(fp, "<f8", nt)
        require(not fp.read(1), f"IGC1 trailing bytes: {path}")
    require(np.all(np.diff(temperatures) > 0) and np.isfinite(omega).all() and
            np.all(omega >= 0), f"IGC1 invalid values: {path}")
    return (np.ascontiguousarray(pairs[:, 0]),
            np.ascontiguousarray(pairs[:, 1]),
            np.ascontiguousarray(temperatures),
            np.ascontiguousarray(omega))


def read_sigma(path: Path, expected_n: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as fp:
        magic, version, n, nb = struct.unpack("<IIii", fp.read(16))
        lo, hi = struct.unpack("<dd", fp.read(16))
        require((magic, version, n, nb) == (0x434D4644, 1, expected_n, NB),
                "CMFD header mismatch")
        flags = np.fromfile(fp, "i1", n).copy()
        offset = 32 + ((n + 7) // 8) * 8
    sigma = np.memmap(path, dtype="<f8", mode="r", offset=offset,
                      shape=(n, NB))
    require(np.isfinite(sigma).all() and np.all(sigma >= 0), "CMFD invalid sigma")
    return flags, sigma


def read_bfc(path: Path, n_levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray]:
    with path.open("rb") as fp:
        magic, version, n, nb, nr, reserved, lo, hi = struct.unpack(BF_HEADER, fp.read(40))
        require((magic, version, n, nb, reserved) ==
                (0x31434642, 1, n_levels, NB, 0), "BFC1 header mismatch")
        offsets = np.fromfile(fp, "<i4", n + 1)
        target = np.fromfile(fp, "<i4", nr)
        state = np.fromfile(fp, "<i4", nr)
        edge = np.fromfile(fp, "<f8", nr)
        upper_g = np.fromfile(fp, "<f8", nr)
        weight = np.fromfile(fp, "<f8", nr)
        sigma_offset = fp.tell()
    require(offsets[0] == 0 and offsets[-1] == nr and
            np.all(np.diff(offsets) >= 0), "BFC1 offsets")
    sigma = np.memmap(path, dtype="<f8", mode="r", offset=sigma_offset,
                      shape=(nr, NB))
    require(np.isfinite(sigma).all() and np.all(sigma >= 0) and
            np.all((state >= 0) & (state <= 2)) and
            np.all((weight > 0) & (weight <= 1)), "BFC1 values")
    return offsets, target, state, edge, upper_g, weight, sigma


def groups(rows: list[dict[str, str]]) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], int]]:
    by: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(rows):
        by.setdefault((int(row["atomic_number"]), int(row["ion_number"])), []).append(i)
    starts = {key: values[0] for key, values in by.items()}
    return by, starts


def read_stages(model: Path, sidecars: Path, Z: int, rows: list[dict[str, str]],
                ionization: dict[tuple[int, int], float],
                by: dict[tuple[int, int], list[int]],
                starts: dict[tuple[int, int], int],
                base) -> tuple[list[dict], np.ndarray, dict]:
    counts = [len(by[(Z, q)]) for q in range(1, 5)]
    base.COUNTS[Z] = counts
    stage_data = []
    table_counts = []
    for q in range(1, 5):
        key = (Z, q)
        frame = [row for row in read_csv(model / "line_list.csv")
                 if int(row["atomic_number"]) == Z and int(row["ion_number"]) == q]
        lo = np.asarray([int(x["level_number_lower"]) for x in frame], dtype=np.int32)
        up = np.asarray([int(x["level_number_upper"]) for x in frame], dtype=np.int32)
        require(np.all((lo >= 0) & (up > lo) & (up < counts[q - 1])),
                f"BB endpoints Z={Z} ion={q}")
        lines = (lo, up, *(np.asarray([float(x[k]) for x in frame], dtype="f8")
                            for k in ("f_lu", "A_ul", "B_lu", "B_ul")))
        nu = np.asarray([float(x["nu"]) for x in frame], dtype="f8")
        cpath = sidecars / f"ige_col_{Z}_{q}_cmfgen.bin"
        if cpath.exists():
            clo, cup, temperatures, omega = read_igc(cpath, Z, q, counts[q - 1])
            table_counts.append(len(clo))
        else:
            clo = cup = np.empty(0, dtype=np.int32)
            temperatures = np.empty(0, dtype="f8")
            omega = np.empty((0, 0), dtype="f8")
            table_counts.append(0)
        stage_data.append(dict(lines=lines, nu=nu,
                               ids=np.asarray([int(x["line_id"]) for x in frame], dtype=np.int64),
                               table=(clo, cup, temperatures, omega)))
    base.TABLES[Z] = table_counts
    limits = np.asarray([ionization[(Z, q)] for q in range(1, 5)], dtype="f8")
    return stage_data, limits, {"counts": counts, "table_counts": table_counts}


def routes_for_z(model: Path, sidecars: Path, Z: int, rows: list[dict[str, str]],
                 by: dict[tuple[int, int], list[int]], starts: dict[tuple[int, int], int],
                 ionization: np.ndarray, allow_unroutable: bool = False) -> tuple[dict, dict]:
    n_total = len(rows)
    counts = [len(by[(Z, q)]) for q in range(1, 5)]
    z_indices = [i for q in range(1, 5) for i in by[(Z, q)]]
    z_start = z_indices[0]
    n = len(z_indices)
    side_offsets, target, state, edge, ug, weight, sigma = read_bfc(
        sidecars / "bf_channels.bin", n_total)
    cmfd_path = next(sidecars.parent.glob("cmfgen_sigma_bf_si_s_*.bin"), None)
    require(cmfd_path is not None, "CMFD full/bound sidecar not found")
    flags, cmfd = read_sigma(cmfd_path, n_total)
    lower = []
    upper = []
    states = []
    edges = []
    gs = []
    weights = []
    sigmas = []
    origins = []
    local_offsets = np.r_[0, np.cumsum(counts)]
    missing = []
    for local, global_level in enumerate(z_indices):
        begin, end = int(side_offsets[global_level]), int(side_offsets[global_level + 1])
        if begin < end:
            for ch in range(begin, end):
                tg = int(target[ch])
                if tg == n_total:
                    tl = n
                else:
                    require(tg in z_indices, f"cross-Z BF target Z={Z}")
                    tl = tg - z_start
                lower.append(local); upper.append(tl); states.append(int(state[ch]))
                edges.append(float(edge[ch])); gs.append(float(ug[ch]))
                weights.append(float(weight[ch])); sigmas.append(np.asarray(sigma[ch]))
                origins.append(ch)
            continue
        q = int(rows[global_level]["ion_number"])
        local_q = local_offsets[q - 1] + int(global_level - by[(Z, q)][0])
        if q < 4:
            tl = local_offsets[q]
        else:
            tl = n
        threshold = ionization[q - 1] - float(rows[global_level]["energy_eV"])
        if threshold <= 0:
            missing.append((global_level, q, threshold))
            continue
        values = np.asarray(cmfd[global_level], dtype="f8")
        positive = bool(np.any(values > 0))
        lower.append(local_q); upper.append(tl); states.append(2 if positive else 0)
        edges.append(threshold * EV_TO_ERG / H_CGS)
        if q < 4:
            gs.append(float(rows[by[(Z, q + 1)][0]]["g"]))
        else:
            gs.append(1.0)
        weights.append(1.0); sigmas.append(values); origins.append(-1)
    if not allow_unroutable:
        require(not missing, f"unroutable BF levels Z={Z}: {missing[:5]} ({len(missing)})")
    out = dict(lower=np.asarray(lower, dtype=np.int32),
               target=np.asarray(upper, dtype=np.int32),
               state=np.asarray(states, dtype=np.int32),
               edge=np.asarray(edges, dtype="f8"),
               g=np.asarray(gs, dtype="f8"),
               weight=np.asarray(weights, dtype="f8"),
               sigma=np.asarray(sigmas, dtype="f8"),
               source_channel=np.asarray(origins, dtype=np.int32))
    counts_by_lower = np.bincount(out["lower"], minlength=n)
    if not allow_unroutable:
        require(np.all(counts_by_lower > 0), f"BF lower-level gap Z={Z}")
    proof = {"routes": len(lower),
             "explicit_routes": int(np.count_nonzero(out["source_channel"] >= 0)),
             "legacy_routes": int(np.count_nonzero(out["source_channel"] < 0)),
             "terminal_routes": int(np.count_nonzero(out["target"] == n)),
             "multi_target_lower_levels": int(np.count_nonzero(counts_by_lower > 1)),
             "unroutable_lower_levels": int(np.count_nonzero(counts_by_lower == 0)),
             "unroutable_records": len(missing)}
    return out, proof


def full_parent(c, active_model: Path, full_rows: list[dict[str, str]],
                by_full: dict[tuple[int, int], list[int]], Z: int, shell: int,
                allow_gap: bool = False) -> tuple[float, np.ndarray, float]:
    active_rows = read_csv(active_model / "levels.csv")
    active_by, _ = groups(active_rows)
    active_offsets = {key: values[0] for key, values in active_by.items()}
    # The sealed parent level array follows the active model's global CSV order.
    parent = []
    for q in range(1, 5):
        key = (Z, q)
        active_group = [row for row in active_rows
                        if (int(row["atomic_number"]), int(row["ion_number"])) == key]
        amap = {row["configuration"].strip().lower(): i
                for i, row in enumerate(active_group)}
        start = active_offsets[key]
        for row in (full_rows[i] for i in by_full[key]):
            j = amap.get(row["configuration"].strip().lower())
            parent.append(float(c["arrays"]["level.f64"][start + j, shell]) if j is not None else 0.0)
    keys = c["keys"]
    ip = [keys.index((Z, q)) for q in range(1, 6)]
    total = float(np.sum(c["arrays"]["ion.f64"][ip, shell], dtype=np.longdouble))
    parent = np.asarray(parent + [float(c["arrays"]["ion.f64"][ip[-1], shell])], dtype="f8")
    require(np.isfinite(parent).all() and np.all(parent >= 0), "full parent invalid")
    gap = total - float(np.sum(parent, dtype=np.longdouble))
    if not allow_gap:
        require(abs(gap) <= 1e-8 * max(total, 1.0), "full parent total mismatch")
    else:
        require(gap >= -1e-8 * max(total, 1.0), "full parent exceeds total")
    return total, parent, gap


def apply_superlevel_source(rows: list[dict[str, str]], source: Path) -> None:
    source_rows = read_csv(source / "levels.csv")
    mapping = {(int(row["atomic_number"]), int(row["ion_number"]),
                row["configuration"].strip().lower()): row["super_level"]
               for row in source_rows}
    for row in rows:
        key = (int(row["atomic_number"]), int(row["ion_number"]),
               row["configuration"].strip().lower())
        require(key in mapping, f"super-level source missing: {key}")
        row["super_level"] = mapping[key]


def collapse_superlevels(matrix: np.ndarray, E: np.ndarray, g: np.ndarray,
                         ions: np.ndarray, super_level: np.ndarray,
                         T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a full-level rate matrix to CMFGEN-style super-levels."""
    keys = [(int(ions[i]), int(super_level[i])) for i in range(len(E))]
    unique = sorted(set(keys))
    group_of = {key: i for i, key in enumerate(unique)}
    group_ids = np.asarray([group_of[key] for key in keys] + [len(unique)], dtype=np.int32)
    ng = len(unique) + 1
    weights = np.zeros(len(E) + 1, dtype="f8")
    weights[-1] = 1.0
    for h, key in enumerate(unique):
        members = np.flatnonzero(group_ids[:-1] == h)
        Emin = float(E[members].min())
        boltz = g[members] * np.exp(-(E[members] - Emin) * EV_TO_ERG /
                                     (K_CGS * T))
        require(np.isfinite(boltz).all() and float(boltz.sum()) > 0,
                f"invalid super-level Boltzmann weights: {key}")
        weights[members] = boltz / boltz.sum()
    aggregated = np.zeros((ng, ng), order="F", dtype="f8")
    for src in range(ng):
        members = np.flatnonzero(group_ids == src)
        contribution = matrix[:, members] @ weights[members]
        np.add.at(aggregated[:, src], group_ids, contribution)
    return aggregated, group_ids, weights, np.asarray(unique, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--sidecars", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--field", type=Path, required=True)
    ap.add_argument("--Z", type=int, choices=(14, 16), required=True)
    ap.add_argument("--shell", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--collapse-super", action="store_true")
    ap.add_argument("--superlevels-from", type=Path)
    ap.add_argument("--allow-unroutable-bf", action="store_true")
    ap.add_argument("--allow-parent-gap", action="store_true")
    ap.add_argument("--ne-override", type=float)
    args = ap.parse_args()
    out = args.output.resolve()
    require(not out.exists(), f"refusing existing output: {out}")
    require(0 <= args.shell < 50, "shell range")
    start = time.monotonic()
    base = load("sealed_element_runner", BASE)
    rows = read_csv(args.model / "levels.csv")
    if args.collapse_super:
        require(args.superlevels_from is not None,
                "--collapse-super requires --superlevels-from")
        apply_superlevel_source(rows, args.superlevels_from.resolve())
    by, starts = groups(rows)
    require(set(by) == {(14, q) for q in range(1, 5)} |
            {(16, q) for q in range(1, 5)}, "Si/S four-stage model required")
    ionization = {(int(r["atomic_number"]), int(r["ion_number"])):
                  float(r["ionization_energy_eV"])
                  for r in read_csv(args.model / "ionization_energies.csv")}
    counts = {z: [len(by[(z, q)]) for q in range(1, 5)] for z in (14, 16)}
    base.COUNTS.update(counts)
    d, adapter, core, identity = base.libraries()
    c = base.adapter.context(args.parent.resolve(), args.field.resolve(), d)
    if args.ne_override is not None:
        require(np.isfinite(args.ne_override) and args.ne_override > 0,
                "--ne-override must be finite and positive")
        ne_array = np.array(c["arrays"]["ne.f64"], dtype="f8", copy=True)
        ne_array[args.shell] = args.ne_override
        c["arrays"] = dict(c["arrays"], **{"ne.f64": ne_array})
    stage_data, limits, stage_proof = read_stages(args.model, args.sidecars, args.Z,
                                                   rows, ionization, by, starts, base)
    routes, route_proof = routes_for_z(args.model, args.sidecars, args.Z, rows,
                                       by, starts, limits,
                                       allow_unroutable=args.allow_unroutable_bf)
    n = sum(stage_proof["counts"])
    z_indices = [i for q in range(1, 5) for i in by[(args.Z, q)]]
    levels = rows
    E = np.asarray([float(levels[i]["energy_eV"]) for i in z_indices], dtype="f8")
    g = np.asarray([int(levels[i]["g"]) for i in z_indices], dtype=np.int32)
    ion = np.asarray([int(levels[i]["ion_number"]) for i in z_indices], dtype=np.int32)
    off = np.r_[0, np.cumsum(stage_proof["counts"])].astype(np.int32)
    cf = __import__("pandas").read_csv(core.CATALOG)
    cf = cf[(cf.Z == args.Z) & (cf.ion_stage_0based == 5)]
    require(len(cf) > 0, "terminal VI catalog missing")
    p = dict(Z=args.Z, E=E, g=g, ion=ion, off=off,
             cm=np.asarray(cf["E_cm-1"], dtype="f8"),
             cg=np.asarray(cf["g"], dtype="f8"), stages=stage_data,
             routes=routes, limits=limits)
    lib = base.Native(args.Z, core)
    fixed = json.loads((base.adapter.FIXED / "state.json").read_text())
    require(fixed["keys"] == c["sm"]["keys"], "fixed geometry/current keys")
    natom, heat, _, _, _, fixed_inputs = adapter.read_extra_inputs(
        dict(sm=fixed, arrays=c["arrays"]), core)
    policy = adapter.policy(core)
    T = float(c["arrays"]["te.f64"][args.shell])
    ne = float(c["arrays"]["ne.f64"][args.shell])
    ca = [0.0, 0.0]
    if args.Z == 14:
        current_levels, _ = d.atomic()
        for i, q in enumerate((1, 2)):
            lo, hi = current_levels["offsets"][(20, q)]
            ca[i] = float(np.sum(c["arrays"]["level.f64"][lo:hi, args.shell], dtype=np.longdouble))
    extra = base.extra_rates(args.Z, policy, T, float(heat[args.shell]),
                             float(natom[args.shell]), *ca)
    total, parent, parent_gap = full_parent(
        c, CURRENT_MODEL, rows, by, args.Z, args.shell,
        allow_gap=args.allow_parent_gap)
    Jfine = base.arr(c["J"][args.shell])
    m, ctx = base.assemble(lib, p, c["grid"], Jfine, T, ne, extra)
    if args.collapse_super:
        super_level = np.asarray([int(rows[i]["super_level"]) for i in z_indices], dtype=np.int32)
        group_matrix, group_ids, group_weights, super_keys = collapse_superlevels(
            m, E, g, ion, super_level, T)
        group_candidate, report = lib.bb.solve(group_matrix, total)
        require(group_candidate is not None and np.isfinite(group_candidate).all() and
                np.all(group_candidate > 0) and
                abs(report["total_relative_error"]) < 1e-12,
                "super-level GTH solve failed")
        candidate = group_weights * group_candidate[group_ids]
        incoming, outgoing, rel = base.residual(m, candidate)
        group_in, group_out, group_rel = base.residual(group_matrix, group_candidate)
    else:
        group_ids = np.arange(len(m), dtype=np.int32)
        group_weights = np.ones(len(m), dtype="f8")
        super_keys = np.empty((0, 2), dtype=np.int32)
        candidate, report = lib.bb.solve(m, total)
        require(candidate is not None and np.isfinite(candidate).all() and
                np.all(candidate > 0) and abs(report["total_relative_error"]) < 1e-12,
                "full atomic GTH solve failed")
        incoming, outgoing, rel = base.residual(m, candidate)
        group_rel = rel
    require(float(group_rel.max()) < 1e-12, "SE aggregate row residual")
    rr, chi, eta = lib.bf(p, ctx["J"], T, ne, ctx["U"], candidate)
    require(np.array_equal(rr, ctx["BF_rates"]), "final BF re-evaluation mismatch")
    stages = np.asarray([float(np.sum(candidate[off[q]:off[q + 1]], dtype=np.longdouble))
                         for q in range(4)] + [float(candidate[-1])])
    result = {
        "variant": args.model.parent.name,
        "Z": args.Z, "shell": args.shell, "n_levels": n,
        "total": total, "stage_population": stages.tolist(),
        "parent_population": parent.tolist(), "Te": T, "ne": ne,
        "parent_gap": parent_gap,
        "allow_unroutable_bf": bool(args.allow_unroutable_bf),
        "allow_parent_gap": bool(args.allow_parent_gap),
        "ne_override": args.ne_override,
        "max_row_relative": float(rel.max()),
        "max_group_row_relative": float(group_rel.max()),
        "collapsed_superlevels": bool(args.collapse_super),
        "superlevel_source": str(args.superlevels_from.resolve()) if args.superlevels_from else None,
        "n_groups": int(group_rel.size), "GTH": report,
        "route_proof": route_proof, "stage_proof": stage_proof,
        "fixed_Te_ne_J": True, "other_elements_fixed": True,
        "production_publication": False,
        "input": {"model": str(args.model.resolve()), "sidecars": str(args.sidecars.resolve()),
                  "parent": c["current_parent"], "field": c["current_field"],
                  "fixed_inputs": fixed_inputs},
        "atomic_identity": identity,
        "wall_seconds": time.monotonic() - start,
        "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    out.mkdir(parents=True)
    np.savez(out / "element.npz", level_population=candidate,
             stage_population=stages, parent_population=parent,
             Te=T, ne=ne, row_in=incoming, row_out=outgoing,
             row_relative=rel, group_row_relative=group_rel,
             group_ids=group_ids, group_weights=group_weights,
             BF_rates=ctx["BF_rates"], BF_J=ctx["J"], partition=ctx["U"])
    (out / "result.json").write_text(json.dumps(result, indent=2,
                                                  ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        raise
