#!/usr/bin/env python3
"""Build isolated BF/collision sidecars for a full CMFGEN Si/S model.

The sidecars deliberately live outside the production model.  Collision data
whose CMFGEN endpoint is a term (rather than a J-resolved level) is expanded
with the CMFGEN convention used by the sealed IGC1 tables:

    Omega(J_l,J_u) = Omega(term_l,term_u)
                     * g(J_l)/G(term_l) * g(J_u)/G(term_u)

BF routes retain one route per lower/target level.  A missing phot entry is
recorded and is left to the runtime legacy fallback; it is never fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


MAGIC_BFC1 = 0x31434642
MAGIC_IGC1 = 0x49474331
NB = 1234
EV_TO_ERG = 1.602176634e-12
H_CGS = 6.62607015e-27


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("cmfgen_builder_sidecar", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Do not allow the known params[0]-as-sigma stand-in in this sidecar.
    mod.SIGMA_EXACT_HYD = True
    return mod


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def term_cfg(builder, value: str) -> str:
    return builder._term_cfg(value)


def exact_or_term(builder, configs: list[str], value: str) -> list[int]:
    norm = builder._norm_cfg(value)
    exact = [i for i, cfg in enumerate(configs) if builder._norm_cfg(cfg) == norm]
    if exact:
        return exact
    term = term_cfg(builder, value)
    return [i for i, cfg in enumerate(configs) if term_cfg(builder, cfg) == term]


def write_igc(path: Path, Z: int, ion: int, T: np.ndarray,
              pairs: list[tuple[int, int, np.ndarray]], nref: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fp:
        fp.write(struct.pack("<IIiiiii", MAGIC_IGC1, 1, Z, ion,
                             len(pairs), len(T), nref))
        fp.write(np.asarray(T, dtype="<f8").tobytes())
        for lo, up, _ in pairs:
            fp.write(struct.pack("<ii", lo, up))
            # The sealed IGC1 layout is pair-interleaved.
            fp.write(np.asarray(_, dtype="<f8").tobytes())


def build_collision(builder, model: Path, out: Path,
                    levels_by_ion: dict[tuple[int, int], list[dict[str, str]]],
                    manifest: dict[tuple[int, int], dict[str, str]],
                    evidence: dict) -> None:
    for (Z, ion), levels in sorted(levels_by_ion.items()):
        rec = manifest[(Z, ion)]
        col_path = Path(rec["col_path"]) if rec.get("col_path") else None
        if col_path is None or not col_path.exists():
            evidence["collisions"].append({"Z": Z, "ion": ion,
                                             "status": "source_missing"})
            continue
        col = builder.parse_col(col_path)
        configs = [r["configuration"] for r in levels]
        g = np.asarray([float(r["g"]) for r in levels], dtype="f8")
        term_groups: dict[str, list[int]] = defaultdict(list)
        for i, cfg in enumerate(configs):
            term_groups[term_cfg(builder, cfg)].append(i)
        exact_groups = {builder._norm_cfg(cfg): [i]
                        for i, cfg in enumerate(configs)}
        accum: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        for cfg_l, cfg_u, omega in col.entries:
            lo_set = exact_groups.get(builder._norm_cfg(cfg_l),
                                      term_groups.get(term_cfg(builder, cfg_l), []))
            up_set = exact_groups.get(builder._norm_cfg(cfg_u),
                                      term_groups.get(term_cfg(builder, cfg_u), []))
            if not lo_set or not up_set:
                raise RuntimeError(f"collision endpoint not found Z={Z} ion={ion}: "
                                   f"{cfg_l}-{cfg_u}")
            G_l = float(g[lo_set].sum())
            G_u = float(g[up_set].sum())
            if G_l <= 0.0 or G_u <= 0.0:
                raise RuntimeError(f"non-positive collision term g Z={Z} ion={ion}")
            om = np.asarray(omega, dtype="f8") * float(col.scale_factor)
            for a in lo_set:
                for b in up_set:
                    if a == b:
                        continue
                    lo, up = sorted((a, b))
                    weight = (g[a] / G_l) * (g[b] / G_u)
                    accum[(lo, up)].append(om * weight)
        pairs: list[tuple[int, int, np.ndarray]] = []
        for key in sorted(accum):
            candidates = accum[key]
            first = candidates[0]
            if any(not np.array_equal(first, other) for other in candidates[1:]):
                raise RuntimeError(f"conflicting collision duplicates Z={Z} ion={ion} pair={key}")
            pairs.append((key[0], key[1], first))
        T = np.asarray(col.T_grid_kK, dtype="f8") * 1.0e4
        if len(T) == 0 or np.any(np.diff(T) <= 0):
            raise RuntimeError(f"invalid collision temperature grid Z={Z} ion={ion}")
        write_igc(out / f"ige_col_{Z}_{ion}_cmfgen.bin", Z, ion, T, pairs, len(levels))
        evidence["collisions"].append({
            "Z": Z, "ion": ion, "status": "built", "source": str(col_path),
            "source_sha256": sha256(col_path), "raw_entries": len(col.entries),
            "expanded_pairs": len(pairs), "n_levels": len(levels),
            "n_T": len(T), "T_K": T.tolist(),
        })


def write_bfc1(path: Path, n_levels: int, routes: list[tuple],
               lo_nu: float, hi_nu: float) -> None:
    routes.sort(key=lambda r: (r[0], r[1]))
    offsets = np.zeros(n_levels + 1, dtype="<i4")
    for route in routes:
        offsets[route[0] + 1] += 1
    offsets = np.cumsum(offsets, dtype="<i4")
    with path.open("xb") as fp:
        fp.write(struct.pack("<IIiiiidd", MAGIC_BFC1, 1, n_levels, NB,
                             len(routes), 0, lo_nu, hi_nu))
        fp.write(offsets.tobytes())
        for index, dtype in ((1, "<i4"), (2, "<i4"),
                             (3, "<f8"), (4, "<f8"), (5, "<f8")):
            fp.write(np.asarray([r[index] for r in routes], dtype=dtype).tobytes())
        for route in routes:
            fp.write(np.asarray(route[6], dtype="<f8").tobytes())


def build_bf(builder, model: Path, out: Path,
             levels_by_ion: dict[tuple[int, int], list[dict[str, str]]],
             manifest: dict[tuple[int, int], dict[str, str]],
             ionization: dict[tuple[int, int], float], evidence: dict) -> None:
    level_index: dict[tuple[int, int, int], int] = {}
    configs: dict[tuple[int, int], list[str]] = {}
    g: dict[tuple[int, int], np.ndarray] = {}
    energy: dict[tuple[int, int], np.ndarray] = {}
    total = 0
    for key in sorted(levels_by_ion):
        rows = levels_by_ion[key]
        configs[key] = [r["configuration"] for r in rows]
        g[key] = np.asarray([float(r["g"]) for r in rows], dtype="f8")
        energy[key] = np.asarray([float(r["energy_eV"]) for r in rows], dtype="f8")
        for i in range(len(rows)):
            level_index[(key[0], key[1], i)] = total + i
        total += len(rows)

    edges = builder.BF_NU_MIN * np.exp(
        np.arange(NB + 1) * np.log(builder.BF_NU_MAX / builder.BF_NU_MIN) / NB)
    dnu = np.diff(edges)
    routes: list[tuple] = []
    route_rows: list[dict] = []
    for (Z, ion), rows in sorted(levels_by_ion.items()):
        phot_path = Path(manifest[(Z, ion)]["phot_path"])
        if not phot_path.exists():
            evidence["bf"].append({"Z": Z, "ion": ion, "status": "source_missing"})
            continue
        phot = builder.parse_phot(phot_path)
        next_key = (Z, ion + 1)
        next_configs = configs.get(next_key, [])
        next_g = g.get(next_key, np.empty(0))
        if next_configs:
            names = {term_cfg(builder, part) for part in phot.final_state.split("/")}
            target_local = [i for i, cfg in enumerate(next_configs)
                            if term_cfg(builder, cfg) in names]
            gsum = float(next_g[target_local].sum()) if target_local else 0.0
            if not target_local or abs(gsum - float(phot.final_state_g)) > 1e-8:
                raise RuntimeError(f"BF final-state target mismatch Z={Z} ion={ion}: "
                                   f"targets={len(target_local)} gsum={gsum} "
                                   f"declared={phot.final_state_g}")
        else:
            target_local = []
            if float(phot.final_state_g) <= 0.0:
                raise RuntimeError(f"BF terminal final-state g invalid Z={Z} ion={ion}")

        exact_entries = {builder._norm_cfg(e.config): e for e in phot.entries}
        term_entries: dict[str, list] = defaultdict(list)
        for entry in phot.entries:
            term_entries[term_cfg(builder, entry.config)].append(entry)
        n_explicit = 0
        n_missing = 0
        for local, row in enumerate(rows):
            entries = []
            exact = exact_entries.get(builder._norm_cfg(row["configuration"]))
            if exact is not None:
                entries = [exact]
            else:
                entries = term_entries.get(term_cfg(builder, row["configuration"]), [])
            if len(entries) > 1:
                raise RuntimeError(f"ambiguous BF entry Z={Z} ion={ion} level={local}")
            if not entries:
                n_missing += 1
                continue
            entry = entries[0]
            E_lower = float(row["energy_eV"])
            targets = target_local if target_local else [total]
            for target in targets:
                if target == total:
                    E_upper = float(phot.final_state_excit_eV)
                    upper_g = float(phot.final_state_g)
                    weight = 1.0
                else:
                    target_global = level_index[(Z, ion + 1, target)]
                    E_upper = float(energy[next_key][target])
                    upper_g = float(next_g[target])
                threshold_eV = ionization[(Z, ion)] - E_lower + E_upper
                if threshold_eV <= 0.0:
                    route_rows.append({"Z": Z, "ion": ion, "lower": local,
                                       "status": "nonpositive_threshold",
                                       "threshold_eV": threshold_eV})
                    n_missing += 1
                    continue
                nu_th = threshold_eV * EV_TO_ERG / H_CGS
                model_sigma = builder._sigma_model(
                    entry.cs_type, entry.energy, entry.sigma_Mb, nu_th,
                    zion=phot.z_screen,
                    nef=builder._cmfgen_nef(Z, phot.z_screen, nu_th),
                )
                if model_sigma is None:
                    n_missing += 1
                    route_rows.append({"Z": Z, "ion": ion, "lower": local,
                                       "status": "unsupported_sigma", "cs_type": entry.cs_type})
                    continue
                fn, nodes, nu_start, tag = model_sigma
                sigma = builder._bin_average_sigma(fn, nodes, edges, dnu, nu_start)
                if not np.isfinite(sigma).all() or np.any(sigma < 0.0):
                    raise RuntimeError(f"invalid BF sigma Z={Z} ion={ion} level={local}")
                if target == total:
                    target_global = total
                else:
                    target_global = level_index[(Z, ion + 1, target)]
                    weight = upper_g / float(phot.final_state_g)
                # BFC1's explicit-state consumer reads sigma directly and does
                # not apply `weight` a second time.  Store the effective
                # target-resolved cross section here, matching the canonical
                # build_bf_channels.py contract used by the native loader.
                sigma *= weight
                state = 2 if np.any(sigma > 0.0) else 1
                lower_global = level_index[(Z, ion, local)]
                routes.append((lower_global, target_global, state, nu_th,
                               upper_g, weight, sigma))
                route_rows.append({"Z": Z, "ion": ion, "lower": local,
                                   "target": target_global, "status": tag,
                                   "cs_type": entry.cs_type,
                                   "threshold_eV": threshold_eV,
                                   "target_g": upper_g, "weight": weight,
                                   "state": state})
                n_explicit += 1
        evidence["bf"].append({"Z": Z, "ion": ion, "status": "built",
                               "source": str(phot_path),
                               "source_sha256": sha256(phot_path),
                               "levels": len(rows), "phot_entries": len(phot.entries),
                               "explicit_routes": n_explicit,
                               "lower_levels_without_entry": n_missing,
                               "final_state": phot.final_state,
                               "final_state_g": float(phot.final_state_g),
                               "terminal": not bool(target_local)})

    write_bfc1(out / "bf_channels.bin", total, routes,
               builder.BF_NU_MIN, builder.BF_NU_MAX)
    with (out / "channels.csv").open("w", newline="") as fp:
        fields = sorted({key for row in route_rows for key in row})
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(route_rows)
    evidence["bf_summary"] = {"n_levels": total, "routes": len(routes),
                               "multi_target_lower_levels": int(sum(
                                   1 for v in _route_counts(routes) if v > 1)),
                               "terminal_routes": int(sum(r[1] == total for r in routes))}


def _route_counts(routes: list[tuple]) -> list[int]:
    counts = defaultdict(int)
    for row in routes:
        counts[row[0]] += 1
    return list(counts.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--builder", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = args.model.resolve()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"refusing existing sidecar output: {out}")
    out.mkdir(parents=True)
    builder = load_builder(args.builder.resolve())
    levels = read_csv(model / "levels.csv")
    manifest_rows = read_csv(model / "atomic_vintage_manifest.csv")
    ie_rows = read_csv(model / "ionization_energies.csv")
    levels_by_ion: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in levels:
        levels_by_ion[(int(row["atomic_number"]), int(row["ion_number"]))].append(row)
    manifest = {(int(r["atomic_number"]), int(r["ion_number"])): r
                for r in manifest_rows}
    ionization = {(int(r["atomic_number"]), int(r["ion_number"])):
                  float(r["ionization_energy_eV"]) for r in ie_rows}
    evidence = {"collisions": [], "bf": []}
    build_collision(builder, model, out, levels_by_ion, manifest, evidence)
    build_bf(builder, model, out, levels_by_ion, manifest, ionization, evidence)
    provenance = {
        "model": str(model),
        "model_files": {name: sha256(model / name) for name in (
            "levels.csv", "line_list.csv", "ionization_energies.csv",
            "atomic_vintage_manifest.csv")},
        "builder": str(args.builder.resolve()),
        "builder_sha256": sha256(args.builder.resolve()),
        "evidence": evidence,
    }
    provenance["outputs"] = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(p for p in out.rglob("*") if p.is_file())
    }
    (out / "sidecar_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "evidence": evidence}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
