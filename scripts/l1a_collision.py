#!/usr/bin/env python3
"""Selected/epoch collision prescriptions versus linked CMFGEN col/osc authority."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from l1a_common import (
    ContractError, QUANTIZATION_RULE_VERSION, differs, endpoint, evidence,
    make_record, manifest_sha, normalize_configuration, read_levels,
    semantic_join, semantic_rows,
)
from l1a_lines import _cmf_rows, load_authority
from cmfgen_parser import parse_col, parse_osc


MAGIC = 0x49474331
TEMPERATURES = (5000.0, 10000.0, 20000.0)
K_B_EV = 8.617333262145e-5
H_CGS = 6.62607015e-27
EV_ERG = 1.602176634e-12
RATE_CONST = 8.63e-6


def _manifest(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Z", "ion0", "n_levels_ref", "n_mapped", "out_bin", "status", "col"}
        if required-set(reader.fieldnames or ()):
            raise ContractError(f"invalid collision manifest: {path}")
        result = {}
        for row in reader:
            key = int(row["Z"]), int(row["ion0"])
            if key in result:
                raise ContractError(f"P05 duplicate collision manifest ion {key}")
            result[key] = row
    return result


def classify_requested_ion(rows: dict[tuple[int, int], dict[str, str]],
                           key: tuple[int, int]) -> dict[str, Any]:
    if key not in rows:
        return {"states": {"present": 0, "missing": 0, "zero": 0, "unsupported": 1},
                "entity_flags": {"ion_present": False, "quantity_present": False,
                                 "counterpart_present": False, "evaluator_supported": False}}
    available = rows[key]["status"] == "OK"
    return {"states": {"present": int(available), "missing": int(not available),
                       "zero": 0, "unsupported": 0},
            "entity_flags": {"ion_present": True, "quantity_present": available,
                             "counterpart_present": available, "evaluator_supported": True}}


def _read_binary(path: Path, expected: tuple[int, int]) \
        -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    with path.open("rb") as stream:
        raw = stream.read(28)
        if len(raw) != 28:
            raise ContractError(f"short collision header: {path}")
        magic, version, z, ion, ntr, nt, nlev = struct.unpack("<IIiiiii", raw)
        if magic != MAGIC or version != 1 or (z, ion) != expected or ntr < 0 or nt < 2 or nlev <= 0:
            raise ContractError(f"invalid collision header: {path}")
        temperatures = np.frombuffer(stream.read(8*nt), dtype="<f8").copy()
        if temperatures.size != nt or np.any(np.diff(temperatures) <= 0):
            raise ContractError(f"invalid collision T grid: {path}")
        values = {}
        for _ in range(ntr):
            pair_raw, omega_raw = stream.read(8), stream.read(8*nt)
            if len(pair_raw) != 8 or len(omega_raw) != 8*nt:
                raise ContractError(f"truncated collision binary: {path}")
            pair = tuple(sorted(struct.unpack("<ii", pair_raw)))
            if pair in values:
                raise ContractError(f"P05 duplicate collision binary pair {expected}/{pair}")
            values[pair] = np.frombuffer(omega_raw, dtype="<f8").copy()
        if stream.read(1):
            raise ContractError(f"trailing collision bytes: {path}")
    return temperatures, values


def _deck_tables(deck: Path) -> tuple[dict, dict, list[Path]]:
    manifest_path = deck / "coldata_cmfgen_manifest.csv"
    rows = _manifest(manifest_path)
    tables = {}
    paths = [manifest_path]
    for key, row in rows.items():
        if row["status"] == "OK":
            binary = deck / row["out_bin"]
            paths.append(binary)
            tables[key] = _read_binary(binary, key)
        elif row["out_bin"]:
            raise ContractError(f"non-OK collision row names binary: {key}")
    return rows, tables, paths


def _cmf_tables(authority: dict) -> tuple[dict, dict, list[Path]]:
    tables = {}
    metadata = {}
    paths = []
    for key, slot in authority.items():
        osc, col = parse_osc(slot["osc"]), parse_col(slot["col"])
        paths.extend([slot["osc"], slot["col"]])
        nf = slot["nf"]
        config_map: dict[str, list[int]] = {}
        for rank, level in enumerate(osc.levels[:nf]):
            config_map.setdefault(normalize_configuration(str(level["config"])), []).append(rank)
        pair_values: dict[tuple[int, int], np.ndarray] = {}
        ambiguous = 0
        for lower, upper, omega in col.entries:
            lowers = config_map.get(normalize_configuration(lower), [])
            uppers = config_map.get(normalize_configuration(upper), [])
            candidates = {tuple(sorted((lo, up))) for lo in lowers for up in uppers if lo != up}
            if len(candidates) != 1:
                ambiguous += 1
                continue
            pair = candidates.pop()
            if pair in pair_values:
                raise ContractError(f"P05 duplicate semantic linked col transition {key}/{pair}")
            pair_values[pair] = np.asarray(omega, dtype=np.float64)*float(col.scale_factor)
        temperatures = np.asarray(col.T_grid_kK, dtype=np.float64)*1e4
        if pair_values and (temperatures.size < 2 or np.any(np.diff(temperatures) <= 0)):
            raise ContractError(f"P01 invalid linked col temperature grid {key}")
        tables[key] = (temperatures, pair_values)
        metadata[key] = {"declared_transitions": int(col.n_transitions),
                         "parsed_transitions": len(col.entries),
                         "semantic_transitions": len(pair_values),
                         "ambiguous_transitions": ambiguous,
                         "OMEGA_SET": float(col.default_omega),
                         "scale_factor": float(col.scale_factor)}
    return tables, metadata, paths


def _interp(grid: np.ndarray, values: np.ndarray, temperature: float) -> float:
    if grid.size != values.size or grid.size < 2:
        raise ContractError("P04 collision grid/value extent mismatch")
    if temperature < grid[0] or temperature > grid[-1]:
        raise ContractError(f"collision temperature {temperature} requires forbidden extrapolation")
    return float(np.interp(temperature, grid, values))


def _ex_e1x(x: float) -> float:
    w = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
    a = (0.2677737343, 8.6347608925, 18.0590169730, 8.5733287401)
    b = (3.9584969228, 21.0996530827, 25.6329561486, 9.5733223454)
    if not x > 0:
        return 0.0
    if x <= 1:
        p = w[-1]
        for coefficient in reversed(w[:-1]):
            p = p*x+coefficient
        return math.exp(x)*(p-math.log(x))
    numerator = denominator = 1.0
    for index in range(3, -1, -1):
        numerator = numerator*x+a[index]
        denominator = denominator*x+b[index]
    return numerator/denominator/x


def _fallback(f_lu: float, g_lower: float, delta_ev: float,
              temperature: float, ion0: int, omega_set: float) -> tuple[str, float]:
    if f_lu <= 1e-5 or delta_ev <= 0 or g_lower <= 0:
        return "OMEGA_SET", omega_set
    x = delta_ev/(K_B_EV*temperature)
    if ion0+1 <= 1:
        gbar = 0.276*_ex_e1x(x) if x <= 14 else 0.066*(1+1.5/x)/math.sqrt(x)
    else:
        gbar = max(0.2, 0.276*_ex_e1x(x))
    fl = delta_ev*EV_ERG/(H_CGS*1e15)
    return "VR", 47.972*gbar*f_lu*g_lower/fl


def _level_attrs(deck: Path) -> dict[tuple[int, int, int], tuple[float, float]]:
    result = {}
    for row in read_levels(deck):
        key = int(row["atomic_number"]), int(row["ion_number"]), int(row["level_number"])
        if key in result:
            raise ContractError(f"P05 duplicate deck level {key}")
        result[key] = float(row["energy_eV"]), float(row["g"])
    return result


def _line_attrs(deck: Path, mapping: dict) -> tuple[dict, int, int]:
    result = {}
    total = 0
    with (deck/"line_list.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            z, ion = int(row["atomic_number"]), int(row["ion_number"])
            lo, up = sorted((int(row["level_number_lower"]), int(row["level_number_upper"])))
            mapped_lo = mapping.get((z, ion, lo))
            mapped_up = mapping.get((z, ion, up))
            total += 1
            if mapped_lo is None or mapped_up is None:
                continue
            if mapped_lo[:2] != mapped_up[:2]:
                raise ContractError(f"P03 cross-ion semantic level mapping {(z,ion,lo,up)}")
            key = z, ion, *sorted((mapped_lo[2], mapped_up[2]))
            if key in result:
                raise ContractError(f"P05 duplicate selected line {key}")
            result[key] = float(row["f_lu"])
    return result, total, total-len(result)


def _remap_tables(tables: dict, mapping: dict) -> tuple[dict, int]:
    result = {}
    unsupported = 0
    for (z, ion), (grid, pairs) in tables.items():
        mapped_pairs = {}
        for (lo, up), values in pairs.items():
            mapped_lo = mapping.get((z, ion, lo))
            mapped_up = mapping.get((z, ion, up))
            if mapped_lo is None or mapped_up is None:
                unsupported += 1
                continue
            pair = tuple(sorted((mapped_lo[2], mapped_up[2])))
            if pair in mapped_pairs:
                raise ContractError(f"P05 duplicate remapped collision transition {(z,ion,pair)}")
            mapped_pairs[pair] = values
        result[(z, ion)] = (grid, mapped_pairs)
    return result, unsupported


def _branch(pair: tuple[int, int], source: tuple | None, f_lu: float) -> str:
    if source is not None and pair in source[1]:
        return "TABULATED"
    return "VR" if f_lu > 1e-5 else "OMEGA_SET"


def _identity(deck: Path, deck_tables: dict, cmf_tables: dict,
              cmf_metadata: dict, mapping: dict) -> dict[str, Any]:
    lines, total, unsupported_lines = _line_attrs(deck, mapping)
    selected_counts = {name: 0 for name in ("TABULATED", "VR", "OMEGA_SET")}
    cmf_counts = dict(selected_counts)
    distance = 0
    for (z, ion, lo, up), f_lu in lines.items():
        pair = lo, up
        selected_branch = _branch(pair, deck_tables.get((z, ion)), f_lu)
        cmf_branch = _branch(pair, cmf_tables.get((z, ion)), f_lu)
        selected_counts[selected_branch] += 1
        cmf_counts[cmf_branch] += 1
        distance += int(selected_branch != cmf_branch)
    authoritative_pairs = {(ion, pair) for ion, source in cmf_tables.items()
                           for pair in source[1]}
    selected_pairs = {(ion, pair) for ion, source in deck_tables.items()
                      for pair in source[1]}
    authoritative = len(authoritative_pairs)
    retained = len(authoritative_pairs & selected_pairs)
    retention = {"numerator": retained, "denominator": authoritative,
                 "value": retained/authoritative if authoritative else "NOT_APPLICABLE"}
    return {"total_lines": total, "comparable_lines": len(lines),
            "unsupported_semantic_lines": unsupported_lines,
            "selected_branch_counts": selected_counts,
            "cmfgen_branch_counts": cmf_counts, "identity_distance": distance,
            "authoritative_tabulated_retention": retention,
            "linked_col_metadata": cmf_metadata}


def _numeric(deck: Path, left_tables: dict, right_tables: dict,
             cmf_metadata: dict, mode: str, threshold: float,
             mapping: dict) -> dict[str, Any]:
    lines, _total, unsupported_lines = _line_attrs(deck, mapping)
    raw_levels = _level_attrs(deck)
    levels = {cmf_key: raw_levels[deck_key] for deck_key, cmf_key in mapping.items()
              if deck_key in raw_levels}
    wanted = set()
    for ion, source in left_tables.items():
        wanted.update((ion[0], ion[1], *pair) for pair in source[1])
    for ion, source in right_tables.items():
        wanted.update((ion[0], ion[1], *pair) for pair in source[1])
    present = missing = mismatch = 0
    max_abs = max_rel = q_max_abs = q_max_rel = 0.0
    max_ulp = 0
    branches = {"TABULATED": 0, "VR": 0, "OMEGA_SET": 0}
    for key in sorted(wanted):
        z, ion, lo, up = key
        f_lu = lines.get(key)
        lower, upper = levels.get((z, ion, lo)), levels.get((z, ion, up))
        if f_lu is None or lower is None or upper is None:
            missing += len(TEMPERATURES)
            continue
        delta_ev = abs(upper[0]-lower[0])
        g_lower = lower[1] if lower[0] <= upper[0] else upper[1]
        omega_set = cmf_metadata.get((z, ion), {}).get("OMEGA_SET", 0.1)
        for temperature in TEMPERATURES:
            values = []
            for source in (left_tables.get((z, ion)), right_tables.get((z, ion))):
                if source is not None and (lo, up) in source[1]:
                    branch = "TABULATED"
                    value = _interp(source[0], source[1][(lo, up)], temperature)
                else:
                    branch, value = _fallback(f_lu, g_lower, delta_ev, temperature, ion, omega_set)
                values.append((branch, value))
            branches[values[1][0]] += 1
            bad, absolute, relative, ulp = differs(values[0][1], values[1][1], mode, threshold)
            mismatch += int(bad)
            present += 1
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative or 0.0)
            max_ulp = max(max_ulp, ulp)
            boltz = math.exp(-delta_ev/(K_B_EV*temperature)) if delta_ev > 0 else 1.0
            qleft = RATE_CONST*values[0][1]*boltz/(g_lower*math.sqrt(temperature))
            qright = RATE_CONST*values[1][1]*boltz/(g_lower*math.sqrt(temperature))
            qabs = abs(qleft-qright)
            qscale = max(abs(qleft), abs(qright))
            q_max_abs = max(q_max_abs, qabs)
            q_max_rel = max(q_max_rel, qabs/qscale if qscale else 0.0)
    return {"present": present, "missing": missing, "mismatch": mismatch,
            "max_abs": max_abs, "max_rel": max_rel, "max_ulp": max_ulp,
            "q_max_abs": q_max_abs, "q_max_rel": q_max_rel,
            "branches": branches, "transition_count": len(wanted),
            "unsupported_semantic_lines": unsupported_lines}


def _numeric_quant(result: dict[str, Any]) -> dict[str, Any]:
    # Both evaluator outputs are binary64; their ULP and relative distances are
    # measured without altering either value.
    return {"applicable": True, "rule_version": QUANTIZATION_RULE_VERSION,
            "left_significant_digits_histogram": {"binary64": result["present"]},
            "right_significant_digits_histogram": {"binary64": result["present"]},
            "overlap_count": result["present"]-result["mismatch"],
            "non_overlap_count": result["mismatch"],
            "max_absolute_interval_gap": result["max_abs"],
            "max_relative_interval_gap": result["max_rel"],
            "historical_rel_1e-6_mismatch": result["mismatch"],
            "interval_rule": "binary64 half-ULP endpoints; relative rule is separately recorded"}


def _numeric_record(*, item_id: str, metric: str, left: dict, right: dict,
                    axis: str, result: dict, members: str, ev: dict,
                    mode: str, threshold: float, provenance: bool = False,
                    measurements: dict | None = None) -> dict:
    denominator = result["present"]+result["missing"]
    return make_record(
        item_id=item_id, metric=metric, left=left, right=right, comparison_axis=axis,
        denominator=denominator, cardinality=result["transition_count"],
        selection="union of tabulated transitions; equal fallback/fallback pairs omitted",
        member_sha=members, states={"present": result["present"], "missing": result["missing"],
                                    "zero": 0, "unsupported": 0},
        threshold_mode=mode, threshold=threshold, digits_left=17, digits_right=17,
        error_abs=result["max_abs"], error_rel=result["max_rel"] if result["present"] else None,
        error_ulp=result["max_ulp"], zero_rule="skip" if result["present"] else "NA",
        join_keys=["atomic_number", "ion_number", "semantic_lower", "semantic_upper", "temperature_K"],
        duplicate_count=0, duplicate_policy="reject", policy_result="semantic transition keys unique",
        evidence_obj=ev, processed=result["present"], unsupported=0,
        outcome="MATCH" if result["mismatch"] == 0 else "DIFFER",
        kind=["NUMERIC"] + (["PROVENANCE"] if provenance else []),
        disposition=["DEFINE" if provenance else "REMEASURE"],
        posedness="UNVERIFIABLE" if provenance else "WELL",
        expected_provenance=provenance, sensitive=True,
        alternatives=["5000 K", "10000 K", "20000 K"],
        coordinate_frame="electron temperature", coordinate_unit="K",
        coordinate_range=[min(TEMPERATURES), max(TEMPERATURES)],
        interpolation="linear only inside tabulated T range", extrapolation="forbidden",
        quantization=_numeric_quant(result),
        measurements={"mismatch_count": result["mismatch"], "branch_counts": result["branches"],
                      "q_ij_max_absolute": result["q_max_abs"],
                      "q_ij_max_relative": result["q_max_rel"], **(measurements or {})})


def run(*, deck: Path, peer: Path, cmfgen_tree: Path, cmfgen_run: Path,
        threshold_mode: str, threshold: float, command: str,
        super_cutoff: int) -> list[dict[str, Any]]:
    del super_cutoff
    authority = load_authority(cmfgen_run, cmfgen_tree)
    cmf_tables, cmf_metadata, cmf_paths = _cmf_tables(authority)
    selected_rows, selected_tables, selected_paths = _deck_tables(deck)
    peer_rows, peer_tables, peer_paths = _deck_tables(peer)
    cmf_level_rows, _parsed = _cmf_rows(authority)
    selected_mapping, selected_ambiguous, selected_join = semantic_join(
        semantic_rows(deck), cmf_level_rows)
    peer_mapping, peer_ambiguous, peer_join = semantic_join(
        semantic_rows(peer), cmf_level_rows)
    selected_tables, selected_unmapped_tables = _remap_tables(selected_tables, selected_mapping)
    peer_tables, peer_unmapped_tables = _remap_tables(peer_tables, peer_mapping)
    if deck.name.endswith("_ftos"):
        legacy, legacy_rows, legacy_tables, legacy_paths, legacy_mapping = peer, peer_rows, peer_tables, peer_paths, peer_mapping
        current, current_rows, current_tables, current_paths, current_mapping = deck, selected_rows, selected_tables, selected_paths, selected_mapping
    elif peer.name.endswith("_ftos"):
        legacy, legacy_rows, legacy_tables, legacy_paths, legacy_mapping = deck, selected_rows, selected_tables, selected_paths, selected_mapping
        current, current_rows, current_tables, current_paths, current_mapping = peer, peer_rows, peer_tables, peer_paths, peer_mapping
    else:
        raise ContractError("P02 epoch-peer roles are ambiguous; one explicit deck must end in _ftos")
    links_path = cmfgen_run/"atomic_links.txt"
    selected_manifest = deck/"coldata_cmfgen_manifest.csv"
    selected_left = endpoint(selected_manifest, "collision-prescription", "pre-runtime atomic input",
                             "selected tabulated/fallback prescription", "selected Lumina deck")
    cmf_right = endpoint(links_path, "collision-prescription", "pre-runtime atomic input",
                         "linked col_data/osc_data prescription", "CMFGEN run atomic links")
    selected_members = manifest_sha([*selected_paths, links_path, *cmf_paths])
    selected_ev = evidence(command, Path(__file__), [*selected_paths, links_path, *cmf_paths],
                           len(selected_rows)+sum(meta["parsed_transitions"] for meta in cmf_metadata.values()))
    selected_identity = _identity(deck, selected_tables, cmf_tables, cmf_metadata, selected_mapping)
    distance = selected_identity["identity_distance"]
    records = [make_record(
        item_id="I1", metric="selected branch census versus CMFGEN",
        left=selected_left, right=cmf_right, comparison_axis="selected_vs_cmfgen",
        denominator=selected_identity["total_lines"], cardinality=selected_identity["total_lines"],
        selection="all selected semantic bound-bound transitions", member_sha=selected_members,
        states={"present": selected_identity["comparable_lines"], "missing": 0, "zero": 0,
                "unsupported": selected_identity["unsupported_semantic_lines"]},
        threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
        error_abs=float(distance), error_rel=distance/selected_identity["total_lines"]
            if selected_identity["total_lines"] else None, error_ulp=0,
        zero_rule="skip" if selected_identity["total_lines"] else "NA",
        join_keys=["semantic_transition", "branch_enum"], duplicate_count=0,
        duplicate_policy="reject", policy_result="exact enum/count comparison",
        evidence_obj=selected_ev, processed=selected_identity["comparable_lines"],
        unsupported=selected_identity["unsupported_semantic_lines"],
        outcome="MATCH" if distance == 0 else "DIFFER", kind=["COVERAGE", "DEFINITION"],
        disposition=["NONE" if distance == 0 else "REMEASURE"],
        measurements=selected_identity)]
    selected_numeric = _numeric(deck, selected_tables, cmf_tables, cmf_metadata,
                                threshold_mode, threshold, selected_mapping)
    records.append(_numeric_record(
        item_id="I1", metric="selected Upsilon_eff(T) and q_ij(T) versus CMFGEN",
        left=selected_left, right=cmf_right, axis="selected_vs_cmfgen",
        result=selected_numeric, members=selected_members, ev=selected_ev,
        mode=threshold_mode, threshold=threshold))

    for label, epoch_deck, rows, tables, paths, mapping in (
            ("legacy", legacy, legacy_rows, legacy_tables, legacy_paths, legacy_mapping),
            ("current", current, current_rows, current_tables, current_paths, current_mapping)):
        identity = _identity(epoch_deck, tables, cmf_tables, cmf_metadata, mapping)
        left_path = epoch_deck/"coldata_cmfgen_manifest.csv"
        epoch_left = endpoint(left_path, "collision-prescription", "pre-runtime atomic input",
                              f"{label} input prescription", f"{label} deck")
        members = manifest_sha([*paths, links_path, *cmf_paths])
        ev = evidence(command, Path(__file__), [*paths, links_path, *cmf_paths],
                      len(rows)+sum(meta["parsed_transitions"] for meta in cmf_metadata.values()))
        distance = identity["identity_distance"]
        records.append(make_record(
            item_id="I19", metric=f"{label} branch identity versus CMFGEN",
            left=epoch_left, right=cmf_right, comparison_axis="selected_vs_cmfgen",
            denominator=identity["total_lines"], cardinality=identity["total_lines"],
            selection=f"all {label} semantic transitions", member_sha=members,
            states={"present": identity["comparable_lines"], "missing": 0, "zero": 0,
                    "unsupported": identity["unsupported_semantic_lines"]},
            threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
            error_abs=float(distance), error_rel=distance/identity["total_lines"] if identity["total_lines"] else None,
            error_ulp=0, zero_rule="skip" if identity["total_lines"] else "NA",
            join_keys=["semantic_transition", "branch_enum"], duplicate_count=0,
            duplicate_policy="reject", policy_result="distance and retention calculated from linked col contents",
            evidence_obj=ev, processed=identity["comparable_lines"],
            unsupported=identity["unsupported_semantic_lines"],
            outcome="MATCH" if distance == 0 else "DIFFER", kind=["PROVENANCE", "DEFINITION"],
            disposition=["DEFINE"], posedness="UNVERIFIABLE", expected_provenance=True,
            measurements={**identity, "runtime_identity": "UNVERIFIABLE without I15 binary attestation",
                          "input_prescription_identity_distance": distance},
        ))

    epoch_result = _numeric(current, legacy_tables, current_tables, cmf_metadata,
                            threshold_mode, threshold, current_mapping)
    epoch_left = endpoint(legacy/"coldata_cmfgen_manifest.csv", "collision-prescription",
                          "pre-runtime atomic input", "legacy prescription", "legacy deck")
    epoch_right = endpoint(current/"coldata_cmfgen_manifest.csv", "collision-prescription",
                           "pre-runtime atomic input", "current prescription", "current deck")
    epoch_members = manifest_sha([*legacy_paths, *current_paths])
    epoch_ev = evidence(command, Path(__file__), [*legacy_paths, *current_paths],
                        len(legacy_rows)+len(current_rows))
    records.append(_numeric_record(
        item_id="I19", metric="legacy-to-current physics change", left=epoch_left,
        right=epoch_right, axis="legacy_vs_current", result=epoch_result,
        members=epoch_members, ev=epoch_ev, mode=threshold_mode, threshold=threshold,
        measurements={"population_weighting": "not applied; physical verdict prohibited"}))
    return records
