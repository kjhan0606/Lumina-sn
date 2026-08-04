#!/usr/bin/env python3
"""Selected baked sigma versus linked, MODEL_SPEC-capped CMFGEN PHOT engine."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np

from l1a_common import (
    ContractError, QUANTIZATION_RULE_VERSION, endpoint, evidence, make_record,
    manifest_sha, normalize_configuration, semantic_join, semantic_rows,
)
from l1a_lines import _cmf_rows, load_authority

from cmfgen_parser import parse_phot


MAGIC = 0x434D4644
H_CGS = 6.62607015e-27
EV_ERG = 1.602176634e-12
SUPPORTED = {1, 7, 20, 21, 22}
UNSUPPORTED = {2, 3, 8}
TARGETS = {
    "I3": (None, "sigma(nu) selected versus CMFGEN PHOT"),
    "I3a": ({(27, 3)}, "sigma(nu) selected versus CMFGEN PHOT Co IV"),
    "I3b": ({(26, 2)}, "sigma(nu) selected versus CMFGEN PHOT Fe III"),
    "I3c": ({(26, 3), (28, 3)}, "sigma(nu) selected versus CMFGEN PHOT Fe IV and Ni IV"),
}
TERM_RE = re.compile(r"\[[^\[\]]*\]\s*$")


class SigmaFile:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            raw = stream.read(32)
            if len(raw) != 32:
                raise ContractError(f"short sigma header: {path}")
            magic, version, self.nlevels, self.nfreq, self.nu_min, self.nu_max = \
                struct.unpack("<IIiidd", raw)
            if magic != MAGIC or version != 1 or self.nlevels < 0 or self.nfreq <= 0:
                raise ContractError(f"invalid sigma header: {path}")
            flags = stream.read(self.nlevels)
        if len(flags) != self.nlevels:
            raise ContractError(f"short sigma flags: {path}")
        self.flags = np.frombuffer(flags, dtype=np.int8).copy()
        padding = (8-self.nlevels % 8) % 8
        self.offset = 32+self.nlevels+padding
        expected = self.offset+8*self.nlevels*self.nfreq
        if path.stat().st_size != expected:
            raise ContractError(f"sigma size mismatch: {path.stat().st_size}/{expected}")
        self.values = np.memmap(path, dtype="<f8", mode="r", offset=self.offset,
                                shape=(self.nlevels, self.nfreq))
        self.edges = self.nu_min*np.exp(np.arange(self.nfreq+1)*
            ((np.log(self.nu_max)-np.log(self.nu_min))/self.nfreq))
        self.centres = np.sqrt(self.edges[:-1]*self.edges[1:])


def _term(value: str) -> str:
    return TERM_RE.sub("", normalize_configuration(value))


def _model(entry, nu_th: float) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, float] | None:
    if nu_th <= 0 or entry.cs_type not in SUPPORTED:
        return None
    params = np.asarray(entry.sigma_Mb, dtype=np.float64)
    if entry.cs_type in {20, 21, 22}:
        energy = np.asarray(entry.energy, dtype=np.float64)
        if energy.size == 0 or params.size != energy.size or np.any(np.diff(energy) < 0):
            return None
        nodes = energy*nu_th
        sigma = params*1e-18
        start = max(nu_th, float(nodes[0]))
        def fn(nu, _nodes=nodes, _sigma=sigma):
            return np.interp(nu, _nodes, _sigma, left=0.0, right=float(_sigma[-1]))
        return fn, nodes, start
    if entry.cs_type == 1 and params.size >= 3:
        s0, beta, exponent = map(float, params[:3])
        if s0 == 0:
            return None
        def fn(nu, _s0=s0, _beta=beta, _exp=exponent, _edge=nu_th):
            out = np.zeros(np.shape(nu))
            mask = np.asarray(nu) >= _edge
            ratio = _edge/np.asarray(nu)[mask]
            out[mask] = 1e-18*_s0*(_beta+(1-_beta)*ratio)*ratio**_exp
            return out
        return fn, np.array([nu_th]), nu_th
    if entry.cs_type == 7 and params.size >= 4:
        s0, beta, exponent = map(float, params[:3])
        edge = nu_th+float(params[3])*1e15
        if s0 == 0:
            return None
        def fn(nu, _s0=s0, _beta=beta, _exp=exponent, _edge=edge):
            out = np.zeros(np.shape(nu))
            mask = np.asarray(nu) >= _edge
            ratio = _edge/np.asarray(nu)[mask]
            out[mask] = 1e-18*_s0*(_beta+(1-_beta)*ratio)*ratio**_exp
            return out
        return fn, np.array([edge]), edge
    return None


def _bin_average(fn: Callable, nodes: np.ndarray, edges: np.ndarray,
                 start: float, n_sub: int = 6) -> np.ndarray:
    output = np.zeros(edges.size-1)
    lo, hi = max(start, float(edges[0])), float(edges[-1])
    if lo >= hi:
        return output
    parts = [np.array([lo, hi]), edges[(edges > lo) & (edges < hi)]]
    internal = nodes[(nodes > lo) & (nodes < hi)]
    if internal.size:
        parts.append(internal)
    log_edges = np.log(edges)
    frac = (np.arange(1, n_sub)/n_sub)[None, :]
    sub = np.exp(log_edges[:-1, None]+np.diff(log_edges)[:, None]*frac).ravel()
    sub = sub[(sub > lo) & (sub < hi)]
    if sub.size:
        parts.append(sub)
    x = np.unique(np.concatenate(parts))
    values = fn(x)
    area = .5*(values[1:]+values[:-1])*np.diff(x)
    bins = np.searchsorted(edges, .5*(x[1:]+x[:-1]))-1
    np.add.at(output, bins, area)
    return output/np.diff(edges)


def _routes(authority: dict, parsed: dict) -> tuple[dict[tuple[int, int, int], list[Any]],
                                                    dict[tuple[int, int, int], set[int]], list[Path]]:
    result: dict[tuple[int, int, int], list[Any]] = {}
    types: dict[tuple[int, int, int], set[int]] = {}
    paths = []
    for key, slot in authority.items():
        z, ion = key
        phot = parse_phot(slot["phot"])
        paths.append(slot["phot"])
        osc = parsed[key]
        exact: dict[str, list[int]] = {}
        terms: dict[str, list[int]] = {}
        for rank, level in enumerate(osc.levels[:slot["nf"]]):
            exact.setdefault(normalize_configuration(str(level["config"])), []).append(rank)
            terms.setdefault(_term(str(level["config"])), []).append(rank)
        for entry in phot.entries:
            targets = exact.get(normalize_configuration(entry.config), [])
            if not targets:
                targets = terms.get(_term(entry.config), [])
            for rank in targets:
                level_key = (z, ion, rank)
                result.setdefault(level_key, []).append(entry)
                types.setdefault(level_key, set()).add(int(entry.cs_type))
    return result, types, paths


def _evaluate(routes: list[Any], osc_level: Any, osc_ion: Any,
              sigma: SigmaFile) -> tuple[np.ndarray, np.ndarray] | None:
    threshold_ev = float(osc_ion.ionization_eV)-float(osc_level["E_cm"])*1.239841984e-4
    threshold = threshold_ev*EV_ERG/H_CGS
    centre = np.zeros(sigma.nfreq)
    average = np.zeros(sigma.nfreq)
    used = 0
    for entry in routes:
        model = _model(entry, threshold)
        if model is None:
            continue
        fn, nodes, start = model
        centre += fn(sigma.centres)
        average += _bin_average(fn, nodes, sigma.edges, start)
        used += 1
    return (centre, average) if used else None


def _compare(left: np.ndarray, right: np.ndarray, mask: np.ndarray,
             threshold: float) -> dict[str, Any]:
    selected = left[mask]
    cmf = right[mask]
    if not selected.size:
        return {"denominator": 0, "mismatch": 0, "max_abs": 0.0, "max_rel": 0.0}
    absolute = np.abs(selected-cmf)
    scale = np.maximum(np.abs(selected), np.abs(cmf))
    relative = absolute/scale
    return {"denominator": int(selected.size),
            "mismatch": int(np.count_nonzero(relative > threshold)),
            "max_abs": float(np.max(absolute)), "max_rel": float(np.max(relative))}


def run(*, deck: Path, peer: Path, cmfgen_tree: Path, cmfgen_run: Path,
        threshold_mode: str, threshold: float, chunk_points: int,
        command: str, super_cutoff: int) -> list[dict[str, Any]]:
    del peer, threshold_mode, super_cutoff
    sigma_path = deck / "cmfgen_sigma_bf.bin"
    sigma = SigmaFile(sigma_path)
    deck_rows = semantic_rows(deck)
    if sigma.nlevels != len(deck_rows):
        raise ContractError(f"P04 sigma/deck level extent mismatch {sigma.nlevels}/{len(deck_rows)}")
    authority = load_authority(cmfgen_run, cmfgen_tree)
    cmf_rows, parsed = _cmf_rows(authority)
    mapping, ambiguous, join_census = semantic_join(deck_rows, cmf_rows)
    routes, route_types, phot_paths = _routes(authority, parsed)
    links_path = cmfgen_run / "atomic_links.txt"
    left = endpoint(sigma_path, "photoionization-cross-section", "deck-bake",
                    "selected baked sigma(nu) in cm^2", "selected Lumina deck")
    right = endpoint(links_path, "photoionization-cross-section", "deck-bake",
                     "linked PHOT evaluator on selected grid", "CMFGEN run atomic links")
    members = manifest_sha([sigma_path, deck/"levels.csv", links_path,
                            cmfgen_run/"MODEL_SPEC", *phot_paths])
    ev = evidence(command, Path(__file__), [sigma_path, links_path, *phot_paths],
                  sigma.nlevels*sigma.nfreq)
    evaluated: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    unsupported_levels: set[tuple[int, int, int]] = set()
    missing_levels: set[tuple[int, int, int]] = set()
    deck_index = {(row["z"], row["ion"], row["rank"]): index
                  for index, row in enumerate(deck_rows)}
    for deck_key, cmf_key in mapping.items():
        entries = routes.get(cmf_key, [])
        types = route_types.get(cmf_key, set())
        supported_entries = [entry for entry in entries if entry.cs_type in SUPPORTED]
        if not supported_entries:
            if types & UNSUPPORTED:
                unsupported_levels.add(deck_key)
            else:
                missing_levels.add(deck_key)
            continue
        z, ion, rank = cmf_key
        value = _evaluate(supported_entries, parsed[(z, ion)].levels[rank], parsed[(z, ion)], sigma)
        if value is None:
            missing_levels.add(deck_key)
        else:
            evaluated[deck_key] = value

    records = []
    rows_per_chunk = max(1, chunk_points//sigma.nfreq)
    for item_id, (target, metric) in TARGETS.items():
        keys = [key for key in sorted(evaluated) if target is None or key[:2] in target]
        centre_tot = {"denominator": 0, "mismatch": 0, "max_abs": 0.0, "max_rel": 0.0}
        average_tot = dict(centre_tot)
        selected_missing_points = 0
        record_denominator = 0
        for start in range(0, len(keys), rows_per_chunk):
            for key in keys[start:start+rows_per_chunk]:
                index = deck_index[key]
                centre, average = evaluated[key]
                positive_c = centre > 0
                positive_a = average > 0
                level_denominator = max(int(np.count_nonzero(positive_c)),
                                        int(np.count_nonzero(positive_a)))
                record_denominator += level_denominator
                if not sigma.flags[index]:
                    selected_missing_points += level_denominator
                left_values = np.asarray(sigma.values[index])
                for totals, right_values, mask in ((centre_tot, centre, positive_c),
                                                    (average_tot, average, positive_a)):
                    result = _compare(left_values, right_values, mask, threshold)
                    totals["denominator"] += result["denominator"]
                    totals["mismatch"] += result["mismatch"]
                    totals["max_abs"] = max(totals["max_abs"], result["max_abs"])
                    totals["max_rel"] = max(totals["max_rel"], result["max_rel"])
        robust = (centre_tot["mismatch"] == 0) == (average_tot["mismatch"] == 0)
        denominator = record_denominator
        outcome = ("MATCH" if centre_tot["mismatch"] == 0 else "DIFFER") if robust else "INCOMPARABLE"
        records.append(make_record(
            item_id=item_id, metric=metric, left=left, right=right,
            comparison_axis="selected_vs_cmfgen", denominator=denominator,
            cardinality=len(keys), selection="supported PHOT routes with CMFGEN sigma>0",
            member_sha=members,
            states={"present": denominator-selected_missing_points,
                    "missing": selected_missing_points, "zero": 0, "unsupported": 0},
            threshold_mode="rel", threshold=threshold, digits_left=17, digits_right=17,
            error_abs=max(centre_tot["max_abs"], average_tot["max_abs"]),
            error_rel=(max(centre_tot["max_rel"], average_tot["max_rel"])
                       if denominator else None), error_ulp=0,
            zero_rule="skip" if denominator else "NA",
            join_keys=["atomic_number", "ion_number", "normalized_configuration",
                       "g_interval", "energy_interval", "frequency_bin"],
            duplicate_count=len(ambiguous), duplicate_policy="ambiguous->unsupported; sum all PHOT routes",
            policy_result="all supported routes evaluated; no first-route selection",
            evidence_obj=ev, processed=denominator, unsupported=0, outcome=outcome,
            kind=["NUMERIC", "PROVENANCE"], disposition=["DEFINE"],
            posedness="UNVERIFIABLE", expected_provenance=True, sensitive=True,
            alternatives=["geometric bin center", "bin average"],
            coordinate_frame="selected Lumina frequency grid", coordinate_unit="Hz",
            coordinate_range=[sigma.nu_min, sigma.nu_max], interpolation="PHOT model evaluation",
            extrapolation="CMFGEN PHOT evaluator convention", bin_edges="log-uniform",
            quantization={"applicable": True, "rule_version": QUANTIZATION_RULE_VERSION,
                          "left_significant_digits_histogram": {"binary64": denominator},
                          "right_significant_digits_histogram": {"binary64": denominator},
                          "overlap_count": denominator-max(centre_tot["mismatch"], average_tot["mismatch"]),
                          "non_overlap_count": max(centre_tot["mismatch"], average_tot["mismatch"]),
                          "max_absolute_interval_gap": max(centre_tot["max_abs"], average_tot["max_abs"]),
                          "max_relative_interval_gap": max(centre_tot["max_rel"], average_tot["max_rel"]),
                          "historical_rel_1e-6_mismatch": max(centre_tot["mismatch"], average_tot["mismatch"]),
                          "interval_rule": "binary64 half-ULP; sampling provenance remains unresolved"},
            measurements={"geometric_bin_center": centre_tot, "bin_average": average_tot,
                          "sampling_verdict_robust": robust, "primary_sampling": None,
                          "primary_sampling_reason": "I15 build attestation absent",
                          "historical_denominator_old_deck": 3953894 if item_id == "I3" else None,
                          "historical_rel_1e-6_mismatch_old_deck": 1233529 if item_id == "I3" else None,
                          "semantic_join": join_census},
        ))

    supported_count = len(evaluated)
    support_denominator = len(mapping)+len(ambiguous)
    records.append(make_record(
        item_id="I3", metric="PHOT evaluator support census", left=left, right=right,
        comparison_axis="coverage_vs_cmfgen", denominator=support_denominator,
        cardinality=support_denominator, selection="semantic joined levels by PHOT evaluator type",
        member_sha=members, states={"present": supported_count, "missing": len(missing_levels),
                                    "zero": 0, "unsupported": len(unsupported_levels)+len(ambiguous)},
        threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
        error_abs=0.0, error_rel=None, error_ulp=0, zero_rule="NA",
        join_keys=["semantic_level", "PHOT_route", "cs_type"], duplicate_count=len(ambiguous),
        duplicate_policy="ambiguous->unsupported", policy_result="types 2/3/8 remain unsupported",
        evidence_obj=ev, processed=supported_count, unsupported=len(unsupported_levels)+len(ambiguous),
        outcome="PARTIAL" if unsupported_levels or ambiguous else "MATCH", kind=["COVERAGE"],
        disposition=["DEFINE" if unsupported_levels else "NONE"],
        measurements={"supported_types": sorted(SUPPORTED), "unsupported_types": sorted(UNSUPPORTED),
                      "unsupported_levels_type_2_3_8": len(unsupported_levels),
                      "all_routes_evaluated": True},
    ))
    selected_present = {
        (row["z"], row["ion"], row["rank"])
        for index, row in enumerate(deck_rows) if sigma.flags[index]
    }
    cmf_present = set(evaluated) | unsupported_levels
    union = selected_present | cmf_present | ambiguous
    common = selected_present & set(evaluated)
    unsupported_cov = len(unsupported_levels | ambiguous)
    missing_cov = len(union)-len(common)-unsupported_cov
    records.append(make_record(
        item_id="I17", metric="sigma semantic-level coverage", left=left, right=right,
        comparison_axis="coverage_vs_cmfgen", denominator=len(union), cardinality=len(union),
        selection="selected addressable/present and linked PHOT semantic-level union",
        member_sha=members, states={"present": len(common), "missing": missing_cov,
                                    "zero": 0, "unsupported": unsupported_cov},
        threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
        error_abs=float(missing_cov), error_rel=missing_cov/len(union) if union else None,
        error_ulp=0, zero_rule="skip" if union else "NA", join_keys=["semantic_level"],
        duplicate_count=len(ambiguous), duplicate_policy="ambiguous->unsupported",
        policy_result="exact set membership", evidence_obj=ev, processed=len(union),
        unsupported=unsupported_cov, outcome="MATCH" if missing_cov == 0 else "DIFFER",
        kind=["COVERAGE"], disposition=["CLOSE" if missing_cov == 0 else "REMEASURE"],
        measurements={"selected_present": len(selected_present),
                      "cmfgen_linked_phot": len(cmf_present), "common": len(common)},
    ))
    return records
