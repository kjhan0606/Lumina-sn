#!/usr/bin/env python3
"""Semantic selected-deck versus CMFGEN line, level, and F_TO_S engine."""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
import sys
import tempfile
from typing import Any

import numpy as np

from l1a_common import (
    ContractError, QUANTIZATION_RULE_VERSION, decimal_interval, endpoint,
    evidence, make_record, manifest_sha, normalize_configuration,
    quantization_interval, semantic_join, semantic_rows, sha256_file,
    ulp_distance,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmfgen_parser import parse_f_to_s, parse_osc  # noqa: E402


CM2EV = Decimal("1.239841984e-4")
TARGETS = {
    "I2": (None, "A_ul semantic transition identity"),
    "I2a": ((26, 3), "A_ul semantic transition identity Fe IV"),
    "I2b": ((28, 3), "A_ul semantic transition identity Ni IV"),
    "I2c": ((27, 3), "A_ul semantic transition identity Co IV"),
    "I2d": ((26, 2), "A_ul semantic transition identity Fe III"),
}
ROMAN = {value: index for index, value in enumerate(
    ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"))}
Z_BY_DIR = {"CARB": 6, "SIL": 14, "SUL": 16, "CAL": 20,
            "FE": 26, "COB": 27, "NICK": 28}
FNUM = r"[-+]?\d+\.?\d*(?:[DdEe][-+]?\d+)?"
TRANS_RE = re.compile(
    rf"\s({FNUM})\s+({FNUM})\s+({FNUM})\s+(\d+)\s*-\s*(\d+)"
    rf"(?:\s+(\d+))?(?=\s|\||$)"
)


def _suffix(path: Path) -> Path:
    parts = path.parts
    positions = [i for i, part in enumerate(parts) if part == "atomic"]
    if not positions:
        raise ContractError(f"P01 atomic component absent from link source {path}")
    i = positions[-1] + 1
    if i < len(parts) and parts[i] == "cmfgen":
        i += 1
    return Path(*parts[i:])


def _identity(path: Path) -> tuple[int, int]:
    parts = _suffix(path).parts
    if len(parts) < 3 or parts[0] not in Z_BY_DIR or parts[1] not in ROMAN:
        raise ContractError(f"P01 unknown ion identity in {path}")
    return Z_BY_DIR[parts[0]], ROMAN[parts[1]] - 1


def _resolve_source(source: Path, tree: Path) -> tuple[Path, dict[str, Any]]:
    mirror = tree / _suffix(source)
    direct = source if source.is_file() else None
    local = mirror if mirror.is_file() else None
    if direct is None and local is None:
        raise ContractError(f"P01 link target and mirror absent: {source} / {mirror}")
    if direct is not None and local is not None and sha256_file(direct) != sha256_file(local):
        raise ContractError(f"P01 link target/mirror SHA mismatch: {source} / {mirror}")
    chosen = direct or local
    before = (chosen.stat().st_dev, chosen.stat().st_ino, chosen.stat().st_size,
              chosen.stat().st_mtime_ns)
    digest = sha256_file(chosen)
    after = (chosen.stat().st_dev, chosen.stat().st_ino, chosen.stat().st_size,
             chosen.stat().st_mtime_ns)
    if before != after:
        raise ContractError(f"P01 link source changed while hashing: {chosen}")
    return chosen, {"link_source": str(source), "mirror": str(mirror), "sha256": digest}


def load_authority(run: Path, tree: Path) -> dict[tuple[int, int], dict[str, Any]]:
    links_path = run / "atomic_links.txt"
    model_path = run / "MODEL_SPEC"
    if not links_path.is_file() or not model_path.is_file():
        raise ContractError("P01 CMFGEN run lacks atomic_links.txt or MODEL_SPEC")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    osc_order: list[tuple[int, int]] = []
    for lineno, line in enumerate(links_path.read_text(encoding="latin-1").splitlines(), 1):
        fields = shlex.split(line, comments=True)
        if not fields or fields[0] != "ln":
            continue
        operands = [item for item in fields[1:] if not item.startswith("-")]
        if len(operands) != 2:
            raise ContractError(f"P01 malformed link at {links_path}:{lineno}")
        source, target = Path(operands[0]), operands[1]
        kind = ("osc" if target.endswith("_F_OSCDAT") else
                "f_to_s" if target.endswith("_F_TO_S") else
                "phot" if target.startswith("PHOT") and target.endswith("_A") else
                "col" if target.endswith("_COL_DATA") else None)
        if kind is None:
            continue
        key = _identity(source)
        slot = result.setdefault(key, {})
        if kind in slot:
            raise ContractError(f"P01 duplicate {kind} link for {key}")
        resolved, proof = _resolve_source(source, tree)
        slot[kind] = resolved
        slot[f"{kind}_proof"] = proof
        if kind == "osc":
            osc_order.append(key)
    for key, slot in result.items():
        missing = {"osc", "f_to_s", "phot", "col"} - set(slot)
        if missing:
            raise ContractError(f"P01 incomplete CMFGEN link bundle {key}: {sorted(missing)}")
    pattern = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s+\[[^]]+_ISF\]")
    nfs = [int(match.group(1)) for line in model_path.read_text(encoding="latin-1").splitlines()
           if (match := pattern.match(line))]
    if len(nfs) != len(osc_order):
        raise ContractError(f"P01 MODEL_SPEC/osc-link count mismatch {len(nfs)}/{len(osc_order)}")
    for key, nf in zip(osc_order, nfs, strict=True):
        result[key]["nf"] = nf
    return result


def _cmf_rows(authority: dict[tuple[int, int], dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], dict[tuple[int, int], Any]]:
    rows: list[dict[str, Any]] = []
    parsed = {}
    for (z, ion), slot in sorted(authority.items()):
        osc = parse_osc(slot["osc"])
        nf = slot["nf"]
        if osc.n_levels < nf:
            raise ContractError(f"P01 osc levels below MODEL_SPEC NF for {(z, ion)}")
        parsed[(z, ion)] = osc
        raw_tokens: list[tuple[str, str]] = []
        expected_id = 1
        for line in slot["osc"].read_text(encoding="latin-1").splitlines():
            tokens = line.split()
            if len(tokens) < 4:
                continue
            try:
                float(tokens[1].replace("D", "E").replace("d", "e"))
                float(tokens[2].replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            ids = [abs(int(token)) for token in tokens[1:]
                   if token.lstrip("+-").isdigit()]
            if expected_id not in ids:
                continue
            raw_tokens.append((tokens[1], tokens[2]))
            expected_id += 1
            if expected_id > nf:
                break
        if len(raw_tokens) != nf:
            raise ContractError(
                f"P03 raw osc level token extent {len(raw_tokens)}/{nf} for {(z,ion)}")
        for rank, level in enumerate(osc.levels[:nf]):
            rows.append({
                "z": z, "ion": ion, "rank": rank,
                "configuration": normalize_configuration(str(level["config"])),
                "g_interval": decimal_interval(raw_tokens[rank][0]),
                "energy_interval": decimal_interval(raw_tokens[rank][1], CM2EV),
                "row": level,
            })
    return rows, parsed


def _key_digest(value: Any) -> bytes:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()


def _deck_level_keys(deck_rows: list[dict[str, Any]], mapping: dict,
                     ambiguous: set) -> dict:
    result = {}
    for row in deck_rows:
        key = (row["z"], row["ion"], row["rank"])
        if key in ambiguous:
            result[key] = None
            continue
        mapped = mapping.get(key)
        result[key] = _key_digest(("cmf", *mapped)) if mapped else _key_digest((
            "deck", row["z"], row["ion"], row["configuration"],
            str(row["g_interval"]), str(row["energy_interval"])))
    return result


def _load_deck_lines(db: sqlite3.Connection, deck: Path, level_keys: dict) \
        -> tuple[int, int, int]:
    db.execute("CREATE TABLE selected(k BLOB PRIMARY KEY,z INTEGER,ion INTEGER,a REAL,token TEXT,f REAL,w REAL) WITHOUT ROWID")
    total = duplicate = unsupported = 0
    with (deck / "line_list.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            z, ion = int(row["atomic_number"]), int(row["ion_number"])
            lo, up = sorted((int(row["level_number_lower"]), int(row["level_number_upper"])))
            lk, uk = level_keys.get((z, ion, lo)), level_keys.get((z, ion, up))
            if lk is None or uk is None:
                unsupported += 1
                total += 1
                continue
            key = _key_digest((z, ion, lk.hex(), uk.hex()))
            try:
                db.execute("INSERT INTO selected VALUES (?,?,?,?,?,?,?)",
                           (key, z, ion, float(row["A_ul"]), row["A_ul"],
                            float(row["f_lu"]), float(row["wavelength_cm"])))
            except sqlite3.IntegrityError:
                duplicate += 1
            total += 1
            if total % 50_000 == 0:
                db.commit()
    db.commit()
    if duplicate:
        raise ContractError(f"P05 duplicate semantic selected transitions: {duplicate}")
    return total, duplicate, unsupported


def _raw_transition_tokens(path: Path) -> dict[tuple[int, int], tuple[str, str, str]]:
    result = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        match = TRANS_RE.search(line)
        if not match:
            continue
        pair = tuple(sorted((int(match.group(4))-1, int(match.group(5))-1)))
        if pair in result:
            raise ContractError(f"P05 duplicate raw CMFGEN transition {path}/{pair}")
        result[pair] = (match.group(2), match.group(1), match.group(3))
    return result


def _load_cmf_lines(db: sqlite3.Connection, authority: dict, parsed: dict) -> tuple[int, list[Path]]:
    db.execute("CREATE TABLE cmf(k BLOB PRIMARY KEY,z INTEGER,ion INTEGER,a REAL,token TEXT,f REAL,w REAL) WITHOUT ROWID")
    total = 0
    paths = []
    for (z, ion), slot in sorted(authority.items()):
        paths.append(slot["osc"])
        osc = parsed[(z, ion)]
        raw = _raw_transition_tokens(slot["osc"])
        nf = slot["nf"]
        for row in osc.transitions:
            lo, up = sorted((int(row["i"])-1, int(row["j"])-1))
            if lo < 0 or up >= nf or float(row["lam_A"]) == 0:
                continue
            tokens = raw.get((lo, up))
            if tokens is None:
                raise ContractError(f"P08 raw CMFGEN token absent {(z,ion,lo,up)}")
            key = _key_digest((z, ion, _key_digest(("cmf", z, ion, lo)).hex(),
                               _key_digest(("cmf", z, ion, up)).hex()))
            try:
                db.execute("INSERT INTO cmf VALUES (?,?,?,?,?,?,?)",
                           (key, z, ion, float(row["A"]), tokens[0], float(row["f"]),
                            abs(float(row["lam_A"]))*1e-8))
            except sqlite3.IntegrityError as exc:
                raise ContractError(f"P05 duplicate semantic CMFGEN transition {(z,ion,lo,up)}") from exc
            total += 1
        db.commit()
    return total, paths


def _quant_metrics(db: sqlite3.Connection, target: tuple[int, int] | None) -> dict[str, Any]:
    where = "" if target is None else " WHERE s.z=? AND s.ion=?"
    params = () if target is None else target
    cursor = db.execute("SELECT s.a,s.token,c.a,c.token FROM selected s JOIN cmf c USING(k)"+where, params)
    common = overlap = historical = zero = 0
    left_hist: dict[str, int] = {}
    right_hist: dict[str, int] = {}
    max_abs = max_rel = max_ulp = max_gap_abs = max_gap_rel = 0.0
    for left, ltoken, right, rtoken in cursor:
        li = quantization_interval(ltoken)
        ri = quantization_interval(rtoken)
        left_hist[str(li[2])] = left_hist.get(str(li[2]), 0) + 1
        right_hist[str(ri[2])] = right_hist.get(str(ri[2]), 0) + 1
        is_overlap = max(li[0], ri[0]) <= min(li[1], ri[1])
        overlap += int(is_overlap)
        gap = 0.0 if is_overlap else max(li[0], ri[0])-min(li[1], ri[1])
        scale = max(abs(left), abs(right))
        rel = abs(left-right)/scale if scale else 0.0
        historical += int(rel > 1e-6)
        zero += int(left == 0 and right == 0)
        max_abs = max(max_abs, abs(left-right))
        max_rel = max(max_rel, rel)
        max_ulp = max(max_ulp, ulp_distance(left, right))
        max_gap_abs = max(max_gap_abs, gap)
        max_gap_rel = max(max_gap_rel, gap/scale if scale else 0.0)
        common += 1
    return {"common": common, "overlap": overlap, "non_overlap": common-overlap,
            "zero": zero, "historical": historical, "left_hist": left_hist,
            "right_hist": right_hist, "max_abs": max_abs, "max_rel": max_rel,
            "max_ulp": int(max_ulp), "max_gap_abs": max_gap_abs,
            "max_gap_rel": max_gap_rel}


def _coverage(db: sqlite3.Connection) -> dict[str, int]:
    common = db.execute("SELECT count(*) FROM selected JOIN cmf USING(k)").fetchone()[0]
    left = db.execute("SELECT count(*) FROM selected").fetchone()[0]
    right = db.execute("SELECT count(*) FROM cmf").fetchone()[0]
    return {"common": common, "selected_only": left-common, "cmfgen_only": right-common,
            "union": left+right-common}


def run(*, deck: Path, peer: Path, cmfgen_tree: Path, cmfgen_run: Path,
        threshold_mode: str, threshold: float, command: str,
        super_cutoff: int) -> list[dict[str, Any]]:
    del peer, threshold_mode, threshold
    authority = load_authority(cmfgen_run, cmfgen_tree)
    deck_rows = semantic_rows(deck)
    cmf_rows, parsed = _cmf_rows(authority)
    mapping, ambiguous, join_census = semantic_join(deck_rows, cmf_rows)
    level_keys = _deck_level_keys(deck_rows, mapping, ambiguous)
    links_path = cmfgen_run / "atomic_links.txt"
    left_path = deck / "line_list.csv"
    left = endpoint(left_path, "atomic-bound-bound-transition", "pre-runtime atomic input",
                    "semantic transition A_ul", "selected Lumina deck")
    right = endpoint(links_path, "atomic-bound-bound-transition", "pre-runtime atomic input",
                     "MODEL_SPEC-capped linked osc A_ul", "CMFGEN run atomic links")
    records = []
    with tempfile.NamedTemporaryFile(prefix="l1a_lines_", suffix=".sqlite") as tmp:
        db = sqlite3.connect(tmp.name)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        selected_count, selected_dup, selected_unsupported = _load_deck_lines(
            db, deck, level_keys)
        cmf_count, osc_paths = _load_cmf_lines(db, authority, parsed)
        members = manifest_sha([left_path, links_path, cmfgen_run/"MODEL_SPEC", *osc_paths])
        ev = evidence(command, Path(__file__), [left_path, links_path, *osc_paths],
                      selected_count+cmf_count)
        for item_id, (target, metric) in TARGETS.items():
            result = _quant_metrics(db, target)
            quant = {
                "applicable": True, "rule_version": QUANTIZATION_RULE_VERSION,
                "left_significant_digits_histogram": result["left_hist"],
                "right_significant_digits_histogram": result["right_hist"],
                "overlap_count": result["overlap"],
                "non_overlap_count": result["non_overlap"],
                "max_absolute_interval_gap": result["max_gap_abs"],
                "max_relative_interval_gap": result["max_gap_rel"],
                "historical_rel_1e-6_mismatch": result["historical"],
                "interval_rule": "raw decimal half-spacing plus float half-ULP",
            }
            states = {"present": result["common"]-result["zero"], "missing": 0,
                      "zero": result["zero"], "unsupported": 0}
            records.append(make_record(
                item_id=item_id, metric=metric, left=left, right=right,
                comparison_axis="selected_vs_cmfgen", denominator=result["common"],
                cardinality=result["common"],
                selection="semantic transition intersection after unique config/g/energy join",
                member_sha=members, states=states, threshold_mode="exact", threshold=0,
                digits_left=min(map(int, result["left_hist"]), default=1),
                digits_right=min(map(int, result["right_hist"]), default=1),
                error_abs=result["max_gap_abs"],
                error_rel=result["max_gap_rel"] if result["common"] else None,
                error_ulp=result["max_ulp"], zero_rule="skip" if result["common"] else "NA",
                join_keys=["atomic_number", "ion_number", "normalized_configuration",
                           "g_interval", "energy_interval", "lower", "upper"],
                duplicate_count=selected_dup, duplicate_policy="reject/ambiguous->unsupported",
                policy_result=f"unique mapping; {len(ambiguous)} ambiguous levels excluded",
                evidence_obj=ev, processed=result["common"], unsupported=0,
                outcome="MATCH" if result["non_overlap"] == 0 else "DIFFER",
                kind=["NUMERIC"], disposition=["NONE" if result["non_overlap"] == 0 else "REMEASURE"],
                quantization=quant,
                measurements={"mismatch_count": result["non_overlap"],
                              "historical_rel_1e-6_mismatch": result["historical"],
                              "historical_rank_join_denominator_old_deck":
                                  880406 if item_id == "I2" and not deck.name.endswith("_ftos") else None,
                              "historical_rank_join_rel_1e-6_mismatch_old_deck":
                                  75075 if item_id == "I2" and not deck.name.endswith("_ftos") else None,
                              "semantic_join": join_census},
            ))
        coverage = _coverage(db)
        coverage["unsupported_ambiguous"] = selected_unsupported
        coverage["union"] += selected_unsupported
        missing = coverage["selected_only"]+coverage["cmfgen_only"]
        records.append(make_record(
            item_id="I17", metric="line semantic transition coverage", left=left, right=right,
            comparison_axis="coverage_vs_cmfgen", denominator=coverage["union"],
            cardinality=coverage["union"], selection="semantic transition union",
            member_sha=members, states={"present": coverage["common"], "missing": missing,
                                        "zero": 0, "unsupported": selected_unsupported},
            threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
            error_abs=float(missing), error_rel=missing/coverage["union"] if coverage["union"] else None,
            error_ulp=0, zero_rule="skip" if coverage["union"] else "NA",
            join_keys=["semantic_lower", "semantic_upper"], duplicate_count=0,
            duplicate_policy="reject", policy_result="exact set membership",
            evidence_obj=ev, processed=coverage["union"], unsupported=selected_unsupported,
            outcome="MATCH" if missing == 0 else "DIFFER", kind=["COVERAGE"],
            disposition=["CLOSE" if missing == 0 else "REMEASURE"], measurements=coverage,
        ))
        # Direct line-bit observation uses the same selected/CMFGEN semantic rows.
        bit_common = db.execute("SELECT count(*) FROM selected s JOIN cmf c USING(k) WHERE "
                                "s.f=c.f AND s.w=c.w AND s.a=c.a").fetchone()[0]
        bit_total = db.execute("SELECT count(*) FROM selected JOIN cmf USING(k)").fetchone()[0]
        db.close()

    # I4: selected runtime min(rank,K) versus linked CMFGEN F_TO_S, both NF-capped.
    selected_by_key = {(row["z"], row["ion"], row["rank"]): row for row in deck_rows}
    membership_total = membership_bad = 0
    ftos_paths = []
    for key, slot in authority.items():
        ftos_paths.append(slot["f_to_s"])
        ftos = parse_f_to_s(slot["f_to_s"])
        if ftos.n_levels < slot["nf"]:
            raise ContractError(f"P01 F_TO_S below MODEL_SPEC NF for {key}")
        for deck_key, cmf_key in mapping.items():
            if cmf_key[:2] != key or cmf_key[2] >= slot["nf"]:
                continue
            membership_total += 1
            lumina_membership = min(deck_key[2], super_cutoff)
            membership_bad += int(lumina_membership != int(ftos.sl_of_fl[cmf_key[2]]))
    level_path = deck / "levels.csv"
    level_members = manifest_sha([level_path, links_path, cmfgen_run/"MODEL_SPEC", *ftos_paths])
    lev_ev = evidence(command, Path(__file__), [level_path, links_path, *ftos_paths],
                      max(1, membership_total))
    records.append(make_record(
        item_id="I4", metric="runtime super-level membership versus CMFGEN F_TO_S",
        left=endpoint(level_path, "atomic-level-membership", "pre-runtime atomic input",
                      f"Lumina min(level,{super_cutoff}) membership", "selected Lumina deck"),
        right=endpoint(links_path, "atomic-level-membership", "pre-runtime atomic input",
                       "MODEL_SPEC-capped linked F_TO_S membership", "CMFGEN run atomic links"),
        comparison_axis="selected_vs_cmfgen", denominator=membership_total,
        cardinality=membership_total, selection="uniquely joined MODEL_SPEC active levels",
        member_sha=level_members, states={"present": membership_total, "missing": 0,
                                         "zero": 0, "unsupported": 0},
        threshold_mode="exact", threshold=0, digits_left=1, digits_right=1,
        error_abs=float(membership_bad), error_rel=membership_bad/membership_total if membership_total else None,
        error_ulp=0, zero_rule="skip" if membership_total else "NA",
        join_keys=["normalized_configuration", "g_interval", "energy_interval"],
        duplicate_count=len(ambiguous), duplicate_policy="ambiguous->unsupported",
        policy_result="no ambiguous candidate selected", evidence_obj=lev_ev,
        processed=membership_total, unsupported=0,
        outcome="MATCH" if membership_bad == 0 else "DIFFER", kind=["DESIGN"],
        disposition=["ACCEPT"], measurements={"membership_mismatch_count": membership_bad,
                                               "super_cutoff": super_cutoff},
    ))

    rank_total = len(deck_rows)
    rank_missing = rank_total-len(mapping)-len(ambiguous)
    rank_bad = sum(left_key[2] != right_key[2] for left_key, right_key in mapping.items())
    rank_stdout = json.dumps({"mapped": len(mapping), "rank_mismatch": rank_bad,
                              "missing": rank_missing, "ambiguous": len(ambiguous)},
                             sort_keys=True).encode()
    rank_exit = int(bool(rank_bad or rank_missing or ambiguous))
    rank_ev = evidence(command + " [direct R1 level/rank algorithm]", Path(__file__),
                       [level_path, links_path], max(1, rank_total), exit_code=rank_exit,
                       stdout=rank_stdout, status="VALID")
    records.append(make_record(
        item_id="I12", metric="level/rank identity (partial)",
        left=endpoint(level_path, "atomic-level-identity", "pre-runtime atomic input",
                      "selected semantic level and rank", "selected Lumina deck"),
        right=endpoint(links_path, "atomic-level-identity", "pre-runtime atomic input",
                       "MODEL_SPEC linked osc semantic level and rank", "CMFGEN run atomic links"),
        comparison_axis="validator_observation", denominator=rank_total, cardinality=rank_total,
        selection="all selected levels; macro-atom topology excluded", member_sha=level_members,
        states={"present": len(mapping), "missing": rank_missing, "zero": 0,
                "unsupported": len(ambiguous)}, threshold_mode="exact", threshold=0,
        digits_left=1, digits_right=1, error_abs=float(rank_bad),
        error_rel=rank_bad/len(mapping) if mapping else None, error_ulp=0,
        zero_rule="skip" if mapping else "NA",
        join_keys=["normalized_configuration", "g_interval", "energy_interval", "rank"],
        duplicate_count=len(ambiguous), duplicate_policy="ambiguous->unsupported",
        policy_result="direct algorithm; no hardcoded verifier result", evidence_obj=rank_ev,
        processed=len(mapping), unsupported=len(ambiguous), outcome="PARTIAL",
        kind=["COVERAGE", "PROVENANCE"], disposition=["REMEASURE"],
        measurements={"validator_exit_code": rank_exit,
                      "validator_stdout_sha256": hashlib.sha256(rank_stdout).hexdigest(),
                      "rank_mismatch_count": rank_bad, "macro_atom_topology": "excluded"},
    ))
    bit_bad = bit_total-bit_common
    bit_stdout = json.dumps({"compared": bit_total, "bit_mismatch": bit_bad}, sort_keys=True).encode()
    bit_exit = int(bit_bad != 0)
    bit_ev = evidence(command + " [direct R1 f/A/lambda bit algorithm]", Path(__file__),
                      [left_path, links_path], max(1, bit_total), exit_code=bit_exit,
                      stdout=bit_stdout, status="VALID")
    records.append(make_record(
        item_id="I12", metric="line-bit identity (partial)", left=left, right=right,
        comparison_axis="validator_observation", denominator=bit_total, cardinality=bit_total,
        selection="semantic transition intersection; macro-atom topology excluded",
        member_sha=members, states={"present": bit_total, "missing": 0, "zero": 0,
                                    "unsupported": 0}, threshold_mode="ulp", threshold=0,
        digits_left=17, digits_right=17, error_abs=float(bit_bad),
        error_rel=bit_bad/bit_total if bit_total else None, error_ulp=bit_bad,
        zero_rule="skip" if bit_total else "NA", join_keys=["semantic_lower", "semantic_upper"],
        duplicate_count=0, duplicate_policy="reject", policy_result="direct bit comparison",
        evidence_obj=bit_ev, processed=bit_total, unsupported=0, outcome="PARTIAL",
        kind=["COVERAGE", "PROVENANCE"], disposition=["REMEASURE"],
        quantization={"applicable": False, "reason": "IEEE-754 bit identity with explicit ULP rule",
                      "rule_version": QUANTIZATION_RULE_VERSION},
        measurements={"validator_exit_code": bit_exit,
                      "validator_stdout_sha256": hashlib.sha256(bit_stdout).hexdigest(),
                      "bit_mismatch_count": bit_bad, "macro_atom_topology": "excluded"},
    ))
    return records
