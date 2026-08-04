#!/usr/bin/env python3
"""Shared schema, provenance, and validation support for the L1-A instrument."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable


VERSION = "4.0.0"
RSS_LIMIT = 1 << 30
QUANTIZATION_RULE_VERSION = "decimal-interval-v1"
COMPARISON_AXES = {
    "selected_vs_cmfgen", "legacy_vs_current", "coverage_vs_cmfgen",
    "validator_observation",
}
EXPECTED_METRICS = {
    ("I1", "selected branch census versus CMFGEN"),
    ("I1", "selected Upsilon_eff(T) and q_ij(T) versus CMFGEN"),
    ("I19", "legacy branch identity versus CMFGEN"),
    ("I19", "current branch identity versus CMFGEN"),
    ("I19", "legacy-to-current physics change"),
    ("I2", "A_ul semantic transition identity"),
    ("I2a", "A_ul semantic transition identity Fe IV"),
    ("I2b", "A_ul semantic transition identity Ni IV"),
    ("I2c", "A_ul semantic transition identity Co IV"),
    ("I2d", "A_ul semantic transition identity Fe III"),
    ("I17", "line semantic transition coverage"),
    ("I4", "runtime super-level membership versus CMFGEN F_TO_S"),
    ("I12", "level/rank identity (partial)"),
    ("I12", "line-bit identity (partial)"),
    ("I3", "sigma(nu) selected versus CMFGEN PHOT"),
    ("I3a", "sigma(nu) selected versus CMFGEN PHOT Co IV"),
    ("I3b", "sigma(nu) selected versus CMFGEN PHOT Fe III"),
    ("I3c", "sigma(nu) selected versus CMFGEN PHOT Fe IV and Ni IV"),
    ("I3", "PHOT evaluator support census"),
    ("I17", "sigma semantic-level coverage"),
}
ID_ENUM = {
    "I1", "I2", "I2a", "I2b", "I2c", "I2d", "I3", "I3a", "I3b",
    "I3c", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I12", "I17",
    "I19",
}
THRESHOLD_MODES = {"exact", "ulp", "abs", "rel"}
POSEDNESS = {"WELL", "ILL", "UNVERIFIABLE", "NOT_APPLICABLE", "UNKNOWN"}
OUTCOMES = {
    "MATCH", "DIFFER", "NO-COUNTERPART", "RESOLVED", "PARTIAL",
    "NOT-ASSESSED", "INCOMPARABLE",
}
KINDS = {"BUG", "DESIGN", "DEFINITION", "COVERAGE", "PROVENANCE", "NUMERIC"}
DISPOSITIONS = {"REPAIR", "ACCEPT", "DEFINE", "REMEASURE", "CLOSE", "NONE"}


class ContractError(RuntimeError):
    """A schema, input, or resource contract was violated."""


def now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def epoch(path: Path) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))


def endpoint(path: Path, role: str, stage: str, quantity: str,
             authority: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "role": role,
        "stage": stage,
        "quantity_definition": quantity,
        "authority": authority,
        "consumed_path": str(path),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
        "epoch": epoch(resolved),
    }


def unknown_attestation() -> dict[str, str]:
    # v3 constraint 12 requires an explicit UNKNOWN, but its printed schema had
    # no status field.  Keep all component values explicit as well.
    return {
        "status": "UNKNOWN",
        "binary_sha": "UNKNOWN",
        "source_tree_sha": "UNKNOWN",
        "dirty_diff_sha": "UNKNOWN",
        "build_command": "UNKNOWN",
        "toolchain": "UNKNOWN",
        "env_manifest": "UNKNOWN",
    }


def evidence(command: str, validator: Path, inputs: Iterable[Path],
             record_count: int, *, status: str = "VALID", exit_code: int = 0,
             negative_control: str = "engine-specific fixture",
             stdout: bytes = b"") -> dict[str, Any]:
    input_paths = [Path(path) for path in inputs]
    return {
        "producer_sha": sha256_file(validator),
        "command": command,
        "exit_code": exit_code,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "validator": str(validator.resolve()),
        "created_at": now_utc(),
        "negative_control": negative_control,
        "input_shas": [sha256_file(path.resolve(strict=True)) for path in input_paths],
        "record_count": int(record_count),
        "evidence_status": status,
    }


def make_record(*, item_id: str, metric: str, left: dict[str, Any],
                right: dict[str, Any], denominator: int, cardinality: int,
                selection: str, member_sha: str, states: dict[str, int],
                threshold_mode: str, threshold: float, digits_left: int,
                digits_right: int, error_abs: float,
                error_rel: float | None, error_ulp: int,
                zero_rule: str, join_keys: list[str], duplicate_count: int,
                duplicate_policy: str, policy_result: str,
                evidence_obj: dict[str, Any], processed: int,
                unsupported: int, outcome: str, kind: list[str],
                disposition: list[str], posedness: str = "WELL",
                sensitive: bool = False, alternatives: list[str] | None = None,
                coordinate_frame: str = "semantic identity",
                coordinate_unit: str = "dimensionless",
                coordinate_range: list[Any] | None = None,
                interpolation: str = "none", extrapolation: str = "forbidden",
                weighting: str = "unweighted", measure: str = "count",
                bin_edges: str = "not-applicable",
                comparison_axis: str = "selected_vs_cmfgen",
                quantization: dict[str, Any] | None = None,
                expected_provenance: bool = False,
                entity_flags: dict[str, bool] | None = None,
                measurements: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item_id,
        "metric": metric,
        "comparison_axis": comparison_axis,
        "left": left,
        "right": right,
        "build_attestation": unknown_attestation(),
        "universe": {
            "selection": selection,
            "denominator": int(denominator),
            "cardinality": int(cardinality),
            "member_manifest_sha": member_sha,
        },
        "coordinate": {
            "frame": coordinate_frame,
            "unit": coordinate_unit,
            "range": coordinate_range or [],
            "interpolation": interpolation,
            "extrapolation": extrapolation,
        },
        "sampling": {
            "rule": "complete selected universe",
            "sensitive": bool(sensitive),
            "alternatives": alternatives or [],
            "weighting": weighting,
            "measure": measure,
            "bin_edges": bin_edges,
        },
        "precision": {
            "digits_left": int(digits_left),
            "digits_right": int(digits_right),
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
            "dtype": "float64" if threshold_mode == "ulp" else "not-applicable",
            "endianness": sys.byteorder if threshold_mode == "ulp" else "not-applicable",
            "ulp_distance_rule": (
                "monotone sign-bit ordered uint64 distance; NaN incomparable"
                if threshold_mode == "ulp" else "not-applicable"
            ),
        },
        "join": {
            "keys": join_keys,
            "normalization": "integers canonical; lower < upper",
            "multiplicity": "one-to-one",
            "duplicate_policy": duplicate_policy,
            "duplicate_count": int(duplicate_count),
            "policy_result": policy_result,
        },
        "states": {key: int(states[key]) for key in
                   ("present", "missing", "zero", "unsupported")},
        "entity_flags": entity_flags or {
            "ion_present": True,
            "quantity_present": True,
            "counterpart_present": True,
            "evaluator_supported": unsupported == 0,
        },
        "error": {
            "absolute": float(error_abs),
            "relative": None if error_rel is None else float(error_rel),
            "ulp": int(error_ulp),
            "zero_denominator_rule": zero_rule,
        },
        "evidence": evidence_obj,
        "resources": {
            "peak_rss_bytes": peak_rss_bytes(),
            "wall_seconds": 0.0,
            "chunk_points": 0,
            "processed": int(processed),
            "unsupported": int(unsupported),
        },
        "verdict": {
            "posedness": posedness,
            "outcome": outcome,
            "kind": kind,
            "disposition": disposition,
        },
        # Extension needed by v3 constraint 4, whose printed schema otherwise
        # has no location in which to mark the mandatory quantization warning.
        "schema_flags": [],
        "quantization": quantization or {
            "applicable": False,
            "reason": "exact categorical/set/count comparison",
            "rule_version": QUANTIZATION_RULE_VERSION,
        },
        "golden": {
            "status": "NOT_REGISTERED", "expected_denominator": None,
            "observed_denominator": int(denominator),
            "expected_historical_mismatch": None,
            "observed_historical_mismatch": None,
            "manifest_key": "", "checksum": "",
        },
        "expected_provenance_result": bool(expected_provenance),
        "run_complete": False,
        "judgment_eligible": False,
    }
    if measurements is not None:
        record["measurements"] = measurements
    return record


def validate_record(record: dict[str, Any]) -> list[str]:
    """Apply the twelve v3 cross-constraints in table order.

    The returned list contains non-fatal warnings.  Every other violation raises
    ContractError and therefore makes the CLI exit nonzero.
    """
    warnings: list[str] = []

    # C01: four mutually exclusive aggregate states exhaust the denominator.
    states = record["states"]
    if sum(int(states[k]) for k in ("present", "missing", "zero", "unsupported")) \
            != int(record["universe"]["denominator"]):
        raise ContractError("C01 states sum differs from universe.denominator")

    # C02: compare like role and like pipeline stage only.
    if (record["left"]["role"] != record["right"]["role"] or
            record["left"]["stage"] != record["right"]["stage"]):
        raise ContractError("C02 left/right role or stage differs")

    # C03: sampling-sensitive metrics must expose at least two alternatives.
    if record["sampling"]["sensitive"] and len(record["sampling"]["alternatives"]) < 2:
        raise ContractError("C03 sensitive sampling lacks two alternatives")

    if record.get("comparison_axis") not in COMPARISON_AXES:
        raise ContractError("P02 comparison_axis is absent or invalid")
    if (record["id"] != "I19" and record["comparison_axis"] != "legacy_vs_current" and
            ("legacy deck" in record["left"]["authority"].lower() or
             "peer" in record["left"]["authority"].lower())):
        raise ContractError("P02 selected/coverage axis uses an epoch peer endpoint")
    if record["comparison_axis"] == "legacy_vs_current" and record["id"] != "I19":
        raise ContractError("P02 epoch axis is allowed only for I19")

    # C04: quantization is a hard judgment gate, never a warning-only flag.
    precision = record["precision"]
    quant = record.get("quantization", {})
    if quant.get("applicable") is False:
        if not quant.get("reason"):
            raise ContractError("C04 exact metric lacks quantization exemption reason")
    elif quant.get("applicable") is True:
        required_quant = {"rule_version", "left_significant_digits_histogram",
                          "right_significant_digits_histogram", "overlap_count",
                          "non_overlap_count", "max_absolute_interval_gap",
                          "max_relative_interval_gap", "historical_rel_1e-6_mismatch"}
        if required_quant - set(quant):
            raise ContractError("C04 measured quantization rule is incomplete")
    else:
        raise ContractError("C04 quantization.applicable is not boolean")

    # C05: ULP mode is fully specified.
    if precision["threshold_mode"] == "ulp":
        for key in ("dtype", "endianness", "ulp_distance_rule"):
            if precision.get(key) in (None, "", "UNKNOWN", "not-applicable"):
                raise ContractError(f"C05 ulp mode lacks {key}")

    # C06: relative error is forbidden when its denominator is zero.
    zero_rule = record["error"]["zero_denominator_rule"]
    if zero_rule not in {"NA", "skip", "absolute_only"}:
        raise ContractError("C06 invalid zero_denominator_rule")
    if zero_rule in {"NA", "absolute_only"} and record["error"]["relative"] is not None:
        raise ContractError("C06 relative error emitted for a zero denominator")

    # C07: epoch mixing is legal only on the preregistered I19 epoch axis.
    if record["left"]["epoch"] != record["right"]["epoch"]:
        if record["comparison_axis"] == "legacy_vs_current":
            record["schema_flags"].append("INTENTIONAL_EPOCH_AXIS")
        elif record["comparison_axis"] == "validator_observation":
            record["schema_flags"].append("VALIDATOR_OBSERVATION_EPOCHS")
    if ("EPOCH_MIXED" in record["schema_flags"] and
            record["comparison_axis"] in {"selected_vs_cmfgen", "coverage_vs_cmfgen"}):
        raise ContractError("C07 accidental epoch mixture on selected/coverage axis")

    # C08: join identity and duplicate-policy application are non-vacuous.
    join = record["join"]
    if not join["keys"]:
        raise ContractError("C08 join.keys is empty")
    if not isinstance(join.get("duplicate_count"), int) or not join.get("duplicate_policy"):
        raise ContractError("C08 duplicate accounting/policy absent")
    if not join.get("policy_result"):
        raise ContractError("C08 duplicate policy result absent")

    # C09: the metric belongs to the declared ID universe.
    if record["id"] not in ID_ENUM:
        raise ContractError(f"C09 unknown id {record['id']!r}")

    # C10: evidence cannot pass vacuously.
    ev = record["evidence"]
    if int(ev["record_count"]) <= 0 or not ev["input_shas"]:
        raise ContractError("C10 evidence is empty")

    # C11: the one-GiB RSS ceiling is a hard gate.
    if int(record["resources"]["peak_rss_bytes"]) > RSS_LIMIT:
        raise ContractError("C11 peak_rss_bytes exceeds 2^30")

    # C12: absent build attestation is explicit UNKNOWN, never blank.
    att = record["build_attestation"]
    required_att = {
        "status", "binary_sha", "source_tree_sha", "dirty_diff_sha",
        "build_command", "toolchain", "env_manifest",
    }
    if required_att - set(att) or any(att[k] == "" for k in required_att):
        raise ContractError("C12 build attestation contains missing/blank values")
    if att["status"] == "UNKNOWN" and any(att[k] != "UNKNOWN" for k in
            required_att - {"status"}):
        raise ContractError("C12 UNKNOWN attestation has partially invented values")

    if record.get("expected_provenance_result"):
        verdict = record["verdict"]
        if verdict["posedness"] != "UNVERIFIABLE" or "PROVENANCE" not in verdict["kind"]:
            raise ContractError("P07 expected provenance result was not preserved")

    verdict = record["verdict"]
    if (verdict["posedness"] not in POSEDNESS or verdict["outcome"] not in OUTCOMES or
            not set(verdict["kind"]).issubset(KINDS) or
            not set(verdict["disposition"]).issubset(DISPOSITIONS)):
        raise ContractError("schema verdict enum violation")
    if precision["threshold_mode"] not in THRESHOLD_MODES:
        raise ContractError("schema threshold_mode enum violation")
    return warnings


def load_golden(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        data = json.load(stream)
    if data.get("version") != VERSION:
        raise ContractError(f"golden version {data.get('version')!r} != {VERSION!r}")
    for key, row in data.get("metrics", {}).items():
        required = {"command", "version", "specimen_id", "comparison_axis",
                    "authority_manifest_sha", "denominator_rule",
                    "quantization_rule_version", "checksum"}
        if required - set(row):
            raise ContractError(f"golden metric {key} lacks {sorted(required-set(row))}")
        payload = {field: row.get(field) for field in (
            "command", "version", "specimen_id", "comparison_axis",
            "authority_manifest_sha", "expected_denominator", "denominator_rule",
            "historical_expected_mismatch", "quantization_rule_version")}
        got = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if got != row["checksum"]:
            raise ContractError(f"golden metric {key} checksum mismatch")
    for key, row in data.get("registrations", {}).items():
        required = {"command", "version", "comparison_axis", "denominator_rule",
                    "quantization_rule_version", "checksum"}
        if required-set(row):
            raise ContractError(f"golden registration {key} lacks {sorted(required-set(row))}")
        payload = {field: row.get(field) for field in (
            "command", "version", "comparison_axis", "denominator_rule",
            "quantization_rule_version", "expected_result")}
        got = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if got != row["checksum"]:
            raise ContractError(f"golden registration {key} checksum mismatch")
    return data


def compare_golden(record: dict[str, Any], golden: dict[str, Any]) -> None:
    specimen = Path(record["left"]["consumed_path"]).parent.name
    key = "|".join((specimen, record["comparison_axis"], record["id"],
                    record["metric"], record["universe"]["member_manifest_sha"],
                    record["quantization"]["rule_version"]))
    expected = golden.get("metrics", {}).get(key)
    if expected is None:
        wildcard_key = "|".join((specimen, record["comparison_axis"], record["id"],
                                  record["metric"], "AUTHORITY_AT_RUNTIME",
                                  record["quantization"]["rule_version"]))
        expected = golden.get("metrics", {}).get(wildcard_key)
        key = wildcard_key
    if expected is None:
        record["golden"]["status"] = "NOT_REGISTERED"
        registration_key = "|".join((record["comparison_axis"], record["id"], record["metric"]))
        registration = golden.get("registrations", {}).get(registration_key)
        if registration is None:
            _compare_contract_golden(record, golden)
            return
        expected_result = registration.get("expected_result")
        provenance_ok = (expected_result != "UNVERIFIABLE/PROVENANCE" or
                         (record["verdict"]["posedness"] == "UNVERIFIABLE" and
                          "PROVENANCE" in record["verdict"]["kind"]))
        actual_key = "|".join((specimen, record["comparison_axis"], record["id"],
                               record["metric"], record["universe"]["member_manifest_sha"],
                               record["quantization"]["rule_version"]))
        runtime_binding = {
            "registration_checksum": registration["checksum"],
            "command": record["evidence"]["command"], "version": VERSION,
            "specimen_id": specimen, "comparison_axis": record["comparison_axis"],
            "id": record["id"], "metric": record["metric"],
            "authority_manifest_sha": record["universe"]["member_manifest_sha"],
            "denominator_rule": registration["denominator_rule"],
            "observed_denominator": record["universe"]["denominator"],
            "quantization_rule_version": record["quantization"]["rule_version"],
        }
        runtime_checksum = hashlib.sha256(json.dumps(
            runtime_binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        record["golden"] = {
            "status": "MATCH" if provenance_ok else "DIFFER",
            "expected_denominator": None,
            "observed_denominator": record["universe"]["denominator"],
            "expected_historical_mismatch": None,
            "observed_historical_mismatch": record.get("measurements", {}).get(
                "historical_rel_1e-6_mismatch"),
            "manifest_key": actual_key,
            "checksum": runtime_checksum,
        }
        _compare_contract_golden(record, golden)
        return
    observed_denominator = record["universe"]["denominator"]
    observed_mismatch = record.get("measurements", {}).get("historical_rel_1e-6_mismatch",
        record.get("measurements", {}).get("mismatch_count"))
    matches = (
        (expected.get("expected_denominator") is None or
         observed_denominator == expected["expected_denominator"]) and
        (expected.get("historical_expected_mismatch") is None or
         observed_mismatch == expected["historical_expected_mismatch"])
    )
    record["golden"] = {
        "status": "MATCH" if matches else "DIFFER",
        "expected_denominator": expected.get("expected_denominator"),
        "observed_denominator": observed_denominator,
        "expected_historical_mismatch": expected.get("historical_expected_mismatch"),
        "observed_historical_mismatch": observed_mismatch,
        "manifest_key": key,
        "checksum": expected["checksum"],
    }
    _compare_contract_golden(record, golden)


def _compare_contract_golden(record: dict[str, Any], golden: dict[str, Any]) -> None:
    measurements = record.setdefault("measurements", {})
    contracts = golden.get("contracts", {})
    pairs: list[tuple[str, str]] = []
    if record["id"] == "I3" and record["metric"] in {
            "sigma(nu) semantic level join", "PHOT evaluator support census"}:
        pairs.append(("unsupported_levels_type_2_3_8",
                      "phot_type_2_3_8_expected_unsupported_levels"))
    if record["id"] == "I19" and record["metric"].startswith("identity axis"):
        pairs.extend([
            ("legacy_mapped_rows", "i19_legacy_mapped_rows_expected"),
            ("current_mapped_rows", "i19_current_mapped_rows_expected"),
            ("legacy_total_mapped_rows", "i19_legacy_total_mapped_expected"),
            ("current_total_mapped_rows", "i19_current_total_mapped_expected"),
        ])
    if not pairs:
        return
    comparisons = {}
    for observed_key, expected_key in pairs:
        expected_value = contracts.get(expected_key)
        observed_value = measurements.get(observed_key)
        comparisons[observed_key] = {
            "expected": expected_value,
            "observed": observed_value,
            "status": "MATCH" if observed_value == expected_value else "DIFFER",
        }
    measurements["golden_contracts"] = comparisons


def read_levels(path: Path) -> list[dict[str, str]]:
    with (path / "levels.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "atomic_number", "ion_number", "level_number", "energy_eV", "g",
        "super_level",
    }
    if not rows or required - set(rows[0]):
        raise ContractError(f"invalid levels.csv below {path}")
    return rows


def normalize_configuration(value: str) -> str:
    """Conservative normalization: case/spacing only; punctuation remains identity."""
    return " ".join(str(value).strip().casefold().split())


def decimal_interval(token: str, scale: Decimal = Decimal(1)) -> tuple[Decimal, Decimal]:
    clean = str(token).strip().replace("D", "E").replace("d", "e")
    try:
        value = Decimal(clean) * scale
    except InvalidOperation as exc:
        raise ContractError(f"invalid decimal token {token!r}") from exc
    exponent = Decimal(clean).as_tuple().exponent
    spacing = (Decimal(10) ** exponent) * abs(scale)
    half = spacing / 2
    return value-half, value+half


def intervals_overlap(left: tuple[Decimal, Decimal],
                      right: tuple[Decimal, Decimal]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def semantic_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_levels(path)
    result = []
    for row in rows:
        if "configuration" not in row:
            raise ContractError(f"levels.csv below {path} lacks configuration")
        result.append({
            "z": int(row["atomic_number"]), "ion": int(row["ion_number"]),
            "rank": int(row["level_number"]),
            "configuration": normalize_configuration(row["configuration"]),
            "g_interval": decimal_interval(row["g"]),
            "energy_interval": decimal_interval(row["energy_eV"]),
            "row": row,
        })
    return result


def semantic_join(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) \
        -> tuple[dict[tuple[int, int, int], tuple[int, int, int]], set[tuple[int, int, int]], dict[str, int]]:
    buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in right_rows:
        buckets.setdefault((row["z"], row["ion"], row["configuration"]), []).append(row)
    mapping = {}
    ambiguous: set[tuple[int, int, int]] = set()
    unmatched = 0
    reverse: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for row in left_rows:
        lkey = (row["z"], row["ion"], row["rank"])
        candidates = [candidate for candidate in buckets.get(
            (row["z"], row["ion"], row["configuration"]), [])
            if intervals_overlap(row["g_interval"], candidate["g_interval"])
            and intervals_overlap(row["energy_interval"], candidate["energy_interval"])]
        if len(candidates) != 1:
            if candidates:
                ambiguous.add(lkey)
            else:
                unmatched += 1
            continue
        candidate = candidates[0]
        rkey = (candidate["z"], candidate["ion"], candidate["rank"])
        mapping[lkey] = rkey
        reverse.setdefault(rkey, []).append(lkey)
    for rkey, lkeys in reverse.items():
        if len(lkeys) > 1:
            ambiguous.update(lkeys)
            for key in lkeys:
                mapping.pop(key, None)
    return mapping, ambiguous, {"mapped": len(mapping), "ambiguous": len(ambiguous),
                                "unmatched": unmatched}


def semantic_level_map(path: Path) -> tuple[dict[tuple[int, int, int], int], list[dict[str, str]]]:
    rows = read_levels(path)
    result: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(rows):
        key = (int(row["atomic_number"]), int(row["ion_number"]),
               int(row["level_number"]))
        if key in result:
            raise ContractError(f"duplicate level identity {key} below {path}")
        result[key] = index
    return result, rows


def significant_digits(token: str) -> int:
    mantissa = str(token).strip().lower().replace("d", "e").split("e", 1)[0]
    digits = "".join(character for character in mantissa if character.isdigit())
    stripped = digits.lstrip("0")
    return len(stripped) if stripped else 1


def quantization_interval(token: str) -> tuple[float, float, int, int, str]:
    clean = str(token).strip().replace("D", "E").replace("d", "e")
    value = float(clean)
    dec = Decimal(clean)
    exponent = dec.as_tuple().exponent
    spacing = float(Decimal(10) ** exponent)
    ulp = math.ulp(value)
    half = abs(spacing) / 2 + ulp / 2
    notation = "scientific" if "e" in clean.lower() else "fixed"
    return value-half, value+half, significant_digits(clean), exponent, notation


def finalize_run(records: list[dict[str, Any]], requested_all: bool,
                 failures: list[str]) -> None:
    keys = [(record["id"], record["metric"]) for record in records]
    complete = not failures and (not requested_all or set(keys) == EXPECTED_METRICS) \
        and len(keys) == len(set(keys))
    if requested_all and set(keys) != EXPECTED_METRICS:
        missing = sorted(EXPECTED_METRICS-set(keys))
        extra = sorted(set(keys)-EXPECTED_METRICS)
        failures.append(f"P09 metric completeness missing={missing} extra={extra}")
        complete = False
    if len(keys) != len(set(keys)):
        failures.append("P09 duplicate metric key")
        complete = False
    for record in records:
        record["run_complete"] = complete
        record["judgment_eligible"] = complete and record["golden"]["status"] != "NOT_REGISTERED"
        if record["golden"]["status"] == "DIFFER":
            record["judgment_eligible"] = False
            failures.append(f"P07 historical golden mismatch {record['id']}/{record['metric']}")


def ulp_distance(left: float, right: float) -> int:
    import struct
    if not (math.isfinite(left) and math.isfinite(right)):
        return (1 << 64) - 1
    a = struct.unpack(">Q", struct.pack(">d", left))[0]
    b = struct.unpack(">Q", struct.pack(">d", right))[0]
    a = (~a & ((1 << 64) - 1)) if a >> 63 else a | (1 << 63)
    b = (~b & ((1 << 64) - 1)) if b >> 63 else b | (1 << 63)
    return abs(a - b)


def differs(left: float, right: float, mode: str, threshold: float) -> tuple[bool, float, float | None, int]:
    absolute = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = absolute / scale if scale else None
    ulp = ulp_distance(left, right)
    if mode == "exact":
        bad = ulp != 0
    elif mode == "ulp":
        bad = ulp > int(threshold)
    elif mode == "abs":
        bad = absolute > threshold
    elif mode == "rel":
        bad = absolute > 0.0 if relative is None else relative > threshold
    else:
        raise ContractError(f"unknown threshold mode {mode}")
    return bad, absolute, relative, ulp
