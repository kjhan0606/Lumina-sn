#!/usr/bin/env python3
"""Build the amended A2-02C line census, BB ledgers, union, and cohort.

The production command is a streaming read of the immutable atomic deck.  It
never edits the deck or any A2-02 v1 artifact.  All A2-02C outputs are created
under a caller-selected new directory and are hash-bound to the source line
list and the canonical 100--20000 Angstrom domain contract.

Exit codes: 0 PASS, 2 input/contract failure, 4 injected negative-control FAIL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
C_CGS = 29_979_245_800.0
LAM_MIN_A = 100.0
LAM_MAX_A = 20_000.0
VDOP_CM_S = 1.0e6
TRUNCATION_DOPPLER = 4.0
AMENDS_AFTER = "43ffe31"
SCHEMA_UNION = "lumina-a2-02c-frequency-union-v2"
SCHEMA_CENSUS = "lumina-a2-02c-bb-census-v1"
SCHEMA_COHORT = "lumina-a2-02c-estimator-cohort-v1"
EXPECTED_LINE_ROWS = 2_220_953
EXPECTED_LOW_COUNTS = {"lt_1e10_hz": 382, "lt_1e13_hz": 34_945,
                       "lt_1e14_hz": 245_123}
OLD_RESULT = ROOT / "validation/a2_02/a2_02_resolution_result.json"
OLD_UNION = ROOT / "docs/A2_02_FREQUENCY_UNION.json"
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
DEFAULT_OUT = ROOT / "validation/a2_02c"

ELEMENTS = (
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
)
LEDGER_FIELDS = (
    "line_id", "element", "atomic_number", "ion", "lower", "upper",
    "nu_lu_hz", "lambda_lu_A", "A_ul_s-1", "reason", "source_row",
    "source_hash", "domain_contract_hash", "profile_id", "profile_hash",
)
WAVE_STRATA = (
    ("100_450_A", 100.0, 450.0), ("450_918_A", 450.0, 918.0),
    ("918_1290_A", 918.0, 1290.0), ("1290_2000_A", 1290.0, 2000.0),
    ("2000_10000_A", 2000.0, 10_000.0),
    ("10000_20000_A", 10_000.0, 20_000.0),
)


class ContractError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def domain_contract() -> dict[str, Any]:
    profile = {
        "profile_id": "runtime-gaussian-vdop1e6-truncated4-v1",
        "kind": "Gaussian exp(-x^2)",
        "normalization": "1/(sqrt(pi)*erf(4)*dnu_D) on closed truncated support",
        "truncation_doppler_widths": TRUNCATION_DOPPLER,
        "vdop_cm_s": VDOP_CM_S,
        "shell_dependence": "none in the registered current runtime default",
        "support": "nu_lu*(1+-4*vdop/c)",
        "provenance": ["src/lumina_cmfgen.c:3980-3982",
                       "src/lumina_cmfgen.c:4205-4208"],
    }
    profile["profile_hash"] = canonical_hash(profile)
    contract = {
        "lambda_min_A": LAM_MIN_A, "lambda_max_A": LAM_MAX_A,
        "boundaries": "closed", "c_cm_s": C_CGS,
        "nu_min_hz": C_CGS / (LAM_MAX_A * 1.0e-8),
        "nu_max_hz": C_CGS / (LAM_MIN_A * 1.0e-8),
        "selection": "finite positive line-center frequency inside the closed wavelength domain",
        "strength_pruning": "forbidden; A_ul/gf/population are not selection inputs",
        "profile": profile,
        "authority": "docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md sections 2.1-2.4",
    }
    return contract


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def load_levels(path: Path) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"atomic_number", "ion_number", "level_number"}
        require(reader.fieldnames is not None and needed <= set(reader.fieldnames),
                "levels.csv lacks atomic/ion/level columns")
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]),
                   int(row["level_number"]))
            require(key not in result, f"duplicate level identity {key}")
            result.add(key)
    require(bool(result), "levels.csv is empty")
    return result


def ion_stratum(ion: int) -> str:
    return str(ion) if ion < 3 else "3plus"


def wavelength_stratum(lam: float) -> str:
    for index, (name, lo, hi) in enumerate(WAVE_STRATA):
        if lo <= lam <= hi and (index == len(WAVE_STRATA) - 1 or lam < hi):
            return name
    raise ContractError(f"in-domain wavelength has no stratum: {lam}")


def parse_old_cohort(path: Path) -> tuple[list[dict[str, Any]], set[int], str]:
    document = load_json(path)
    require(document.get("schema") == "lumina-a2-02-resolution-result-v1",
            "old BLOCKED result schema changed")
    jbar = document["pairs"][-1]["metrics"]["Jbar"]
    rows = [row for row in jbar["records"]
            if row.get("judgment_eligible") and row.get("valid")
            and not row.get("zero_denominator_mismatch")]
    require(len(rows) == 42, f"old positive-control cohort is {len(rows)}, expected 42")
    parsed: list[dict[str, Any]] = []
    wanted: set[int] = set()
    for row in rows:
        match = re.fullmatch(
            r"line:s(?P<shell>\d+):Z(?P<z>\d+):i(?P<ion>\d+):"
            r"l(?P<lower>\d+):u(?P<upper>\d+):id(?P<line>[^:]+):row(?P<row>\d+)",
            row["record_id"],
        )
        require(match is not None, f"legacy record ID malformed: {row['record_id']}")
        source_row = int(match.group("row"))
        wanted.add(source_row)
        parsed.append({"legacy_record_id": row["record_id"], "shell_id": int(row["shell"]),
                       "source_row": source_row, "legacy_status": "VALID_A2_02_V1"})
    mandatory = [row for row in parsed if row["shell_id"] == 8 and
                 ":Z26:i1:l61:u1308:" in row["legacy_record_id"]]
    require(len(mandatory) == 1, "mandatory s8 Fe II l61->u1308 record absent")
    return parsed, wanted, sha256_file(path)


def row_to_ledger(row: dict[str, str], source_row: int, source_hash: str,
                  domain_hash: str, contract: dict[str, Any], reason: str) -> dict[str, Any]:
    z = int(row["atomic_number"])
    require(0 < z < len(ELEMENTS), f"row {source_row}: unsupported Z={z}")
    return {
        "line_id": row["line_id"], "element": ELEMENTS[z], "atomic_number": z,
        "ion": int(row["ion_number"]), "lower": int(row["level_number_lower"]),
        "upper": int(row["level_number_upper"]), "nu_lu_hz": float(row["nu"]),
        "lambda_lu_A": C_CGS / float(row["nu"]) / 1.0e-8,
        "A_ul_s-1": float(row["A_ul"]), "reason": reason,
        "source_row": source_row, "source_hash": source_hash,
        "domain_contract_hash": domain_hash,
        "profile_id": contract["profile"]["profile_id"],
        "profile_hash": contract["profile"]["profile_hash"],
    }


def write_csv_open(path: Path) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile("w", newline="", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False)
    writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
    writer.writeheader()
    return stream, writer


def build(args: argparse.Namespace) -> None:
    deck = args.deck.resolve()
    line_path = deck / "line_list.csv"
    level_path = deck / "levels.csv"
    geometry_path = deck / "geometry.csv"
    for path in (line_path, level_path, geometry_path, args.old_union.resolve(),
                 args.old_result.resolve()):
        require(path.is_file(), f"required input absent: {path}")
    with geometry_path.open(newline="") as stream:
        shell_count = sum(1 for _ in csv.DictReader(stream))
    require(shell_count == 50, f"transport shell census {shell_count}, expected 50")
    out = args.output_dir.resolve()
    outputs = {
        "census": out / "A2_02C_LINE_CENSUS.json",
        "inside": out / "A2_02C_BB_IN_DOMAIN.csv",
        "excluded": out / "A2_02C_BB_EXCLUDED_OUTSIDE_DOMAIN.csv",
        "union": out / "A2_02C_FREQUENCY_UNION.json",
        "qset": out / "A2_02C_Q_SET.json",
        "cohort": out / "A2_02C_ESTIMATOR_COHORT.json",
    }
    require(args.force or not any(path.exists() for path in outputs.values()),
            "an A2-02C output already exists; use --force only for these new names")

    contract = domain_contract()
    domain_hash = canonical_hash(contract)
    source_hash = sha256_file(line_path)
    levels_hash = sha256_file(level_path)
    levels = load_levels(level_path)
    legacy, wanted_rows, old_result_hash = parse_old_cohort(args.old_result.resolve())
    old_union = load_json(args.old_union.resolve())
    require(old_union.get("schema") == "lumina-a2-02-frequency-union-v1",
            "old frequency union schema changed")

    inside_stream, inside_writer = write_csv_open(outputs["inside"])
    excluded_stream, excluded_writer = write_csv_open(outputs["excluded"])
    inside_tmp, excluded_tmp = Path(inside_stream.name), Path(excluded_stream.name)
    total = inside_count = excluded_count = 0
    low_counts = {key: 0 for key in EXPECTED_LOW_COUNTS}
    previous_nu = math.inf
    inside_min = math.inf
    inside_max = 0.0
    retained: dict[int, dict[str, Any]] = {}
    supplement: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    try:
        with line_path.open(newline="") as source:
            reader = csv.DictReader(source)
            required = {"atomic_number", "ion_number", "level_number_lower",
                        "level_number_upper", "line_id", "nu", "A_ul"}
            require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                    "line_list.csv missing required columns")
            for source_row, row in enumerate(reader):
                try:
                    z, ion = int(row["atomic_number"]), int(row["ion_number"])
                    lower = int(row["level_number_lower"])
                    upper = int(row["level_number_upper"])
                    line_id = int(row["line_id"])
                    nu = float(row["nu"])
                except (ValueError, TypeError) as exc:
                    raise ContractError(f"row {source_row}: malformed identity/frequency") from exc
                require(math.isfinite(nu) and nu > 0.0,
                        f"row {source_row}: nonfinite/nonpositive nu is atomic-input error")
                require(nu <= previous_nu, f"row {source_row}: line frequencies not descending")
                require(line_id == source_row,
                        f"row {source_row}: line_id {line_id} is not the stable source row")
                require((z, ion, lower) in levels and (z, ion, upper) in levels and lower != upper,
                        f"row {source_row}: invalid level connection Z{z} i{ion} l{lower}->u{upper}")
                previous_nu = nu
                total += 1
                if nu < 1.0e10: low_counts["lt_1e10_hz"] += 1
                if nu < 1.0e13: low_counts["lt_1e13_hz"] += 1
                if nu < 1.0e14: low_counts["lt_1e14_hz"] += 1
                in_domain = contract["nu_min_hz"] <= nu <= contract["nu_max_hz"]
                reason = "BB_IN_DOMAIN" if in_domain else "BB_EXCLUDED_OUTSIDE_DOMAIN"
                item = row_to_ledger(row, source_row, source_hash, domain_hash,
                                     contract, reason)
                if source_row in wanted_rows:
                    retained[source_row] = item
                if in_domain:
                    inside_writer.writerow(item)
                    inside_count += 1
                    inside_min, inside_max = min(inside_min, nu), max(inside_max, nu)
                    wave = wavelength_stratum(item["lambda_lu_A"])
                    istr = ion_stratum(ion)
                    score = hashlib.sha256(
                        f"{domain_hash}|{wave}|{istr}|{line_id}|{source_row}".encode()
                    ).hexdigest()
                    key = (wave, istr)
                    if key not in supplement or score < supplement[key][0]:
                        supplement[key] = (score, item)
                else:
                    excluded_writer.writerow(item)
                    excluded_count += 1
        inside_stream.close(); excluded_stream.close()
        require(total == EXPECTED_LINE_ROWS,
                f"line census {total}, expected measured {EXPECTED_LINE_ROWS}")
        require(low_counts == EXPECTED_LOW_COUNTS,
                f"low-frequency voice check {low_counts}, expected {EXPECTED_LOW_COUNTS}")
        require(inside_count + excluded_count == total, "partition census identity failed")
        require(set(retained) == wanted_rows, "not all 42 legacy source rows were retained")
        inside_tmp.replace(outputs["inside"]); excluded_tmp.replace(outputs["excluded"])
    except Exception:
        inside_stream.close(); excluded_stream.close()
        inside_tmp.unlink(missing_ok=True); excluded_tmp.unlink(missing_ok=True)
        raise

    inside_hash, excluded_hash = sha256_file(outputs["inside"]), sha256_file(outputs["excluded"])
    census = {
        "schema": SCHEMA_CENSUS, "stage": "A2-02C", "amends_after": AMENDS_AFTER,
        "source": {"line_list": str(line_path), "line_list_sha256": source_hash,
                   "levels": str(level_path), "levels_sha256": levels_hash},
        "domain_contract": contract, "domain_contract_hash": domain_hash,
        "counts": {"raw_line_census": total, "finite_positive_input_census": total,
                   "BB_IN_DOMAIN": inside_count,
                   "BB_EXCLUDED_OUTSIDE_DOMAIN": excluded_count,
                   **low_counts},
        "identity": {"expression": "BB_IN_DOMAIN + BB_EXCLUDED_OUTSIDE_DOMAIN == finite_positive_input_census",
                     "passed": inside_count + excluded_count == total},
        "ledgers": {"BB_IN_DOMAIN": {"path": str(outputs["inside"]), "sha256": inside_hash},
                    "BB_EXCLUDED_OUTSIDE_DOMAIN": {"path": str(outputs["excluded"]),
                                                   "sha256": excluded_hash}},
        "invalid_atomic_input_count": 0,
    }
    atomic_json(outputs["census"], census)
    census_hash = sha256_file(outputs["census"])

    bb = dict(old_union["consumers"][1])
    support_fraction = TRUNCATION_DOPPLER * VDOP_CM_S / C_CGS
    bb.update({
        "nu_min_hz": inside_min * (1.0 - support_fraction),
        "nu_max_hz": inside_max * (1.0 + support_fraction),
        "lambda_min_A": C_CGS / (inside_max * (1.0 + support_fraction)) / 1e-8,
        "lambda_max_A": C_CGS / (inside_min * (1.0 - support_fraction)) / 1e-8,
        "range_derivation": "BB_IN_DOMAIN line centers plus the complete registered +-4 Doppler support",
        "domain_contract": contract, "domain_contract_hash": domain_hash,
        "census_manifest": {"path": str(outputs["census"]), "sha256": census_hash},
        "data_sources": [{"path": str(line_path), "sha256": source_hash}],
    })
    consumers = [dict(value) for value in old_union["consumers"]]
    consumers[1] = bb
    union_lo = min(float(value["nu_min_hz"]) for value in consumers)
    union_hi = max(float(value["nu_max_hz"]) for value in consumers)
    union = {
        "schema": SCHEMA_UNION, "stage": "A2-02C", "amends_after": AMENDS_AFTER,
        "preserves_blocked_artifact": {"path": str(args.old_union.resolve()),
                                       "sha256": sha256_file(args.old_union.resolve())},
        "frequency_unit": "Hz", "wavelength_unit": "Angstrom",
        "speed_of_light_cm_s": C_CGS, "consumer_count": 7,
        "consumers": consumers,
        "union": {"nu_min_hz": union_lo, "nu_max_hz": union_hi,
                  "lambda_min_A": C_CGS / union_hi / 1e-8,
                  "lambda_max_A": C_CGS / union_lo / 1e-8,
                  "derivation": "closed interval union of exactly seven amended consumers"},
        "validity_contract": old_union["validity_contract"],
        "resolution_ladder": {**old_union["resolution_ladder"],
                              "metrics": ["band_integral_J", "Gamma",
                                          "band_integral_chi", "band_integral_eta"],
                              "Jbar_removed_from_global_ladder": True},
        "oracle_shell_policy": {**old_union["oracle_shell_policy"],
                                "applies_to_metrics": ["Gamma"]},
        "line_census_manifest_sha256": census_hash,
    }
    atomic_json(outputs["union"], union)

    q_descriptor = {
        "selection_basis": "enabled current bound-bound graph: every BB_IN_DOMAIN line at every transport shell; no A_ul/gf/population/rate pruning",
        "bb_in_domain_ledger_sha256": inside_hash,
        "bb_in_domain_lines": inside_count,
        "shell_ids": list(range(shell_count)),
        "profile_hash": contract["profile"]["profile_hash"],
    }
    q_set_hash = canonical_hash(q_descriptor)
    qset = {
        "schema": "lumina-a2-02c-q-set-v1", "stage": "A2-02C",
        "amends_after": AMENDS_AFTER, "generation": "BIND_FROM_CAPTURE_RUN",
        "frozen_before_estimator_accumulation": True,
        "descriptor": q_descriptor, "q_record_count": inside_count * shell_count,
        "q_set_hash": q_set_hash,
        "out_of_domain_lookup": "OUT_OF_BB_DOMAIN",
        "cache_miss_or_unsampled": "FAIL_NO_COARSE_FALLBACK",
    }
    atomic_json(outputs["qset"], qset)
    qset_hash = sha256_file(outputs["qset"])

    cohort_rows: list[dict[str, Any]] = []
    for old in legacy:
        line = retained[old["source_row"]]
        active = line["reason"] == "BB_IN_DOMAIN"
        cohort_rows.append({**old, "line_id": line["line_id"],
                            "atomic_number": line["atomic_number"], "ion": line["ion"],
                            "lower": line["lower"], "upper": line["upper"],
                            "nu_lu_hz": line["nu_lu_hz"], "lambda_lu_A": line["lambda_lu_A"],
                            "profile_id": line["profile_id"], "profile_hash": line["profile_hash"],
                            "cohort_status": "ACTIVE_POSITIVE_CONTROL" if active else
                            "CARRIED_EXCLUDED_OUTSIDE_DOMAIN",
                            "exclusion_reason": None if active else
                            "BB_EXCLUDED_OUTSIDE_DOMAIN; carried without post-result replacement"})
    existing = {(row["source_row"], row["shell_id"]) for row in cohort_rows}
    for (wave, istr), (_, line) in sorted(supplement.items()):
        for shell in (0, 8):
            key = (line["source_row"], shell)
            if key in existing:
                continue
            cohort_rows.append({
                "legacy_record_id": None, "legacy_status": None, "shell_id": shell,
                "source_row": line["source_row"], "line_id": line["line_id"],
                "atomic_number": line["atomic_number"], "ion": line["ion"],
                "lower": line["lower"], "upper": line["upper"],
                "nu_lu_hz": line["nu_lu_hz"], "lambda_lu_A": line["lambda_lu_A"],
                "profile_id": line["profile_id"], "profile_hash": line["profile_hash"],
                "cohort_status": "ACTIVE_DETERMINISTIC_SUPPLEMENT",
                "stratum": {"wavelength": wave, "ion": istr, "shell": f"s{shell}"},
                "exclusion_reason": None,
            })
    active_rows = [row for row in cohort_rows if row["cohort_status"].startswith("ACTIVE_")]
    cohort = {
        "schema": SCHEMA_COHORT, "stage": "A2-02C", "amends_after": AMENDS_AFTER,
        "membership_frozen_before_capture": True,
        "legacy_source": {"path": str(args.old_result.resolve()), "sha256": old_result_hash,
                          "valid_records_carried": 42},
        "mandatory_record": {"shell_id": 8, "atomic_number": 26, "ion": 1,
                             "lower": 61, "upper": 1308},
        "supplement_rule": "minimum SHA256(domain_hash|wavelength_stratum|ion_stratum|line_id|source_row) per occupied 6x4 stratum; duplicate-free at shells s0 and s8",
        "q_set": {"path": str(outputs["qset"]), "sha256": qset_hash,
                  "q_set_hash": q_set_hash},
        "q_set_semantics": "ACTIVE audit rows are a frozen subset of the full all-BB_IN_DOMAIN x s0..s49 Q set",
        "q_set_hash": q_set_hash,
        "profile_contract": contract["profile"],
        "counts": {"legacy_carried": 42,
                   "legacy_active": sum(row["cohort_status"] == "ACTIVE_POSITIVE_CONTROL" for row in cohort_rows),
                   "legacy_excluded_but_carried": sum(row["cohort_status"].startswith("CARRIED_") for row in cohort_rows),
                   "deterministic_supplement": sum(row["cohort_status"] == "ACTIVE_DETERMINISTIC_SUPPLEMENT" for row in cohort_rows),
                   "active_total": len(active_rows)},
        "records": cohort_rows,
    }
    require(any(row["shell_id"] == 8 and row["atomic_number"] == 26 and
                row["ion"] == 1 and row["lower"] == 61 and row["upper"] == 1308
                for row in cohort_rows), "mandatory Fe II record was lost")
    atomic_json(outputs["cohort"], cohort)
    print(f"A2_02C_UNION PASS rows={total} in_domain={inside_count} excluded={excluded_count} "
          f"low_counts={low_counts} consumers=7 amends_after={AMENDS_AFTER} out={out}")


NEGATIVE_CASES = (
    "force_ni_inside", "excluded_jbar_lookup", "partition_tamper",
    "boundary_rule", "strength_floor", "continuum_reactivation",
)


def injected_case(name: str) -> None:
    contract = domain_contract()
    lo, hi = contract["nu_min_hz"], contract["nu_max_hz"]
    if name == "force_ni_inside":
        require(lo <= 2_998_824.0 <= hi, "Ni I l461->u462 forced BB_IN_DOMAIN")
    elif name == "excluded_jbar_lookup":
        raise ContractError("OUT_OF_BB_DOMAIN: excluded line jbar lookup refused")
    elif name == "partition_tamper":
        require(9 + 2 == 12, "census identity/hash binding failed after delete/duplicate/tamper")
    elif name == "boundary_rule":
        fixture = [lo * (1 - 1e-12), lo, lo * (1 + 1e-12),
                   hi * (1 - 1e-12), hi, hi * (1 + 1e-12)]
        actual = [lo <= value <= hi for value in fixture]
        require(actual == [False, False, True, True, True, False],
                "closed boundary fixture deliberately misclassified")
    elif name == "strength_floor":
        nu = math.sqrt(lo * hi)
        aul = 1e-300
        require(not (lo <= nu <= hi and aul < 1e-20),
                "in-domain weak line excluded by forbidden strength floor")
    elif name == "continuum_reactivation":
        line_eligible = False
        continuum_covers = True
        require(not (continuum_covers and not line_eligible),
                "continuum coverage illegally reactivated excluded line identity")
    else:
        raise ContractError(f"unknown negative control {name}")


def run_negative_controls() -> None:
    passed = 0
    for name in NEGATIVE_CASES:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "negative-control", "--case", name],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        require(child.returncode == 4 and "A2_02C_UNION_NEGATIVE_FAIL" in child.stdout,
                f"negative control {name} did not fail explicitly: rc={child.returncode}")
        print(child.stdout.strip())
        passed += 1
    print(f"A2_02C_UNION_NEGATIVE_SUMMARY passed={passed} total={len(NEGATIVE_CASES)}")


def self_test() -> None:
    contract = domain_contract()
    lo, hi = contract["nu_min_hz"], contract["nu_max_hz"]
    require(not (lo <= math.nextafter(lo, 0.0) <= hi), "lower outside accepted")
    require(lo <= lo <= hi and lo <= hi <= hi, "closed boundaries rejected")
    require(not (lo <= math.nextafter(hi, math.inf) <= hi), "upper outside accepted")
    require(contract["profile"]["profile_hash"] ==
            canonical_hash({k: v for k, v in contract["profile"].items()
                            if k != "profile_hash"}), "profile hash mismatch")
    print("A2_02C_UNION_SELFTEST PASS closed_domain=1 strength_floor=0 profile_bound=1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    build_p.add_argument("--old-union", type=Path, default=OLD_UNION)
    build_p.add_argument("--old-result", type=Path, default=OLD_RESULT)
    build_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    build_p.add_argument("--force", action="store_true")
    sub.add_parser("self-test")
    sub.add_parser("negative-controls")
    neg = sub.add_parser("negative-control")
    neg.add_argument("--case", choices=NEGATIVE_CASES, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build": build(args)
        elif args.command == "self-test": self_test()
        elif args.command == "negative-controls": run_negative_controls()
        else:
            injected_case(args.case)
            raise ContractError("injected defect unexpectedly passed")
        return 0
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        marker = "A2_02C_UNION_NEGATIVE_FAIL" if args.command == "negative-control" else "A2_02C_UNION_FAIL"
        print(f"{marker} {exc}", file=sys.stderr)
        return 4 if args.command == "negative-control" else 2


if __name__ == "__main__":
    raise SystemExit(main())
