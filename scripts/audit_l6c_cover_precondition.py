#!/usr/bin/env python3
"""Offline, fail-closed precondition audit for DET-L6C-COVER.

The sealed run roots are evidence and are therefore opened read-only.  Every
negative control writes only to a TemporaryDirectory scratch copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


A210_L6C_PROBE = "A210_L6C_PROBE"
SOURCE_REVISION = "8526d2c"
BASELINE_REVISION = "5d711d06"
FOSSIL_REVISION = "dd9f7c18"
EXPECTED_BINARY_SHA256 = (
    "b9a30a81ebea57f9fa857d192107dd85aeb04ab1308f27b1a68cf45f1a69af99"
)
DEFAULT_L6_ROOT = Path(
    "/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/"
    "sprim_l6_20260821T054111Z_probe"
)
DEFAULT_IDSEAL_ROOT = Path(
    "/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/"
    "idseal_20260820T044703Z_a209"
)
TARGET_Z = frozenset({26, 27, 28})
TARGET_PAIRS = frozenset((z, 1) for z in TARGET_Z)
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
EV_TO_ERG = 1.602176634e-12
K_BOLTZMANN = 1.380649e-16
H_PLANCK = 6.62607015e-27
FOUR_PI = 12.56637061435917295385057353311801153679
C_LIGHT = 2.99792458e10
SOBOLEV_COEFF = 2.6540281e-02
LTE_TEMPERATURE_K = 10020.0


class AuditError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Level:
    energy_eV: float
    g: int


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require_regular(path: Path, reason: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise AuditError(reason, str(path))


def sha256(path: Path) -> str:
    require_regular(path, "UNSAFE_OR_MISSING_FILE")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(repo: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError("SOURCE_BLOB_UNAVAILABLE", f"{revision}:{path}")
    return result.stdout


def parse_c_int_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"static\s+const\s+int\s+{re.escape(name)}(?:\[[^\]]*\])?\s*=\s*"
        rf"\{{(?P<body>.*?)\}}\s*;",
        source,
        re.S,
    )
    if not match:
        raise AuditError("TABLE_PARSE_FAILED", name)
    tokens = re.findall(r"[-+]?\d+", match.group("body"))
    if not tokens:
        raise AuditError("TABLE_PARSE_FAILED", f"{name}:empty")
    return [int(token) for token in tokens]


def parse_mapping_tables(source: str) -> dict[str, frozenset[tuple[int, int]]]:
    tables: dict[str, frozenset[tuple[int, int]]] = {}
    for label, z_name, ion_name in (
        ("base", "NLTE_TARGET_Z", "NLTE_TARGET_ION"),
        ("ION4", "NLTE_TARGET_Z4", "NLTE_TARGET_ION4"),
    ):
        z_values = parse_c_int_array(source, z_name)
        ion_values = parse_c_int_array(source, ion_name)
        if len(z_values) != len(ion_values):
            raise AuditError("TABLE_LENGTH_MISMATCH", label)
        tables[label] = frozenset(zip(z_values, ion_values))
    anchors = (
        "for (int l = 0; l < atom->n_levels; l++)",
        "nlte->global_to_nlte_level[l] = g;",
        "for (int line = 0; line < atom->n_lines; line++)",
        "nlte->nlte_line_map[line] = i;",
    )
    missing = [anchor for anchor in anchors if anchor not in source]
    if missing:
        raise AuditError("PROJECTION_LOOP_UNVERIFIED", missing[0])
    return tables


def parse_exports(path: Path) -> dict[str, str]:
    require_regular(path, "ENV_EXPORTS_MISSING")
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuditError("ENV_EXPORTS_UNREADABLE", str(path)) from exc
    for number, line in enumerate(lines, 1):
        try:
            words = shlex.split(line, posix=True)
        except ValueError as exc:
            raise AuditError("ENV_EXPORT_PARSE_FAILED", f"{path}:{number}") from exc
        if len(words) != 3 or words[:2] != ["declare", "-x"] or "=" not in words[2]:
            raise AuditError("ENV_EXPORT_PARSE_FAILED", f"{path}:{number}")
        name, value = words[2].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise AuditError("ENV_EXPORT_PARSE_FAILED", f"{path}:{number}:name")
        if name in result:
            raise AuditError("ENV_EXPORT_DUPLICATE", name)
        result[name] = value
    return result


def c_atoi(token: str) -> int:
    match = re.match(r"[+-]?\d+", token)
    return int(match.group(0)) if match else 0


def parse_skip_mask(value: str | None) -> frozenset[int]:
    """Single transcription shared by both SKIP_Z variables.

    The C owners copy at most 255 bytes, split on comma/space/tab, call atoi,
    and accept only 0 < Z < 100.
    """
    if value is None or value == "":
        return frozenset()
    try:
        raw = value.encode("ascii", errors="strict")[:255].decode("ascii")
    except UnicodeError as exc:
        raise AuditError("SKIPZ_PARSE_NONASCII") from exc
    tokens = [token for token in re.split(r"[, \t]+", raw) if token]
    return frozenset(z for z in map(c_atoi, tokens) if 0 < z < 100)


def check_environment(path: Path) -> dict[str, Any]:
    exports = parse_exports(path)
    forbidden = (
        "LUMINA_NLTE_STAGE4",
        "LUMINA_NLTE_ELEMENT_WIDE",
        "LUMINA_SUPER_LEVELS",
    )
    present = [name for name in forbidden if name in exports]
    if present:
        raise AuditError("PROMOTION_ENV_PRESENT", ",".join(present))
    masks: dict[str, list[int]] = {}
    for name in ("LUMINA_NLTE_SKIP_Z", "LUMINA_OPACITY_SKIP_Z"):
        parsed = parse_skip_mask(exports.get(name))
        overlap = sorted(parsed & TARGET_Z)
        masks[name] = sorted(parsed)
        if overlap:
            raise AuditError("SKIPZ_TARGET_OVERLAP", f"{name}:{overlap}")
    return {"path": str(path), "forbidden_present": [], "skip_masks": masks}


def row_identities(path: Path) -> list[tuple[int, int]]:
    require_regular(path, "SEALED_STDERR_MISSING")
    identities: list[tuple[int, int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if not line.startswith(ROW_PREFIX):
                    continue
                values = dict(KEY_VALUE.findall(line))
                try:
                    identities.append((int(values["Z"]), int(values["ion"])))
                except (KeyError, ValueError) as exc:
                    raise AuditError("SEALED_ROW_PARSE_FAILED") from exc
    except (OSError, UnicodeError) as exc:
        raise AuditError("SEALED_STDERR_UNREADABLE", str(path)) from exc
    return identities


def coverage_count(
    identities: Iterable[tuple[int, int]], mapping: frozenset[tuple[int, int]]
) -> tuple[int, int]:
    rows = list(identities)
    return sum(identity in mapping for identity in rows), len(rows)


def require_coverage(
    identities: Iterable[tuple[int, int]], mapping: frozenset[tuple[int, int]]
) -> dict[str, int]:
    mapped, total = coverage_count(identities, mapping)
    if total == 0 or mapped != total:
        raise AuditError("COVERAGE_ABSENT", f"mapped={mapped}/{total}")
    return {"mapped": mapped, "total": total}


def compensated_sum(values: Iterable[float]) -> float:
    total = 0.0
    compensation = 0.0
    for value in values:
        updated = total + value
        if abs(total) >= abs(value):
            compensation += (total - updated) + value
        else:
            compensation += (value - updated) + total
        total = updated
    return total + compensation


def level_fractions(
    levels: dict[tuple[int, int, int], Level], temperature: float
) -> dict[tuple[int, int, int], float]:
    grouped: dict[tuple[int, int], list[tuple[int, Level]]] = defaultdict(list)
    for (z, ion, number), level in levels.items():
        grouped[(z, ion)].append((number, level))
    fractions: dict[tuple[int, int, int], float] = {}
    for (z, ion), entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        e0 = min(level.energy_eV for _, level in entries)
        terms: list[float] = []
        for _, level in entries:
            x = (level.energy_eV - e0) * EV_TO_ERG / (K_BOLTZMANN * temperature)
            terms.append(float(level.g) * math.exp(-x) if x < 745.0 else 0.0)
        partition = compensated_sum(terms)
        if not math.isfinite(partition) or partition <= 0.0:
            raise AuditError("PC2_INVALID_PARTITION", f"Z={z},ion={ion}")
        for (number, _), term in zip(entries, terms):
            fraction = term / partition
            if not math.isfinite(fraction) or fraction < 0.0:
                raise AuditError("PC2_INVALID_LEVEL_FRACTION", f"Z={z},ion={ion}")
            fractions[(z, ion, number)] = fraction
    return fractions


def read_target_levels(path: Path) -> dict[tuple[int, int, int], Level]:
    require_regular(path, "LEVELS_CSV_MISSING")
    result: dict[tuple[int, int, int], Level] = {}
    unavailable: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"atomic_number", "ion_number", "level_number", "energy_eV", "g"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AuditError("LEVELS_FIELDS_MISSING", str(sorted(required)))
            for row_number, row in enumerate(reader, 2):
                try:
                    z = int(row["atomic_number"])
                    ion = int(row["ion_number"])
                    if z not in TARGET_Z or ion not in (1, 2, 3):
                        continue
                    key = (z, ion, int(row["level_number"]))
                    level = Level(float(row["energy_eV"]), int(row["g"]))
                    if (not math.isfinite(level.energy_eV) or level.g <= 0 or
                            key in result):
                        raise ValueError
                    result[key] = level
                except (KeyError, TypeError, ValueError):
                    unavailable.append({"row": row_number, "reason": "INVALID_LEVEL"})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuditError("LEVELS_PARSE_FAILED", str(path)) from exc
    if unavailable:
        raise AuditError("LEVELS_PARSE_INCOMPLETE", json.dumps(unavailable, sort_keys=True))
    return result


def zero_chi_hazard_census(
    levels_path: Path, lines_path: Path, time_explosion_s: float,
    temperature: float = LTE_TEMPERATURE_K,
) -> dict[str, Any]:
    if not math.isfinite(time_explosion_s) or time_explosion_s <= 0.0:
        raise AuditError("PC2_INVALID_TIME_EXPLOSION")
    levels = read_target_levels(levels_path)
    fractions = level_fractions(levels, temperature)
    counts = {str(ion): 0 for ion in (1, 2, 3)}
    candidates = {str(ion): 0 for ion in (1, 2, 3)}
    hazards: dict[str, list[dict[str, Any]]] = {str(ion): [] for ion in (1, 2, 3)}
    unavailable: list[dict[str, Any]] = []
    require_regular(lines_path, "LINE_LIST_CSV_MISSING")
    try:
        with lines_path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "atomic_number", "ion_number", "level_number_lower",
                "level_number_upper", "line_id", "f_lu", "wavelength_cm",
                "A_ul", "nu",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AuditError("LINE_LIST_FIELDS_MISSING", str(sorted(required)))
            for row_number, row in enumerate(reader, 2):
                try:
                    z = int(row["atomic_number"])
                    ion = int(row["ion_number"])
                    if z not in TARGET_Z or ion not in (1, 2, 3):
                        continue
                    lower_number = int(row["level_number_lower"])
                    upper_number = int(row["level_number_upper"])
                    lower = levels[(z, ion, lower_number)]
                    upper = levels[(z, ion, upper_number)]
                    n_lower = fractions[(z, ion, lower_number)]
                    n_upper = fractions[(z, ion, upper_number)]
                    f_lu = float(row["f_lu"])
                    wavelength_cm = float(row["wavelength_cm"])
                    a_ul = float(row["A_ul"])
                    nu = float(row["nu"])
                    line_id = int(row["line_id"])
                    numeric = (f_lu, wavelength_cm, a_ul, nu)
                    if not all(math.isfinite(value) for value in numeric):
                        raise ValueError
                    difference = n_lower - (float(lower.g) / float(upper.g)) * n_upper
                    tau = (SOBOLEV_COEFF * f_lu * wavelength_cm *
                           time_explosion_s * difference)
                    eta = n_upper * a_ul * H_PLANCK * nu / FOUR_PI
                    if not math.isfinite(tau) or not math.isfinite(eta) or eta < 0.0:
                        raise ValueError
                    candidates[str(ion)] += 1
                    if tau == 0.0 and eta > 0.0:
                        counts[str(ion)] += 1
                        hazards[str(ion)].append({
                            "row": row_number,
                            "line_id": line_id,
                            "Z": z,
                            "ion": ion,
                            "lower_level": lower_number,
                            "upper_level": upper_number,
                            "lower_energy_eV": lower.energy_eV,
                            "upper_energy_eV": upper.energy_eV,
                            "n_lower_unit_ion_density": n_lower,
                            "n_upper_unit_ion_density": n_upper,
                            "tau_unit_ion_density": tau,
                            "eta_unit_ion_density": eta,
                        })
                except (KeyError, TypeError, ValueError):
                    unavailable.append({"row": row_number, "reason": "INVALID_OR_MISSING_LINE_FIELD"})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuditError("LINE_LIST_PARSE_FAILED", str(lines_path)) from exc
    return {
        "temperature_K": temperature,
        "time_explosion_s": time_explosion_s,
        "normalization": "unit positive ion density; zero-chi condition is homogeneous",
        "candidate_rows": candidates,
        "hazard_counts": counts,
        "hazard_rows": hazards,
        "unavailable_rows": unavailable,
        "unavailable_count": len(unavailable),
    }


def synthetic_hazard(degenerate: bool) -> int:
    levels = {
        (26, 1, 0): Level(0.0, 2),
        (26, 1, 1): Level(0.0 if degenerate else 1.0, 4),
    }
    fractions = level_fractions(levels, LTE_TEMPERATURE_K)
    lower = levels[(26, 1, 0)]
    upper = levels[(26, 1, 1)]
    difference = (
        fractions[(26, 1, 0)] -
        (float(lower.g) / float(upper.g)) * fractions[(26, 1, 1)]
    )
    tau = SOBOLEV_COEFF * 0.1 * 5.0e-5 * 1.0e6 * difference
    eta = fractions[(26, 1, 1)] * 1.0e8 * H_PLANCK * 5.0e14 / FOUR_PI
    return int(tau == 0.0 and eta > 0.0)


def source_identity(repo: Path, source_revision: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", "--no-ext-diff",
         source_revision, BASELINE_REVISION, "--", "src", "tests", "Makefile"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout or result.stderr:
        detail = (result.stdout + result.stderr).strip().splitlines()
        raise AuditError("SOURCE_IDENTITY_MISMATCH", (detail or [source_revision])[0])
    return {
        "source_revision": source_revision,
        "baseline_revision": BASELINE_REVISION,
        "diff_empty": True,
    }


def load_time_explosion(model_dir: Path) -> float:
    config = model_dir / "config.json"
    require_regular(config, "MODEL_CONFIG_MISSING")
    try:
        value = float(json.loads(config.read_text(encoding="utf-8"))["time_explosion_s"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AuditError("MODEL_CONFIG_INVALID") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise AuditError("MODEL_CONFIG_INVALID")
    return value


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    source = git_blob(args.repo_root, args.source_revision, "src/lumina_plasma.c")
    mappings = parse_mapping_tables(source)
    if not TARGET_PAIRS.issubset(mappings["base"]):
        raise AuditError("COVERAGE_ABSENT", "target table membership")
    environment = [
        check_environment(root / "input" / "resolved_lumina.exports")
        for root in (args.l6_run_root, args.idseal_run_root)
    ]
    identities = row_identities(args.l6_run_root / "stderr.log")
    if len(identities) != 594 or any(z not in TARGET_Z or ion != 3 for z, ion in identities):
        raise AuditError("SEALED_594_ROW_ANCHOR_MISMATCH", f"rows={len(identities)}")
    base_mapped, total = coverage_count(identities, mappings["base"])
    ion4_mapped, _ = coverage_count(identities, mappings["ION4"])
    if base_mapped != 0 or ion4_mapped != total:
        raise AuditError(
            "SEALED_MAPPING_DISCRIMINATOR_MISMATCH",
            f"base={base_mapped}/{total},ION4={ion4_mapped}/{total}",
        )
    model_dir = args.l6_run_root / "input" / "model"
    pc2 = zero_chi_hazard_census(
        model_dir / "levels.csv",
        model_dir / "line_list.csv",
        load_time_explosion(model_dir),
    )
    findings: list[str] = []
    blocking_reasons: list[str] = []
    if pc2["unavailable_count"]:
        blocking_reasons.append("PC2_PARSE_INCOMPLETE")
    if pc2["hazard_counts"]["1"] != 0:
        blocking_reasons.append("ION1_ZERO_CHI_HAZARD")
    if pc2["hazard_counts"]["2"] == 0:
        findings.append("ION2_HAZARD_ANCHOR_ZERO")
    if pc2["hazard_counts"]["3"] != 0:
        blocking_reasons.append("ION3_HAZARD_ANCHOR_NONZERO")
    identity = source_identity(args.repo_root, args.source_revision)
    binary = args.binary or args.l6_run_root / "input" / "lumina_cuda"
    actual_binary_sha = sha256(binary)
    if actual_binary_sha != EXPECTED_BINARY_SHA256:
        raise AuditError("BINARY_SHA_MISMATCH", actual_binary_sha)
    return {
        "schema": "DET_L6C_COVER_PRECONDITION_V1",
        "mode": A210_L6C_PROBE,
        "status": "FAIL" if blocking_reasons else "PASS",
        "blocking_reasons": blocking_reasons,
        "pc1": {
            "target_pairs": [list(pair) for pair in sorted(TARGET_PAIRS)],
            "base_table_contains_targets": True,
            "projection_loops_uncut": True,
            "sealed_environment": environment,
            "sealed_mapping_discriminator": {
                "rows": total,
                "base_mapped": base_mapped,
                "ION4_mapped": ion4_mapped,
            },
        },
        "pc2": pc2,
        "pc5": {
            "source_identity": identity,
            "binary_path": str(binary),
            "binary_sha256": actual_binary_sha,
        },
        "findings": findings,
        "physical_values_modified": False,
        "sealed_roots_written": False,
    }


def expect_failure(name: str, wanted: str, operation: Any) -> None:
    try:
        operation()
    except AuditError as exc:
        if exc.reason != wanted:
            raise AuditError(f"{name}_WRONG_REASON", f"{exc.reason}!={wanted}") from exc
        print(
            f"{name} inject=1 status=FAIL reason={exc.reason}"
            + (f" detail={exc.detail}" if exc.detail else "")
        )
    else:
        raise AuditError(f"{name}_INJECTION_ACCEPTED")


def selftest(args: argparse.Namespace) -> int:
    source = git_blob(args.repo_root, args.source_revision, "src/lumina_plasma.c")
    mappings = parse_mapping_tables(source)
    sealed_stderr = args.l6_run_root / "stderr.log"
    identities = row_identities(sealed_stderr)
    if len(identities) != 594:
        raise AuditError("NC_P1_SEALED_ROW_COUNT", str(len(identities)))

    expect_failure(
        "NC-P1", "COVERAGE_ABSENT",
        lambda: require_coverage(identities, mappings["base"]),
    )
    restored_identities = [(z, 1) for z, _ in identities]
    restored = require_coverage(restored_identities, mappings["base"])
    print(f"NC-P1 remove=TARGET_ION_1 status=PASS mapped={restored['mapped']}/{restored['total']}")

    missing = frozenset(pair for pair in mappings["base"] if pair != (26, 1))
    target_fixture = [(26, 1), (27, 1), (28, 1)]
    expect_failure(
        "NC-P2", "COVERAGE_ABSENT",
        lambda: require_coverage(target_fixture, missing),
    )
    restored = require_coverage(target_fixture, mappings["base"])
    print(f"NC-P2 remove=RESTORE_26_1 status=PASS mapped={restored['mapped']}/{restored['total']}")

    hazard = synthetic_hazard(degenerate=True)
    if hazard != 1:
        raise AuditError("NC-P3_HAZARD_NOT_DETECTED")
    print(f"NC-P3 inject=DEGENERATE_LEVEL_PAIR status=FAIL reason=ZERO_CHI_HAZARD hazards={hazard}")
    clean = synthetic_hazard(degenerate=False)
    if clean != 0:
        raise AuditError("NC-P3_REMOVAL_NONZERO")
    print(f"NC-P3 remove=NONDEGENERATE_PAIR status=PASS hazards={clean}")

    sealed_exports = args.l6_run_root / "input" / "resolved_lumina.exports"
    require_regular(sealed_exports, "NC_P4_SEALED_EXPORTS_MISSING")
    with tempfile.TemporaryDirectory(prefix="l6c-ncp4-") as directory:
        scratch = Path(directory) / "resolved_lumina.exports"
        shutil.copyfile(sealed_exports, scratch)
        healthy_text = scratch.read_text(encoding="utf-8", errors="strict")
        if 'declare -x LUMINA_NLTE_SKIP_Z="14"' not in healthy_text:
            raise AuditError("NC_P4_SEALED_ANCHOR_MISSING")
        scratch.write_text(
            healthy_text.replace(
                'declare -x LUMINA_NLTE_SKIP_Z="14"',
                'declare -x LUMINA_NLTE_SKIP_Z="14,26"',
            ),
            encoding="utf-8",
        )
        expect_failure("NC-P4", "SKIPZ_TARGET_OVERLAP", lambda: check_environment(scratch))
        scratch.write_text(healthy_text, encoding="utf-8")
        restored_env = check_environment(scratch)
        masks = restored_env["skip_masks"]
        print(
            "NC-P4 remove=RESTORE_SKIPZ status=PASS "
            f"nlte={masks['LUMINA_NLTE_SKIP_Z']} opacity={masks['LUMINA_OPACITY_SKIP_Z']}"
        )

    expect_failure(
        "NC-PC5", "SOURCE_IDENTITY_MISMATCH",
        lambda: source_identity(args.repo_root, FOSSIL_REVISION),
    )
    source_identity(args.repo_root, args.source_revision)
    print("NC-PC5 remove=SOURCE_REVISION status=PASS diff=EMPTY")
    print("DET_L6C_COVER_PRECONDITION_SELFTEST_PASS controls=NC-P1,NC-P2,NC-P3,NC-P4,NC-PC5")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--l6-run-root", type=Path, default=DEFAULT_L6_ROOT)
    parser.add_argument("--idseal-run-root", type=Path, default=DEFAULT_IDSEAL_ROOT)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.selftest:
            return selftest(args)
        if args.report is None:
            raise AuditError("REPORT_REQUIRED")
        report = run_audit(args)
        atomic_write_json(args.report, report)
    except (AuditError, OSError, UnicodeError) as exc:
        reason = exc.reason if isinstance(exc, AuditError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, AuditError) else str(exc)
        if args.report is not None:
            atomic_write_json(args.report, {
                "schema": "DET_L6C_COVER_PRECONDITION_V1",
                "status": "FAIL",
                "reason": reason,
                "detail": detail,
                "physical_values_modified": False,
                "sealed_roots_written": False,
            })
        print(
            f"DET_L6C_COVER_PRECONDITION_FAIL reason={reason}"
            + (f" detail={detail}" if detail else ""),
            file=sys.stderr,
        )
        return 4
    if report["status"] != "PASS":
        print(
            "DET_L6C_COVER_PRECONDITION_FAIL "
            f"reasons={','.join(report['blocking_reasons'])} report={args.report}",
            file=sys.stderr,
        )
        return 4
    print(
        "DET_L6C_COVER_PRECONDITION_PASS "
        f"ion1_hazards={report['pc2']['hazard_counts']['1']} "
        f"ion2_hazards={report['pc2']['hazard_counts']['2']} "
        f"ion3_hazards={report['pc2']['hazard_counts']['3']} "
        f"report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
