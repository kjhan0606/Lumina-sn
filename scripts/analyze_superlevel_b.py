#!/usr/bin/env python3
"""SUPERLEVEL_B_PROMOTION offline reader for G-RC and G-2 through G-7.

Only Python's standard library is imported.  In particular, no earlier Lumina
reader is imported: the predecessor is re-read from its raw logs and its
registered JSON before this reader is allowed to measure B-1/B-2.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUPERLEVEL_B_PROMOTION = True
SCHEMA = "DET_SUPERLEVEL_B_READER_V1"
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
ZERO_PREFIX = "[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
R7_PREFIX = "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED"
PHYSICS_PREFIX = "[PHYSICS-COMPARISON]"
ACTIVE_BANNER = "[NLTE] Super-levels: ACTIVE (13015 FL -> 1173 SL across ions)"
OFF_BANNER = "off (identity)"
CUTOFF_BANNER = "[ARTIS super-cutoff] K=100: 21919 levels lumped"
BLOCKED_BANNER = "[A2-07] within-super-level blocked"
FROZEN_TOLERANCE = 1.0e-13
EV_TO_ERG = 1.602176634e-12
K_BOLTZMANN = 1.380649e-16
SUPER_CUTOFF = 100
ANCHOR_LINE = 669_992
ANCHOR_BITS = (
    {
        "tau_validity": "2",
        "n_upper": "0x1.d1534dba20c26p-12",
        "A_ul": "0x1.0b6516044e436p-1",
        "emission_per_sr": "0x1.3adcec9caadc0p-52",
    },
    {
        "tau_validity": "2",
        "n_upper": "0x1.daa141c1395e6p-12",
        "A_ul": "0x1.0b6516044e436p-1",
        "emission_per_sr": "0x1.4128acfd24836p-52",
    },
)
REGISTERED_FF = 95 / 885
REGISTERED_EXP_MIN = 2.928690524872657e-6
REGISTERED_FPC = {"BL": 95, "EXP": 284, "MIX": 506}
REGISTERED_NEG = 46
REGISTERED_ZERO = (221_382, 221_382)
PATH_KEYS = {
    "LUMINA_CMFGEN_SIGMA_BF",
    "LUMINA_DEPOSITION_FILE",
    "LUMINA_IONIZATION_REFERENCE_FILE",
    "LUMINA_MODEL_DIR",
    "LUMINA_PHYSICS_COMPARISON_DIR",
    "LUMINA_TOPION_LEVELS_FILE",
}
MIRROR_FILES = (
    "work/lumina_plasma_state.csv",
    "work/lumina_spectrum.csv",
    "work/lumina_spectrum_comoving.csv",
    "work/physics_comparison/physics_DET_iter0000.manifest.json",
    "work/physics_comparison/physics_DET_iter0000.shell.csv",
    "work/physics_comparison/physics_DET_iter0000.spectral.csv",
    "work/physics_comparison/physics_DET_iter0001.manifest.json",
    "work/physics_comparison/physics_DET_iter0001.shell.csv",
    "work/physics_comparison/physics_DET_iter0001.spectral.csv",
)
SEALS = (
    {
        "name": "census",
        "root": Path("/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1"),
        "manifest": "ZO_RUN_SEAL.sha",
        "metadata": "ZO_RUN_SEAL.txt",
        "entries": 136,
        "schema": "A210_ZO_CENSUS_RUN_SEAL_V1",
    },
    {
        "name": "R-II",
        "root": Path("/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1"),
        "manifest": "RII_RUN_SEAL.sha",
        "metadata": "RII_RUN_SEAL.txt",
        "entries": 135,
        "schema": "A210_Z2A_RII_RUN_SEAL_V1",
    },
)
READ_ONLY_ROOTS = tuple(spec["root"] for spec in SEALS) + (
    Path("/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe"),
    Path("/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover"),
)


class ReaderError(RuntimeError):
    def __init__(self, reason: str, detail: Any = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def named_failure(exc: BaseException, context: str) -> ReaderError:
    """Convert every catchable analysis failure into a stable reader error."""
    if isinstance(exc, ReaderError):
        return exc
    if isinstance(exc, KeyboardInterrupt):
        kind = "INTERRUPTED"
    elif isinstance(exc, UnicodeError):
        kind = "ENCODING_FAILURE"
    elif isinstance(exc, OSError):
        kind = "IO_FAILURE"
    else:
        kind = "UNEXPECTED_FAILURE"
    return ReaderError(f"{context}_{kind}", {
        "exception": type(exc).__name__, "message": str(exc),
    })


@dataclass
class LogBlock:
    rows: list[dict[str, str]] = field(default_factory=list)
    zero_rows: list[dict[str, str]] = field(default_factory=list)
    zero_count: int = 0
    anchors: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedLog:
    blocks: list[LogBlock]
    r7: list[dict[str, str]]
    physics_commits: list[dict[str, str]]
    blocked: int
    isolated: int


@dataclass(frozen=True)
class Level:
    z: int
    ion: int
    number: int
    energy_ev: float
    g: float


def regular_file(path: Path, reason: str = "MISSING_OR_UNSAFE_REGULAR_FILE") -> None:
    if not path.is_file() or path.is_symlink():
        raise ReaderError(reason, str(path))


def read_text(path: Path) -> str:
    regular_file(path)
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ReaderError("TEXT_READ_FAILED", f"{path}:{exc}") from exc


def validate_output_path(
    path: Path, protected_roots: tuple[Path, ...] = READ_ONLY_ROOTS,
) -> Path:
    """Reject protected destinations and every symlink in an output path.

    The lexical walk happens before ``resolve`` so an intermediate symlink
    cannot hide the destination that will receive chmod/os.replace.  The
    resolved containment check is a second, fail-closed boundary check.
    """
    if not path.is_absolute():
        raise ReaderError("OUTPUT_PATH_NOT_ABSOLUTE", str(path))
    normalized = Path(os.path.normpath(os.fspath(path)))
    cursor = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        cursor /= part
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ReaderError("OUTPUT_PATH_LSTAT_FAILED", f"{cursor}:{exc}") from exc
        if stat.S_ISLNK(mode):
            reason = "OUTPUT_PATH_SYMLINK" if cursor == normalized else "OUTPUT_ANCESTOR_SYMLINK"
            raise ReaderError(reason, str(cursor))
    try:
        resolved = normalized.resolve(strict=False)
    except OSError as exc:
        raise ReaderError("OUTPUT_PATH_RESOLVE_FAILED", f"{normalized}:{exc}") from exc
    for protected in protected_roots:
        protected_resolved = protected.resolve(strict=False)
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise ReaderError("OUTPUT_INSIDE_READ_ONLY_SEAL", {
                "path": str(normalized), "seal": str(protected),
            })
    return normalized


def prepare_output_dir(
    path: Path, protected_roots: tuple[Path, ...] = READ_ONLY_ROOTS,
) -> Path:
    path = validate_output_path(path, protected_roots)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReaderError("OUTPUT_DIRECTORY_CREATE_FAILED", f"{path}:{exc}") from exc
    validate_output_path(path, protected_roots)
    if not path.is_dir():
        raise ReaderError("OUTPUT_DIRECTORY_NOT_DIRECTORY", str(path))
    return path


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name in result:
            raise ReaderError("DUPLICATE_FIELD", name)
        result[name] = value
    return result


def exact_int(row: dict[str, str], name: str) -> int:
    raw = row.get(name)
    if raw is None:
        raise ReaderError("FIELD_MISSING", name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReaderError("FIELD_INVALID", f"{name}={raw}") from exc
    if str(value) != raw:
        raise ReaderError("FIELD_INVALID", f"{name}={raw}")
    return value


def finite_float(row: dict[str, str], name: str) -> float:
    raw = row.get(name)
    if raw is None:
        raise ReaderError("FIELD_MISSING", name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ReaderError("FIELD_INVALID", f"{name}={raw}") from exc
    if not math.isfinite(value):
        raise ReaderError("FIELD_NONFINITE", f"{name}={raw}")
    return value


def parse_log(path: Path, keep_zero_rows: bool) -> ParsedLog:
    text = read_text(path)
    blocks: list[LogBlock] = []
    pending_rows: list[dict[str, str]] = []
    pending_zero: list[dict[str, str]] = []
    anchors: list[dict[str, str]] = []
    zero_count = 0
    r7: list[dict[str, str]] = []
    commits: list[dict[str, str]] = []
    blocked = 0
    isolated = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            if line.startswith(ZERO_PREFIX):
                row = fields(line)
                row["_log_line"] = str(line_number)
                zero_count += 1
                if keep_zero_rows:
                    pending_zero.append(row)
                if row.get("line") == str(ANCHOR_LINE):
                    anchors.append(row)
            elif line.startswith(ROW_PREFIX):
                row = fields(line)
                row["_log_line"] = str(line_number)
                pending_rows.append(row)
            elif line.startswith(SUMMARY_PREFIX):
                summary = fields(line)
                if summary.get("phase") != "REQUESTED_TE" or summary.get("target_ion") != "1":
                    continue
                summary["_log_line"] = str(line_number)
                blocks.append(LogBlock(
                    rows=pending_rows,
                    zero_rows=pending_zero,
                    zero_count=zero_count,
                    anchors=anchors,
                    summary=summary,
                ))
                pending_rows, pending_zero, anchors, zero_count = [], [], [], 0
            elif line.startswith(R7_PREFIX):
                r7.append(fields(line))
            elif line.startswith(PHYSICS_PREFIX) and "status=COMMITTED" in line:
                commits.append(fields(line))
            elif "[A2-10][LINE-SATURATION-BLOCKED]" in line:
                blocked += 1
            elif "[A2-10][LINE-SATURATION-ISOLATED]" in line:
                isolated += 1
        except ReaderError as exc:
            raise ReaderError(exc.reason, {"line": line_number, "detail": exc.detail}) from exc
    if pending_rows or pending_zero or zero_count or anchors:
        raise ReaderError("ORPHAN_RECORDS_AFTER_FINAL_SUMMARY", {
            "rows": len(pending_rows), "zero_rows": zero_count,
        })
    return ParsedLog(blocks, r7, commits, blocked, isolated)


def summary_public(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if key != "_log_line"}


def check_anchor(block: LogBlock, iteration: int) -> None:
    if len(block.anchors) != 1:
        raise ReaderError("ANCHOR_ROW_COUNT_MISMATCH", {
            "iteration": iteration, "count": len(block.anchors),
        })
    row = block.anchors[0]
    for name, expected in ANCHOR_BITS[iteration].items():
        raw = row.get(name)
        if raw is None:
            raise ReaderError("ANCHOR_FIELD_MISSING", name)
        if name == "tau_validity":
            actual = str(exact_int(row, name))
        else:
            actual = finite_float(row, name).hex()
        if actual != expected:
            raise ReaderError("ANCHOR_FIELD_MISMATCH", {
                "iteration": iteration, "field": name,
                "actual": actual, "expected": expected,
            })


def validate_registered_f1b(path: Path) -> dict[str, Any]:
    regular_file(path, "F1B_JSON_UNSAFE")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReaderError("F1B_JSON_INVALID", str(exc)) from exc
    rows = data.get("matched_rows")
    if not isinstance(rows, list):
        raise ReaderError("F1B_MATCHED_ROWS_MISSING")
    seen: set[int] = set()
    fpc = Counter()
    neg: list[dict[str, Any]] = []
    frozen = 0
    fpc_d: list[float] = []
    exp_d: list[float] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("line"), int):
            raise ReaderError("F1B_MATCHED_ROW_INVALID")
        line = row["line"]
        if line in seen:
            raise ReaderError("F1B_MATCHED_LINE_DUPLICATE", line)
        seen.add(line)
        axis, class_name = row.get("chi_axis"), row.get("class")
        if axis == "FPC":
            if class_name not in ("BL", "EXP", "MIX"):
                raise ReaderError("F1B_FPC_CLASS_INVALID", class_name)
            fpc[class_name] += 1
            value = row.get("capture_d")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ReaderError("F1B_D_INVALID", line)
            value = float(value)
            fpc_d.append(value)
            if class_name == "EXP":
                exp_d.append(value)
            frozen += value <= FROZEN_TOLERANCE
        elif axis == "NEG":
            neg.append(row)
    if dict(fpc) != REGISTERED_FPC:
        raise ReaderError("F1B_FPC_CENSUS_MISMATCH", dict(fpc))
    if len(neg) != REGISTERED_NEG or any(row.get("class") != "BL" for row in neg):
        raise ReaderError("F1B_NEG_CENSUS_MISMATCH", len(neg))
    if len(fpc_d) != 885 or frozen != 95:
        raise ReaderError("F1B_FF_COUNTS_MISMATCH", {"frozen": frozen, "total": len(fpc_d)})
    observed_ff = frozen / len(fpc_d)
    try:
        registered_ff = float(data["ff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReaderError("F1B_FF_INVALID") from exc
    if observed_ff.hex() != REGISTERED_FF.hex() or registered_ff.hex() != REGISTERED_FF.hex():
        raise ReaderError("F1B_FF_BIT_MISMATCH", observed_ff.hex())
    if not exp_d or min(exp_d).hex() != REGISTERED_EXP_MIN.hex():
        raise ReaderError("F1B_EXP_MIN_BIT_MISMATCH", min(exp_d) if exp_d else None)
    registered_anchor = data.get("gates", {}).get("RD-6", {}).get("anchor_bits_by_iteration")
    if registered_anchor != list(ANCHOR_BITS):
        raise ReaderError("F1B_ANCHOR_REGISTRATION_MISMATCH")
    return {
        "data": data,
        "fpc_counts": dict(fpc),
        "negative_count": len(neg),
        "ff": observed_ff,
        "exp_min": min(exp_d),
        "bl_lines": sorted(row["line"] for row in rows
                           if row.get("chi_axis") == "FPC" and row.get("class") == "BL"),
        "neg_lines": sorted(row["line"] for row in rows if row.get("chi_axis") == "NEG"),
        "by_line": {row["line"]: row for row in rows},
    }


def predecessor_calibration(rii_root: Path, census_root: Path, f1b_json: Path) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    rii_parsed: ParsedLog | None = None
    for name, root in (("R-II", rii_root), ("census", census_root)):
        parsed = parse_log(root / "stderr.log", keep_zero_rows=False)
        if len(parsed.blocks) != 2:
            raise ReaderError("G_RC_SUMMARY_COUNT_MISMATCH", {"source": name, "count": len(parsed.blocks)})
        counts = tuple(block.zero_count for block in parsed.blocks)
        if counts != REGISTERED_ZERO:
            raise ReaderError("G_RC_ZERO_COUNT_MISMATCH", {"source": name, "counts": counts})
        for iteration, block in enumerate(parsed.blocks):
            check_anchor(block, iteration)
        observations[name] = {
            "zero_rows": list(counts),
            "anchor_bits": list(ANCHOR_BITS),
        }
        if name == "R-II":
            rii_parsed = parsed
    if rii_parsed is None:  # Defensive: the loop above is the registered source list.
        raise ReaderError("G_RC_RII_RAW_SOURCE_MISSING")

    # Recompute the F1b predecessor measurements from the sealed R-II ROW
    # stream.  The JSON is a registry/roster, never the source of these values.
    raw_rows, raw_observation = current_matched(rii_parsed)
    raw_fpc = {name: raw_observation["matched_FPC"].get(name, 0)
               for name in ("BL", "EXP", "MIX")}
    raw_neg = {name: raw_observation["matched_NEG"].get(name, 0)
               for name in ("BL", "EXP", "MIX")}
    raw_exp = [record["capture_d"] for record in raw_rows.values()
               if record["chi_axis"] == "FPC" and record["class"] == "EXP"]
    if raw_fpc != REGISTERED_FPC:
        raise ReaderError("G_RC_RAW_FPC_CENSUS_MISMATCH", raw_fpc)
    if raw_neg != {"BL": REGISTERED_NEG, "EXP": 0, "MIX": 0}:
        raise ReaderError("G_RC_RAW_NEG_CENSUS_MISMATCH", raw_neg)
    raw_ff = raw_observation["ff"]
    if raw_ff is None or raw_ff.hex() != REGISTERED_FF.hex():
        raise ReaderError("G_RC_RAW_FF_BIT_MISMATCH", None if raw_ff is None else raw_ff.hex())
    if not raw_exp or min(raw_exp).hex() != REGISTERED_EXP_MIN.hex():
        raise ReaderError("G_RC_RAW_EXP_MIN_BIT_MISMATCH", min(raw_exp) if raw_exp else None)

    f1b = validate_registered_f1b(f1b_json)
    registered_rows = f1b["by_line"]
    if set(raw_rows) != set(registered_rows):
        raise ReaderError("G_RC_RAW_ROSTER_MISMATCH", {
            "raw_only": sorted(set(raw_rows) - set(registered_rows)),
            "json_only": sorted(set(registered_rows) - set(raw_rows)),
        })
    for line, raw in raw_rows.items():
        registered = registered_rows[line]
        raw_axis = None if raw["chi_axis"] == "HAZ" else raw["chi_axis"]
        if (raw_axis != registered.get("chi_axis") or
                raw["class"] != registered.get("class")):
            raise ReaderError("G_RC_RAW_AXIS_CLASS_MISMATCH", line)
        raw_d, registered_d = raw.get("capture_d"), registered.get("capture_d")
        if raw_d is None or registered_d is None:
            if raw_d is not None or registered_d is not None:
                raise ReaderError("G_RC_RAW_D_MISSING", line)
        elif (not isinstance(registered_d, (int, float)) or
              raw_d.hex() != float(registered_d).hex()):
            raise ReaderError("G_RC_RAW_D_BIT_MISMATCH", line)
        if raw.get("capture_frozen") != registered.get("capture_frozen"):
            raise ReaderError("G_RC_RAW_FROZEN_MISMATCH", line)
    return {
        "status": "PASS",
        "ff_fraction": "95/885",
        "ff": raw_ff,
        "fpc_counts": raw_fpc,
        "negative_count": sum(raw_neg.values()),
        "exp_min": min(raw_exp),
        "measurement_source": "sealed R-II raw stderr ROW stream",
        "registered_json_role": "bitwise cross-check and transition roster only",
        "raw_predecessors": observations,
        "f1b": f1b,
    }


def parse_exports(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
    for line_number, line in enumerate(read_text(path).splitlines(), 1):
        match = pattern.fullmatch(line)
        if not match:
            continue
        name, value = match.groups()
        if name in result:
            raise ReaderError("EXPORT_DUPLICATE", {"line": line_number, "key": name})
        result[name] = value
    return result


def validate_export_delta(base_root: Path, run_root: Path, mode: str) -> dict[str, Any]:
    base = parse_exports(base_root / "input/resolved_lumina.exports")
    run = parse_exports(run_root / "input/resolved_lumina.exports")
    expected = set(PATH_KEYS) | {"LUMINA_SUPER_LEVELS"}
    if mode == "B-1":
        expected |= {"LUMINA_A210_SPRODUCER_CAPTURE", "LUMINA_A210_INDEPENDENT_CAPTURE"}
    actual = {key for key in set(base) | set(run) if base.get(key) != run.get(key)}
    if actual != expected:
        raise ReaderError("G2_EXPORT_DELTA_KEYSET", {
            "mode": mode, "actual": sorted(actual), "expected": sorted(expected),
        })
    if "LUMINA_SUPER_LEVELS" in base or run.get("LUMINA_SUPER_LEVELS") != "1":
        raise ReaderError("G2_SUPERLEVEL_DELTA_INVALID")
    for key in PATH_KEYS:
        if str(base_root) not in base[key] or run[key] != base[key].replace(str(base_root), str(run_root)):
            raise ReaderError("G2_PATH_RELOCATION_INVALID", key)
    for key in ("LUMINA_A210_SPRODUCER_CAPTURE", "LUMINA_A210_INDEPENDENT_CAPTURE"):
        wanted = "0" if mode == "B-1" else "1"
        if base.get(key) != "1" or run.get(key) != wanted:
            raise ReaderError("G2_CAPTURE_DELTA_INVALID", {"mode": mode, "key": key})
    return {
        "mode": mode,
        "keys": sorted(actual),
        "values": {key: {"before": base.get(key), "after": run.get(key)} for key in sorted(actual)},
    }


def atomic_text(
    path: Path, text: str, protected_roots: tuple[Path, ...] = READ_ONLY_ROOTS,
) -> None:
    path = validate_output_path(path, protected_roots)
    prepare_output_dir(path.parent, protected_roots)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o640)
        validate_output_path(path, protected_roots)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(
    path: Path, value: Any, protected_roots: tuple[Path, ...] = READ_ONLY_ROOTS,
) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", protected_roots)


def make_verdict_view(source: Path, target: Path) -> None:
    target.mkdir()
    (target / ".run_start").touch()
    shutil.copyfile(source / "stdout.log", target / "stdout.log")
    for name in ("lumina_ion_pops.csv", "lumina_plasma_state.csv", "lumina_spectrum_formal.csv"):
        candidate = source / name
        if not candidate.is_file() or candidate.is_symlink():
            candidate = source / "work" / name
        regular_file(candidate, "VERDICT_GATE_CORE_OUTPUT_UNSAFE")
        shutil.copyfile(candidate, target / name)


def run_verdict_gate(
    gate_path: Path, baseline: Path, run_root: Path, mode: str, output_dir: Path,
) -> dict[str, Any]:
    regular_file(gate_path, "VERDICT_GATE_SCRIPT_UNSAFE")
    delta = validate_export_delta(baseline, run_root, mode)
    run_exports = parse_exports(run_root / "input/resolved_lumina.exports")
    tokens: list[str] = []
    for key in delta["keys"]:
        value = run_exports[key]
        tokens.append(("+" if key == "LUMINA_SUPER_LEVELS" else "") + f"{key}={value}")
    tokens.append("argv")
    spec = ",".join(tokens)
    with tempfile.TemporaryDirectory(prefix="slb_verdict_", dir=output_dir) as scratch_name:
        scratch = Path(scratch_name)
        base_view, run_view = scratch / "base", scratch / "run"
        make_verdict_view(baseline, base_view)
        make_verdict_view(run_root, run_view)
        command = [
            sys.executable, "-B", str(gate_path), str(run_view),
            "--baseline", str(base_view), "--intended-diff", spec,
            "--strict-env", "--no-write",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    atomic_text(
        output_dir / f"verdict_gate_{mode}.log",
        f"command={command!r}\nrc={completed.returncode}\n--- stdout ---\n"
        f"{completed.stdout}\n--- stderr ---\n{completed.stderr}",
    )
    if completed.returncode != 0:
        raise ReaderError("G2_VERDICT_GATE_RC", {"mode": mode, "rc": completed.returncode})
    return {"rc": completed.returncode, "intended_diff": spec, "exports": delta}


def gate3_banner(run_root: Path) -> dict[str, Any]:
    stdout = read_text(run_root / "stdout.log")
    stderr = read_text(run_root / "stderr.log")
    lines = [line.strip() for line in stdout.splitlines()]
    active_lines = [line for line in lines if line.startswith("[NLTE] Super-levels: ACTIVE")]
    unregistered_active = [line for line in active_lines if line != ACTIVE_BANNER]
    if unregistered_active:
        raise ReaderError("G3_ACTIVE_DIMENSION_UNREGISTERED", unregistered_active)
    active = sum(line == ACTIVE_BANNER for line in lines)
    off = sum(OFF_BANNER in line for line in lines)
    cutoff = sum(CUTOFF_BANNER in line for line in lines)
    blocked = (stdout + "\n" + stderr).count(BLOCKED_BANNER)
    reasons: list[str] = []
    if active != 1:
        reasons.append("BANNER_ACTIVE_COUNT")
    if off != 0:
        reasons.append("BANNER_OFF_IDENTITY_PRESENT")
    if cutoff < 1:
        reasons.append("CUTOFF_BANNER_MISSING")
    if blocked != 0:
        reasons.append("WITHIN_SL_BLOCKED_PRESENT")
    if reasons:
        raise ReaderError("G3_BANNER_MODE_CONTRACT", reasons)
    return {"active": active, "off_identity": off, "cutoff_markers": cutoff, "blocked": blocked}


def parse_footer(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in result:
            raise ReaderError("RUN_FOOTER_DUPLICATE_KEY", key)
        result[key] = value
    return result


def gate4_completion(run_root: Path, parsed: ParsedLog) -> dict[str, Any]:
    reasons: list[str] = []
    if read_text(run_root / "model.rc").strip() != "0":
        reasons.append("MODEL_RC_NOT_ZERO")
    footer = parse_footer(run_root / "RUN_FOOTER.txt")
    if footer.get("schema") != "LUMINA_DET_FLIGHT_V1":
        reasons.append("RUN_FOOTER_SCHEMA")
    if footer.get("outer_iterations") != "2":
        reasons.append("RUN_FOOTER_ITERATIONS")
    argv = footer.get("argv", "").split()
    expected_argv = [
        str(run_root / "input/lumina_cuda"), str(run_root / "input/model"),
        "1", "2", "spectrum", "nlte",
    ]
    if argv != expected_argv:
        reasons.append("RUN_FOOTER_ARGV")
    if len(parsed.blocks) != 2:
        reasons.append("SUMMARY_COUNT")
    for block in parsed.blocks:
        if block.summary.get("complete") != "1":
            reasons.append("SUMMARY_COMPLETE")
    r7_by_iter = Counter(row.get("iter") for row in parsed.r7 if "te_generation" in row)
    if r7_by_iter != Counter({"0": 1, "1": 1}):
        reasons.append("R7_TE_GENERATION_MARKERS")
    if len(parsed.physics_commits) != 2:
        reasons.append("PHYSICS_COMPARISON_COMMIT_COUNT")
    if parsed.blocked or parsed.isolated:
        reasons.append("BLOCKED_OR_ISOLATED_PRESENT")
    if reasons:
        raise ReaderError("G4_COMPLETION_CONTRACT", sorted(set(reasons)))
    return {
        "model_rc": 0,
        "footer_schema": footer["schema"],
        "argv": footer.get("argv"),
        "summaries": [summary_public(block.summary) for block in parsed.blocks],
        "r7_te_generation_by_iter": dict(r7_by_iter),
        "physics_commits": len(parsed.physics_commits),
    }


def load_levels(path: Path) -> tuple[list[Level], dict[tuple[int, int, int], Level]]:
    regular_file(path, "LEVELS_CSV_UNSAFE")
    global_levels: list[Level] = []
    by_key: dict[tuple[int, int, int], Level] = {}
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"atomic_number", "ion_number", "level_number", "energy_eV", "g"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ReaderError("LEVELS_COLUMNS_MISSING")
            for row_number, row in enumerate(reader, 2):
                try:
                    level = Level(
                        int(row["atomic_number"]), int(row["ion_number"]),
                        int(row["level_number"]), float(row["energy_eV"]), float(row["g"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReaderError("LEVELS_FIELD_INVALID", row_number) from exc
                if not all(math.isfinite(value) for value in (level.energy_ev, level.g)) or level.g <= 0:
                    raise ReaderError("LEVELS_FIELD_INVALID", row_number)
                key = (level.z, level.ion, level.number)
                if key in by_key:
                    raise ReaderError("LEVELS_KEY_DUPLICATE", key)
                global_levels.append(level)
                by_key[key] = level
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, ReaderError):
            raise
        raise ReaderError("LEVELS_READ_FAILED", str(exc)) from exc
    return global_levels, by_key


def load_te_by_iteration(run_root: Path) -> tuple[float, float]:
    values: list[float] = []
    for iteration in (0, 1):
        path = run_root / f"work/physics_comparison/physics_DET_iter{iteration:04d}.shell.csv"
        regular_file(path, "SHELL_COMPARISON_UNSAFE")
        try:
            with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
                reader = csv.DictReader(stream)
                matches = [row for row in reader if row.get("shell_id") == "0"]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ReaderError("SHELL_COMPARISON_READ_FAILED", str(exc)) from exc
        if len(matches) != 1:
            raise ReaderError("SHELL0_TEMPERATURE_COUNT", {"iteration": iteration, "count": len(matches)})
        raw = matches[0].get("T_e_K")
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ReaderError("SHELL0_TEMPERATURE_INVALID", {"iteration": iteration, "value": raw}) from exc
        if not math.isfinite(value) or value <= 0:
            raise ReaderError("SHELL0_TEMPERATURE_INVALID", value)
        values.append(value)
    return values[0], values[1]


def exp_pair_collision(
    row: dict[str, str], temperature: float, global_levels: list[Level],
    by_key: dict[tuple[int, int, int], Level],
) -> dict[str, Any]:
    lower_index, upper_index = exact_int(row, "lower_global"), exact_int(row, "upper_global")
    if not (0 <= lower_index < len(global_levels) and 0 <= upper_index < len(global_levels)):
        raise ReaderError("GLOBAL_LEVEL_INDEX_INVALID", {"lower": lower_index, "upper": upper_index})
    lower, upper = global_levels[lower_index], global_levels[upper_index]
    z, ion = exact_int(row, "Z"), exact_int(row, "ion")
    if (lower.z, lower.ion) != (z, ion) or (upper.z, upper.ion) != (z, ion):
        raise ReaderError("GLOBAL_LEVEL_IDENTITY_MISMATCH", row.get("line"))
    anchor = by_key.get((z, ion, SUPER_CUTOFF))
    if anchor is None:
        raise ReaderError("SUPERLEVEL_ANCHOR_LEVEL_MISSING", (z, ion, SUPER_CUTOFF))
    equal_energy = lower.energy_ev.hex() == upper.energy_ev.hex()
    exponent_lower = math.exp(-(lower.energy_ev - anchor.energy_ev) * EV_TO_ERG /
                              (K_BOLTZMANN * temperature))
    exponent_upper = math.exp(-(upper.energy_ev - anchor.energy_ev) * EV_TO_ERG /
                              (K_BOLTZMANN * temperature))
    return {
        "line": exact_int(row, "line"),
        "lower": {"number": lower.number, "energy_hex": lower.energy_ev.hex(), "g_hex": lower.g.hex()},
        "upper": {"number": upper.number, "energy_hex": upper.energy_ev.hex(), "g_hex": upper.g.hex()},
        "temperature_hex": temperature.hex(),
        "equal_energy_bits": equal_energy,
        "lower_exp_hex": exponent_lower.hex(),
        "upper_exp_hex": exponent_upper.hex(),
        "exp_bit_collision": exponent_lower.hex() == exponent_upper.hex(),
    }


def gate5_hazard(
    runs: dict[str, tuple[Path, ParsedLog]], global_levels: list[Level],
    by_key: dict[tuple[int, int, int], Level],
) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": {}, "residual": 0, "unattributed": []}
    for name, (root, parsed) in runs.items():
        temperatures = load_te_by_iteration(root)
        run_observation: list[dict[str, Any]] = []
        for iteration, block in enumerate(parsed.blocks):
            try:
                announced = exact_int(block.summary, "zero_opacity_emitting_rows")
                sprobe = exact_int(block.summary, "sprobe_undefined_rows")
            except ReaderError as exc:
                raise ReaderError("G5_SUMMARY_COUNTER_INVALID", {"run": name, "detail": exc.detail}) from exc
            if announced != block.zero_count or sprobe != block.zero_count:
                raise ReaderError("G5_ZERO_COUNTER_ROW_MISMATCH", {
                    "run": name, "iteration": iteration, "rows": block.zero_count,
                    "zero_counter": announced, "sprobe_counter": sprobe,
                })
            attributions: list[dict[str, Any]] = []
            for row in block.zero_rows:
                observation = exp_pair_collision(row, temperatures[iteration], global_levels, by_key)
                if observation["equal_energy_bits"]:
                    observation["attribution"] = "SLB-R_EQUAL_ENERGY_CONTRADICTION"
                    result["unattributed"].append(observation)
                elif observation["exp_bit_collision"]:
                    observation["attribution"] = "EXP_BINARY64_BIT_COLLISION"
                else:
                    observation["attribution"] = "SLB-R_EXP_COLLISION_NOT_REPRODUCED"
                    result["unattributed"].append(observation)
                attributions.append(observation)
            result["residual"] += block.zero_count
            run_observation.append({
                "iteration": iteration, "zero_rows": block.zero_count,
                "zero_opacity_emitting_rows": announced,
                "sprobe_undefined_rows": sprobe,
                "attributions": attributions,
            })
        result["runs"][name] = run_observation
    result["exact_zero_expected"] = result["residual"] == 0
    result["all_residuals_attributed"] = not result["unattributed"]
    return result


def classify_row(row: dict[str, str]) -> str:
    validity = exact_int(row, "tau_validity")
    if validity == 2:
        return "HAZ"
    if validity != 1:
        raise ReaderError("TAU_VALIDITY_INVALID", validity)
    lower, upper = exact_int(row, "lower_level"), exact_int(row, "upper_level")
    if lower >= SUPER_CUTOFF and upper >= SUPER_CUTOFF:
        return "BL"
    if lower < SUPER_CUTOFF and upper < SUPER_CUTOFF:
        return "EXP"
    return "MIX"


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ReaderError("EMPTY_DISTRIBUTION")
    position = probability * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def distribution(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values), "minimum": min(values),
        "q10": percentile(values, 0.10), "median": percentile(values, 0.50),
        "q90": percentile(values, 0.90), "maximum": max(values),
    }


def current_matched(parsed: ParsedLog) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if len(parsed.blocks) != 2:
        raise ReaderError("G6_SUMMARY_COUNT_MISMATCH", len(parsed.blocks))
    tables: list[dict[int, dict[str, str]]] = []
    for iteration, block in enumerate(parsed.blocks):
        table: dict[int, dict[str, str]] = {}
        for row in block.rows:
            line = exact_int(row, "line")
            if line in table:
                raise ReaderError("G6_SELECTED_LINE_DUPLICATE", {"iteration": iteration, "line": line})
            table[line] = row
        tables.append(table)
    intersection = sorted(set(tables[0]) & set(tables[1]))
    matched: dict[int, dict[str, Any]] = {}
    class_counts = Counter()
    neg_counts = Counter()
    d_by_class: dict[str, list[float]] = {"BL": [], "EXP": [], "MIX": []}
    for line in intersection:
        left, right = tables[0][line], tables[1][line]
        left_class, right_class = classify_row(left), classify_row(right)
        if left_class != right_class:
            raise ReaderError("G6_MATCHED_CLASS_CHANGED_ACROSS_ITERATIONS", line)
        chis = (finite_float(left, "chi_effective"), finite_float(right, "chi_effective"))
        if left_class == "HAZ":
            axis = "HAZ"
        elif chis[0] > 0 and chis[1] > 0:
            axis = "FPC"
            class_counts[left_class] += 1
        elif chis[0] < 0 and chis[1] < 0:
            axis = "NEG"
            neg_counts[left_class] += 1
        else:
            axis = "CROSS_OR_ZERO"
        record: dict[str, Any] = {
            "line": line, "class": left_class, "chi_axis": axis,
            "raw_by_iteration": (left, right), "capture_d": None, "capture_frozen": None,
        }
        if axis in ("FPC", "NEG"):
            if any(row.get(name) != "1" for row in (left, right) for name in (
                    "producer_terms_defined", "producer_raw_defined", "independent_fields_defined")):
                raise ReaderError("G6_CAPTURE_FIELDS_UNDEFINED", line)
            eta0, eta1 = finite_float(left, "producer_eta"), finite_float(right, "producer_eta")
            tau0, tau1 = finite_float(left, "producer_tau_eff"), finite_float(right, "producer_tau_eff")
            if eta0 == 0 or tau0 == 0 or tau1 == 0:
                raise ReaderError("G6_D_ZERO_DENOMINATOR", line)
            value = abs((eta1 / eta0) / (tau1 / tau0) - 1.0)
            if not math.isfinite(value):
                raise ReaderError("G6_D_NONFINITE", line)
            record["capture_d"] = value
            record["capture_frozen"] = value <= FROZEN_TOLERANCE
            if axis == "FPC":
                d_by_class[left_class].append(value)
        matched[line] = record
    all_fpc = [value for values in d_by_class.values() for value in values]
    observation = {
        "matched_FPC": dict(class_counts),
        "matched_NEG": dict(neg_counts),
        "d_distribution_by_class": {name: distribution(values) for name, values in d_by_class.items()},
        "ff": (sum(value <= FROZEN_TOLERANCE for value in all_fpc) / len(all_fpc)
               if all_fpc else None),
        "intersection": len(intersection),
    }
    return matched, observation


def gate6_transitions(
    parsed: ParsedLog, f1b: dict[str, Any], run_root: Path,
    global_levels: list[Level], by_key: dict[tuple[int, int, int], Level],
) -> dict[str, Any]:
    current, observation = current_matched(parsed)
    source_rows = f1b["data"]["matched_rows"]
    bl_lines = validate_tracking_population(source_rows, "FPC", "BL", 95)
    neg_lines = validate_tracking_population(source_rows, "NEG", "BL", 46)
    transitions: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for population, destination in ((bl_lines, transitions), (neg_lines, negative)):
        for line in population:
            before = f1b["by_line"][line]
            after = current.get(line)
            destination.append({
                "line": line,
                "before_axis": before.get("chi_axis"),
                "before_class": before.get("class"),
                "before_d": before.get("capture_d"),
                "before_frozen": before.get("capture_frozen"),
                "after_present": after is not None,
                "after_axis": after.get("chi_axis") if after else None,
                "after_class": after.get("class") if after else None,
                "after_d": after.get("capture_d") if after else None,
                "after_frozen": after.get("capture_frozen") if after else None,
            })
    validate_transition_table(transitions, bl_lines, f1b["by_line"], current)
    validate_transition_table(negative, neg_lines, f1b["by_line"], current)
    bl_residual = [record for record in current.values()
                   if record["chi_axis"] == "FPC" and record["class"] == "BL" and
                   record["capture_d"] is not None and record["capture_d"] <= FROZEN_TOLERANCE]
    neg_residual = [record for record in current.values() if record["chi_axis"] == "NEG"]
    temperatures = load_te_by_iteration(run_root)
    te_frozen = temperatures[0].hex() == temperatures[1].hex()
    residual_attributions: list[dict[str, Any]] = []
    unattributed: list[dict[str, Any]] = []
    for record in bl_residual:
        item = {"line": record["line"], "kind": "BL_FROZEN", "temperature_bits_equal": te_frozen}
        item["attribution"] = "SHELL_TE_BINARY64_FROZEN" if te_frozen else "SLB-R_TE_NOT_FROZEN"
        (residual_attributions if te_frozen else unattributed).append(item)
    for record in neg_residual:
        collisions = [
            exp_pair_collision(record["raw_by_iteration"][iteration], temperatures[iteration],
                               global_levels, by_key)
            for iteration in (0, 1)
        ]
        named = all(item["exp_bit_collision"] and not item["equal_energy_bits"] for item in collisions)
        item = {
            "line": record["line"], "kind": "NEG", "iterations": collisions,
            "attribution": "EXP_BINARY64_BIT_COLLISION" if named else "SLB-R_NEG_NOT_EXPLAINED",
        }
        (residual_attributions if named else unattributed).append(item)
    observation.update({
        "transitions_95": transitions,
        "negative_tracking_46": negative,
        "bl_residual": len(bl_residual),
        "negative_residual": len(neg_residual),
        "shell0_temperature_hex": [value.hex() for value in temperatures],
        "residual_attributions": residual_attributions,
        "unattributed": unattributed,
        "all_residuals_attributed": not unattributed,
    })
    return observation


def validate_transition_table(
    rows: list[dict[str, Any]], population: list[int],
    before_by_line: dict[int, dict[str, Any]], after_by_line: dict[int, dict[str, Any]],
) -> None:
    if [row.get("line") for row in rows] != population:
        raise ReaderError("G6_TRANSITION_MISATTRIBUTED", "line population/order")
    for row, line in zip(rows, population):
        before, after = before_by_line[line], after_by_line.get(line)
        expected = {
            "line": line,
            "before_axis": before.get("chi_axis"),
            "before_class": before.get("class"),
            "before_d": before.get("capture_d"),
            "before_frozen": before.get("capture_frozen"),
            "after_present": after is not None,
            "after_axis": after.get("chi_axis") if after else None,
            "after_class": after.get("class") if after else None,
            "after_d": after.get("capture_d") if after else None,
            "after_frozen": after.get("capture_frozen") if after else None,
        }
        if row != expected:
            raise ReaderError("G6_TRANSITION_MISATTRIBUTED", line)


def validate_tracking_population(
    rows: list[dict[str, Any]], axis: str, class_name: str, expected_count: int,
) -> list[int]:
    selected = [row for row in rows
                if row.get("chi_axis") == axis and row.get("class") == class_name]
    lines = [row.get("line") for row in selected]
    if (len(selected) != expected_count or any(not isinstance(line, int) for line in lines) or
            len(set(lines)) != expected_count):
        raise ReaderError("G6_PREDECESSOR_TRACKING_POPULATION", {
            "axis": axis, "class": class_name, "count": len(selected),
            "unique": len(set(lines)), "expected": expected_count,
        })
    return sorted(lines)


def sha256(path: Path) -> str:
    regular_file(path, "MIRROR_FILE_UNSAFE")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run_seal(spec: dict[str, Any]) -> dict[str, Any]:
    root: Path = spec["root"]
    manifest, metadata = root / spec["manifest"], root / spec["metadata"]
    regular_file(manifest, "G7_RUN_SEAL_MANIFEST_UNSAFE")
    regular_file(metadata, "G7_RUN_SEAL_METADATA_UNSAFE")
    manifest_lines = read_text(manifest).splitlines()
    if len(manifest_lines) != spec["entries"]:
        raise ReaderError("G7_RUN_SEAL_LINE_COUNT", {"name": spec["name"], "count": len(manifest_lines)})
    metadata_lines = read_text(metadata).splitlines()
    for wanted in (f"schema={spec['schema']}", f"run_root={root}", "DO_NOT_MUTATE=1"):
        if metadata_lines.count(wanted) != 1:
            raise ReaderError("G7_RUN_SEAL_METADATA", {"name": spec["name"], "token": wanted})
    completed = subprocess.run(
        ["sha256sum", "-c", spec["manifest"]], cwd=root,
        capture_output=True, text=True, check=False,
    )
    output_lines = completed.stdout.splitlines() + completed.stderr.splitlines()
    ok = sum(line.endswith(": OK") for line in output_lines)
    non_ok = len(output_lines) - ok
    if completed.returncode != 0 or ok != spec["entries"] or non_ok != 0:
        raise ReaderError("G7_RUN_SEAL_CHECK", {
            "name": spec["name"], "rc": completed.returncode, "ok": ok, "non_ok": non_ok,
        })
    return {"name": spec["name"], "root": str(root), "entries": ok, "non_ok": non_ok}


class PostRunSealGuard:
    """Run every registered post-run seal check exactly once, even on failure."""

    def __init__(
        self, seal_specs: tuple[dict[str, Any], ...] = SEALS,
        seal_verifier: Any = verify_run_seal,
    ) -> None:
        self.seal_specs = seal_specs
        self.seal_verifier = seal_verifier
        self.checked = False
        self.results: list[dict[str, Any]] = []
        self.failure: ReaderError | None = None

    def verify(self) -> list[dict[str, Any]]:
        if self.checked:
            if self.failure is not None:
                raise self.failure
            return self.results
        self.checked = True
        errors: list[dict[str, Any]] = []
        for spec in self.seal_specs:
            try:
                self.results.append(self.seal_verifier(spec))
            except BaseException as exc:
                failure = named_failure(exc, "G7_POST_RUN_SEAL")
                errors.append({
                    "name": spec.get("name"), "reason": failure.reason,
                    "detail": failure.detail,
                })
        if errors:
            self.failure = ReaderError("G7_POST_RUN_SEAL_FAILURE", {
                "seal_results": self.results, "seal_errors": errors,
            })
            raise self.failure
        return self.results


def compare_mirror(
    left: Path, right: Path, logs: dict[str, ParsedLog],
    left_name: str = "B-1", right_name: str = "B-2",
) -> dict[str, Any]:
    summaries1 = [summary_public(block.summary) for block in logs[left_name].blocks]
    summaries2 = [summary_public(block.summary) for block in logs[right_name].blocks]
    if summaries1 != summaries2:
        raise ReaderError("G7_SUMMARY_MIRROR_MISMATCH")
    artifacts: list[dict[str, str]] = []
    for rel in MIRROR_FILES:
        digest1, digest2 = sha256(left / rel), sha256(right / rel)
        if digest1 != digest2:
            raise ReaderError("G7_ARTIFACT_MIRROR_MISMATCH", rel)
        artifacts.append({"path": rel, left_name: digest1, right_name: digest2})
    return {"summaries": summaries1, "artifacts": artifacts}


def gate7_mirror(
    b1: Path, b2: Path, logs: dict[str, ParsedLog],
    seal_specs: tuple[dict[str, Any], ...] = SEALS, seal_verifier: Any = verify_run_seal,
    mirror_comparer: Any = compare_mirror, seal_guard: PostRunSealGuard | None = None,
) -> dict[str, Any]:
    mirror: dict[str, Any] | None = None
    mirror_error: ReaderError | None = None
    try:
        mirror = mirror_comparer(b1, b2, logs)
    except BaseException as exc:
        mirror_error = named_failure(exc, "G7_MIRROR")

    # G-7 requires the post-run seal check at termination, including every
    # mirror-failure branch.  Do not allow an earlier comparison exception to
    # skip either seal.
    guard = seal_guard if seal_guard is not None else PostRunSealGuard(seal_specs, seal_verifier)
    try:
        seals = guard.verify()
    except ReaderError as seal_error:
        detail = dict(seal_error.detail)
        detail["mirror_error"] = None if mirror_error is None else mirror_error.reason
        raise ReaderError(seal_error.reason, detail) from seal_error
    if mirror_error is not None:
        raise ReaderError(mirror_error.reason, {
            "mirror_detail": mirror_error.detail, "post_run_seals": seals,
        })
    assert mirror is not None
    return {**mirror, "post_run_seals": seals}


def bake_anchor(matched: dict[int, dict[str, Any]]) -> dict[str, Any]:
    candidates = [line for line, row in matched.items() if row["chi_axis"] == "FPC"]
    if not candidates:
        raise ReaderError("ANCHOR_BAKE_FPC_INTERSECTION_EMPTY")
    line = min(candidates)
    record = matched[line]
    values: list[dict[str, str]] = []
    for iteration, row in enumerate(record["raw_by_iteration"]):
        fields_out = {"tau_validity": str(exact_int(row, "tau_validity"))}
        for name in ("n_upper", "A_ul", "emission_per_sr"):
            fields_out[name] = finite_float(row, name).hex()
        values.append({"iteration": str(iteration), **fields_out})
    return {
        "selection": "minimum line in two-iteration FPC intersection with chi_effective>0 on both",
        "line": line,
        "fields": values,
    }


def write_csv(
    path: Path, rows: list[dict[str, Any]],
    protected_roots: tuple[Path, ...] = READ_ONLY_ROOTS,
) -> None:
    if not rows:
        raise ReaderError("CSV_ROWS_EMPTY", str(path))
    columns = list(rows[0])
    path = validate_output_path(path, protected_roots)
    prepare_output_dir(path.parent, protected_roots)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o640)
        validate_output_path(path, protected_roots)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def gate_result(function: Any, *args: Any) -> dict[str, Any]:
    try:
        return {"status": "PASS", "observation": function(*args), "reasons": []}
    except ReaderError as exc:
        return {"status": "FAIL", "observation": None, "reasons": [exc.reason], "detail": exc.detail}


def close_analysis_failure(
    exc: BaseException, seal_guard: PostRunSealGuard | None,
) -> ReaderError:
    """Name a failure and, if necessary, close the post-run seal obligation."""
    failure = named_failure(exc, "ANALYSIS")
    if seal_guard is None or seal_guard.checked:
        return failure
    try:
        seals = seal_guard.verify()
    except ReaderError as seal_error:
        detail = dict(seal_error.detail)
        detail["analysis_error"] = {"reason": failure.reason, "detail": failure.detail}
        return ReaderError(seal_error.reason, detail)
    return ReaderError(failure.reason, {
        "analysis_detail": failure.detail, "post_run_seals": seals,
    })


def analyze_full(
    args: argparse.Namespace, seal_guard: PostRunSealGuard | None = None,
) -> tuple[dict[str, Any], int]:
    output_dir = prepare_output_dir(args.output_dir)
    calibration = predecessor_calibration(args.rii_root, args.census_root, args.f1b_json)
    f1b = calibration.pop("f1b")
    logs = {
        "B-1": parse_log(args.b1_root / "stderr.log", keep_zero_rows=True),
        "B-2": parse_log(args.b2_root / "stderr.log", keep_zero_rows=True),
    }
    if args.b3_root is not None:
        logs["B-3"] = parse_log(args.b3_root / "stderr.log", keep_zero_rows=True)
    levels, by_key = load_levels(args.b2_root / "input/model/levels.csv")

    g2 = gate_result(lambda: {
        "B-1": run_verdict_gate(args.verdict_gate, args.rii_root, args.b1_root, "B-1", output_dir),
        "B-2": run_verdict_gate(args.verdict_gate, args.rii_root, args.b2_root, "B-2", output_dir),
    })
    g3 = gate_result(lambda: {
        "B-1": gate3_banner(args.b1_root), "B-2": gate3_banner(args.b2_root),
    })
    g4 = gate_result(lambda: {
        "B-1": gate4_completion(args.b1_root, logs["B-1"]),
        "B-2": gate4_completion(args.b2_root, logs["B-2"]),
    })
    g5 = gate_result(gate5_hazard, {
        "B-1": (args.b1_root, logs["B-1"]), "B-2": (args.b2_root, logs["B-2"]),
    }, levels, by_key)
    g6 = gate_result(gate6_transitions, logs["B-2"], f1b, args.b2_root, levels, by_key)
    g7 = gate_result(
        gate7_mirror, args.b1_root, args.b2_root, logs,
        SEALS, verify_run_seal, compare_mirror, seal_guard,
    )
    gates = {
        "G-RC": {"status": "PASS", "observation": calibration, "reasons": []},
        "G-2": g2, "G-3": g3, "G-4": g4, "G-5": g5, "G-6": g6, "G-7": g7,
    }
    failed = [name for name, gate in gates.items() if gate["status"] != "PASS"]
    e4_expectation = (gate_result(require_e4_expected_zero, g5["observation"])
                      if g5["status"] == "PASS" else {
                          "status": "BLOCKED", "observation": None,
                          "reasons": ["G5_NOT_AVAILABLE"],
                      })
    b3_closure: dict[str, Any] | None = None
    mirror_failure_reasons = {
        "G7_SUMMARY_MIRROR_MISMATCH", "G7_ARTIFACT_MIRROR_MISMATCH", "MIRROR_FILE_UNSAFE",
        "G7_MIRROR_IO_FAILURE", "G7_MIRROR_ENCODING_FAILURE", "G7_MIRROR_INTERRUPTED",
        "G7_MIRROR_UNEXPECTED_FAILURE",
    }
    mirror_failed = "G-7" in failed and bool(set(g7["reasons"]) & mirror_failure_reasons)
    if args.b3_root is not None:
        if not mirror_failed:
            b3_closure = {
                "status": "FAIL", "reason": "B3_OUTSIDE_SLB_F_D",
                "disposition": "SLB-X_UNAUTHORIZED_REPEAT_INPUT",
            }
        else:
            b3_gates = {
                "G-2": gate_result(run_verdict_gate, args.verdict_gate, args.rii_root,
                                     args.b3_root, "B-3", output_dir),
                "G-3": gate_result(gate3_banner, args.b3_root),
                "G-4": gate_result(gate4_completion, args.b3_root, logs["B-3"]),
                "G-5": gate_result(gate5_hazard, {
                    "B-3": (args.b3_root, logs["B-3"]),
                }, levels, by_key),
            }
            b3_failed = [name for name, gate in b3_gates.items() if gate["status"] != "PASS"]
            repeat = (gate_result(compare_mirror, args.b2_root, args.b3_root, logs, "B-2", "B-3")
                      if not b3_failed else None)
            if b3_failed:
                disposition = "B3_INVALID_STAGING_OR_RUN"
            elif repeat is not None and repeat["status"] == "PASS":
                disposition = "B2_B3_MATCH_STAGING_ACCIDENT_INDICATED"
            else:
                disposition = "B2_B3_MISMATCH_NONDETERMINISM_INDICATED"
            b3_closure = {
                "status": "PASS" if not b3_failed else "FAIL",
                "gates": b3_gates, "failed_gates": b3_failed,
                "repeat_comparison": repeat, "disposition": disposition,
            }
    residual = None
    if b3_closure is not None and b3_closure.get("disposition") == "SLB-X_UNAUTHORIZED_REPEAT_INPUT":
        branch, subbranch, status = "SLB-X", "B3_OUTSIDE_SLB_F_D", "BLOCKED"
    elif "G-3" in failed and "G3_ACTIVE_DIMENSION_UNREGISTERED" in g3["reasons"]:
        branch, subbranch, status = "SLB-X", "ACTIVE_DIMENSION_UNREGISTERED", "BLOCKED"
    elif not failed:
        e4 = g5["observation"]["residual"]
        e5_bl = g6["observation"]["bl_residual"]
        e5_neg = g6["observation"]["negative_residual"]
        residual = {"E-4": e4, "E-5_BL": e5_bl, "E-5_NEG": e5_neg}
        if e4 == 0 and e5_bl == 0 and e5_neg == 0:
            branch, subbranch, status = "SLB-A", None, "PASS"
        elif (g5["observation"]["all_residuals_attributed"] and
              g6["observation"]["all_residuals_attributed"]):
            branch, subbranch, status = "SLB-PR", "SLB-P", "PASS"
        else:
            branch, subbranch, status = "SLB-PR", "SLB-R", "BLOCKED"
    elif mirror_failed:
        branch, subbranch, status = "SLB-F", "SLB-F-D", "FAIL"
    else:
        branch, subbranch, status = "SLB-F", None, "FAIL"

    anchor = None
    artifact_hashes = None
    if g6["status"] == "PASS":
        current, _ = current_matched(logs["B-2"])
        anchor = bake_anchor(current)
        write_csv(output_dir / "slb_transition_95.csv", g6["observation"]["transitions_95"])
        write_csv(output_dir / "slb_negative_46.csv", g6["observation"]["negative_tracking_46"])
        atomic_json(output_dir / "slb_anchor_bake.json", anchor)
    if g7["status"] == "PASS":
        artifact_hashes = g7["observation"]["artifacts"]
        atomic_json(output_dir / "slb_artifact_sha256.json", artifact_hashes)
    report = {
        "schema": SCHEMA,
        "status": status,
        "branch": branch,
        "subbranch": subbranch,
        "failed_gates": failed,
        "gates": gates,
        "expectations": {"E-4": e4_expectation},
        "residual": residual,
        "new_anchor": anchor,
        "artifact_sha256": artifact_hashes,
        "b3_closure": b3_closure,
        "physical_values_modified": False,
        "clamp": 0, "floor": 0, "cap": 0, "jitter": 0, "repair": 0,
    }
    atomic_json(output_dir / "superlevel_b_report.json", report)
    return report, 0 if status == "PASS" else 1


def synthetic_log(active: bool = True, summaries: int = 2, zero: bool = False) -> tuple[str, str]:
    stdout_lines = [
        ACTIVE_BANNER if active else "[NLTE] Super-levels: off (identity) (13015 FL -> 1173 SL across ions)",
        CUTOFF_BANNER,
    ]
    stderr_lines: list[str] = []
    for iteration in range(summaries):
        stderr_lines.append(
            f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 rank=1 line=200 Z=26 ion=1 "
            "lower_level=1 upper_level=2 lower_global=0 upper_global=1 "
            "tau_validity=1 chi_effective=1 producer_terms_defined=1 "
            "producer_raw_defined=1 independent_fields_defined=1 "
            f"producer_eta={1 + iteration} producer_tau_eff=1 "
            "n_upper=1 A_ul=1 emission_per_sr=1"
        )
        if zero:
            stderr_lines.append(
                f"{ZERO_PREFIX} phase=REQUESTED_TE shell=0 line=7 Z=26 ion=1 "
                "lower_global=0 upper_global=1 tau_validity=2 n_upper=1 A_ul=1 emission_per_sr=1"
            )
        stderr_lines.extend([
            f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 target_ion=1 complete=1 "
            f"zero_opacity_emitting_rows={1 if zero else 0} sprobe_undefined_rows={1 if zero else 0}",
            f"{R7_PREFIX} lane=DET iter={iteration} te_generation={iteration + 1}->{iteration + 2}",
            f"{PHYSICS_PREFIX} lane=DET iter={iteration} status=COMMITTED",
        ])
    return "\n".join(stdout_lines) + "\n", "\n".join(stderr_lines) + "\n"


def parsed_text(text: str) -> ParsedLog:
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "stderr.log"
        path.write_text(text, encoding="utf-8")
        return parse_log(path, keep_zero_rows=True)


def require_e4_expected_zero(observation: dict[str, Any]) -> None:
    if observation["residual"] != 0:
        raise ReaderError("E4_HAZARD_RESIDUAL", observation["residual"])


def synthetic_f1b() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in range(1_000, 1_095):
        rows.append({
            "line": line, "chi_axis": "FPC", "class": "BL",
            "capture_d": 0.0, "capture_frozen": True,
        })
    for line in range(2_000, 2_046):
        rows.append({
            "line": line, "chi_axis": "NEG", "class": "BL",
            "capture_d": 0.0, "capture_frozen": True,
        })
    return {"data": {"matched_rows": rows}, "by_line": {row["line"]: row for row in rows}}


def write_synthetic_exports(base: Path, run: Path, mode: str) -> None:
    base_values = {key: f"{base}/{key}" for key in PATH_KEYS}
    base_values.update({
        "LUMINA_A210_SPRODUCER_CAPTURE": "1",
        "LUMINA_A210_INDEPENDENT_CAPTURE": "1",
    })
    run_values = {key: value.replace(str(base), str(run)) for key, value in base_values.items()}
    run_values["LUMINA_SUPER_LEVELS"] = "1"
    if mode == "B-1":
        run_values["LUMINA_A210_SPRODUCER_CAPTURE"] = "0"
        run_values["LUMINA_A210_INDEPENDENT_CAPTURE"] = "0"
    for root, values in ((base, base_values), (run, run_values)):
        path = root / "input/resolved_lumina.exports"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f'declare -x {key}="{values[key]}"\n' for key in sorted(values)),
                        encoding="utf-8")


def make_synthetic_run(root: Path, stdout: str, stderr: str) -> None:
    (root / "input/model").mkdir(parents=True)
    (root / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "stderr.log").write_text(stderr, encoding="utf-8")
    (root / "model.rc").write_text("0\n", encoding="utf-8")
    (root / "RUN_FOOTER.txt").write_text(
        "schema=LUMINA_DET_FLIGHT_V1\nouter_iterations=2\n"
        f"argv={root / 'input/lumina_cuda'} {root / 'input/model'} 1 2 spectrum nlte\n",
        encoding="utf-8",
    )
    (root / "input/model/levels.csv").write_text(
        "atomic_number,ion_number,level_number,energy_eV,g\n"
        "26,1,1,1.0,2.0\n26,1,2,2.0,4.0\n26,1,100,0.0,1.0\n",
        encoding="utf-8",
    )
    for iteration in (0, 1):
        shell = root / f"work/physics_comparison/physics_DET_iter{iteration:04d}.shell.csv"
        shell.parent.mkdir(parents=True, exist_ok=True)
        shell.write_text("shell_id,T_e_K\n0,10020\n", encoding="utf-8")
    for rel in MIRROR_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"synthetic mirror payload {rel}\n", encoding="utf-8")


def healthy_synthetic_analysis(scratch: Path) -> dict[str, Any]:
    scratch.mkdir(parents=True)
    stdout, stderr = synthetic_log()
    baseline, b1, b2 = scratch / "baseline", scratch / "B-1", scratch / "B-2"
    baseline.mkdir()
    for root in (b1, b2):
        make_synthetic_run(root, stdout, stderr)
    write_synthetic_exports(baseline, b1, "B-1")
    write_synthetic_exports(baseline, b2, "B-2")
    return validate_synthetic_fixture({
        "baseline": baseline, "b1": b1, "b2": b2,
        "f1b": synthetic_f1b(), "stdout": stdout, "stderr": stderr,
    })


def validate_synthetic_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    baseline, b1, b2 = fixture["baseline"], fixture["b1"], fixture["b2"]
    logs = {
        "B-1": parse_log(b1 / "stderr.log", keep_zero_rows=True),
        "B-2": parse_log(b2 / "stderr.log", keep_zero_rows=True),
    }
    levels, by_key = load_levels(b2 / "input/model/levels.csv")
    validate_export_delta(baseline, b1, "B-1")
    validate_export_delta(baseline, b2, "B-2")
    gate3_banner(b1); gate3_banner(b2)
    gate4_completion(b1, logs["B-1"]); gate4_completion(b2, logs["B-2"])
    hazard = gate5_hazard({"B-1": (b1, logs["B-1"]), "B-2": (b2, logs["B-2"])},
                          levels, by_key)
    require_e4_expected_zero(hazard)
    f1b = fixture["f1b"]
    transitions = gate6_transitions(logs["B-2"], f1b, b2, levels, by_key)
    current, _ = current_matched(logs["B-2"])
    bake_anchor(current)
    gate7_mirror(b1, b2, logs, seal_specs=())
    fixture.update({
        "logs": logs,
        "levels": levels, "by_key": by_key, "f1b": f1b,
        "transitions": transitions,
    })
    return fixture


def selftest() -> int:
    failures = 0

    def emit(
        name: str, injection: str, named_reason: str, healthy: bool,
        rejected: bool, restored: bool,
    ) -> None:
        nonlocal failures
        print(f"{name} inject=NONE status={'PASS' if healthy else 'FAIL'} "
              f"reason={'HEALTHY_GATE_PASS' if healthy else 'HEALTHY_GATE_FAILED'}")
        print(f"{name} inject={injection} status={'FAIL' if rejected else 'UNEXPECTED_PASS'} "
              f"reason={named_reason if rejected else 'NOT_DETECTED'}")
        print(f"{name} remove=RESTORE_FIXTURE status={'PASS' if restored else 'FAIL'} "
              f"reason={'PASS' if restored else 'RESTORE_FAILED'}")
        if not healthy or not rejected or not restored:
            failures += 1

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        try:
            fixture = healthy_synthetic_analysis(scratch / "fixture")
            healthy = True
        except ReaderError as exc:
            print(f"NC-S0 inject=NONE status=FAIL reason={exc.reason}")
            return 1

        s0_path = fixture["b2"] / MIRROR_FILES[0]
        s0_payload = s0_path.read_bytes()
        s0_path.unlink()
        rejected = False
        try:
            validate_synthetic_fixture(fixture)
        except ReaderError as exc:
            rejected = exc.reason == "MIRROR_FILE_UNSAFE"
        s0_path.write_bytes(s0_payload)
        restored = gate_result(validate_synthetic_fixture, fixture)["status"] == "PASS"
        emit("NC-S0", "MISSING_REGISTERED_MIRROR_FILE", "MIRROR_FILE_UNSAFE",
             healthy, rejected, restored)

        clean, clean_stdout, clean_stderr = fixture["b1"], fixture["stdout"], fixture["stderr"]

        bad_stdout, _ = synthetic_log(active=False)
        healthy = gate_result(gate3_banner, clean)["status"] == "PASS"
        (clean / "stdout.log").write_text(bad_stdout, encoding="utf-8")
        rejected = False
        try:
            gate3_banner(clean)
        except ReaderError as exc:
            rejected = exc.reason == "G3_BANNER_MODE_CONTRACT"
        (clean / "stdout.log").write_text(clean_stdout, encoding="utf-8")
        try:
            gate3_banner(clean); restored = True
        except ReaderError:
            restored = False
        emit("NC-S1", "OFF_IDENTITY_BANNER", "G3_BANNER_MODE_CONTRACT",
             healthy, rejected, restored)

        _, truncated = synthetic_log(summaries=1)
        healthy = gate_result(gate4_completion, clean, parsed_text(clean_stderr))["status"] == "PASS"
        rejected = False
        try:
            gate4_completion(clean, parsed_text(truncated))
        except ReaderError as exc:
            rejected = exc.reason == "G4_COMPLETION_CONTRACT"
        restored = gate_result(gate4_completion, clean, parsed_text(clean_stderr))["status"] == "PASS"
        emit("NC-S2", "TRUNCATED_SUMMARY", "G4_COMPLETION_CONTRACT",
             healthy, rejected, restored)

        footer = clean / "RUN_FOOTER.txt"
        clean_footer = read_text(footer)
        healthy = gate_result(gate4_completion, clean, parsed_text(clean_stderr))["status"] == "PASS"
        footer.write_text(clean_footer.replace("argv=", "argv=/extra/argument ", 1), encoding="utf-8")
        rejected = False
        try:
            gate4_completion(clean, parsed_text(clean_stderr))
        except ReaderError as exc:
            rejected = exc.reason == "G4_COMPLETION_CONTRACT" and "RUN_FOOTER_ARGV" in exc.detail
        footer.write_text(clean_footer, encoding="utf-8")
        restored = gate_result(gate4_completion, clean, parsed_text(clean_stderr))["status"] == "PASS"
        emit("NC-S2-ARGV", "LEADING_EXTRA_ARGV", "G4_COMPLETION_CONTRACT:RUN_FOOTER_ARGV",
             healthy, rejected, restored)

        _, hazard = synthetic_log(zero=True)
        healthy_observation = gate5_hazard({"B-1": (clean, parsed_text(clean_stderr))},
                                           fixture["levels"], fixture["by_key"])
        healthy = gate_result(require_e4_expected_zero, healthy_observation)["status"] == "PASS"
        hazard_observation = gate5_hazard({"B-1": (clean, parsed_text(hazard))},
                                          fixture["levels"], fixture["by_key"])
        rejected = False
        try:
            require_e4_expected_zero(hazard_observation)
        except ReaderError as exc:
            rejected = exc.reason == "E4_HAZARD_RESIDUAL"
        restored = gate_result(require_e4_expected_zero, healthy_observation)["status"] == "PASS"
        emit("NC-S3", "ZERO_ROW", "E4_HAZARD_RESIDUAL", healthy, rejected, restored)

        actual_base, actual_run = fixture["baseline"], fixture["b2"]
        run_env = actual_run / "input/resolved_lumina.exports"
        text_run = read_text(run_env)
        healthy = gate_result(validate_export_delta, actual_base, actual_run, "B-2")["status"] == "PASS"
        with run_env.open("a", encoding="utf-8") as stream:
            stream.write('declare -x LUMINA_EXTRA_UNREGISTERED="1"\n')
        rejected = False
        try:
            validate_export_delta(actual_base, actual_run, "B-2")
        except ReaderError as exc:
            rejected = exc.reason == "G2_EXPORT_DELTA_KEYSET"
        run_env.write_text(text_run, encoding="utf-8")
        try:
            validate_export_delta(actual_base, actual_run, "B-2"); restored = True
        except ReaderError:
            restored = False
        emit("NC-S4", "EXTRA_ENV_KEY", "G2_EXPORT_DELTA_KEYSET", healthy, rejected, restored)

        transitions = fixture["transitions"]["transitions_95"]
        population = fixture["f1b"]["data"]["matched_rows"][:95]
        population_lines = [row["line"] for row in population]
        current, _ = current_matched(fixture["logs"]["B-2"])
        healthy = True
        try:
            validate_transition_table(transitions, population_lines, fixture["f1b"]["by_line"], current)
        except ReaderError:
            healthy = False
        saved_after_class = transitions[0]["after_class"]
        transitions[0]["after_class"] = "MIX"
        rejected = False
        try:
            validate_transition_table(transitions, population_lines, fixture["f1b"]["by_line"], current)
        except ReaderError as exc:
            rejected = exc.reason == "G6_TRANSITION_MISATTRIBUTED"
        transitions[0]["after_class"] = saved_after_class
        restored = gate_result(validate_transition_table, transitions, population_lines,
                               fixture["f1b"]["by_line"], current)["status"] == "PASS"
        emit("NC-S5", "AFTER_TRANSITION_MISATTRIBUTION", "G6_TRANSITION_MISATTRIBUTED",
             healthy, rejected, restored)

        bad_dimension = clean_stdout.replace(
            ACTIVE_BANNER, "[NLTE] Super-levels: ACTIVE (13015 FL -> 1174 SL across ions)")
        healthy = gate_result(gate3_banner, clean)["status"] == "PASS"
        (clean / "stdout.log").write_text(bad_dimension, encoding="utf-8")
        rejected = False
        try:
            gate3_banner(clean)
        except ReaderError as exc:
            rejected = exc.reason == "G3_ACTIVE_DIMENSION_UNREGISTERED"
        (clean / "stdout.log").write_text(clean_stdout, encoding="utf-8")
        restored = gate_result(gate3_banner, clean)["status"] == "PASS"
        emit("NC-S6-X", "ACTIVE_1174_SL", "G3_ACTIVE_DIMENSION_UNREGISTERED",
             healthy, rejected, restored)

        b1_mirror, b2_mirror, mirror_logs = fixture["b1"], fixture["b2"], fixture["logs"]
        seal_calls: list[str] = []
        fake_seals = ({"name": "census"}, {"name": "R-II"})
        def fake_seal(spec: dict[str, Any]) -> dict[str, Any]:
            seal_calls.append(spec["name"])
            return {"name": spec["name"], "status": "PASS"}
        normal_raw_leak = False
        try:
            healthy = (gate7_mirror(b1_mirror, b2_mirror, mirror_logs,
                                    fake_seals, fake_seal)["post_run_seals"] != [])
        except BaseException:
            healthy, normal_raw_leak = False, True
        normal_calls = list(seal_calls)
        injected_path = b2_mirror / MIRROR_FILES[0]
        injected_path.write_bytes(injected_path.read_bytes() + b"X")
        seal_calls.clear()
        digest_raw_leak = False
        try:
            gate7_mirror(b1_mirror, b2_mirror, mirror_logs, fake_seals, fake_seal)
            rejected = False
        except ReaderError as exc:
            rejected = (exc.reason == "G7_ARTIFACT_MIRROR_MISMATCH" and
                        seal_calls == ["census", "R-II"])
        except BaseException:
            rejected, digest_raw_leak = False, True
        digest_calls = list(seal_calls)
        shutil.copyfile(b1_mirror / MIRROR_FILES[0], injected_path)
        seal_calls.clear()
        restored = (gate_result(gate7_mirror, b1_mirror, b2_mirror, mirror_logs,
                                fake_seals, fake_seal)["status"] == "PASS" and
                    seal_calls == ["census", "R-II"])
        emit("NC-S7", "MIRROR_ONE_BYTE_MISMATCH", "G7_ARTIFACT_MIRROR_MISMATCH",
             healthy, rejected, restored)
        print(f"C07-G7-NORMAL post_seal_calls={len(normal_calls)} "
              f"names={','.join(normal_calls)} exception_leak={int(normal_raw_leak)}")
        print(f"C07-G7-DIGEST-MISMATCH post_seal_calls={len(digest_calls)} "
              f"names={','.join(digest_calls)} reason=G7_ARTIFACT_MIRROR_MISMATCH "
              f"exception_leak={int(digest_raw_leak)}")

        original_mode = stat.S_IMODE(injected_path.stat().st_mode)
        injected_path.chmod(0o000)
        seal_calls.clear()
        permission_raw_leak = False
        permission_exception = ""
        try:
            gate7_mirror(b1_mirror, b2_mirror, mirror_logs, fake_seals, fake_seal)
            permission_rejected = False
        except ReaderError as exc:
            mirror_detail = exc.detail.get("mirror_detail", {})
            permission_exception = mirror_detail.get("exception", "")
            permission_rejected = (exc.reason == "G7_MIRROR_IO_FAILURE" and
                                   permission_exception == "PermissionError" and
                                   seal_calls == ["census", "R-II"])
        except BaseException:
            permission_rejected, permission_raw_leak = False, True
        permission_calls = list(seal_calls)
        injected_path.chmod(original_mode)
        seal_calls.clear()
        permission_restored = (
            gate_result(gate7_mirror, b1_mirror, b2_mirror, mirror_logs,
                        fake_seals, fake_seal)["status"] == "PASS" and
            seal_calls == ["census", "R-II"]
        )
        emit("NC-S7-PERMISSION", "MIRROR_FILE_CHMOD_000", "G7_MIRROR_IO_FAILURE",
             healthy, permission_rejected, permission_restored)
        print(f"C07-G7-PERMISSION post_seal_calls={len(permission_calls)} "
              f"names={','.join(permission_calls)} exception={permission_exception} "
              f"reason=G7_MIRROR_IO_FAILURE "
              f"exception_leak={int(permission_raw_leak)}")

        invalid_utf8 = scratch / "invalid_utf8.log"
        invalid_utf8.write_bytes(b"\xff")
        try:
            read_text(invalid_utf8)
            encoding_failure: BaseException = AssertionError("invalid UTF-8 was accepted")
        except BaseException as exc:
            encoding_failure = exc
        failure_cases: tuple[tuple[str, BaseException, str], ...] = (
            ("NC-S7-DISK-FULL", OSError(errno.ENOSPC, "synthetic disk full"),
             "ANALYSIS_IO_FAILURE"),
            ("NC-S7-ENCODING", encoding_failure, "TEXT_READ_FAILED"),
            ("NC-S7-INTERRUPT", KeyboardInterrupt(), "ANALYSIS_INTERRUPTED"),
        )
        for name, injected, expected_reason in failure_cases:
            seal_calls.clear()
            guard = PostRunSealGuard(fake_seals, fake_seal)
            failure = close_analysis_failure(injected, guard)
            closed = (failure.reason == expected_reason and
                      seal_calls == ["census", "R-II"])
            emit(name, type(injected).__name__, expected_reason, True, closed, closed)
            print(f"C07-{name.removeprefix('NC-S7-')} post_seal_calls={len(seal_calls)} "
                  f"names={','.join(seal_calls)} reason={failure.reason} exception_leak=0")

        seal_calls.clear()
        def failing_seal(spec: dict[str, Any]) -> dict[str, Any]:
            seal_calls.append(spec["name"])
            if spec["name"] == "census":
                raise PermissionError("synthetic seal read failure")
            return {"name": spec["name"], "status": "PASS"}
        try:
            PostRunSealGuard(fake_seals, failing_seal).verify()
            seal_failure_closed = False
        except ReaderError as exc:
            seal_failure_closed = (exc.reason == "G7_POST_RUN_SEAL_FAILURE" and
                                   seal_calls == ["census", "R-II"])
        emit("NC-S7-SEAL-FAIL", "CENSUS_SEAL_PERMISSION_ERROR",
             "G7_POST_RUN_SEAL_FAILURE", True, seal_failure_closed, seal_failure_closed)
        print(f"C07-SEAL-FAIL post_seal_calls={len(seal_calls)} "
              f"names={','.join(seal_calls)} reason=G7_POST_RUN_SEAL_FAILURE exception_leak=0")

        b3 = scratch / "B-3"
        shutil.copytree(b2_mirror, b3)
        b3_logs = {"B-2": mirror_logs["B-2"], "B-3": mirror_logs["B-2"]}
        healthy = gate_result(compare_mirror, b2_mirror, b3, b3_logs, "B-2", "B-3")["status"] == "PASS"
        b3_bad = b3 / MIRROR_FILES[1]
        b3_bad.write_bytes(b3_bad.read_bytes() + b"X")
        rejected = False
        try:
            compare_mirror(b2_mirror, b3, b3_logs, "B-2", "B-3")
        except ReaderError as exc:
            rejected = exc.reason == "G7_ARTIFACT_MIRROR_MISMATCH"
        shutil.copyfile(b2_mirror / MIRROR_FILES[1], b3_bad)
        restored = gate_result(compare_mirror, b2_mirror, b3, b3_logs,
                               "B-2", "B-3")["status"] == "PASS"
        emit("NC-S7-B3", "B3_ONE_BYTE_MISMATCH", "G7_ARTIFACT_MIRROR_MISMATCH",
             healthy, rejected, restored)

        model_seal = scratch / "model_seal"
        model_seal.mkdir()
        protected = (model_seal,)
        outside = scratch / "safe_output/report.json"
        atomic_json(outside, {"status": "PASS"}, protected)
        healthy = outside.is_file()
        sealed_report = model_seal / "report.json"
        sealed_report.write_text("sealed\n", encoding="utf-8")
        sealed_report.chmod(0o444)
        sealed_inode = sealed_report.stat().st_ino
        rejected = False
        try:
            atomic_json(sealed_report, {"status": "FAIL"}, protected)
        except ReaderError as exc:
            rejected = exc.reason == "OUTPUT_INSIDE_READ_ONLY_SEAL"
        restored = (sealed_report.stat().st_ino == sealed_inode and
                    stat.S_IMODE(sealed_report.stat().st_mode) == 0o444 and
                    sealed_report.read_text(encoding="utf-8") == "sealed\n")
        emit("NC-OUT-SEAL", "OUTPUT_INSIDE_MODEL_SEAL", "OUTPUT_INSIDE_READ_ONLY_SEAL",
             healthy, rejected, restored)

        real_parent = scratch / "real_parent"
        real_parent.mkdir()
        link_parent = scratch / "link_parent"
        link_parent.symlink_to(real_parent, target_is_directory=True)
        healthy_path = real_parent / "report.json"
        atomic_json(healthy_path, {"status": "PASS"}, protected)
        healthy = healthy_path.is_file()
        rejected = False
        try:
            atomic_json(link_parent / "through.json", {"status": "FAIL"}, protected)
        except ReaderError as exc:
            rejected = exc.reason == "OUTPUT_ANCESTOR_SYMLINK"
        atomic_json(real_parent / "restored.json", {"status": "PASS"}, protected)
        restored = not (real_parent / "through.json").exists() and (real_parent / "restored.json").is_file()
        emit("NC-OUT-ANCESTOR", "INTERMEDIATE_SYMLINK", "OUTPUT_ANCESTOR_SYMLINK",
             healthy, rejected, restored)

        dangling = real_parent / "dangling_report.json"
        dangling.symlink_to(real_parent / "missing_target.json")
        healthy = (real_parent / "restored.json").is_file()
        rejected = False
        try:
            atomic_json(dangling, {"status": "FAIL"}, protected)
        except ReaderError as exc:
            rejected = exc.reason == "OUTPUT_PATH_SYMLINK"
        dangling.unlink()
        atomic_json(dangling, {"status": "PASS"}, protected)
        restored = dangling.is_file() and not dangling.is_symlink()
        emit("NC-OUT-DANGLING", "DANGLING_OUTPUT_SYMLINK", "OUTPUT_PATH_SYMLINK",
             healthy, rejected, restored)

        error_target = model_seal / "superlevel_b_reader_error.json"
        healthy_error = scratch / "safe_output/superlevel_b_reader_error.json"
        atomic_json(healthy_error, {"status": "FAIL", "reason": "SYNTHETIC"}, protected)
        healthy = healthy_error.is_file()
        rejected = False
        try:
            atomic_json(error_target, {"status": "FAIL", "reason": "SYNTHETIC"}, protected)
        except ReaderError as exc:
            rejected = exc.reason == "OUTPUT_INSIDE_READ_ONLY_SEAL"
        atomic_json(scratch / "safe_output/error_restored.json", {"status": "PASS"}, protected)
        restored = not error_target.exists()
        emit("NC-OUT-ERROR", "ERROR_REPORT_INSIDE_MODEL_SEAL", "OUTPUT_INSIDE_READ_ONLY_SEAL",
             healthy, rejected, restored)

    print(f"SUPERLEVEL_B_ANALYZER_SELFTEST {'PASS' if failures == 0 else 'FAIL'} "
          "controls=NC-S0,NC-S1,NC-S2,NC-S2-ARGV,NC-S3,NC-S4,NC-S5,"
          "NC-S6-X,NC-S7,NC-S7-PERMISSION,NC-S7-DISK-FULL,NC-S7-ENCODING,"
          "NC-S7-INTERRUPT,NC-S7-SEAL-FAIL,NC-S7-B3,NC-OUT-SEAL,NC-OUT-ANCESTOR,"
          "NC-OUT-DANGLING,NC-OUT-ERROR "
          f"failed={failures}")
    return 0 if failures == 0 else 1


def mutate_registered_fpc_byte(path: Path, line_id: int) -> tuple[int, bytes]:
    line_token = f" line={line_id} ".encode("ascii")
    field_token = b"producer_eta="
    with path.open("r+b") as stream:
        offset = 0
        for raw in stream:
            if (raw.startswith(ROW_PREFIX.encode("ascii")) and line_token in raw and
                    b"producer_terms_defined=1" in raw and field_token in raw):
                start = raw.index(field_token) + len(field_token)
                end = raw.find(b" ", start)
                if end < 0:
                    end = len(raw.rstrip(b"\r\n"))
                for index in range(start, end):
                    if raw[index:index + 1] in b"0123456789":
                        absolute = offset + index
                        original = raw[index:index + 1]
                        replacement = b"1" if original != b"1" else b"2"
                        stream.seek(absolute)
                        stream.write(replacement)
                        stream.flush()
                        os.fsync(stream.fileno())
                        return absolute, original
            offset += len(raw)
    raise ReaderError("G_RC_SELFTEST_MUTATION_TARGET_MISSING", line_id)


def g_rc_selftest(args: argparse.Namespace) -> int:
    required = ("rii_root", "census_root", "f1b_json", "scratch_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ReaderError("ARGUMENT_REQUIRED", missing)
    scratch_root = prepare_output_dir(args.scratch_root)
    if not str(scratch_root).startswith("/gpfs/"):
        raise ReaderError("G_RC_SELFTEST_SCRATCH_NOT_GPFS", str(scratch_root))
    f1b = validate_registered_f1b(args.f1b_json)
    line_id = f1b["bl_lines"][0]
    with tempfile.TemporaryDirectory(prefix="slb_g_rc_", dir=scratch_root) as scratch_name:
        copied_root = Path(scratch_name) / "R-II-copy"
        copied_root.mkdir()
        source = args.rii_root / "stderr.log"
        regular_file(source, "G_RC_SOURCE_LOG_UNSAFE")
        copied = copied_root / "stderr.log"
        shutil.copyfile(source, copied)
        predecessor_calibration(copied_root, args.census_root, args.f1b_json)
        print("NC-RC inject=NONE status=PASS reason=G_RC_RAW_PREDECESSOR_REPRODUCED")
        offset, original = mutate_registered_fpc_byte(copied, line_id)
        rejected = False
        named_reason = "NOT_DETECTED"
        try:
            predecessor_calibration(copied_root, args.census_root, args.f1b_json)
        except ReaderError as exc:
            named_reason = exc.reason
            rejected = exc.reason.startswith("G_RC_RAW_")
        print(f"NC-RC inject=ONE_BYTE_RAW_RII_COPY status={'FAIL' if rejected else 'UNEXPECTED_PASS'} "
              f"reason={named_reason} line={line_id} offset={offset}")
        with copied.open("r+b") as stream:
            stream.seek(offset)
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        restored = True
        try:
            predecessor_calibration(copied_root, args.census_root, args.f1b_json)
        except ReaderError:
            restored = False
        print(f"NC-RC remove=RESTORE_ONE_BYTE status={'PASS' if restored else 'FAIL'} "
              "reason=G_RC_RAW_PREDECESSOR_REPRODUCED")
    print(f"SUPERLEVEL_B_G_RC_SELFTEST {'PASS' if rejected and restored else 'FAIL'} controls=NC-RC")
    return 0 if rejected and restored else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-g-rc", action="store_true")
    parser.add_argument("--mode", choices=("g-rc", "analyze"))
    parser.add_argument("--rii-root", type=Path)
    parser.add_argument("--census-root", type=Path)
    parser.add_argument("--f1b-json", type=Path)
    parser.add_argument("--b1-root", type=Path)
    parser.add_argument("--b2-root", type=Path)
    parser.add_argument("--b3-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--verdict-gate", type=Path,
        default=Path(__file__).resolve().parent / "verdict_gate.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    seal_guard: PostRunSealGuard | None = None
    try:
        if args.selftest_g_rc:
            return g_rc_selftest(args)
        required = ("mode", "rii_root", "census_root", "f1b_json", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ReaderError("ARGUMENT_REQUIRED", missing)
        if args.mode == "g-rc":
            args.output_dir = prepare_output_dir(args.output_dir)
            calibration = predecessor_calibration(args.rii_root, args.census_root, args.f1b_json)
            calibration.pop("f1b")
            report = {"schema": SCHEMA, "mode": "G-RC", "status": "PASS", "calibration": calibration}
            atomic_json(args.output_dir / "superlevel_b_g_rc.json", report)
            print("SUPERLEVEL_B_G_RC_PASS ff=95/885 BL=95 EXP=284 MIX=506 NEG=46 zero=221382x2")
            return 0
        if args.b1_root is None or args.b2_root is None:
            raise ReaderError("ARGUMENT_REQUIRED", ["b1_root", "b2_root"])
        seal_guard = PostRunSealGuard()
        args.output_dir = prepare_output_dir(args.output_dir)
        report, rc = analyze_full(args, seal_guard)
        print(f"SUPERLEVEL_B_ANALYSIS_{report['status']} branch={report['branch']} "
              f"subbranch={report['subbranch']} report={args.output_dir / 'superlevel_b_report.json'}")
        return rc
    except BaseException as caught:
        exc = close_analysis_failure(caught, seal_guard)
        error = {"schema": SCHEMA, "status": "FAIL", "reason": exc.reason, "detail": exc.detail}
        output_error: ReaderError | None = None
        if getattr(args, "output_dir", None) is not None:
            try:
                atomic_json(args.output_dir / "superlevel_b_reader_error.json", error)
            except BaseException as write_exc:
                output_error = named_failure(write_exc, "ERROR_REPORT")
        reported = output_error if output_error is not None else exc
        print(f"SUPERLEVEL_B_READER_FAIL reason={reported.reason} detail={reported.detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
