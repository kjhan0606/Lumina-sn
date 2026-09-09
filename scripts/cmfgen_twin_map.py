#!/usr/bin/env python3
"""Offline reader for DET-CMFGEN-TWIN-MAP.

The reader deliberately has no acquisition path.  Missing observations are
published as INPUT_PENDING; values are never interpolated, floored, clamped,
or renormalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA = "CMFGEN_TWIN_MAP_REV4_V1"
TARGET_ELEMENTS = {"ca": 20, "co": 27, "fe": 26, "ni": 28, "si": 14, "s": 16}
REFERENCE_NAMES = ["phys_toy06_cmfgen.txt"] + [
    f"ionfrac_{name}_toy06_cmfgen.txt" for name in TARGET_ELEMENTS
]
BASE_BINARY_SHA256 = "ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059"
CONTRACT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md")
REAL_BASE = Path("/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on")
REAL_REFERENCE = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06")
# Sealed source convention: SRC/src/lumina.h defines AMU, consumed by
# SRC/src/lumina_plasma.c when constructing the production element total.
AMU_G = Decimal("1.660539066e-24")
PUBLICATION_FIELDS = frozenset(("T", "ne", "Saha", "NLTE"))


class Failure(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def fail(reason: str, detail: str = "") -> None:
    raise Failure(reason, detail)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_regular(path: Path, reason: str = "INPUT_NOT_SAFE_REGULAR") -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        fail(reason, str(path))


def require_real_path(raw: str, *, base: Path | None = None, must_exist: bool = True) -> Path:
    """Reject lexical aliases and every symlink component, then return realpath."""
    if not raw.startswith("/") or raw == "/" or os.path.normpath(raw) != raw:
        fail("PATH_NOT_CANONICAL_ABSOLUTE", raw)
    path = Path(raw)
    probe = path if must_exist else path.parent
    if must_exist and not path.exists():
        fail("PATH_MISSING", raw)
    if not probe.exists() or not probe.is_dir():
        fail("PATH_PARENT_MISSING", str(probe))
    cursor = Path("/")
    for part in probe.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            fail("PATH_SYMLINK_COMPONENT", str(cursor))
    real = Path(os.path.realpath(path if must_exist else path.parent))
    candidate = real if must_exist else real / path.name
    if base is not None:
        base_real = Path(os.path.realpath(base))
        if candidate == base_real or base_real in candidate.parents:
            fail("OUTPUT_INSIDE_BASE", raw)
    return candidate


def validate_output_parents(evidence: Path, base: Path) -> None:
    for rel in ("inputs", "offline", "run", "result"):
        require_real_path(str(evidence / rel), base=base)


def decimal_token(text: str, *, nonnegative: bool = True) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        fail("NONFINITE_OR_INVALID_NUMBER", text)
    if not value.is_finite() or (nonnegative and value < 0):
        fail("NEGATIVE_OR_NONFINITE_VALUE", text)
    return value


def token_interval(text: str) -> tuple[Decimal, Decimal]:
    """Interval implied by round-to-nearest at the token's printed place."""
    value = decimal_token(text)
    if value == 0:
        return value, value
    unit = Decimal(1).scaleb(value.as_tuple().exponent)
    half = unit / 2
    return value - half, value + half


def ratio_interval(a_token: str, b_token: str) -> tuple[Decimal, Decimal]:
    alo, ahi = token_interval(a_token)
    blo, bhi = token_interval(b_token)
    if alo <= 0 or blo <= 0:
        fail("LOG_DOMAIN_NONPOSITIVE", f"numerator={a_token} denominator={b_token}")
    return alo / bhi, ahi / blo


def dex_interval(a_token: str, b_token: str) -> tuple[float, float]:
    lo, hi = ratio_interval(a_token, b_token)
    return math.log10(float(lo)), math.log10(float(hi))


def scalar_metric(a_token: str, b_token: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "lumina_raw": a_token,
        "cmfgen_raw": b_token,
        "ratio": None,
        "dex": None,
        "dex_lo": None,
        "dex_hi": None,
        "status": "VALID",
        "reason": "",
    }
    try:
        a = decimal_token(a_token)
        b = decimal_token(b_token)
    except Failure as exc:
        result.update(status="INVALID_VALUE", reason=exc.reason)
        return result
    if b == 0:
        result.update(status="UNDEFINED", reason="CMFGEN_DENOMINATOR_ZERO")
    elif a == 0:
        result.update(ratio="0", status="VALID", reason="LOG_UNDEFINED_NONPOSITIVE")
    else:
        ratio = a / b
        dlo, dhi = dex_interval(a_token, b_token)
        result.update(ratio=str(ratio), dex=math.log10(float(ratio)), dex_lo=dlo, dex_hi=dhi)
    return result


def compound_log_metric(
    numerator: Iterable[str | Decimal], denominator: Iterable[str | Decimal]
) -> dict[str, Any]:
    """A signed log ratio with printed-token intervals and exact derived factors."""
    result: dict[str, Any] = {
        "ratio": None, "dex": None, "dex_lo": None, "dex_hi": None,
        "status": "VALID", "reason": "",
    }

    def factor(value: str | Decimal) -> tuple[Decimal, Decimal, Decimal]:
        if isinstance(value, Decimal):
            if not value.is_finite() or value < 0:
                fail("NEGATIVE_OR_NONFINITE_VALUE", str(value))
            return value, value, value
        parsed = decimal_token(value)
        lo, hi = token_interval(value)
        return parsed, lo, hi

    try:
        numerators = [factor(value) for value in numerator]
        denominators = [factor(value) for value in denominator]
    except Failure as exc:
        result.update(status="INVALID_VALUE", reason=exc.reason)
        return result
    if any(value <= 0 for value, _lo, _hi in numerators + denominators):
        result.update(status="UNDEFINED", reason="LOG_DOMAIN_NONPOSITIVE")
        return result
    n_value = math.prod(value for value, _lo, _hi in numerators)
    d_value = math.prod(value for value, _lo, _hi in denominators)
    n_lo = math.prod(lo for _value, lo, _hi in numerators)
    n_hi = math.prod(hi for _value, _lo, hi in numerators)
    d_lo = math.prod(lo for _value, lo, _hi in denominators)
    d_hi = math.prod(hi for _value, _lo, hi in denominators)
    if min(n_lo, d_lo) <= 0:
        result.update(status="UNDEFINED", reason="PRINTED_INTERVAL_CROSSES_ZERO")
        return result
    ratio = n_value / d_value
    result.update(
        ratio=str(ratio),
        dex=math.log10(float(ratio)),
        dex_lo=math.log10(float(n_lo / d_hi)),
        dex_hi=math.log10(float(n_hi / d_lo)),
    )
    return result


def equal_decimal_tokens(a: str, b: str) -> bool | None:
    try:
        return decimal_token(a) == decimal_token(b)
    except Failure:
        return None


def _safe_output_target(parent_fd: int, name: str, display: Path) -> None:
    """Reject links/shared inodes without resolving the parent again."""
    try:
        meta = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(meta.st_mode):
        fail("OUTPUT_PATH_NOT_REGULAR", str(display))
    if meta.st_nlink != 1:
        fail("OUTPUT_PATH_SHARED_INODE", str(display))


def _create_output_tree(path: Path, base: Path, children: Iterable[str]) -> None:
    """Create a new output root and its children through pinned directory fds."""
    if not path.is_absolute() or path == Path("/"):
        fail("OUTPUT_ROOT_UNSAFE", str(path))
    base_meta = os.stat(base, follow_symlinks=False)
    base_key = (base_meta.st_dev, base_meta.st_ino)
    parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    root_fd = -1
    created = False
    created_children: list[str] = []
    complete = False
    try:
        for component in path.parent.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
            meta = os.fstat(parent_fd)
            if (meta.st_dev, meta.st_ino) == base_key:
                fail("OUTPUT_INSIDE_BASE", str(path))
        os.mkdir(path.name, 0o750, dir_fd=parent_fd)
        created = True
        root_fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        for child in children:
            if not child or "/" in child or child in (".", ".."):
                fail("OUTPUT_CHILD_UNSAFE", child)
            os.mkdir(child, 0o750, dir_fd=root_fd)
            created_children.append(child)
        complete = True
    except FileExistsError:
        fail("EVIDENCE_NOT_NEW_SAFE_ABSOLUTE", str(path))
    except OSError as exc:
        fail("OUTPUT_ROOT_UNSAFE", f"{path}:{exc.filename or exc}")
    finally:
        if created and not complete and root_fd >= 0:
            for child in reversed(created_children):
                os.rmdir(child, dir_fd=root_fd)
            os.rmdir(path.name, dir_fd=parent_fd)
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)


def _atomic_output(path: Path, write: Callable[[Any], None], *, newline: str | None = None) -> None:
    """Publish through a fixed directory fd and an exclusively-created inode."""
    if not path.is_absolute():
        fail("OUTPUT_PARENT_UNSAFE", str(path.parent))
    parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parent.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
    except OSError:
        os.close(parent_fd)
        fail("OUTPUT_PARENT_UNSAFE", str(path.parent))
    temporary_name = f".{path.name}.{secrets.token_hex(16)}"
    fd = -1
    try:
        _safe_output_target(parent_fd, path.name, path)
        fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = os.fstat(fd)
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as stream:
            fd = -1
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o640)
        current = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (created.st_dev, created.st_ino):
            fail("OUTPUT_TEMPORARY_REPLACED", str(path))
        _safe_output_target(parent_fd, path.name, path)
        os.replace(temporary_name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def text_write(path: Path, text: str) -> None:
    _atomic_output(path, lambda stream: stream.write(text))


def json_write(path: Path, value: Any) -> None:
    text_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def csv_text(fields: list[str], rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def csv_write(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    text = csv_text(fields, rows)
    _atomic_output(path, lambda stream: stream.write(text), newline="")


@dataclass(frozen=True)
class RawRow:
    epoch: Decimal
    row_index: int
    source_line: int
    tokens: tuple[str, ...]

    @property
    def velocity(self) -> Decimal:
        return decimal_token(self.tokens[0])


def parse_cmfgen(path: Path, columns: int | None) -> dict[Decimal, list[RawRow]]:
    safe_regular(path)
    blocks: dict[Decimal, list[RawRow]] = {}
    epoch: Decimal | None = None
    expected_rows: int | None = None
    rows: list[RawRow] = []

    def finish() -> None:
        nonlocal rows, epoch, expected_rows
        if epoch is None:
            return
        if expected_rows is None or len(rows) != expected_rows:
            fail("CMFGEN_NVEL_MISMATCH", f"path={path} epoch={epoch} actual={len(rows)} expected={expected_rows}")
        if epoch in blocks:
            fail("CMFGEN_DUPLICATE_EPOCH", f"path={path} epoch={epoch}")
        velocities = [row.velocity for row in rows]
        if len(set(velocities)) != len(velocities):
            fail("CMFGEN_DUPLICATE_VELOCITY", f"path={path} epoch={epoch}")
        blocks[epoch] = rows
        rows = []
        epoch = None
        expected_rows = None

    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            text = raw.strip()
            if text.startswith("#TIME:"):
                finish()
                epoch = decimal_token(text.split(":", 1)[1].strip())
            elif text.startswith("#NVEL:"):
                if epoch is None or expected_rows is not None:
                    fail("CMFGEN_BLOCK_HEADER_INVALID", f"path={path} line={line_number}")
                expected_rows = int(text.split(":", 1)[1].strip())
            elif text and not text.startswith("#"):
                if epoch is None or expected_rows is None:
                    fail("CMFGEN_DATA_OUTSIDE_BLOCK", f"path={path} line={line_number}")
                tokens = tuple(text.split())
                if (columns is not None and len(tokens) != columns) or (columns is None and len(tokens) < 2):
                    fail("CMFGEN_COLUMN_COUNT", f"path={path} line={line_number} actual={len(tokens)} expected={columns or 'velocity+stages'}")
                if rows and len(tokens) != len(rows[0].tokens):
                    fail("CMFGEN_COLUMN_COUNT", f"path={path} line={line_number} block_columns_changed")
                for token in tokens:
                    try:
                        Decimal(token)
                    except InvalidOperation:
                        fail("CMFGEN_NUMBER_PARSE_INVALID", f"path={path} line={line_number} token={token}")
                rows.append(RawRow(epoch, len(rows), line_number, tokens))
    finish()
    if not blocks:
        fail("CMFGEN_NO_BLOCKS", str(path))
    return blocks


def read_csv_strict(path: Path, fields: set[str]) -> list[dict[str, str]]:
    safe_regular(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not fields.issubset(reader.fieldnames):
            fail("CSV_SCHEMA_MISMATCH", f"path={path} fields={reader.fieldnames}")
        rows = list(reader)
    return rows


def read_exports(path: Path) -> dict[str, str]:
    safe_regular(path)
    result: dict[str, str] = {}
    prefix = "declare -x "
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.startswith(prefix) or "=" not in raw:
            fail("EXPORT_LINE_INVALID", f"path={path} line={line_number}")
        key, quoted = raw[len(prefix):].split("=", 1)
        if key in result or len(quoted) < 2 or quoted[0] != '"' or quoted[-1] != '"':
            fail("EXPORT_LINE_INVALID", f"path={path} line={line_number}")
        result[key] = quoted[1:-1]
    return result


def effective_nlte_levels(run: Path) -> dict[tuple[int, int], set[tuple[int, int]]]:
    # LUMINA_NLTE_SKIP_Z controls nebular-tau/line-source writing; it does not
    # remove an ion from the NLTE solve or from the individual-level writer.
    exports = read_exports(run / "input/resolved_lumina.exports")
    if exports.get("LUMINA_NLTE_STAGE4", "0") not in ("", "0") or exports.get("LUMINA_NLTE_ELEMENT_WIDE", "0") not in ("", "0"):
        fail("NLTE_PRODUCTION_LAYOUT_OUTSIDE_REGISTERED_LANE")
    identity_path = run / "input/current_source_identity.json"
    safe_regular(identity_path)
    repository = Path(json.loads(identity_path.read_text(encoding="utf-8")).get("repository", ""))
    source_path = repository / "src/lumina_plasma.c"
    safe_regular(source_path)
    source_text = source_path.read_text(encoding="utf-8")
    arrays: list[list[int]] = []
    for name in ("NLTE_TARGET_Z", "NLTE_TARGET_ION"):
        match = re.search(rf"static const int {name}\[\]\s*=\s*\{{(.*?)\}};", source_text, re.S)
        if match is None: fail("NLTE_TARGET_SOURCE_DEFINITION_MISSING", name)
        arrays.append([int(token) for token in re.findall(r"-?\d+", match.group(1))])
    if len(arrays[0]) != len(arrays[1]): fail("NLTE_TARGET_SOURCE_DEFINITION_INVALID")
    targets = set(zip(*arrays))
    active_rows = read_csv_strict(run / "input/model/active_ions.csv", {"atomic_number", "ion_number"})
    active = {(int(row["atomic_number"]), int(row["ion_number"])) for row in active_rows}
    level_rows = read_csv_strict(run / "input/model/levels.csv", {"atomic_number", "ion_number", "level_number"})
    expected: dict[tuple[int, int], set[tuple[int, int]]] = {}
    local_seen: dict[tuple[int, int], set[int]] = {}
    for global_idx, row in enumerate(level_rows):
        pair = (int(row["atomic_number"]), int(row["ion_number"]))
        if pair not in active or pair not in targets:
            continue
        level_idx = int(row["level_number"])
        if level_idx in local_seen.setdefault(pair, set()):
            fail("REGISTERED_LEVEL_DUPLICATE", f"pair={pair} level={level_idx}")
        local_seen[pair].add(level_idx)
        expected.setdefault(pair, set()).add((level_idx, global_idx))
    return expected


def production_saha_stages(model_root: Path) -> dict[int, set[int]]:
    rows = read_csv_strict(model_root / "input/model/ionization_energies.csv", {"atomic_number", "ion_number"})
    source: dict[int, set[int]] = {}
    for row in rows:
        source.setdefault(int(row["atomic_number"]), set()).add(int(row["ion_number"]))
    result: dict[int, set[int]] = {}
    for z in TARGET_ELEMENTS.values():
        ions = source.get(z)
        if not ions:
            fail("SAHA_PRODUCTION_SET_MISSING", str(z))
        first, last = min(ions), max(ions)
        if ions != set(range(first, last + 1)):
            fail("SAHA_PRODUCTION_SET_NONCONTIGUOUS", str(z))
        result[z] = set(range(first, last + 2))
    return result


def model_epoch(base: Path) -> Decimal:
    path = base / "input/model/config.json"
    safe_regular(path)
    value = json.loads(path.read_text(encoding="utf-8")).get("target_epoch_d")
    if value is None:
        fail("MODEL_EPOCH_MISSING", str(path))
    return decimal_token(str(value))


def geometry(base: Path) -> list[dict[str, Any]]:
    rows = read_csv_strict(base / "input/model/geometry.csv", {"shell_id", "v_inner", "v_outer"})
    result = []
    seen: set[int] = set()
    for row in rows:
        shell = int(row["shell_id"])
        if shell in seen:
            fail("DUPLICATE_SHELL", str(shell))
        seen.add(shell)
        inner = decimal_token(row["v_inner"]) / Decimal(100000)
        outer = decimal_token(row["v_outer"]) / Decimal(100000)
        if inner >= outer:
            fail("SHELL_BOUNDS_INVALID", str(shell))
        result.append({"shell_id": shell, "v_inner_km_s": inner, "v_outer_km_s": outer})
    result.sort(key=lambda item: item["shell_id"])
    if [item["shell_id"] for item in result] != list(range(len(result))):
        fail("SHELL_COVERAGE_INVALID")
    return result


def reference_stage_set(path: Path, element: str) -> set[int]:
    nstages: int | None = None
    labels: list[str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text.startswith("#NSTAGES:"):
            try: nstages = int(text.split(":", 1)[1])
            except ValueError: fail("REFERENCE_STAGE_HEADER_INVALID", str(path))
        elif text.startswith("#vel_mid"):
            labels = text.split()[1:]
        elif text and not text.startswith("#"):
            break
    expected = [f"{element}{stage}" for stage in range(nstages or 0)]
    if not nstages or labels != expected:
        fail("REFERENCE_STAGE_HEADER_INVALID", str(path))
    return set(range(nstages))


def load_ion_reference(path: Path, element: str) -> tuple[set[int], dict[Decimal, list[RawRow]]]:
    """Read the declared stages and reject any block whose rows contradict them."""
    stages = reference_stage_set(path, element)
    blocks = parse_cmfgen(path, len(stages) + 1)
    return stages, blocks


def select_available_anchors(
    shells: list[dict[str, Any]], phys: list[RawRow]
) -> tuple[list[dict[str, Any]], list[int]]:
    result: list[dict[str, Any]] = []
    missing: list[int] = []
    for shell in shells:
        matches = [row for row in phys if shell["v_inner_km_s"] <= row.velocity < shell["v_outer_km_s"]]
        if not matches:
            missing.append(shell["shell_id"])
            continue
        center = (shell["v_inner_km_s"] + shell["v_outer_km_s"]) / Decimal(2)
        ranked = sorted(matches, key=lambda row: (abs(row.velocity - center), row.velocity, row.row_index))
        selected = ranked[0]
        tied = [row for row in matches if abs(row.velocity - center) == abs(selected.velocity - center)]
        result.append({
            **shell,
            "v_center_km_s": center,
            "sample_distance_km_s": abs(selected.velocity - center),
            "tie_break": "LOWER_VELOCITY_THEN_SOURCE_ORDER" if len(tied) > 1 else "UNIQUE_NEAREST",
            "row": selected,
        })
    return result, missing


def base_identity(base: Path, reference_hashes: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    pending: list[str] = []
    identity_path = base / "input/current_source_identity.json"
    footer_path = base / "RUN_FOOTER.txt"
    binary_path = base / "input/lumina_cuda"
    exports_path = base / "input/resolved_lumina.exports"
    for path in (identity_path, footer_path, binary_path, exports_path):
        safe_regular(path)
    recorded = json.loads(identity_path.read_text(encoding="utf-8"))
    actual_binary = sha256(binary_path)
    footer = dict(
        line.split("=", 1) for line in footer_path.read_text(encoding="utf-8").splitlines() if "=" in line
    )
    if actual_binary != BASE_BINARY_SHA256 or recorded.get("binary_sha256") != actual_binary or footer.get("binary_sha256") != actual_binary:
        fail("BINARY_IDENTITY_MISMATCH")
    source_checks = []
    repository = Path(recorded.get("repository", ""))
    for member in recorded.get("source_files", []):
        source = repository / member["path"]
        ok = source.is_file() and not source.is_symlink() and sha256(source) == member["sha256"]
        source_checks.append({"path": member["path"], "sha256": member["sha256"], "verified": ok})
        if not ok:
            pending.append(f"source_identity:{member['path']}")
    return {
        "schema": SCHEMA,
        "base": str(base),
        "base_binary": {"path": str(binary_path), "sha256": actual_binary},
        "source_identity": recorded,
        "source_checks": source_checks,
        "effective_exports": {"path": str(exports_path), "sha256": sha256(exports_path)},
        "reference_files": reference_hashes,
        "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT) if CONTRACT.is_file() else None},
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "pending": pending,
    }, pending


def source_writer_contract(identity: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    pending: list[str] = []
    repo = Path(identity["source_identity"].get("repository", ""))
    cuda = repo / "src/lumina_cuda.cu"
    plasma = repo / "src/lumina_plasma.c"
    observations = {}
    for name, path, needles in (
        ("plasma_state", cuda, ["lumina_plasma_state.csv", "plasma.n_electron"]),
        ("saha_stage", cuda, ["lumina_ion_pops.csv", "ion_number_density"]),
        ("nlte_level", plasma, ["nlte_levels_iter%03d.csv", "population_required_generation"]),
    ):
        if not path.is_file() or path.is_symlink():
            observations[name] = {"status": "INPUT_PENDING", "reason": "SOURCE_MISSING"}
            pending.append(f"writer_source:{name}")
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        found = {needle: text.count(needle) for needle in needles}
        observations[name] = {"path": str(path), "sha256": sha256(path), "tokens": found}
        if any(count == 0 for count in found.values()):
            pending.append(f"writer_token:{name}")
    # The available writers have unlike naming/generation contracts.  Source
    # presence alone cannot certify that their final files share one epoch.
    pending.append("final_publication_generation_link")
    return {
        "schema": SCHEMA,
        "observations": observations,
        "authority": {"lumina_ion_pops.csv": "SAHA_STAGE", "nlte_levels_iterNNN.csv": "NLTE_LEVEL_SUM", "lumina_plasma_state.csv": "PUBLIC_NE_TE"},
        "finding": "writer existence verified where available; common final publication generation is not encoded",
        "qualification_required": ["final_nlte_relative_path", "final_population_generation", "public_T_ne_Saha_NLTE_same_generation_evidence"],
        "pending": sorted(set(pending)),
    }, sorted(set(pending))


def model_inventory(base: Path, shells: list[dict[str, Any]]) -> dict[str, Any]:
    abundance_rows = read_csv_strict(base / "input/model/abundances.csv", {"atomic_number"})
    density_rows = read_csv_strict(base / "input/model/density.csv", {"shell_id", "rho"})
    mass_rows = read_csv_strict(base / "input/model/atom_masses.csv", {"atomic_number", "mass_amu"})
    masses = {int(row["atomic_number"]): decimal_token(row["mass_amu"]) for row in mass_rows}
    density_tokens = {int(row["shell_id"]): row["rho"] for row in density_rows}
    densities = {shell: decimal_token(token) for shell, token in density_tokens.items()}
    production_totals: dict[tuple[int, int], Decimal] = {}
    comparison_totals: dict[tuple[int, int], Decimal] = {}
    for row in abundance_rows:
        z = int(row["atomic_number"])
        if z not in masses:
            fail("ATOM_MASS_MISSING", str(z))
        for key, token in row.items():
            if key == "atomic_number" or token is None:
                continue
            shell = int(key)
            if shell not in densities:
                fail("DENSITY_SHELL_MISSING", str(shell))
            x = decimal_token(token)
            production_totals[(z, shell)] = densities[shell] * x / (masses[z] * AMU_G)
            comparison_mass = Decimal(56) if z in (26, 27, 28) else masses[z]
            comparison_totals[(z, shell)] = densities[shell] * x / (comparison_mass * AMU_G)
    production_atom_totals: dict[int, Decimal] = {}
    comparison_atom_totals: dict[int, Decimal] = {}
    for (z, shell), value in production_totals.items():
        production_atom_totals[shell] = production_atom_totals.get(shell, Decimal(0)) + value
        comparison_atom_totals[shell] = comparison_atom_totals.get(shell, Decimal(0)) + comparison_totals[(z, shell)]
    unnormalized: dict[int, Decimal] = {}
    for shell in shells:
        shell_id = shell["shell_id"]
        if shell_id not in densities:
            fail("DENSITY_SHELL_MISSING", str(shell_id))
        unnormalized[shell_id] = densities[shell_id] * (
            shell["v_outer_km_s"] ** 3 - shell["v_inner_km_s"] ** 3
        )
    total_weight = sum(unnormalized.values(), Decimal(0))
    if total_weight <= 0:
        fail("MASS_WEIGHT_NORMALIZATION_UNDEFINED")
    return {
        "production_element_totals": production_totals,
        "comparison_element_totals": comparison_totals,
        "production_atom_totals": production_atom_totals,
        "comparison_atom_totals": comparison_atom_totals,
        "density_tokens": density_tokens,
        "mass_weights": {shell: value / total_weight for shell, value in unnormalized.items()},
        "mass_convention": {
            "production": "sealed_model_atom_masses.csv",
            "comparison": {"Fe": "56", "Co": "56", "Ni": "56", "other": "sealed_model_atom_masses.csv"},
        },
    }


ANCHOR_FIELDS = [
    "shell_id", "v_inner_km_s", "v_outer_km_s", "v_center_km_s", "epoch_d",
    "cmfgen_row_index_0", "phys_source_line", "phys_raw_row", "vel_mid_km_s",
    "sample_distance_km_s", "tie_break", "temperature_K_raw", "cmfgen_rho_raw",
    "cmfgen_ne_raw", "cmfgen_natom_raw",
]
MAP_FIELDS = [
    "shell_id", "element", "Z", "stage", "authority", "status", "reason", "epoch_d",
    "v_center_km_s", "vel_mid_km_s", "sample_distance_km_s",
    "temperature_registered_K", "temperature_public_K", "cmfgen_raw", "population_raw",
    "population_components", "lumina_raw", "delta_f", "percentage_point",
    "N_Z_model", "N_Z_production", "N_Z_comparison", "N_Z_production_to_comparison",
    "N_Z_production_to_comparison_status", "N_Z_dump",
    "top_bin_semantics", "top_bin_mapping",
    "reference_ne_raw", "lumina_ne_raw", "ne_ratio", "d_e", "d_e_lo", "d_e_hi",
    "rho_model_raw", "rho_model_source", "cmfgen_rho_raw", "cmfgen_rho_source", "d_rho", "d_rho_lo", "d_rho_hi",
    "n_atom_production", "n_atom_production_source", "n_atom_model", "n_atom_model_source",
    "n_atom_production_to_model", "n_atom_production_to_model_status",
    "cmfgen_natom_raw", "cmfgen_natom_source", "n_atom_ratio", "n_atom_dex", "n_atom_dex_lo", "n_atom_dex_hi",
    "z_production", "q_comparison", "zbar_cmfgen", "z_ratio", "d_Z", "d_Z_lo", "d_Z_hi",
    "mass_weight", "abs_delta_T", "nlte_expected_levels", "nlte_observed_levels",
    "nlte_level_coverage", "n_ion_total_raw", "R_SN", "condition_differences",
]


def map_row(**values: Any) -> dict[str, Any]:
    answer = {field: "" for field in MAP_FIELDS}
    answer.update(values)
    return answer


def cell_template(
    shell: int, element: str, z: int | str, stage: int | str,
    epoch: Decimal | str, refrow: RawRow, public_t: str, cmfgen_raw: str,
) -> dict[str, Any]:
    row = map_row(**{
        "shell_id": shell, "element": element, "Z": z, "stage": stage,
        "epoch_d": str(epoch), "vel_mid_km_s": refrow.tokens[0],
        "temperature_registered_K": refrow.tokens[1],
        "temperature_public_K": public_t, "cmfgen_raw": cmfgen_raw,
        "condition_differences": "density_composition;radiation_field;iterations;atomic_data_level_truncation;inner_boundary;gamma_specific_heat;diagnostic_radiation",
    })
    return row


def public_state_row(
    item: dict[str, Any], public: dict[str, str] | None, inventory: dict[str, Any],
    epoch: Decimal, *, generation_known: bool, abs_delta_t: Decimal | None = None,
) -> dict[str, Any]:
    shell = item["shell_id"]
    ref: RawRow = item["row"]
    density_token = inventory["density_tokens"].get(shell, "")
    production_atom_total = inventory["production_atom_totals"].get(shell)
    comparison_atom_total = inventory["comparison_atom_totals"].get(shell)
    common = {
        "shell_id": shell, "element": "electron", "authority": "PUBLIC_NE",
        "epoch_d": str(epoch), "v_center_km_s": str(item["v_center_km_s"]),
        "vel_mid_km_s": ref.tokens[0], "sample_distance_km_s": str(item["sample_distance_km_s"]),
        "temperature_registered_K": ref.tokens[1], "temperature_public_K": public["T_e"] if public else "",
        "reference_ne_raw": ref.tokens[3], "lumina_ne_raw": public["n_e"] if public else "",
        "rho_model_raw": density_token, "cmfgen_rho_raw": ref.tokens[2],
        "n_atom_production": str(production_atom_total) if production_atom_total is not None else "",
        "n_atom_model": str(comparison_atom_total) if comparison_atom_total is not None else "",
        "n_atom_production_to_model": (
            str(production_atom_total / comparison_atom_total)
            if production_atom_total is not None and comparison_atom_total is not None
            and production_atom_total > 0 and comparison_atom_total > 0 else ""
        ),
        "n_atom_production_to_model_status": (
            "VALID" if production_atom_total is not None and comparison_atom_total is not None
            and production_atom_total > 0 and comparison_atom_total > 0 else "UNDEFINED"
        ),
        "cmfgen_natom_raw": ref.tokens[4], "mass_weight": str(inventory["mass_weights"].get(shell, "")),
        "rho_model_source": "sealed input/model/density.csv",
        "cmfgen_rho_source": f"phys_toy06_cmfgen.txt:{ref.source_line}",
        "n_atom_production_source": "derived sum over full sealed composition; sealed atom_masses.csv",
        "n_atom_model_source": "derived sum over full sealed composition; comparison Fe/Co/Ni A=56",
        "cmfgen_natom_source": f"phys_toy06_cmfgen.txt:{ref.source_line}",
        "abs_delta_T": str(abs_delta_t) if abs_delta_t is not None else "",
        "condition_differences": "density_composition;radiation_field;iterations;atomic_data_level_truncation;inner_boundary;gamma_specific_heat;diagnostic_radiation",
    }
    if public is None or production_atom_total is None or comparison_atom_total is None or not density_token:
        return map_row(**common, status="INPUT_PENDING", reason="PUBLIC_OR_MODEL_ATOM_CELL_MISSING")
    ne = scalar_metric(public["n_e"], ref.tokens[3])
    rho = scalar_metric(density_token, ref.tokens[2])
    natom = compound_log_metric([comparison_atom_total], [ref.tokens[4]])
    zmetric = compound_log_metric([public["n_e"], ref.tokens[4]], [comparison_atom_total, ref.tokens[3]])
    try:
        z_prod = decimal_token(public["n_e"]) / production_atom_total if production_atom_total > 0 else None
        q_cmp = decimal_token(public["n_e"]) / comparison_atom_total if comparison_atom_total > 0 else None
        ref_atom = decimal_token(ref.tokens[4])
        z_c = decimal_token(ref.tokens[3]) / ref_atom if ref_atom > 0 else None
    except Failure:
        z_prod = q_cmp = z_c = None
    te_match = equal_decimal_tokens(public["T_e"], ref.tokens[1])
    status = "VALID"
    reasons: list[str] = []
    for metric in (ne, rho, natom, zmetric):
        if metric["status"] != "VALID":
            status = "INVALID_VALUE" if metric["status"] == "INVALID_VALUE" else "UNDEFINED"
            reasons.append(metric["reason"])
    if te_match is None:
        status = "INVALID_VALUE"; reasons.append("INVALID_PUBLIC_T")
    elif te_match is False:
        if status == "VALID": status = "CONDITION_MISMATCH"
        reasons.append("T_NOT_TWIN")
    if not generation_known:
        if status not in ("INVALID_VALUE", "UNDEFINED"): status = "INPUT_PENDING"
        reasons.append("PUBLICATION_GENERATION_UNKNOWN")
    return map_row(
        **common, status=status, reason=";".join(dict.fromkeys(reasons)),
        cmfgen_raw=ref.tokens[3], lumina_raw=public["n_e"],
        ne_ratio=ne["ratio"] or "", d_e=ne["dex"] if ne["dex"] is not None else "",
        d_e_lo=ne["dex_lo"] if ne["dex_lo"] is not None else "", d_e_hi=ne["dex_hi"] if ne["dex_hi"] is not None else "",
        d_rho=rho["dex"] if rho["dex"] is not None else "", d_rho_lo=rho["dex_lo"] if rho["dex_lo"] is not None else "",
        d_rho_hi=rho["dex_hi"] if rho["dex_hi"] is not None else "",
        n_atom_ratio=natom["ratio"] or "", n_atom_dex=natom["dex"] if natom["dex"] is not None else "",
        n_atom_dex_lo=natom["dex_lo"] if natom["dex_lo"] is not None else "",
        n_atom_dex_hi=natom["dex_hi"] if natom["dex_hi"] is not None else "",
        z_production=str(z_prod) if z_prod is not None else "",
        q_comparison=str(q_cmp) if q_cmp is not None else "", zbar_cmfgen=str(z_c) if z_c is not None else "",
        z_ratio=zmetric["ratio"] or "", d_Z=zmetric["dex"] if zmetric["dex"] is not None else "",
        d_Z_lo=zmetric["dex_lo"] if zmetric["dex_lo"] is not None else "",
        d_Z_hi=zmetric["dex_hi"] if zmetric["dex_hi"] is not None else "",
    )


def qualify_observation(fields: dict[str, Any], te_match: bool | None, generation_known: bool = True) -> dict[str, Any]:
    answer = dict(fields)
    reasons = [reason for reason in str(answer.get("reason", "")).split(";") if reason]
    if te_match is None:
        reasons.append("INVALID_PUBLIC_T")
        if answer.get("status") == "VALID": answer["status"] = "INVALID_VALUE"
    elif te_match is False:
        reasons.append("T_NOT_TWIN")
        if answer.get("status") == "VALID": answer["status"] = "CONDITION_MISMATCH"
    if not generation_known:
        reasons.append("PUBLICATION_GENERATION_UNKNOWN")
        if answer.get("status") not in ("INVALID_VALUE", "NOT_SOLVED"): answer["status"] = "INPUT_PENDING"
    answer["reason"] = ";".join(dict.fromkeys(reasons))
    return answer


def _absolute_interval(lo: Decimal, hi: Decimal) -> tuple[Decimal, Decimal]:
    if lo <= 0 <= hi:
        return Decimal(0), max(abs(lo), abs(hi))
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def aggregate_ez(rows: list[dict[str, Any]], shell_count: int) -> dict[str, Any]:
    cells = [row for row in rows if row.get("authority") == "PUBLIC_NE" and row.get("d_Z") not in (None, "")]
    by_shell = {int(row["shell_id"]): row for row in cells}
    if len(by_shell) != shell_count:
        return {"M_Z": None, "M_Z_interval": None, "S_Z": None, "S_Z_interval": None, "cell_count": len(by_shell), "sign_transitions": None}
    m_value = m_lo = m_hi = s_value = s_lo = s_hi = Decimal(0)
    signs: list[int] = []
    for shell in range(shell_count):
        row = by_shell[shell]
        weight = Decimal(str(row["mass_weight"])); value = Decimal(str(row["d_Z"]))
        lo = Decimal(str(row["d_Z_lo"])); hi = Decimal(str(row["d_Z_hi"]))
        abs_lo, abs_hi = _absolute_interval(lo, hi)
        m_value += weight * abs(value); m_lo += weight * abs_lo; m_hi += weight * abs_hi
        s_value += weight * value; s_lo += weight * lo; s_hi += weight * hi
        signs.append(1 if value > 0 else (-1 if value < 0 else 0))
    transitions = sum(1 for left, right in zip(signs, signs[1:]) if left and right and left != right)
    return {
        "M_Z": str(m_value), "M_Z_interval": [str(m_lo), str(m_hi)],
        "S_Z": str(s_value), "S_Z_interval": [str(s_lo), str(s_hi)],
        "cell_count": len(by_shell), "sign_transitions": transitions,
    }


def paired_ez_structure(
    base_rows: list[dict[str, Any]], twin_rows: list[dict[str, Any]], shell_count: int
) -> dict[str, Any]:
    base = {int(row["shell_id"]): row for row in base_rows if row.get("authority") == "PUBLIC_NE" and row.get("d_Z") not in (None, "")}
    twin = {int(row["shell_id"]): row for row in twin_rows if row.get("authority") == "PUBLIC_NE" and row.get("d_Z") not in (None, "")}
    census = {"improved": 0, "worsened": 0, "representation_unresolved": 0}
    details: list[dict[str, Any]] = []
    for shell in range(shell_count):
        if shell not in base or shell not in twin:
            details.append({"shell_id": shell, "status": "INPUT_PENDING"})
            continue
        b_lo, b_hi = _absolute_interval(Decimal(str(base[shell]["d_Z_lo"])), Decimal(str(base[shell]["d_Z_hi"])))
        t_lo, t_hi = _absolute_interval(Decimal(str(twin[shell]["d_Z_lo"])), Decimal(str(twin[shell]["d_Z_hi"])))
        improvement = abs(Decimal(str(base[shell]["d_Z"]))) - abs(Decimal(str(twin[shell]["d_Z"])))
        i_lo, i_hi = b_lo - t_hi, b_hi - t_lo
        label = "improved" if i_lo > 0 else ("worsened" if i_hi < 0 else "representation_unresolved")
        census[label] += 1
        details.append({
            "shell_id": shell, "I_s": str(improvement), "I_s_interval": [str(i_lo), str(i_hi)],
            "classification": label, "abs_delta_T": base[shell].get("abs_delta_T", ""),
            "mass_weight": base[shell].get("mass_weight", ""),
        })
    strata: list[dict[str, Any]] = []
    complete = [item for item in details if item.get("status") != "INPUT_PENDING"]
    ordered = sorted(complete, key=lambda item: (Decimal(str(item["abs_delta_T"])), item["shell_id"]))
    for start in range(0, len(ordered), 10):
        members = ordered[start:start + 10]
        if not members: continue
        twin_subset = [twin[item["shell_id"]] for item in members]
        m = s = Decimal(0)
        counts = {"improved": 0, "worsened": 0, "representation_unresolved": 0}
        for item, row in zip(members, twin_subset):
            weight = Decimal(str(row["mass_weight"])); dz = Decimal(str(row["d_Z"]))
            m += weight * abs(dz); s += weight * dz; counts[item["classification"]] += 1
        strata.append({
            "rank_start": start, "shell_ids": [item["shell_id"] for item in members],
            "weight_sum": str(sum((Decimal(str(row["mass_weight"])) for row in twin_subset), Decimal(0))),
            "M_Z": str(m), "S_Z": str(s), "paired_sign_counts": counts,
        })
    return {"paired_sign_counts": census, "cells": details, "abs_delta_T_strata": strata}


def prepare(args: argparse.Namespace) -> int:
    base = require_real_path(args.base)
    reference = Path(args.reference_dir)
    if not base.is_dir():
        fail("BASE_NOT_SAFE_ABSOLUTE", str(base))
    if Path(args.evidence).exists() or Path(args.evidence).is_symlink():
        fail("EVIDENCE_NOT_NEW_SAFE_ABSOLUTE", args.evidence)
    evidence = require_real_path(args.evidence, base=base, must_exist=False)
    _create_output_tree(evidence, base, ("inputs", "offline", "run", "result"))
    validate_output_parents(evidence, base)

    pending: list[str] = []
    anchor_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    epoch = model_epoch(base)
    shells = geometry(base)
    anchors: list[dict[str, Any]] = []
    ions: dict[str, list[RawRow]] = {}
    phys_path = reference / REFERENCE_NAMES[0]
    phys: list[RawRow] = []
    if not phys_path.is_file() or phys_path.is_symlink():
        pending.append(f"reference:{REFERENCE_NAMES[0]}")
    else:
        phys_blocks = parse_cmfgen(phys_path, 5)
        hashes[REFERENCE_NAMES[0]] = sha256(phys_path)
        if epoch not in phys_blocks:
            pending.append(f"reference_epoch:{REFERENCE_NAMES[0]}")
        else:
            phys = phys_blocks[epoch]
            anchors, missing_shells = select_available_anchors(shells, phys)
            pending.extend(f"reference_anchor:shell={shell}" for shell in missing_shells)
    phys_keys = {(item["row"].epoch, item["row"].velocity) for item in anchors}
    for element in TARGET_ELEMENTS:
        name = f"ionfrac_{element}_toy06_cmfgen.txt"
        path = reference / name
        if not path.is_file() or path.is_symlink():
            pending.append(f"reference:{name}")
            continue
        try:
            _stages, blocks = load_ion_reference(path, element)
        except Failure as exc:
            pending.append(f"reference_invalid:{name}:{exc.reason}")
            continue
        hashes[name] = sha256(path)
        if epoch not in blocks:
            pending.append(f"reference_epoch:{name}")
            continue
        rows = blocks[epoch]
        available = {(row.epoch, row.velocity) for row in rows}
        missing_selected = sorted(str(velocity) for block_epoch, velocity in phys_keys if block_epoch == epoch and (block_epoch, velocity) not in available)
        if missing_selected:
            pending.append(f"reference_key_join:{name}:missing={','.join(missing_selected)}")
        ions[element] = rows
    identity, identity_pending = base_identity(base, hashes)
    pending.extend(identity_pending)
    writer, writer_pending = source_writer_contract(identity)
    pending.extend(writer_pending)
    pending.append("actual_fixed_te_loader_cpu_consumption")

    for item in anchors:
        row: RawRow = item["row"]
        for label, token in zip(("T", "rho", "ne", "natom"), row.tokens[1:5]):
            try: decimal_token(token)
            except Failure as exc: pending.append(f"reference_phys_value:shell={item['shell_id']}:{label}:{exc.reason}")
        anchor_rows.append({
            "shell_id": item["shell_id"], "v_inner_km_s": str(item["v_inner_km_s"]),
            "v_outer_km_s": str(item["v_outer_km_s"]), "v_center_km_s": str(item["v_center_km_s"]),
            "epoch_d": str(row.epoch),
            "cmfgen_row_index_0": row.row_index, "phys_source_line": row.source_line,
            "phys_raw_row": " ".join(row.tokens),
            "vel_mid_km_s": row.tokens[0], "sample_distance_km_s": str(item["sample_distance_km_s"]),
            "tie_break": item["tie_break"], "temperature_K_raw": row.tokens[1],
            "cmfgen_rho_raw": row.tokens[2], "cmfgen_ne_raw": row.tokens[3], "cmfgen_natom_raw": row.tokens[4],
        })
    csv_write(evidence / "inputs/anchors.csv", ANCHOR_FIELDS, anchor_rows)
    profile = evidence / "inputs/fixed_te_profile.txt"
    text_write(profile, "".join(f"{item['shell_id']} {item['row'].tokens[1]}\n" for item in anchors))

    plasma_path = base / "work/lumina_plasma_state.csv"
    plasma_by_shell: dict[int, dict[str, str]] = {}
    if plasma_path.is_file() and not plasma_path.is_symlink():
        for row in read_csv_strict(plasma_path, {"shell_id", "T_e", "n_e"}):
            shell = int(row["shell_id"])
            if shell in plasma_by_shell:
                fail("DUPLICATE_SHELL", f"plasma:{shell}")
            plasma_by_shell[shell] = row
    else:
        pending.append("base:lumina_plasma_state.csv")
    missing_plasma_shells = {item["shell_id"] for item in anchors} - set(plasma_by_shell)
    pending.extend(f"base:lumina_plasma_state.csv:shell={shell}" for shell in sorted(missing_plasma_shells))
    inventory = model_inventory(base, shells)
    for item in anchors:
        shell = item["shell_id"]
        public = plasma_by_shell.get(shell)
        abs_delta_t = None
        if public is not None:
            try: abs_delta_t = abs(decimal_token(item["row"].tokens[1]) - decimal_token(public["T_e"]))
            except Failure: pass
        base_rows.append(public_state_row(item, public, inventory, epoch, generation_known=True, abs_delta_t=abs_delta_t))
    csv_write(evidence / "offline/base_map.csv", MAP_FIELDS, base_rows)
    json_write(evidence / "inputs/identity.json", identity)
    json_write(evidence / "offline/writer_contract.json", writer)
    base_ez = aggregate_ez(base_rows, len(shells))
    top_semantics = {element: {"value": "INPUT_PENDING", "source": None} for element in TARGET_ELEMENTS}
    pending.extend(f"top_bin_semantics:{element}" for element in TARGET_ELEMENTS)
    expectation = {
        "schema": SCHEMA,
        "pending": sorted(set(pending)),
        "epoch_d": str(epoch), "shell_count": len(shells), "reference_dir": str(reference.resolve()) if reference.exists() else str(reference),
        "identity_sha256": sha256(evidence / "inputs/identity.json"), "anchors_sha256": sha256(evidence / "inputs/anchors.csv"),
        "fixed_te_profile_sha256": sha256(profile), "writer_contract_sha256": sha256(evidence / "offline/writer_contract.json"),
        "rules": {"decrease": "upper(M_Z_twin) < lower(M_Z_base)", "increase": "lower(M_Z_twin) > upper(M_Z_base)", "otherwise": "REPRESENTATION_UNRESOLVED"},
        "base_e_z": base_ez,
        "base_self_comparison": paired_ez_structure(base_rows, base_rows, len(shells)),
        "reference_top_bin_semantics": top_semantics,
        "mass_convention": inventory["mass_convention"],
        "required_outputs": ["lumina_plasma_state.csv", "lumina_ion_pops.csv", "nlte_levels_iterNNN.csv"],
        "registered_outer_iterations": 2,
        "saha_production_stages": {str(z): sorted(stages) for z, stages in production_saha_stages(base).items()},
        "nlte_production_ions": {str(z): sorted(ion for zz, ion in effective_nlte_levels(base) if zz == z) for z in TARGET_ELEMENTS.values()},
    }
    json_write(evidence / "offline/expectations.json", expectation)
    input_size = sum(path.stat().st_size for path in (base / "input").rglob("*") if path.is_file() and not path.is_symlink())
    work_size = sum(path.stat().st_size for path in (base / "work").rglob("*") if path.is_file() and not path.is_symlink())
    level_count = len(read_csv_strict(base / "input/model/levels.csv", {"atomic_number", "ion_number", "level_number"}))
    active_count = len(read_csv_strict(base / "input/model/active_ions.csv", {"atomic_number", "ion_number"}))
    estimated = len(shells) * active_count * 48 * 3 + len(shells) * level_count * 100 * 3
    resource = {
        "schema": SCHEMA, "limits": {"gpu_hour": 8, "allocated_cpu_hour": 128, "offline_cpu_hour": 1, "storage_bytes": 16 * 1024**3},
        "measured": {"base_input_bytes": input_size, "base_work_bytes": work_size, "shells": len(shells), "active_ions": active_count, "levels": level_count},
        "dump_upper_bound_formula_bytes": estimated,
        "future_storage_upper_bounds_bytes": {"run_logs": None, "result_and_summary": None, "reproduction_and_audit": None},
        "pending": ["new_T_runtime", "peak_memory", "hardware_acceptance", "offline_elapsed_and_memory_measurement", "free_space_at_new_root", "run_logs_storage_upper_bound", "result_and_summary_storage_upper_bound", "reproduction_and_audit_storage_upper_bound"],
    }
    json_write(evidence / "offline/resource_card.json", resource)
    print(f"CMFGEN_TWIN_MAP_PREPARE shells={len(anchors)}/{len(shells)} pending={len(expectation['pending'])}")
    for reason in expectation["pending"]:
        print(f"INPUT_PENDING reason={reason}")
    return 0


def _fraction_fields(numerator_token: str, denominator: Decimal, reference_token: str) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "population_raw": numerator_token, "lumina_raw": numerator_token, "cmfgen_raw": reference_token,
        "delta_f": "", "percentage_point": "",
        "status": "VALID", "reason": "",
    }
    try:
        numerator = decimal_token(numerator_token)
        reference = decimal_token(reference_token)
    except Failure as exc:
        answer.update(status="INVALID_VALUE", reason=exc.reason)
        return answer
    if denominator == 0:
        answer.update(status="UNDEFINED", reason="ELEMENT_DENOMINATOR_ZERO")
        return answer
    fraction = numerator / denominator
    delta = fraction - reference
    answer.update(lumina_raw=str(fraction), delta_f=str(delta), percentage_point=str(Decimal(100) * delta))
    return answer


def writer_generation_record(writer: dict[str, Any]) -> dict[str, Any] | None:
    generations = writer.get("generations")
    if (
        writer.get("public_T_ne_Saha_NLTE_same_generation_evidence") is not True
        or not isinstance(generations, dict)
        or set(generations) != PUBLICATION_FIELDS
    ):
        return None
    def exact_integer(value: Any) -> int:
        if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (int, str)):
            raise ValueError
        parsed = int(value)
        if str(parsed) != str(value):
            raise ValueError
        return parsed
    try:
        final = exact_integer(writer["final_population_generation"])
        parsed = {key: exact_integer(value) for key, value in generations.items()}
    except (KeyError, TypeError, ValueError):
        return None
    if any(value != final for value in parsed.values()):
        return {
            "final_generation": final,
            "generations": parsed,
            "visibility": "CONTRADICTORY_GENERATIONS",
            "reason": "PUBLICATION_GENERATION_CONTRADICTION",
        }
    return {"final_generation": final, "generations": parsed, "visibility": "PUBLIC_FINAL", "reason": ""}


def _same_number(actual: Any, expected: Decimal | float) -> bool:
    if actual in (None, ""):
        return False
    try:
        if isinstance(expected, Decimal):
            return Decimal(str(actual)) == expected
        return float(actual) == expected
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return False


def verify_map_metrics(rows: Any) -> None:
    """Independently recompute every present metric field from recorded raw values."""
    if not isinstance(rows, list) or not rows:
        fail("METRIC_EVIDENCE_MISSING")
    checked = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("authority") not in ("PUBLIC_NE", "SAHA_STAGE", "NLTE_LEVEL_SUM"):
            fail("METRIC_ROW_SCHEMA")
        if row["authority"] == "PUBLIC_NE":
            value_token = row.get("lumina_ne_raw", ""); reference_token = row.get("reference_ne_raw", "")
            if not value_token or not reference_token:
                if row.get("status") in ("VALID", "CONDITION_MISMATCH"):
                    fail("METRIC_CELL_EVIDENCE_MISSING", f"shell={row.get('shell_id')}")
                continue
            try:
                model_atom = Decimal(str(row.get("n_atom_model", "")))
                production_atom = Decimal(str(row.get("n_atom_production", "")))
            except InvalidOperation:
                if row.get("status") in ("VALID", "CONDITION_MISMATCH"): fail("MODEL_ATOM_EVIDENCE_MISSING")
                continue
            metrics = [
                (scalar_metric(value_token, reference_token), ("ne_ratio", "d_e", "d_e_lo", "d_e_hi")),
                (scalar_metric(row.get("rho_model_raw", ""), row.get("cmfgen_rho_raw", "")), (None, "d_rho", "d_rho_lo", "d_rho_hi")),
                (compound_log_metric([model_atom], [row.get("cmfgen_natom_raw", "")]), ("n_atom_ratio", "n_atom_dex", "n_atom_dex_lo", "n_atom_dex_hi")),
                (compound_log_metric([value_token, row.get("cmfgen_natom_raw", "")], [model_atom, reference_token]), ("z_ratio", "d_Z", "d_Z_lo", "d_Z_hi")),
            ]
            for metric, fields in metrics:
                if metric["status"] != "VALID":
                    continue
                expected_values = (Decimal(str(metric["ratio"])), metric["dex"], metric["dex_lo"], metric["dex_hi"])
                for field, expected in zip(fields, expected_values):
                    if field is not None and not _same_number(row.get(field), expected):
                        fail("METRIC_FIELD_MISMATCH", f"field={field} shell={row.get('shell_id')}")
            cmf_atom = Decimal(str(row["cmfgen_natom_raw"]))
            if production_atom > 0 and not _same_number(row.get("z_production"), Decimal(value_token) / production_atom): fail("Z_PRODUCTION_FIELD_MISMATCH", f"shell={row.get('shell_id')}")
            if model_atom > 0 and not _same_number(row.get("q_comparison"), Decimal(value_token) / model_atom): fail("Q_COMPARISON_FIELD_MISMATCH", f"shell={row.get('shell_id')}")
            if production_atom > 0 and model_atom > 0 and not _same_number(row.get("n_atom_production_to_model"), production_atom / model_atom): fail("ATOM_TOTAL_RATIO_MISMATCH", f"shell={row.get('shell_id')}")
            if row.get("n_atom_production_to_model_status") != ("VALID" if production_atom > 0 and model_atom > 0 else "UNDEFINED"):
                fail("ATOM_TOTAL_RATIO_STATUS_MISMATCH", f"shell={row.get('shell_id')}")
            if cmf_atom > 0 and not _same_number(row.get("zbar_cmfgen"), Decimal(reference_token) / cmf_atom): fail("ZBAR_FIELD_MISMATCH", f"side=cmfgen shell={row.get('shell_id')}")
            checked += 1
        else:
            reference_token = row.get("cmfgen_raw", "")
            numerator_token, denominator_token = row.get("population_raw", ""), row.get("N_Z_production", "")
            if row.get("N_Z_model") != denominator_token:
                fail("N_Z_MODEL_ALIAS_MISMATCH", f"shell={row.get('shell_id')} Z={row.get('Z')}")
            try:
                production = Decimal(str(denominator_token))
                comparison = Decimal(str(row.get("N_Z_comparison", "")))
            except InvalidOperation:
                production = comparison = Decimal(-1)
            expected_ratio = production / comparison if production > 0 and comparison > 0 else None
            if expected_ratio is not None and not _same_number(row.get("N_Z_production_to_comparison"), expected_ratio):
                fail("ELEMENT_TOTAL_RATIO_MISMATCH", f"shell={row.get('shell_id')} Z={row.get('Z')}")
            if (denominator_token or row.get("N_Z_comparison")) and row.get("N_Z_production_to_comparison_status") != ("VALID" if expected_ratio is not None else "UNDEFINED"):
                fail("ELEMENT_TOTAL_RATIO_STATUS_MISMATCH", f"shell={row.get('shell_id')} Z={row.get('Z')}")
            if not numerator_token or not denominator_token or not reference_token:
                if row.get("status") in ("VALID", "CONDITION_MISMATCH"):
                    fail("METRIC_CELL_EVIDENCE_MISSING", f"shell={row.get('shell_id')} Z={row.get('Z')}")
                continue
            try:
                numerator, denominator, reference = Decimal(str(numerator_token)), Decimal(str(denominator_token)), Decimal(str(reference_token))
            except InvalidOperation:
                continue
            if not all(value.is_finite() and value >= 0 for value in (numerator, denominator, reference)):
                continue
            if denominator == 0:
                checked += 1; continue
            fraction = numerator / denominator; delta = fraction - reference
            for field, expected in (("lumina_raw", fraction), ("delta_f", delta), ("percentage_point", Decimal(100) * delta)):
                if not _same_number(row.get(field), expected): fail("FRACTION_FIELD_MISMATCH", f"field={field} shell={row.get('shell_id')} Z={row.get('Z')} stage={row.get('stage')}")
            if row["authority"] == "NLTE_LEVEL_SUM" and row.get("R_SN") not in (None, ""):
                nion = [token for token in str(row.get("n_ion_total_raw", "")).split(";") if token]
                if not nion or any(Decimal(token) != Decimal(nion[0]) for token in nion):
                    fail("R_SN_SOURCE_INVALID", f"shell={row.get('shell_id')} Z={row.get('Z')}")
                expected_rsn = numerator / Decimal(nion[0])
                if not _same_number(row.get("R_SN"), expected_rsn):
                    fail("R_SN_MISMATCH", f"shell={row.get('shell_id')} Z={row.get('Z')} stage={row.get('stage')}")
            checked += 1
    if checked == 0:
        fail("METRIC_EVIDENCE_MISSING")


def _independent_token_interval(text: str) -> tuple[Decimal, Decimal]:
    value = Decimal(text)
    if not value.is_finite() or value <= 0:
        fail("LOG_DOMAIN_NONPOSITIVE", text)
    half = Decimal(1).scaleb(value.as_tuple().exponent) / 2
    return value - half, value + half


def exports_delta_is_registered(base: Path, run: Path) -> bool:
    old = read_exports(base / "input/resolved_lumina.exports")
    new = read_exports(run / "input/resolved_lumina.exports")
    base_s, run_s = str(base), str(run)
    expected = {key: value.replace(base_s, run_s) for key, value in old.items()}
    expected["LUMINA_FIXED_TE_PROFILE"] = f"{run_s}/input/fixed_te_profile.txt"
    expected.pop("LUMINA_RADEQ_DIAG_TE_K", None)
    expected.update(LUMINA_ION_POP_DUMP="1", LUMINA_ION_POP_DUMP_ITER="1", LUMINA_NLTE_LEVEL_DUMP="1")
    return new == expected


def bounded_integer(value: Any, name: str, upper: int) -> int:
    if isinstance(value, bool):
        fail("RESOURCE_VALUE_INVALID", name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail("RESOURCE_VALUE_INVALID", name)
    if str(parsed) != str(value) or parsed < 0 or parsed > upper:
        fail("RESOURCE_VALUE_OUT_OF_RANGE", name)
    return parsed


def _tree_manifest(root: Path) -> str:
    lines: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        rel = path.relative_to(root)
        if path.is_symlink():
            lines.append(f"SYMLINK {os.readlink(path)}  ./{rel}\n")
        elif path.is_file():
            lines.append(f"{sha256(path)}  ./{rel}\n")
    return "".join(lines)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()) if root.is_dir() else 0


def analyze(args: argparse.Namespace) -> int:
    evidence = require_real_path(args.evidence)
    contract = Path(args.contract)
    run_arg = Path(args.run)
    identity_path = evidence / "inputs/identity.json"
    anchors_path = evidence / "inputs/anchors.csv"
    writer_path = evidence / "offline/writer_contract.json"
    resource_path = evidence / "offline/resource_card.json"
    profile_path = evidence / "inputs/fixed_te_profile.txt"
    for path in (contract, identity_path, anchors_path, writer_path, resource_path, profile_path):
        safe_regular(path)
    if contract.resolve() != (evidence / "offline/expectations.json").resolve():
        fail("EXPECTATIONS_PATH_MISMATCH")
    expectations = json.loads(contract.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    writer = json.loads(writer_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    base = require_real_path(identity["base"])
    validate_output_parents(evidence, base)
    run = require_real_path(args.run, base=base, must_exist=run_arg.exists())
    pending = list(expectations.get("pending", []))
    pending.extend(writer.get("pending", []))
    if expectations.get("identity_sha256") != sha256(identity_path): fail("SEALED_IDENTITY_CHANGED")
    if expectations.get("anchors_sha256") != sha256(anchors_path): fail("SEALED_ANCHORS_CHANGED")
    if expectations.get("fixed_te_profile_sha256") != sha256(profile_path): fail("SEALED_TE_PROFILE_CHANGED")
    if expectations.get("writer_contract_sha256") != sha256(writer_path): fail("SEALED_WRITER_CONTRACT_CHANGED")
    run_exists = run.is_dir()
    if not run_exists: pending.append("run_root_missing")

    files: list[dict[str, Any]] = []
    required = ["work/lumina_plasma_state.csv", "work/lumina_ion_pops.csv"]
    final_nlte_rel = writer.get("final_nlte_relative_path")
    if not final_nlte_rel:
        pending.append("writer_contract:final_nlte_relative_path")
    else:
        required.append(final_nlte_rel)
    for rel in required:
        path = run / rel
        if path.is_file() and not path.is_symlink():
            files.append({"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size})
        else:
            pending.append(f"run_output:{rel}")
    stage_path = run / "STAGE_IDENTITY.txt"
    footer_path = run / "RUN_FOOTER.txt"
    before_path = evidence / "inputs/base_before.sha256"
    stage_identity: dict[str, str] = {}
    footer: dict[str, str] = {}
    for path, target, reason in ((stage_path, stage_identity, "stage_identity"), (footer_path, footer, "run_footer")):
        if path.is_file() and not path.is_symlink():
            target.update(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
        else:
            pending.append(reason)
    before_manifest = before_path.read_text(encoding="utf-8") if before_path.is_file() and not before_path.is_symlink() else ""
    if not before_manifest: pending.append("base_before_manifest")

    reference = Path(expectations["reference_dir"])
    epoch = Decimal(expectations["epoch_d"])
    shells = geometry(run) if run_exists else []
    expected_anchor_rows = read_csv_strict(anchors_path, set(ANCHOR_FIELDS))
    anchors: list[dict[str, Any]] = []
    current_hashes: dict[str, str] = {}
    ions: dict[str, list[RawRow]] = {}
    phys_epoch_rows: list[RawRow] = []
    phys_path = reference / REFERENCE_NAMES[0]
    if phys_path.is_file() and not phys_path.is_symlink():
        blocks = parse_cmfgen(phys_path, 5); current_hashes[REFERENCE_NAMES[0]] = sha256(phys_path)
        if epoch in blocks and shells:
            phys_epoch_rows = blocks[epoch]
            anchors, missing_shells = select_available_anchors(shells, phys_epoch_rows)
            pending.extend(f"reference_anchor:shell={shell}" for shell in missing_shells)
        else: pending.append("reference_phys_epoch_or_geometry")
    else: pending.append(f"reference:{REFERENCE_NAMES[0]}")
    phys_keys = {(item["row"].epoch, item["row"].velocity) for item in anchors}
    for element in TARGET_ELEMENTS:
        name = f"ionfrac_{element}_toy06_cmfgen.txt"; path = reference / name
        if not path.is_file() or path.is_symlink(): pending.append(f"reference:{name}"); continue
        try:
            _stages, blocks = load_ion_reference(path, element)
        except Failure as exc:
            pending.append(f"reference_invalid:{name}:{exc.reason}"); continue
        current_hashes[name] = sha256(path)
        rows = blocks.get(epoch)
        if rows is None:
            pending.append(f"reference_epoch:{name}"); continue
        available = {(row.epoch, row.velocity) for row in rows}
        missing_selected = sorted(str(velocity) for block_epoch, velocity in phys_keys if (block_epoch, velocity) not in available)
        if missing_selected:
            pending.append(f"reference_key_join:{name}:missing={','.join(missing_selected)}")
        ions[element] = rows
    anchor_match = len(anchors) == len(expected_anchor_rows) and all(
        item["row"].row_index == int(saved["cmfgen_row_index_0"])
        and item["row"].velocity == Decimal(saved["vel_mid_km_s"])
        and item["v_center_km_s"] == Decimal(saved["v_center_km_s"])
        and item["sample_distance_km_s"] == Decimal(saved["sample_distance_km_s"])
        and item["tie_break"] == saved["tie_break"]
        for item, saved in zip(anchors, expected_anchor_rows)
    )
    profile_lines = profile_path.read_text(encoding="utf-8").splitlines()
    wanted_profile = [f"{item['shell_id']} {item['row'].tokens[1]}" for item in anchors]

    plasma: dict[int, dict[str, str]] = {}
    plasma_path = run / "work/lumina_plasma_state.csv"
    if plasma_path.is_file() and not plasma_path.is_symlink():
        for row in read_csv_strict(plasma_path, {"shell_id", "T_e", "n_e"}):
            shell = int(row["shell_id"])
            if shell in plasma: fail("DUPLICATE_SHELL", f"plasma:{shell}")
            plasma[shell] = row

    pops: dict[tuple[int, int, int], str] = {}
    pop_path = run / "work/lumina_ion_pops.csv"
    if pop_path.is_file() and not pop_path.is_symlink():
        for row in read_csv_strict(pop_path, {"shell_id", "Z", "stage", "n_ion"}):
            key = (int(row["shell_id"]), int(row["Z"]), int(row["stage"]))
            if key in pops: fail("DUPLICATE_STAGE_POPULATION", str(key))
            pops[key] = row["n_ion"]
    inventory = model_inventory(run, shells) if run_exists else {
        "production_element_totals": {}, "comparison_element_totals": {},
        "production_atom_totals": {}, "comparison_atom_totals": {},
        "density_tokens": {}, "mass_weights": {}, "mass_convention": {},
    }
    totals = inventory["production_element_totals"]
    comparison_totals = inventory["comparison_element_totals"]
    if expectations.get("mass_convention") != inventory["mass_convention"]:
        pending.append("MASS_CONVENTION_CHANGED")
    registered_saha_sets = {int(z): set(stages) for z, stages in expectations.get("saha_production_stages", {}).items()}
    registered_nlte_sets = {int(z): set(stages) for z, stages in expectations.get("nlte_production_ions", {}).items()}
    saha_sets = production_saha_stages(run) if run_exists else registered_saha_sets
    nlte_sets = {z: set(ions_for_z) for z, ions_for_z in registered_nlte_sets.items()}
    dump_totals: dict[tuple[int, int], Decimal] = {}
    dump_intervals: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    invalid_pop_keys: set[tuple[int, int, int]] = set()
    production_population_failures: list[dict[str, Any]] = []
    for (shell, z, stage_index), token in pops.items():
        try:
            value = decimal_token(token)
        except Failure as exc:
            invalid_pop_keys.add((shell, z, stage_index))
            production_population_failures.append({
                "shell_id": shell, "Z": z, "stage": stage_index,
                "raw_value": token, "reason": exc.reason,
                "aggregate_disposition": "EXCLUDED_FROM_N_Z_DUMP_BECAUSE_INVALID",
            })
            continue
        dump_totals[(z, shell)] = dump_totals.get((z, shell), Decimal(0)) + value
    for (shell, z, _stage), token in pops.items():
        try:
            lo, hi = token_interval(token)
        except Failure:
            continue
        old_lo, old_hi = dump_intervals.get((z, shell), (Decimal(0), Decimal(0)))
        dump_intervals[(z, shell)] = old_lo + lo, old_hi + hi
    contaminated_saha_totals = {(z, shell) for shell, z, _stage in invalid_pop_keys}
    for key in contaminated_saha_totals:
        # A partial sum is not the observed production total.
        dump_totals.pop(key, None)
        dump_intervals.pop(key, None)
    denominator_mismatch: set[tuple[int, int]] = set()
    for key, value in totals.items():
        if key[0] in TARGET_ELEMENTS.values() and key not in dump_totals:
            continue
        if key[0] in TARGET_ELEMENTS.values():
            if any(z == key[0] and shell == key[1] for shell, z, _stage in invalid_pop_keys):
                continue
            lo, hi = dump_intervals[key]
            if not (lo <= value <= hi): denominator_mismatch.add(key)

    nlte_path = run / final_nlte_rel if final_nlte_rel else run / "work/INPUT_PENDING"
    nlte_rows = read_csv_strict(nlte_path, {"Z", "ion", "shell", "level_idx", "global_idx", "n_pop", "T_e", "n_ion_total", "population_generation"}) if nlte_path.is_file() and not nlte_path.is_symlink() else []
    generation_record = writer_generation_record(writer)
    generation_known = generation_record is not None and generation_record["visibility"] == "PUBLIC_FINAL"
    final_generation = generation_record["final_generation"] if generation_known else None
    if generation_record is not None and not generation_known:
        pending.append(generation_record["reason"])
    nlte_values: dict[tuple[int, int, int], list[str]] = {}
    nlte_ion_totals: dict[tuple[int, int, int], list[str]] = {}
    nlte_cell_failures: dict[tuple[int, int, int], list[str]] = {}
    seen_levels: set[tuple[int, int, int, int]] = set()
    observed_levels: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for row in nlte_rows:
        try: row_generation = bounded_integer(row["population_generation"], "population_generation", 10**9)
        except Failure: row_generation = -1
        key = (int(row["shell"]), int(row["Z"]), int(row["ion"]))
        if final_generation is None or row_generation != final_generation:
            pending.append(f"WRITER_GENERATION_MISMATCH:{key}")
            nlte_cell_failures.setdefault(key, []).append("WRITER_GENERATION_MISMATCH")
        level_key = (*key, int(row["level_idx"]))
        if level_key in seen_levels: fail("NLTE_LEVEL_DUPLICATE", str(level_key))
        seen_levels.add(level_key)
        observed_levels.setdefault(key, set()).add((int(row["level_idx"]), int(row["global_idx"])))
        if key[0] not in plasma or equal_decimal_tokens(row["T_e"], plasma[key[0]]["T_e"]) is not True:
            pending.append(f"NLTE_T_PUBLICATION_MISMATCH:{key}")
            nlte_cell_failures.setdefault(key, []).append("NLTE_T_PUBLICATION_MISMATCH")
        saha_token = pops.get(key)
        if saha_token is None or equal_decimal_tokens(row["n_ion_total"], saha_token) is not True:
            pending.append(f"NLTE_SAHA_STAGE_LINK_MISMATCH:{key}")
            nlte_cell_failures.setdefault(key, []).append("NLTE_SAHA_STAGE_LINK_MISMATCH")
        nlte_values.setdefault(key, []).append(row["n_pop"])
        nlte_ion_totals.setdefault(key, []).append(row["n_ion_total"])
    expected_nlte = {key: levels for key, levels in effective_nlte_levels(run).items() if key[0] in TARGET_ELEMENTS.values()} if run_exists else {}
    actual_nlte_sets = {z: {ion for zz, ion in expected_nlte if zz == z} for z in TARGET_ELEMENTS.values()}
    if actual_nlte_sets != registered_nlte_sets: pending.append("NLTE_PRODUCTION_SET_CHANGED")
    for (z, ion), registered_levels in expected_nlte.items():
        for shell in range(len(shells)):
            if z in TARGET_ELEMENTS.values() and observed_levels.get((shell, z, ion), set()) != registered_levels:
                nlte_cell_failures.setdefault((shell, z, ion), []).append("NLTE_LEVEL_COVERAGE_MISSING")
    unexpected = {(z, ion) for (_shell, z, ion) in observed_levels} - set(expected_nlte)
    if unexpected: pending.append(f"NLTE_UNEXPECTED_ION:{sorted(unexpected)}")
    registered_level_cells = sorted((shell, z, ion, level, global_idx) for (z, ion), levels in expected_nlte.items() for shell in range(len(shells)) for level, global_idx in levels)
    observed_level_cells = sorted((shell, z, ion, level, global_idx) for (shell, z, ion), levels in observed_levels.items() for level, global_idx in levels)
    observed_stage_cells = sorted(key for key in pops if key[1] in TARGET_ELEMENTS.values())
    registered_stage_cells = sorted((item["shell_id"], z, stage) for item in anchors for z in TARGET_ELEMENTS.values() for stage in saha_sets.get(z, set()))

    zero_denominator_state: dict[tuple[int, int], tuple[str, str]] = {}
    for (z, shell), denominator in totals.items():
        if z not in TARGET_ELEMENTS.values() or denominator != 0:
            continue
        tokens: list[str] = []
        complete = True
        for stage_index in sorted(saha_sets.get(z, set())):
            token = pops.get((shell, z, stage_index))
            if token is None:
                complete = False
            else:
                tokens.append(token)
        for ion in sorted(nlte_sets.get(z, set())):
            key = (shell, z, ion)
            expected_levels = expected_nlte.get((z, ion), set())
            if observed_levels.get(key, set()) != expected_levels or key not in nlte_values:
                complete = False
            else:
                tokens.extend(nlte_values[key])
        if not complete:
            continue
        try:
            values = [decimal_token(token) for token in tokens]
        except Failure:
            zero_denominator_state[(z, shell)] = ("INVALID_VALUE", "INVALID_POPULATION_WITH_ZERO_DENOMINATOR")
        else:
            zero_denominator_state[(z, shell)] = (
                ("INVALID_VALUE", "POSITIVE_POPULATION_WITH_ZERO_DENOMINATOR")
                if any(value > 0 for value in values)
                else ("UNDEFINED", "ELEMENT_ABSENT")
            )

    ion_lookup = {name: {row.velocity: row for row in rows} for name, rows in ions.items()}
    base_map_path = evidence / "offline/base_map.csv"
    base_rows = read_csv_strict(base_map_path, set(MAP_FIELDS))
    base_by_shell = {int(row["shell_id"]): row for row in base_rows if row["authority"] == "PUBLIC_NE"}
    output_rows: list[dict[str, Any]] = []
    reference_row_checks: list[dict[str, Any]] = []
    for element in TARGET_ELEMENTS:
        record = expectations.get("reference_top_bin_semantics", {}).get(element, {})
        if not isinstance(record, dict) or record.get("value") not in ("CUMULATIVE", "INDIVIDUAL") or not record.get("source"):
            pending.append(f"TOP_BIN_SEMANTICS_PENDING:{element}")
    conditions = "density_composition;radiation_field;iterations;atomic_data_level_truncation;inner_boundary;gamma_specific_heat;diagnostic_radiation"
    for item in anchors:
        shell = item["shell_id"]; ref = item["row"]; public = plasma.get(shell)
        try:
            intervention = abs(Decimal(str(base_by_shell[shell]["temperature_public_K"])) - decimal_token(ref.tokens[1]))
        except (KeyError, InvalidOperation, Failure):
            intervention = None
        output_rows.append(public_state_row(item, public, inventory, epoch, generation_known=generation_known, abs_delta_t=intervention))
        te_match = equal_decimal_tokens(public["T_e"], ref.tokens[1]) if public else None
        for element, z in TARGET_ELEMENTS.items():
            cmf = ion_lookup.get(element, {}).get(ref.velocity)
            if cmf is None:
                output_rows.append({**cell_template(shell, element, z, "", epoch, ref, public["T_e"] if public else "", ""), "authority": "SAHA_STAGE", "status": "INPUT_PENDING", "reason": "REFERENCE_ELEMENT_MISSING"})
                continue
            fraction_tokens = list(cmf.tokens[1:])
            try:
                lows_highs = [token_interval(token) for token in fraction_tokens]
                row_sum = sum((decimal_token(token) for token in fraction_tokens), Decimal(0))
            except Failure as exc:
                reference_row_checks.append({
                    "shell_id": shell, "element": element, "raw_tokens": fraction_tokens,
                    "raw_sum": None, "sum_interval": None, "includes_one": None,
                    "neutral_fraction_raw": cmf.tokens[1], "status": "INVALID_VALUE", "reason": exc.reason,
                })
                pending.append(f"REFERENCE_ROW_VALUE_INVALID:shell={shell}:element={element}")
            else:
                sum_lo = sum((lo for lo, _hi in lows_highs), Decimal(0))
                sum_hi = sum((hi for _lo, hi in lows_highs), Decimal(0))
                reference_row_checks.append({
                    "shell_id": shell, "element": element, "raw_sum": str(row_sum),
                    "sum_interval": [str(sum_lo), str(sum_hi)], "includes_one": sum_lo <= 1 <= sum_hi,
                    "neutral_fraction_raw": cmf.tokens[1], "status": "OBSERVED", "reason": "",
                })
            denominator = totals.get((z, shell)); comparison_denominator = comparison_totals.get((z, shell)); dumped = dump_totals.get((z, shell))
            denominator_state = zero_denominator_state.get((z, shell))
            q_top = len(cmf.tokens) - 2
            semantics_record = expectations.get("reference_top_bin_semantics", {}).get(element, {})
            semantics = semantics_record.get("value") if isinstance(semantics_record, dict) and semantics_record.get("source") else None
            for stage_index in range(q_top + 1):
                top = stage_index == q_top
                common = map_row(
                    shell_id=shell, element=element, Z=z, stage=stage_index, epoch_d=str(epoch),
                    v_center_km_s=str(item["v_center_km_s"]), vel_mid_km_s=ref.tokens[0],
                    sample_distance_km_s=str(item["sample_distance_km_s"]), temperature_registered_K=ref.tokens[1],
                    temperature_public_K=public["T_e"] if public else "",
                    N_Z_model=str(denominator) if denominator is not None else "",
                    N_Z_production=str(denominator) if denominator is not None else "",
                    N_Z_comparison=str(comparison_denominator) if comparison_denominator is not None else "",
                    N_Z_production_to_comparison=(
                        str(denominator / comparison_denominator)
                        if denominator is not None and comparison_denominator is not None
                        and denominator > 0 and comparison_denominator > 0 else ""
                    ),
                    N_Z_production_to_comparison_status=(
                        "VALID" if denominator is not None and comparison_denominator is not None
                        and denominator > 0 and comparison_denominator > 0 else "UNDEFINED"
                    ),
                    N_Z_dump=str(dumped) if dumped is not None else "", mass_weight=str(inventory["mass_weights"].get(shell, "")),
                    abs_delta_T=str(intervention) if intervention is not None else "", condition_differences=conditions,
                    top_bin_semantics=(semantics or "INPUT_PENDING") if top else "NOT_TOP_BIN",
                )
                reference_token = cmf.tokens[stage_index + 1]
                if top and semantics not in ("CUMULATIVE", "INDIVIDUAL"):
                    output_rows.append({**common, "authority": "SAHA_STAGE", "status": "INPUT_PENDING", "reason": "TOP_BIN_SEMANTICS_PENDING", "cmfgen_raw": reference_token})
                else:
                    mapped_saha = sorted(q for q in saha_sets.get(z, set()) if (q >= q_top if top and semantics == "CUMULATIVE" else q == stage_index))
                    tokens = [pops.get((shell, z, q)) for q in mapped_saha]
                    if not mapped_saha:
                        output_rows.append({**common, "authority": "SAHA_STAGE", "status": "NOT_SOLVED", "reason": "STAGE_NOT_IN_SAHA_PRODUCTION_SET", "cmfgen_raw": reference_token, "top_bin_mapping": ""})
                    elif any(token is None for token in tokens) or denominator is None:
                        output_rows.append({**common, "authority": "SAHA_STAGE", "status": "INPUT_PENDING", "reason": "STAGE_NOT_DUMPED", "cmfgen_raw": reference_token, "population_components": ";".join(f"q={q}:{token or 'MISSING'}" for q, token in zip(mapped_saha, tokens)), "top_bin_mapping": ";".join(map(str, mapped_saha)) if top else str(stage_index)})
                    else:
                        assert all(token is not None for token in tokens)
                        try:
                            numerator = sum((decimal_token(str(token)) for token in tokens), Decimal(0))
                        except Failure as exc:
                            fields = {"status": "INVALID_VALUE", "reason": exc.reason, "population_raw": ";".join(str(token) for token in tokens), "lumina_raw": "", "cmfgen_raw": reference_token, "delta_f": "", "percentage_point": ""}
                        else:
                            fields = _fraction_fields(str(numerator), denominator, reference_token)
                        fields["population_components"] = ";".join(f"q={q}:{token}" for q, token in zip(mapped_saha, tokens))
                        fields["top_bin_mapping"] = ";".join(map(str, mapped_saha)) if top else str(stage_index)
                        if denominator_state: fields.update(status=denominator_state[0], reason=denominator_state[1])
                        if (z, shell) in contaminated_saha_totals:
                            reasons = [reason for reason in fields.get("reason", "").split(";") if reason]
                            reasons.append("INVALID_PRODUCTION_POPULATION")
                            fields.update(status="INVALID_VALUE", reason=";".join(dict.fromkeys(reasons)))
                        if (z, shell) in denominator_mismatch: fields.update(status="INVALID_VALUE", reason="ELEMENT_DENOMINATOR_MISMATCH")
                        fields = qualify_observation(fields, te_match, generation_known)
                        output_rows.append({**common, "authority": "SAHA_STAGE", **fields})
                if top and semantics not in ("CUMULATIVE", "INDIVIDUAL"):
                    output_rows.append({**common, "authority": "NLTE_LEVEL_SUM", "status": "INPUT_PENDING", "reason": "TOP_BIN_SEMANTICS_PENDING", "cmfgen_raw": reference_token})
                else:
                    mapped_nlte = sorted(q for q in nlte_sets.get(z, set()) if (q >= q_top if top and semantics == "CUMULATIVE" else q == stage_index))
                    if not mapped_nlte:
                        output_rows.append({**common, "authority": "NLTE_LEVEL_SUM", "status": "NOT_SOLVED", "reason": "ION_NOT_IN_NLTE_SET", "cmfgen_raw": reference_token})
                    else:
                        all_level_tokens = [token for q in mapped_nlte for token in nlte_values.get((shell, z, q), [])]
                        expected_count = sum(len(expected_nlte.get((z, q), set())) for q in mapped_nlte)
                        observed_count = sum(len(observed_levels.get((shell, z, q), set())) for q in mapped_nlte)
                        coverage = "COMPLETE" if observed_count == expected_count else "INCOMPLETE"
                        failures = [reason for q in mapped_nlte for reason in nlte_cell_failures.get((shell, z, q), [])]
                        summed: Decimal | None = None
                        if denominator is None or not all_level_tokens:
                            fields = {"status": "INPUT_PENDING", "reason": "NLTE_LEVEL_DUMP_MISSING", "cmfgen_raw": reference_token}
                        else:
                            try: summed = sum((decimal_token(token) for token in all_level_tokens), Decimal(0))
                            except Failure as exc:
                                fields = {"status": "INVALID_VALUE", "reason": "INVALID_NLTE_LEVEL_VALUE", "population_raw": ";".join(all_level_tokens), "cmfgen_raw": reference_token}
                                failures.append(exc.reason)
                            else:
                                fields = _fraction_fields(str(summed), denominator, reference_token)
                            if denominator_state: fields.update(status=denominator_state[0], reason=denominator_state[1])
                        fields.update(
                            population_components=";".join(f"q={q}" for q in mapped_nlte),
                            top_bin_mapping=";".join(map(str, mapped_nlte)) if top else str(stage_index),
                            nlte_expected_levels=expected_count, nlte_observed_levels=observed_count,
                            nlte_level_coverage=coverage,
                        )
                        if len(mapped_nlte) == 1:
                            nion_tokens = nlte_ion_totals.get((shell, z, mapped_nlte[0]), [])
                            fields["n_ion_total_raw"] = ";".join(nion_tokens)
                            if summed is not None and nion_tokens and all(equal_decimal_tokens(nion_tokens[0], token) is True for token in nion_tokens):
                                ratio = scalar_metric(str(summed), nion_tokens[0])
                                fields["R_SN"] = ratio["ratio"] or ""
                                if ratio["status"] != "VALID" or ratio["ratio"] in (None, ""):
                                    failures.append("R_SN_UNDEFINED")
                            elif summed is not None:
                                failures.append("NLTE_N_ION_TOTAL_INCONSISTENT")
                        if coverage != "COMPLETE": failures.append("NLTE_LEVEL_COVERAGE_MISSING")
                        if failures and fields.get("status") != "INVALID_VALUE":
                            fields.update(status="INPUT_PENDING", reason=";".join(dict.fromkeys(failures)))
                        if (z, shell) in denominator_mismatch: fields.update(status="INVALID_VALUE", reason="ELEMENT_DENOMINATOR_MISMATCH")
                        fields = qualify_observation(fields, te_match, generation_known)
                        output_rows.append({**common, "authority": "NLTE_LEVEL_SUM", **fields})

    verify_map_metrics(output_rows)
    verify_map_metrics(base_rows)
    base_ez = aggregate_ez(base_rows, len(shells))
    twin_ez = aggregate_ez(output_rows, len(shells))
    if expectations.get("base_e_z") != base_ez:
        pending.append("BASE_EXPECTATION_AGGREGATE_MISMATCH")
    paired_structure = paired_ez_structure(base_rows, output_rows, len(shells))
    closure_observations: list[dict[str, Any]] = []
    for item in anchors:
        shell = item["shell_id"]
        for element, z in TARGET_ELEMENTS.items():
            cmf = ion_lookup.get(element, {}).get(item["row"].velocity)
            if cmf is None: continue
            q_top = len(cmf.tokens) - 2
            semantics_record = expectations.get("reference_top_bin_semantics", {}).get(element, {})
            semantics = semantics_record.get("value") if isinstance(semantics_record, dict) and semantics_record.get("source") else None
            production_tokens = [pops.get((shell, z, q)) for q in sorted(saha_sets.get(z, set()))]
            denominator = totals.get((z, shell))
            if semantics not in ("CUMULATIVE", "INDIVIDUAL"):
                closure_observations.append({"shell_id": shell, "element": element, "status": "INPUT_PENDING", "reason": "TOP_BIN_SEMANTICS_PENDING", "consistent": None})
                continue
            if denominator is None or denominator <= 0 or not production_tokens or any(token is None for token in production_tokens):
                closure_observations.append({"shell_id": shell, "element": element, "status": "INPUT_PENDING", "reason": "SAHA_CLOSURE_INPUT_MISSING", "consistent": None})
                continue
            try:
                production_values = [decimal_token(str(token)) for token in production_tokens]
                production_intervals = [token_interval(str(token)) for token in production_tokens]
                reference_values = [decimal_token(token) for token in cmf.tokens[1:]]
                reference_intervals = [token_interval(token) for token in cmf.tokens[1:]]
            except Failure as exc:
                closure_observations.append({"shell_id": shell, "element": element, "status": "INVALID_VALUE", "reason": exc.reason, "consistent": None})
                continue
            production_sum = sum(production_values, Decimal(0)) / denominator
            reference_sum = sum(reference_values, Decimal(0))
            residual = production_sum - reference_sum
            production_lo = sum((lo for lo, _hi in production_intervals), Decimal(0)) / denominator
            production_hi = sum((hi for _lo, hi in production_intervals), Decimal(0)) / denominator
            reference_lo = sum((lo for lo, _hi in reference_intervals), Decimal(0))
            reference_hi = sum((hi for _lo, hi in reference_intervals), Decimal(0))
            residual_lo = production_lo - reference_hi
            residual_hi = production_hi - reference_lo
            closure_observations.append({
                "shell_id": shell, "element": element, "status": "OBSERVED",
                "mapping": semantics,
                "production_sum_over_N_Z_production": str(production_sum),
                "production_sum_interval": [str(production_lo), str(production_hi)],
                "reference_sum": str(reference_sum),
                "reference_sum_interval": [str(reference_lo), str(reference_hi)],
                "N_Z_production": str(denominator),
                "production_stage_raw": [str(token) for token in production_tokens],
                "reference_stage_raw": list(cmf.tokens[1:]),
                "residual": str(residual),
                "residual_interval": [str(residual_lo), str(residual_hi)],
                "consistent": residual_lo <= 0 <= residual_hi,
                "neutral_fraction_raw": cmf.tokens[1],
            })
    if any(item["includes_one"] is False for item in reference_row_checks): pending.append("REFERENCE_ROW_SUM_INTERVAL_EXCLUDES_ONE")
    if any(item.get("status") == "OBSERVED" and item.get("consistent") is False for item in closure_observations):
        pending.append("SAHA_CLOSURE_MISMATCH")

    current_manifest = _tree_manifest(base)
    after_path = evidence / "inputs/base_after_run.sha256"
    if current_manifest != before_manifest: fail("BASE_SEAL_CHANGED")

    if not anchor_match: pending.append("ANCHOR_SELECTION_CHANGED")
    if profile_lines != wanted_profile: pending.append("TE_PROFILE_NOT_DIRECT_COPY")
    if footer.get("fixed_te_loader_consumed") != "1": pending.append("TE_LOADER_EVIDENCE_MISSING")
    if current_hashes != identity.get("reference_files"):
        pending.append("REFERENCE_IDENTITY_MISMATCH")
    run_binary = run / "input/lumina_cuda"
    actual_run_binary = sha256(run_binary) if run_binary.is_file() and not run_binary.is_symlink() else None
    if actual_run_binary != identity.get("base_binary", {}).get("sha256") or footer.get("binary_sha256") != actual_run_binary:
        pending.append("RUN_BINARY_IDENTITY_MISMATCH")
    for member in identity.get("source_identity", {}).get("source_files", []):
        source = Path(identity["source_identity"].get("repository", "")) / member.get("path", "")
        if not source.is_file() or source.is_symlink() or sha256(source) != member.get("sha256"):
            pending.append(f"SOURCE_IDENTITY_MISMATCH:{member.get('path', '')}")
    exports_exact = False
    try:
        exports_exact = exports_delta_is_registered(base, run)
    except Failure:
        pass
    if not exports_exact: pending.append("EXPORT_DELTA_NOT_EXACT")
    if observed_stage_cells != registered_stage_cells: pending.append("SAHA_PRODUCTION_COVERAGE_MISMATCH")
    if observed_level_cells != registered_level_cells: pending.append("NLTE_LEVEL_COVERAGE_MISMATCH")
    if denominator_mismatch: pending.append("ELEMENT_DENOMINATOR_MISMATCH")

    execution: dict[str, Any] = {
        "registered_runs": footer.get("registered_runs"),
        "outer_iterations": footer.get("outer_iterations"),
        "terminal_state": footer.get("terminal_state"),
        "elapsed_seconds": footer.get("elapsed_seconds"),
        "allocated_gpus": footer.get("allocated_gpus"),
        "allocated_cpus": footer.get("allocated_cpus"),
        "required_outputs": required,
        "observed_outputs": [item["path"] for item in files],
        "exports_delta_exact": exports_exact,
    }
    try:
        elapsed = int(footer["elapsed_seconds"])
        gpu_count = int(footer["allocated_gpus"])
        cpu_count = int(footer["allocated_cpus"])
        if footer.get("terminal_state") != "SUCCESS": pending.append("RUN_TERMINAL_FAILURE")
        if footer.get("registered_runs") != "1": pending.append("RUN_COUNT_MISMATCH")
        if footer.get("outer_iterations") != str(expectations.get("registered_outer_iterations")): pending.append("ITERATION_COUNT_MISMATCH")
        if elapsed > 4 * 3600 or gpu_count != 2 or gpu_count * elapsed > 8 * 3600: pending.append("GPU_TIME_LIMIT_EXCEEDED")
        if cpu_count > 32 or cpu_count * elapsed > 128 * 3600: pending.append("CPU_TIME_LIMIT_EXCEEDED")
    except (KeyError, TypeError, ValueError):
        pending.append("RUN_RESOURCE_RECORD_INVALID")
    if set(item["path"] for item in files) != set(required): pending.append("REQUIRED_OUTPUT_MISSING")
    if resource.get("pending"): pending.extend(f"resource:{reason}" for reason in resource["pending"])

    valid_z_rows = [row for row in output_rows if row["authority"] == "PUBLIC_NE" and row["d_Z"] != ""]
    missing = [{"shell_id": row["shell_id"], "element": row["element"], "stage": row["stage"], "authority": row["authority"], "status": row["status"], "reason": row["reason"]} for row in output_rows if row["status"] != "VALID"]
    mass_rows = [row for row in output_rows if row["authority"] == "PUBLIC_NE" and row["n_atom_dex"] != ""]
    extrema = None
    if valid_z_rows:
        extreme = max(valid_z_rows, key=lambda row: abs(float(row["d_Z"])))
        extrema = {"shell_id": extreme["shell_id"], "d_Z": extreme["d_Z"], "d_Z_interval": [extreme["d_Z_lo"], extreme["d_Z_hi"]], "vel_mid_km_s": extreme["vel_mid_km_s"]}
    summary = {
        "schema": SCHEMA,
        "measurement_conventions": {
            "fractions": "N_Z_production from sealed atom_masses.csv",
            "production_electron_per_atom": "z_production = n_e/n_atom_production",
            "comparison_electron_per_atom": "q_comparison = n_e/n_atom_model; Fe/Co/Ni comparison mass A=56",
            "e_z_operand": "q_comparison/(cmfgen_ne/cmfgen_natom)",
        },
        "pending": sorted(set(pending)),
        "outputs": {"run": str(run), "files": files, "publication_generation_record": generation_record},
        "production_population_failures": sorted(production_population_failures, key=lambda item: (item["shell_id"], item["Z"], item["stage"])),
        "missing": missing,
        "e_z_operands": {"base": base_ez, "twin": twin_ez},
        "paired_structure": paired_structure,
        "reference_row_checks": reference_row_checks,
        "saha_closure_observations": closure_observations,
        "top_bin_semantics": expectations.get("reference_top_bin_semantics", {}),
        "coverage": {
            "reference_core_rows_below_3900_km_s": sum(1 for row in phys_epoch_rows if row.velocity < 3900),
            "reference_core_velocity_range_km_s": [str(min((row.velocity for row in phys_epoch_rows if row.velocity < 3900), default="")), str(max((row.velocity for row in phys_epoch_rows if row.velocity < 3900), default=""))],
            "saha_production_stages": {str(z): sorted(stages) for z, stages in saha_sets.items()},
            "nlte_production_ions": {str(z): sorted(stages) for z, stages in nlte_sets.items()},
            "reference_stage_sets": {element: list(range(len(next(iter(rows)).tokens) - 1)) if rows else [] for element, rows in ions.items()},
        },
        "mass_convention_check": {
            "convention": inventory["mass_convention"], "cell_count": len(mass_rows),
            "max_abs_n_atom_dex": max((abs(float(row["n_atom_dex"])) for row in mass_rows), default=None),
            "n_atom_production_to_model": {
                str(row["shell_id"]): row["n_atom_production_to_model"] for row in mass_rows
            },
        },
        "extrema": extrema,
        "identity_observations": {"registered_reference_hashes": identity.get("reference_files"), "observed_reference_hashes": current_hashes, "registered_binary_sha256": identity.get("base_binary", {}).get("sha256"), "run_binary_sha256": actual_run_binary, "stage_exports_sha256": stage_identity.get("effective_exports_sha256"), "run_exports_sha256": sha256(run / "input/resolved_lumina.exports") if (run / "input/resolved_lumina.exports").is_file() else None},
        "execution_observations": execution,
        "independent_reproduction": {"shared_inputs": ["reference files", "geometry", "run raw outputs", "sealed identity records"], "shared_implementation": [], "production_parser_arithmetic_aggregation_reused": False, "required_method": "driver-authored direct csv/decimal/hashlib commands recorded with stdout, stderr, rc and hashes"},
    }
    map_text = csv_text(MAP_FIELDS, output_rows)
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    planned = {after_path: current_manifest, evidence / "result/divergence_map.csv": map_text, evidence / "result/summary.json": summary_text}
    cap = bounded_integer(resource.get("limits", {}).get("storage_bytes"), "storage_bytes", 16 * 1024**3)
    current_total = _tree_bytes(run) + _tree_bytes(evidence)
    planned_bytes = sum(len(text.encode("utf-8")) for text in planned.values())
    peak_total = current_total + planned_bytes
    if peak_total > cap:
        fail("STORAGE_CAP_EXCEEDED", f"current={current_total} temporary={planned_bytes} peak={peak_total} cap={cap}")
    available = os.statvfs(evidence).f_bavail * os.statvfs(evidence).f_frsize
    if planned_bytes > available:
        fail("FREE_SPACE_SHORTAGE", f"temporary={planned_bytes} available={available}")
    for path, text in planned.items():
        text_write(path, text)
    final_total = _tree_bytes(run) + _tree_bytes(evidence)
    if final_total > cap:
        fail("STORAGE_CAP_EXCEEDED_AFTER_WRITE", f"actual={final_total} cap={cap}")
    print(f"CMFGEN_TWIN_MAP_ANALYZE rows={len(output_rows)} pending={len(summary['pending'])} missing={len(missing)} peak_storage_bytes={peak_total} actual_storage_bytes={final_total}")
    return 0


# Selftests exercise the arithmetic, parsers, observation path, and write/resource
# controls directly.  They do not construct facts dictionaries or gate outcomes.


def _assert_arithmetic() -> None:
    metric = scalar_metric("2.000", "1.000")
    if (metric["ratio"], metric["dex"], metric["dex_lo"], metric["dex_hi"]) != (
        "2", 0.3010299956639812, 0.30070431549731025, 0.3013557572608802,
    ):
        fail("SELFTEST_SCALAR_FIXED_EXPECTATION")
    zero = scalar_metric("0", "1.000")
    if zero["ratio"] != "0" or zero["dex"] is not None:
        fail("SELFTEST_ZERO_FIXED_EXPECTATION")
    if scalar_metric("0", "0")["status"] != "UNDEFINED":
        fail("SELFTEST_UNDEFINED_FIXED_EXPECTATION")
    fraction = _fraction_fields("1.000", Decimal("4.000"), "0.200")
    if (fraction["lumina_raw"], fraction["delta_f"], fraction["percentage_point"]) != ("0.25", "0.050", "5.000"):
        fail("SELFTEST_FRACTION_FIXED_EXPECTATION")
    multiple = qualify_observation(scalar_metric("-Infinity", "1"), False)
    if multiple["status"] != "INVALID_VALUE" or set(multiple["reason"].split(";")) != {"NEGATIVE_OR_NONFINITE_VALUE", "T_NOT_TWIN"}:
        fail("SELFTEST_MULTIPLE_REASON_PRECEDENCE")


def _fixture_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    csv_write(path, fields, rows)


def _replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        fail("SELFTEST_MUTATION_TARGET_MISSING", old)
    text_write(path, text.replace(old, new, 1))


def _private_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    global BASE_BINARY_SHA256
    base, reference, evidence, run, repo = (root / name for name in ("base", "reference", "evidence", "run", "source"))
    for path in (base / "input/model", base / "work", reference, repo / "src"):
        path.mkdir(parents=True, exist_ok=True)
    binary = base / "input/lumina_cuda"; text_write(binary, "private fixture binary\n")
    BASE_BINARY_SHA256 = sha256(binary)
    cuda = repo / "src/lumina_cuda.cu"; plasma_source = repo / "src/lumina_plasma.c"
    text_write(cuda, "lumina_plasma_state.csv plasma.n_electron lumina_ion_pops.csv ion_number_density\n")
    target_z = ",".join(str(z) for z in TARGET_ELEMENTS.values() for _ in (1, 2))
    target_ion = ",".join(str(ion) for _z in TARGET_ELEMENTS.values() for ion in (1, 2))
    text_write(plasma_source, f"static const int NLTE_TARGET_Z[] = {{{target_z}}};\nstatic const int NLTE_TARGET_ION[] = {{{target_ion}}};\nnlte_levels_iter%03d.csv population_required_generation\n")
    source_files = [{"path": "src/lumina_cuda.cu", "sha256": sha256(cuda)}, {"path": "src/lumina_plasma.c", "sha256": sha256(plasma_source)}]
    json_write(base / "input/current_source_identity.json", {"binary_sha256": BASE_BINARY_SHA256, "repository": str(repo), "source_files": source_files})
    text_write(base / "RUN_FOOTER.txt", f"binary_sha256={BASE_BINARY_SHA256}\n")
    text_write(base / "input/resolved_lumina.exports", f'declare -x LUMINA_MODEL_DIR="{base}/input/model"\ndeclare -x LUMINA_FIXED_TE_PROFILE="/old/profile"\ndeclare -x LUMINA_RADEQ_DIAG_TE_K="10020"\n')
    json_write(base / "input/model/config.json", {"target_epoch_d": "1.0"})
    _fixture_csv(base / "input/model/geometry.csv", ["shell_id", "v_inner", "v_outer"], [{"shell_id": 0, "v_inner": 0, "v_outer": 100000}, {"shell_id": 1, "v_inner": 100000, "v_outer": 200000}])
    fixture_rho = str(AMU_G * 56)
    _fixture_csv(base / "input/model/density.csv", ["shell_id", "rho"], [{"shell_id": 0, "rho": fixture_rho}, {"shell_id": 1, "rho": fixture_rho}])
    fixture_masses = {14: "56", 16: "56", 20: "56", 26: "58", 27: "59", 28: "60", 8: "16"}
    _fixture_csv(base / "input/model/atom_masses.csv", ["atomic_number", "mass_amu"], [{"atomic_number": z, "mass_amu": mass} for z, mass in fixture_masses.items()])
    abundances = [{"atomic_number": z, "0": "0.1", "1": "0.1"} for z in TARGET_ELEMENTS.values()]
    abundances.append({"atomic_number": 8, "0": "0.4", "1": "0.4"})
    _fixture_csv(base / "input/model/abundances.csv", ["atomic_number", "0", "1"], abundances)
    _fixture_csv(base / "input/model/ionization_energies.csv", ["atomic_number", "ion_number", "ionization_energy_eV"], [{"atomic_number": z, "ion_number": ion, "ionization_energy_eV": ion * 10} for z in TARGET_ELEMENTS.values() for ion in (1, 2)])
    _fixture_csv(base / "input/model/active_ions.csv", ["atomic_number", "ion_number"], [{"atomic_number": z, "ion_number": ion} for z in TARGET_ELEMENTS.values() for ion in (1, 2)])
    level_fixture = [{"atomic_number": z, "ion_number": ion, "level_number": level} for z in TARGET_ELEMENTS.values() for ion in (1, 2) for level in (0, 1)]
    _fixture_csv(base / "input/model/levels.csv", ["atomic_number", "ion_number", "level_number"], level_fixture)
    plasma = [{"shell_id": 0, "T_e": "10000.000000", "n_e": "2.000"}, {"shell_id": 1, "T_e": "12000.000000", "n_e": "2.000"}]
    production_denominators = {
        z: Decimal(56) * Decimal("0.1") / Decimal(fixture_masses[z])
        for z in TARGET_ELEMENTS.values()
    }
    stage_populations = {
        z: {1: production_denominators[z] * Decimal("0.2"), 2: production_denominators[z] * Decimal("0.3"), 3: production_denominators[z] * Decimal("0.5")}
        for z in TARGET_ELEMENTS.values()
    }
    pops = [
        {"shell_id": shell, "Z": z, "stage": stage, "n_ion": str(value)}
        for shell in range(2) for z in TARGET_ELEMENTS.values()
        for stage, value in stage_populations[z].items()
    ]
    _fixture_csv(base / "work/lumina_plasma_state.csv", ["shell_id", "T_e", "n_e"], plasma)
    _fixture_csv(base / "work/lumina_ion_pops.csv", ["shell_id", "Z", "stage", "n_ion"], pops)
    phys_rows = (
        f"0.1 9.0000E+03 {fixture_rho} 2.000 2.000\n"
        f"0.6 1.0000E+04 {fixture_rho} 2.000 2.000\n"
        f"1.4 1.2000E+04 {fixture_rho} 2.000 2.000\n"
        f"1.6 1.3000E+04 {fixture_rho} 2.000 2.000\n"
    )
    text_write(reference / REFERENCE_NAMES[0], "#TIME: 1.0\n#NVEL: 4\n" + phys_rows)
    for element in TARGET_ELEMENTS:
        values = "\n".join(f"{velocity} 0.100 0.400 0.500" for velocity in ("0.1", "0.6", "1.4", "1.6"))
        text_write(reference / f"ionfrac_{element}_toy06_cmfgen.txt", f"#NSTAGES: 3\n#TIME: 1.0\n#NVEL: 4\n#vel_mid[km/s] {element}0 {element}1 {element}2\n{values}\n")
    prepare(argparse.Namespace(base=str(base), reference_dir=str(reference), evidence=str(evidence)))
    writer_path = evidence / "offline/writer_contract.json"
    writer = json.loads(writer_path.read_text(encoding="utf-8")); writer.update(pending=[], final_nlte_relative_path="work/nlte_levels_iter002.csv", final_population_generation=2, public_T_ne_Saha_NLTE_same_generation_evidence=True, generations={key: 2 for key in PUBLICATION_FIELDS}); json_write(writer_path, writer)
    resource_path = evidence / "offline/resource_card.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8")); resource["pending"] = []; json_write(resource_path, resource)
    expectations_path = evidence / "offline/expectations.json"
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    expectations.update(
        pending=[], writer_contract_sha256=sha256(writer_path),
        reference_top_bin_semantics={element: {"value": "CUMULATIVE", "source": "selftest-explicit-definition"} for element in TARGET_ELEMENTS},
    )
    json_write(expectations_path, expectations)
    shutil.copytree(base / "input", run / "input"); (run / "work").mkdir()
    text_write(run / "input/fixed_te_profile.txt", (evidence / "inputs/fixed_te_profile.txt").read_text(encoding="utf-8"))
    staged = {key: value.replace(str(base), str(run)) for key, value in read_exports(base / "input/resolved_lumina.exports").items()}; staged["LUMINA_FIXED_TE_PROFILE"] = f"{run}/input/fixed_te_profile.txt"; staged.pop("LUMINA_RADEQ_DIAG_TE_K", None); staged.update(LUMINA_ION_POP_DUMP="1", LUMINA_ION_POP_DUMP_ITER="1", LUMINA_NLTE_LEVEL_DUMP="1")
    text_write(run / "input/resolved_lumina.exports", "".join(f'declare -x {key}="{value}"\n' for key, value in staged.items()))
    _fixture_csv(run / "work/lumina_plasma_state.csv", ["shell_id", "T_e", "n_e"], [{"shell_id": 0, "T_e": "10000.000000", "n_e": "1.100"}, {"shell_id": 1, "T_e": "12000.000000", "n_e": "1.100"}])
    _fixture_csv(run / "work/lumina_ion_pops.csv", ["shell_id", "Z", "stage", "n_ion"], pops)
    nlte = [
        {"Z": row["atomic_number"], "ion": row["ion_number"], "shell": shell,
         "level_idx": row["level_number"], "global_idx": global_idx,
         "n_pop": str(stage_populations[row["atomic_number"]][row["ion_number"]] / 2),
         "T_e": "10000.000000" if shell == 0 else "12000.000000",
         "n_ion_total": str(stage_populations[row["atomic_number"]][row["ion_number"]]), "population_generation": 2}
        for shell in range(2) for global_idx, row in enumerate(level_fixture)
    ]
    _fixture_csv(run / "work/nlte_levels_iter002.csv", ["Z", "ion", "shell", "level_idx", "global_idx", "n_pop", "T_e", "n_ion_total", "population_generation"], nlte)
    text_write(run / "STAGE_IDENTITY.txt", f"effective_exports_sha256={sha256(run / 'input/resolved_lumina.exports')}\n")
    text_write(run / "RUN_FOOTER.txt", f"outer_iterations=2\nbinary_sha256={BASE_BINARY_SHA256}\nterminal_state=SUCCESS\nregistered_runs=1\nelapsed_seconds=100\nallocated_gpus=2\nallocated_cpus=32\nfixed_te_loader_consumed=1\n")
    text_write(evidence / "inputs/base_before.sha256", _tree_manifest(base))
    return base, reference, evidence, run, expectations_path


def _read_results(evidence: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    return read_csv_strict(evidence / "result/divergence_map.csv", set(MAP_FIELDS)), json.loads((evidence / "result/summary.json").read_text(encoding="utf-8"))


def _production_path_selftest(emit: bool) -> dict[str, Any]:
    global BASE_BINARY_SHA256, _tree_bytes
    saved_binary = BASE_BINARY_SHA256
    saved_tree_bytes = _tree_bytes
    observations: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="cmfgen-twin-e2e.") as raw:
            base, reference, evidence, run, expectations = _private_fixture(Path(raw))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, summary = _read_results(evidence)
            if len(rows) != 74 or any(key in summary for key in ("gates", "branch", "hypothesis", "status")):
                fail("PRODUCTION_PATH_OBSERVATION_SHAPE")
            anchors = read_csv_strict(evidence / "inputs/anchors.csv", set(ANCHOR_FIELDS))
            if [row["vel_mid_km_s"] for row in anchors] != ["0.6", "1.4"] or anchors[1]["tie_break"] != "LOWER_VELOCITY_THEN_SOURCE_ORDER":
                fail("MIDPOINT_NEAREST_SELECTION_NOT_CONSUMED")
            if summary["e_z_operands"]["twin"]["cell_count"] != 2 or not any(row["authority"] == "SAHA_STAGE" and row["stage"] == "2" and row["delta_f"] == "0.300" and row["top_bin_mapping"] == "2;3" for row in rows):
                fail("PRODUCTION_PATH_RAW_VALUE_OR_COVERAGE")
            public_row = next(row for row in rows if row["authority"] == "PUBLIC_NE" and row["shell_id"] == "0")
            if (Decimal(public_row["n_atom_model"]) != Decimal("2.0")
                or Decimal(public_row["n_atom_production"]) == Decimal(public_row["n_atom_model"])
                or Decimal(public_row["cmfgen_natom_raw"]) != Decimal("2.000")
                or Decimal(public_row["d_rho"]) != 0
                or "zbar_lumina" in public_row
                or Decimal(public_row["q_comparison"]) != Decimal(public_row["lumina_ne_raw"]) / Decimal(public_row["n_atom_model"])
                or Decimal(public_row["z_production"]) != Decimal(public_row["lumina_ne_raw"]) / Decimal(public_row["n_atom_production"])):
                fail("REV4_NATOM_OR_DENSITY_NOT_CONSUMED")
            fe_row = next(row for row in rows if row["authority"] == "SAHA_STAGE" and row["element"] == "fe" and row["shell_id"] == "0" and row["stage"] == "1")
            if (fe_row["N_Z_model"] != fe_row["N_Z_production"]
                or Decimal(fe_row["N_Z_production"]) == Decimal(fe_row["N_Z_comparison"])
                or Decimal(fe_row["N_Z_production_to_comparison"]) != Decimal(56) / Decimal(58)
                or "ELEMENT_DENOMINATOR_MISMATCH" in summary["pending"]):
                fail("REV4_ELEMENT_DENOMINATORS_NOT_CONSUMED")
            if not any(row["authority"] == "NLTE_LEVEL_SUM" and row["R_SN"] == "1" and row["nlte_level_coverage"] == "COMPLETE" for row in rows):
                fail("R_SN_OR_LEVEL_COVERAGE_NOT_CONSUMED")
            if not all(item["includes_one"] for item in summary["reference_row_checks"]) or not all(item.get("consistent") for item in summary["saha_closure_observations"] if item["status"] == "OBSERVED"):
                fail("REFERENCE_SUM_OR_CLOSURE_NOT_CONSUMED")
            prepared_contract = json.loads(expectations.read_text(encoding="utf-8"))
            self_counts = prepared_contract["base_self_comparison"]["paired_sign_counts"]
            paired_counts = summary["paired_structure"]["paired_sign_counts"]
            if self_counts != {"improved": 0, "worsened": 0, "representation_unresolved": 2} or sum(paired_counts.values()) != 2 or len(summary["paired_structure"]["abs_delta_T_strata"]) != 1:
                fail("MASS_WEIGHTED_PAIRED_OR_STRATIFIED_OUTPUT_MISSING")
            observations.append("REV4_TWO_DENOMINATORS_AND_74_CELLS midpoint=0.6,1.4 top_delta=0.300 n_atom_model=2.0 R_SN=1")

            writer_path = evidence / "offline/writer_contract.json"
            original_writer = writer_path.read_text(encoding="utf-8")
            original_expectations = expectations.read_text(encoding="utf-8")
            writer_data = json.loads(original_writer)
            writer_data["generations"]["ne"] = 1
            json_write(writer_path, writer_data)
            expectation_data = json.loads(original_expectations)
            expectation_data["writer_contract_sha256"] = sha256(writer_path)
            json_write(expectations, expectation_data)
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            generation_rows, generation_summary = _read_results(evidence)
            generation_record = generation_summary["outputs"]["publication_generation_record"]
            if generation_record.get("visibility") != "CONTRADICTORY_GENERATIONS" or generation_record.get("generations", {}).get("ne") != 1:
                fail("CONTRADICTORY_GENERATION_RECORD_NOT_PRESERVED")
            if not all("PUBLICATION_GENERATION_UNKNOWN" in row["reason"] for row in generation_rows if row["status"] != "NOT_SOLVED"):
                fail("CONTRADICTORY_GENERATION_CELL_NOT_DOWNGRADED")
            observations.append("CONTRADICTORY_GENERATION_PRESERVED final=2 ne=1 visibility=CONTRADICTORY_GENERATIONS")
            text_write(writer_path, original_writer); text_write(expectations, original_expectations)

            pop_path = run / "work/lumina_ion_pops.csv"
            original_population_dump = pop_path.read_text(encoding="utf-8")
            closure_data = json.loads(expectations.read_text(encoding="utf-8"))
            original_closure_expectations = expectations.read_text(encoding="utf-8")
            closure_data["reference_top_bin_semantics"]["ca"] = {"value": "INDIVIDUAL", "source": "selftest-explicit-definition"}
            json_write(expectations, closure_data)
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, direct_summary = _read_results(evidence)
            direct_closure = next(item for item in direct_summary["saha_closure_observations"] if item["shell_id"] == 0 and item["element"] == "ca")
            if direct_closure["mapping"] != "INDIVIDUAL" or direct_closure["consistent"] is not True:
                fail("DIRECT_TOP_CLOSURE_NOT_RECOMPUTED")
            observations.append("DIRECT_TOP_CLOSURE_OBSERVED residual_interval_includes_zero=1")
            text_write(expectations, original_closure_expectations)

            _replace_text(pop_path, "0,20,3,0.05", "0,20,3,0.01")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, closure_failure_summary = _read_results(evidence)
            disturbed = next(item for item in closure_failure_summary["saha_closure_observations"] if item["shell_id"] == 0 and item["element"] == "ca")
            reference_check = next(item for item in closure_failure_summary["reference_row_checks"] if item["shell_id"] == 0 and item["element"] == "ca")
            if disturbed["consistent"] is not False or reference_check["includes_one"] is not True or "SAHA_CLOSURE_MISMATCH" not in closure_failure_summary["pending"]:
                fail("PRODUCTION_CLOSURE_DISTURBANCE_NOT_OBSERVED")
            observations.append(f"PRODUCTION_CLOSURE_MISMATCH_PRESERVED reference_includes_one=1 residual={disturbed['residual']}")
            text_write(pop_path, original_population_dump)

            _replace_text(pop_path, "0,20,3,0.05", "0,20,3,-0.125")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            invalid_rows, invalid_summary = _read_results(evidence)
            failure = next((item for item in invalid_summary["production_population_failures"] if item["shell_id"] == 0 and item["Z"] == 20 and item["stage"] == 3), None)
            if failure is None or failure["raw_value"] != "-0.125" or failure["aggregate_disposition"] != "EXCLUDED_FROM_N_Z_DUMP_BECAUSE_INVALID":
                fail("OUTSIDE_COMPARISON_FAILURE_NOT_PRESERVED")
            if not any(row["authority"] == "SAHA_STAGE" and row["element"] == "ca" and row["shell_id"] == "0" and row["status"] == "INVALID_VALUE" and row["N_Z_dump"] == "" and "INVALID_PRODUCTION_POPULATION" in row["reason"] for row in invalid_rows):
                fail("OUTSIDE_COMPARISON_FAILURE_NOT_VISIBLE_IN_MAP")
            observations.append("OUTSIDE_COMPARISON_FAILURE_PRESERVED shell=0 Z=20 stage=3 raw=-0.125 disposition=EXCLUDED_FROM_N_Z_DUMP_BECAUSE_INVALID")
            text_write(pop_path, original_population_dump)

            nlte_path = run / "work/nlte_levels_iter002.csv"; original_nlte = nlte_path.read_text(encoding="utf-8")
            for invalid_token in ("NaN", "Infinity", "-Infinity", "-0.010"):
                _replace_text(nlte_path, ",0.01,10000.000000,0.02,2\n", f",{invalid_token},10000.000000,0.02,2\n")
                analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
                invalid_nlte_rows, invalid_nlte_summary = _read_results(evidence)
                invalid_nlte = next(row for row in invalid_nlte_rows if row["authority"] == "NLTE_LEVEL_SUM" and row["element"] == "ca" and row["shell_id"] == "0" and row["stage"] == "1")
                if (invalid_nlte["status"] != "INVALID_VALUE" or invalid_token not in invalid_nlte["population_raw"]
                    or invalid_nlte["R_SN"] != "" or not invalid_nlte_summary["outputs"]["files"]):
                    fail("NONFINITE_NLTE_CELL_NOT_LOCALIZED", invalid_token)
                observations.append(f"INVALID_NLTE_TOKEN_PRESERVED token={invalid_token} R_SN=blank map_and_summary=written")
                text_write(nlte_path, original_nlte)
            _replace_text(nlte_path, ",2\n", ",1\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, _ = _read_results(evidence)
            if not any(row["authority"] == "NLTE_LEVEL_SUM" and "WRITER_GENERATION_MISMATCH" in row["reason"] and row["status"] == "INPUT_PENDING" for row in rows):
                fail("PREVIOUS_GENERATION_NOT_PRESERVED")
            observations.append("PREVIOUS_GENERATION_PRESERVED")
            text_write(nlte_path, original_nlte)

            _replace_text(nlte_path, ",0.02,2\n", ",0.021,2\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, _ = _read_results(evidence)
            if not any("NLTE_SAHA_STAGE_LINK_MISMATCH" in row["reason"] for row in rows): fail("SAHA_LINK_FAILURE_NOT_PRESERVED")
            observations.append("SAHA_LINK_FAILURE_PRESERVED"); text_write(nlte_path, original_nlte)

            _replace_text(nlte_path, ",0.01,10000.000000,0.02,2\n", ",0.005,10000.000000,0.02,2\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, _ = _read_results(evidence)
            rsn_row = next(row for row in rows if row["authority"] == "NLTE_LEVEL_SUM" and row["element"] == "ca" and row["shell_id"] == "0" and row["stage"] == "1")
            if rsn_row["R_SN"] != "0.75" or rsn_row["population_raw"] != "0.015": fail("R_SN_GAP_NOT_OBSERVED")
            observations.append("R_SN_GAP_OBSERVED shell=0 Z=20 stage=1 R_SN=0.75"); text_write(nlte_path, original_nlte)

            nlte_lines = original_nlte.splitlines()
            removed = next(index for index, line in enumerate(nlte_lines) if line.startswith("20,1,0,0,0,"))
            text_write(nlte_path, "\n".join(nlte_lines[:removed] + nlte_lines[removed + 1:]) + "\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, _ = _read_results(evidence)
            coverage_row = next(row for row in rows if row["authority"] == "NLTE_LEVEL_SUM" and row["element"] == "ca" and row["shell_id"] == "0" and row["stage"] == "1")
            if coverage_row["nlte_level_coverage"] != "INCOMPLETE" or coverage_row["R_SN"] != "0.5" or "NLTE_LEVEL_COVERAGE_MISSING" not in coverage_row["reason"]:
                fail("PARTIAL_LEVEL_COVERAGE_NOT_PRESERVED_WITH_R_SN")
            observations.append("PARTIAL_LEVEL_COVERAGE_PRESERVED shell=0 expected=2 observed=1 R_SN=0.5"); text_write(nlte_path, original_nlte)

            expectation_data = json.loads(expectations.read_text(encoding="utf-8")); original_expectations_semantics = expectations.read_text(encoding="utf-8")
            expectation_data["reference_top_bin_semantics"]["ca"] = {"value": "INPUT_PENDING", "source": None}
            json_write(expectations, expectation_data)
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            rows, _ = _read_results(evidence)
            ca_top = [row for row in rows if row["element"] == "ca" and row["stage"] == "2"]
            if not ca_top or any(row["status"] != "INPUT_PENDING" or row["delta_f"] for row in ca_top): fail("UNKNOWN_TOP_BIN_WAS_NUMERIC")
            if not any(row["element"] == "ca" and row["stage"] == "1" and row["delta_f"] for row in rows): fail("UNKNOWN_TOP_BIN_ERASED_DIRECT_CELL")
            observations.append("TOP_BIN_SEMANTICS_PENDING_PRESERVED without_numeric_top=1 lower_stage_preserved=1")
            text_write(expectations, original_expectations_semantics)

            ca = reference / "ionfrac_ca_toy06_cmfgen.txt"; original_ca = ca.read_text(encoding="utf-8")
            text_write(ca, original_ca.replace("#NVEL: 4", "#NVEL: 5").replace("1.6 0.100 0.400 0.500", "1.6 0.100 0.400 0.500\n2.5 0.100 0.400 0.500"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            extra_rows, _ = _read_results(evidence)
            if len(extra_rows) != 74: fail("UNSELECTED_REFERENCE_ROW_ERASED_MAP")
            observations.append("UNSELECTED_REFERENCE_ROW_IGNORED"); text_write(ca, original_ca)

            text_write(ca, original_ca.replace("0.6 0.100 0.400 0.500", "0.6 0.100 0.400 0.400"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, bad_sum_summary = _read_results(evidence)
            bad_sum = next(item for item in bad_sum_summary["reference_row_checks"] if item["element"] == "ca" and item["shell_id"] == 0)
            if bad_sum["includes_one"] or "REFERENCE_ROW_SUM_INTERVAL_EXCLUDES_ONE" not in bad_sum_summary["pending"]:
                fail("REFERENCE_ROW_SUM_FAILURE_NOT_CONSUMED")
            observations.append("REFERENCE_ROW_SUM_FAILURE_OBSERVED shell=0 element=ca raw_sum=0.900")
            text_write(ca, original_ca)

            text_write(ca, original_ca.replace("0.6 0.100 0.400 0.500", "0.6 0.100 0.400 NaN"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            invalid_reference_rows, invalid_reference_summary = _read_results(evidence)
            if "REFERENCE_ROW_VALUE_INVALID:shell=0:element=ca" not in invalid_reference_summary["pending"] or not any(row["element"] == "ca" and row["shell_id"] == "0" and row["cmfgen_raw"] == "NaN" and row["status"] == "INVALID_VALUE" for row in invalid_reference_rows):
                fail("NONFINITE_REFERENCE_VALUE_NOT_PRESERVED")
            observations.append("NONFINITE_REFERENCE_VALUE_PRESERVED shell=0 element=ca raw=NaN")
            text_write(ca, original_ca)

            text_write(ca, original_ca.replace("#NSTAGES: 3", "#NSTAGES: 4"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            header_rows, _ = _read_results(evidence)
            if not any(row["element"] == "ca" and row["reason"] == "REFERENCE_ELEMENT_MISSING" for row in header_rows) or not any(row["element"] == "fe" and row["cmfgen_raw"] for row in header_rows):
                fail("HEADER_CONTRADICTION_NOT_LOCALIZED")
            observations.append("HEADER_CONTRADICTION_LOCALIZED"); text_write(ca, original_ca)

            abundance = run / "input/model/abundances.csv"; pops = run / "work/lumina_ion_pops.csv"
            original_abundance, original_pops = abundance.read_text(encoding="utf-8"), pops.read_text(encoding="utf-8")
            lines = original_abundance.splitlines(); lines = [line.replace("20,0.1,0.1", "20,0,0.1") for line in lines]; text_write(abundance, "\n".join(lines) + "\n")
            text_write(pops, original_pops.replace("0,20,1,0.02", "0,20,1,0").replace("0,20,2,0.03", "0,20,2,0").replace("0,20,3,0.05", "0,20,3,0"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            absent_rows, _ = _read_results(evidence)
            if not any(row["element"] == "ca" and row["shell_id"] == "0" and row["status"] == "INVALID_VALUE" and row["reason"] == "POSITIVE_POPULATION_WITH_ZERO_DENOMINATOR" for row in absent_rows):
                fail("CONTRADICTORY_ABSENCE_NOT_PRESERVED")
            observations.append("CONTRADICTORY_ABSENCE_PRESERVED")
            text_write(pops, original_pops.replace("0,20,1,0.02", "0,20,1,NaN"))
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            nan_rows, _ = _read_results(evidence)
            if not any(row["element"] == "ca" and row["shell_id"] == "0" and row["status"] == "INVALID_VALUE" and row["reason"] == "INVALID_POPULATION_WITH_ZERO_DENOMINATOR" for row in nan_rows): fail("NONFINITE_ABSENCE_NOT_PRESERVED")
            observations.append("NONFINITE_ABSENCE_PRESERVED"); text_write(abundance, original_abundance); text_write(pops, original_pops)

            plasma_path = run / "work/lumina_plasma_state.csv"; original_plasma = plasma_path.read_text(encoding="utf-8")
            _replace_text(plasma_path, "10000.000000", "10001.000000")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            t_rows, _ = _read_results(evidence)
            if not any(row["authority"] == "PUBLIC_NE" and row["shell_id"] == "0" and "T_NOT_TWIN" in row["reason"] for row in t_rows): fail("PUBLIC_T_MISMATCH_NOT_PRESERVED")
            observations.append("PUBLIC_T_MISMATCH_PRESERVED"); text_write(plasma_path, original_plasma)

            source = Path(json.loads((base / "input/current_source_identity.json").read_text(encoding="utf-8"))["repository"]) / "src/lumina_cuda.cu"
            original_source = source.read_text(encoding="utf-8"); text_write(source, original_source + "changed\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, changed_summary = _read_results(evidence)
            if not any(reason.startswith("SOURCE_IDENTITY_MISMATCH") for reason in changed_summary["pending"]): fail("SOURCE_MUTATION_NOT_OBSERVED")
            observations.append("SOURCE_MUTATION_OBSERVED"); text_write(source, original_source)

            run_binary = run / "input/lumina_cuda"; original_binary = run_binary.read_text(encoding="utf-8"); text_write(run_binary, original_binary + "changed\n")
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, changed_summary = _read_results(evidence)
            if "RUN_BINARY_IDENTITY_MISMATCH" not in changed_summary["pending"]: fail("RUN_BINARY_MUTATION_NOT_OBSERVED")
            observations.append("RUN_BINARY_MUTATION_OBSERVED"); text_write(run_binary, original_binary)

            contract_data = json.loads(expectations.read_text(encoding="utf-8")); original_interval = contract_data["base_e_z"]["M_Z_interval"]; contract_data["base_e_z"]["M_Z_interval"] = ["9", "10"]; json_write(expectations, contract_data)
            analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            _, changed_summary = _read_results(evidence)
            if changed_summary["e_z_operands"]["base"]["M_Z_interval"] == ["9", "10"] or "BASE_EXPECTATION_AGGREGATE_MISMATCH" not in changed_summary["pending"]: fail("BASE_AGGREGATE_MUTATION_CONSUMED")
            observations.append("BASE_E_Z_AGGREGATE_RECOMPUTED"); contract_data["base_e_z"]["M_Z_interval"] = original_interval; json_write(expectations, contract_data)

            result_before = (evidence / "result/divergence_map.csv").read_bytes(), (evidence / "result/summary.json").read_bytes()
            contract_data = json.loads(expectations.read_text(encoding="utf-8")); original_cap = contract_data.copy(); resource_path = evidence / "offline/resource_card.json"; resource = json.loads(resource_path.read_text(encoding="utf-8")); resource["limits"]["storage_bytes"] = 1; json_write(resource_path, resource)
            try: analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            except Failure as exc:
                if exc.reason != "STORAGE_CAP_EXCEEDED": raise
            else: fail("POST_CHECK_OUTPUT_CREATED")
            if result_before != ((evidence / "result/divergence_map.csv").read_bytes(), (evidence / "result/summary.json").read_bytes()): fail("STORAGE_REJECTION_CHANGED_OUTPUT")
            observations.append("STORAGE_REJECTED_BEFORE_OUTPUT")

            resource = json.loads(resource_path.read_text(encoding="utf-8")); resource["limits"]["storage_bytes"] = 16 * 1024**3; json_write(resource_path, resource)
            cap = resource["limits"]["storage_bytes"]
            _tree_bytes = lambda root: cap - 1024 if root == evidence else 0
            try: analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
            except Failure as exc:
                if exc.reason != "STORAGE_CAP_EXCEEDED" or "temporary=" not in exc.detail or "peak=" not in exc.detail: raise
                observations.append(f"REPLACEMENT_TEMPORARY_PEAK_REJECTED {exc.detail}")
            else: fail("REPLACEMENT_TEMPORARY_PEAK_ACCEPTED")
            finally: _tree_bytes = saved_tree_bytes

            # Independent oracle: read the fixture's raw model, population, and
            # reference files directly.  No production parser, metric, or
            # aggregation function is called.
            with (evidence / "result/divergence_map.csv").open(newline="", encoding="utf-8") as stream:
                direct = next(row for row in csv.DictReader(stream) if row["authority"] == "SAHA_STAGE" and row["element"] == "fe" and row["shell_id"] == "0" and row["stage"] == "2")
            with (run / "work/lumina_ion_pops.csv").open(newline="", encoding="utf-8") as stream:
                raw_population = sum((Decimal(row["n_ion"]) for row in csv.DictReader(stream) if row["shell_id"] == "0" and row["Z"] == "26" and int(row["stage"]) >= 2), Decimal(0))
            with (run / "input/model/density.csv").open(newline="", encoding="utf-8") as stream:
                raw_density = next(row["rho"] for row in csv.DictReader(stream) if row["shell_id"] == "0")
            with (run / "input/model/abundances.csv").open(newline="", encoding="utf-8") as stream:
                raw_abundance = next(row["0"] for row in csv.DictReader(stream) if row["atomic_number"] == "26")
            with (run / "input/model/atom_masses.csv").open(newline="", encoding="utf-8") as stream:
                raw_mass = next(row["mass_amu"] for row in csv.DictReader(stream) if row["atomic_number"] == "26")
            raw_reference = next(line.split()[3] for line in (reference / "ionfrac_fe_toy06_cmfgen.txt").read_text(encoding="utf-8").splitlines() if line.startswith("0.6 "))
            independent_total = Decimal(raw_density) * Decimal(raw_abundance) / (Decimal(raw_mass) * Decimal("1.660539066e-24"))
            if Decimal(raw_mass) == 56: fail("SELFTEST_IGE_MASS_DID_NOT_DIFFER_FROM_OVERRIDE")
            independent_delta = raw_population / independent_total - Decimal(raw_reference)
            if independent_delta != Decimal(direct["delta_f"]): fail("INDEPENDENT_DIRECT_ARITHMETIC")
            observations.append("INDEPENDENT_DIRECT_ARITHMETIC")
    finally:
        BASE_BINARY_SHA256 = saved_binary
        _tree_bytes = saved_tree_bytes
    if emit:
        for item in observations: print(f"SELFTEST observation={item}")
        print("CMFGEN_TWIN_MAP_E2E_OBSERVED prepare_rc=0 analyze_rc=0 rows=74 negative_observations=24")
    return {"observations": observations}


def _write_boundary_selftest(emit: bool) -> str:
    with tempfile.TemporaryDirectory(prefix="cmfgen-twin-output-selftest.") as raw:
        root = Path(raw); protected = root / "protected"; text_write(protected, "sealed\n"); os.chmod(protected, 0o440)
        attacked = root / "result.json"
        def swap(stream: Any) -> None:
            stream.write("bad\n")
            temporary = next(root.glob(".result.json.*")); temporary.unlink(); temporary.symlink_to(protected)
        try: _atomic_output(attacked, swap)
        except Failure as exc:
            if exc.reason != "OUTPUT_TEMPORARY_REPLACED": raise
        else: fail("OUTPUT_SWAP_ACCEPTED")
        mode = stat.S_IMODE(protected.stat().st_mode)
        if mode != 0o440 or protected.read_text(encoding="utf-8") != "sealed\n": fail("PROTECTED_TARGET_CHANGED_BEFORE_REJECTION")
        protected_dir = root / "protected-dir"; protected_dir.mkdir()
        protected_output = protected_dir / "out"; text_write(protected_output, "sealed-ancestor\n"); os.chmod(protected_output, 0o440)
        route = root / "route"; (route / "result").mkdir(parents=True)
        held = root / "held"; route.rename(held); route.symlink_to(protected_dir, target_is_directory=True)
        try: text_write(route / "out", "bad\n")
        except Failure as exc:
            if exc.reason != "OUTPUT_PARENT_UNSAFE": raise
        else: fail("ANCESTOR_PATH_SWAP_ACCEPTED")
        if protected_output.read_text(encoding="utf-8") != "sealed-ancestor\n" or stat.S_IMODE(protected_output.stat().st_mode) != 0o440:
            fail("ANCESTOR_PATH_SWAP_CHANGED_PROTECTED")

        nested_protected = root / "nested-protected"; nested_protected.mkdir()
        nested_output = nested_protected / "out"; text_write(nested_output, "sealed-nested\n"); os.chmod(nested_output, 0o440)
        nested = root / "nested" / "one"; nested.mkdir(parents=True)
        (nested / "two").symlink_to(nested_protected, target_is_directory=True)
        try: text_write(nested / "two" / "out", "bad\n")
        except Failure as exc:
            if exc.reason != "OUTPUT_PARENT_UNSAFE": raise
        else: fail("NESTED_ANCESTOR_LINK_ACCEPTED")
        if nested_output.read_text(encoding="utf-8") != "sealed-nested\n" or stat.S_IMODE(nested_output.stat().st_mode) != 0o440:
            fail("NESTED_ANCESTOR_LINK_CHANGED_PROTECTED")

        creation_base = root / "creation-base"; creation_base.mkdir()
        marker = creation_base / "sealed"; marker.write_text("sealed\n", encoding="utf-8"); os.chmod(marker, 0o440)
        creation_route = root / "creation-route"; creation_route.symlink_to(creation_base, target_is_directory=True)
        try: _create_output_tree(creation_route / "fresh-evidence", creation_base, ("inputs",))
        except Failure as exc:
            if exc.reason != "OUTPUT_ROOT_UNSAFE": raise
        else: fail("OUTPUT_ROOT_ANCESTOR_SWAP_ACCEPTED")
        if (creation_base / "fresh-evidence").exists(): fail("OUTPUT_ROOT_ANCESTOR_SWAP_CREATED_ENTRY")

        nested_route = root / "creation-nested"; nested_route.mkdir()
        (nested_route / "one").symlink_to(creation_base, target_is_directory=True)
        try: _create_output_tree(nested_route / "one/fresh-evidence", creation_base, ("inputs",))
        except Failure as exc:
            if exc.reason != "OUTPUT_ROOT_UNSAFE": raise
        else: fail("OUTPUT_ROOT_NESTED_LINK_ACCEPTED")
        if (creation_base / "fresh-evidence").exists(): fail("OUTPUT_ROOT_NESTED_LINK_CREATED_ENTRY")

        normal_root = root / "normal-evidence"
        _create_output_tree(normal_root, creation_base, ("inputs", "offline", "run", "result"))
        if not all((normal_root / child).is_dir() for child in ("inputs", "offline", "run", "result")):
            fail("OUTPUT_ROOT_NORMAL_CREATION_FAILED")
        if marker.read_text(encoding="utf-8") != "sealed\n" or stat.S_IMODE(marker.stat().st_mode) != 0o440:
            fail("OUTPUT_ROOT_CREATION_CHANGED_PROTECTED")
    if emit:
        print("SELFTEST observation=TEMPORARY_SWAP_REJECTED protected_mode_before=0440 protected_mode_after=0440 protected_content=unchanged")
        print("SELFTEST observation=ANCESTOR_PATH_SWAP_REJECTED protected_mode=0440 protected_content=sealed-ancestor")
        print("SELFTEST observation=NESTED_ANCESTOR_LINK_REJECTED protected_mode=0440 protected_content=sealed-nested")
        print("SELFTEST inject=OUTPUT_ROOT_ANCESTOR_SWAP observation=rejected first_write=none protected_entries=unchanged")
        print("SELFTEST inject=OUTPUT_ROOT_NESTED_LINK observation=rejected first_write=none protected_entries=unchanged")
        print("SELFTEST control=OUTPUT_ROOT_NORMAL_CREATION observation=accepted children=4")
    return "TEMPORARY_SWAP_REJECTED"


def _real_input_prepare_selftest(emit: bool) -> str:
    """Consume the registered 100-row reference blocks and sealed 50-shell base read-only."""
    if not REAL_BASE.is_dir() or not REAL_REFERENCE.is_dir():
        fail("SELFTEST_REAL_INPUT_PENDING", f"base={REAL_BASE} reference={REAL_REFERENCE}")
    with tempfile.TemporaryDirectory(prefix="cmfgen-rev3-real-input.") as raw:
        evidence = Path(raw) / "evidence"
        prepare(argparse.Namespace(base=str(REAL_BASE), reference_dir=str(REAL_REFERENCE), evidence=str(evidence)))
        anchors = read_csv_strict(evidence / "inputs/anchors.csv", set(ANCHOR_FIELDS))
        base_rows = read_csv_strict(evidence / "offline/base_map.csv", set(MAP_FIELDS))
        expectations = json.loads((evidence / "offline/expectations.json").read_text(encoding="utf-8"))
        expected_pending = {"final_publication_generation_link", "actual_fixed_te_loader_cpu_consumption"} | {f"top_bin_semantics:{element}" for element in TARGET_ELEMENTS}
        if len(anchors) != 50 or len(base_rows) != 50 or set(expectations["pending"]) != expected_pending:
            fail("SELFTEST_REAL_INPUT_SHAPE_OR_PENDING_MISMATCH")
        for row in anchors:
            inner = Decimal(row["v_inner_km_s"]); outer = Decimal(row["v_outer_km_s"])
            center = Decimal(row["v_center_km_s"]); velocity = Decimal(row["vel_mid_km_s"])
            if center != (inner + outer) / 2 or Decimal(row["sample_distance_km_s"]) != abs(velocity - center) or not (inner <= velocity < outer):
                fail("SELFTEST_REAL_INPUT_ANCHOR_RELATION_MISMATCH", f"shell={row['shell_id']}")
        if expectations["base_e_z"]["cell_count"] != 50:
            fail("SELFTEST_REAL_BASE_E_Z_INCOMPLETE")
        first, last = anchors[0]["vel_mid_km_s"], anchors[-1]["vel_mid_km_s"]
    if emit:
        print(f"SELFTEST observation=REAL_REFERENCE_PREPARE_CONSUMED anchors=50 base_e_z_cells=50 first={first} last={last} pending=8")
    return "REAL_REFERENCE_PREPARE_CONSUMED"


def run_selftest(emit: bool = True, include_e2e: bool = True) -> dict[str, Any]:
    _assert_arithmetic()
    boundary = _write_boundary_selftest(emit)
    production = _production_path_selftest(emit) if include_e2e else {"observations": []}
    real_input = _real_input_prepare_selftest(emit) if include_e2e else "SKIPPED"
    report = {"schema": SCHEMA, "arithmetic": "fixed constants matched", "write_boundary": boundary, "production_path": production, "real_input_prepare": real_input}
    if emit: print(f"CMFGEN_TWIN_MAP_SELFTEST_PASS observations={len(production['observations']) + 8}")
    return report

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="command")
    prep = sub.add_parser("prepare")
    prep.add_argument("--base", required=True); prep.add_argument("--reference-dir", required=True); prep.add_argument("--evidence", required=True)
    ana = sub.add_parser("analyze")
    ana.add_argument("--contract", required=True); ana.add_argument("--run", required=True); ana.add_argument("--evidence", required=True)
    return ap


def main() -> int:
    ap = parser(); args = ap.parse_args()
    if args.selftest:
        if args.command is not None: ap.error("--selftest accepts no command")
        run_selftest(True); return 0
    if args.command == "prepare": return prepare(args)
    if args.command == "analyze": return analyze(args)
    ap.error("choose --selftest, prepare, or analyze")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"CMFGEN_TWIN_MAP_FAIL reason={exc.reason} detail={exc.detail}", file=sys.stderr)
        raise SystemExit(70)
