#!/usr/bin/env python3
"""Write/check the A2-00 CMFGEN oracle-eligibility manifest.

The contract inventories only immediate children of the requested run directory.
It never follows symlinks.  mtimes are recorded for diagnosis but deliberately
excluded from every verdict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Iterable


SCHEMA = "lumina-cmfgen-oracle-manifest-v1"
ATTESTATION = "CMFGEN_ORACLE_ATTESTATION.json"
ROLES = {
    "oracle-data",
    "oracle-metadata",
    "provenance",
    "run-log",
    "scratch",
    "unclassified",
}

RC_OK = 0
RC_MANIFEST = 10
RC_INVENTORY = 11
RC_SIZE = 12
RC_HASH = 13
RC_RECORD_SCHEMA = 14
RC_UNCLASSIFIED = 15
RC_OPHYS = 16

FLOAT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)

ORACLE_DATA_EXACT = {
    "ADIABAT_CHK",
    "CHI_DATA",
    "CONT_FREQ",
    "EDDFACTOR",
    "ETA_DATA",
    "GAMMAS",
    "GENCOOL",
    "JH_AT_CURRENT_TIME",
    "LINEHEAT",
    "MEANOPAC",
    "NEG_OPAC",
    "NETRATE",
    "OBSFLUX",
    "OBS_FREQ",
    "RVTJ",
    "SN_HYDRO_FOR_NEXT_MODEL",
    "TOTRATE",
}

ORACLE_METADATA_EXACT = {
    ATTESTATION,
    "IN_ITS",
    "LEVEL_SL_STEQ_LINKS",
    "MODEL",
    "MODEL_SPEC",
    "SN_HYDRO_DATA",
    "SPECIES_MASSES",
    "VADAT",
    "atomic_links.txt",
    "model_spec_isf.txt",
}

RUN_LOG_EXACT = {
    "COLLISION_SUMMARY",
    "CORRECTION_LINK",
    "CORRECTION_SUM",
    "GREY_CHK",
    "GREY_SCL_FACOUT",
    "GreyDiagnostics",
    "MASS_FRACTION_SUM_CHK",
    "MOD_SUM",
    "MOM_J_ERRORS",
    "MU_VALUE_CHK",
    "NEW_SN_R_GRID",
    "NUM_DECAYS_INFO",
    "OLD_SN_R_GRID",
    "PHOT_SUMMARY_FILE",
    "R_GRID_SELECTION",
    "SN_DATA_INPUT_CHK",
    "TIMING",
    "TWO_PHOT_SUM",
    "WARNINGS",
    "gamma_feiii_coiii_formingshells.csv",
    "jnu_918_1290_formingshells.csv",
}

SCRATCH_EXACT = {
    "CFDAT_OUT",
    "CMFGEN_PID",
    "CUR_MODEL_DATA",
    "JEW",
    "J_COMP",
    "POINT1",
    "POINT2",
    "SCRTEMP",
    "STEQ_VALS",
    "crashbak_1948_lte_coldstart",
    "unconv_it40_bak",
    "unconv_resume_diverged",
}

LINK_EXACT = {
    "GBF_N_DATA",
    "H2_IS_LINE_LIST",
    "HI_IS_LINE_LIST",
    "HYD_L_DATA",
    "MAINGEN_OPTIONS",
    "MAINGEN_OPT_DESC",
    "NUC_DECAY_DATA",
    "PLT_JH_OPTIONS",
    "PLT_JH_OPT_DESC",
    "PLT_SPEC_OPTIONS",
    "PLT_SPEC_OPT_DESC",
    "RS_XRAY_FLUXES",
    "SOL_ABUND",
    "TWO_PHOT_DATA",
    "WR_F_OPTIONS",
    "WR_F_OPT_DESC",
    "XRAY_PHOT_FITS",
}


class ContractError(Exception):
    """A fail-closed contract error carrying a stable exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_link(path: Path) -> str:
    return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()


def iso_mtime(stat_result: os.stat_result) -> str:
    value = dt.datetime.fromtimestamp(stat_result.st_mtime, dt.timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def classify(path: Path) -> tuple[str, str]:
    """Return a strict role and the matching rule; unknown names stay unknown."""

    name = path.name
    is_directory = path.is_dir() and not path.is_symlink()
    if is_directory and re.fullmatch(r"__pycache__", name):
        return "scratch", "python-bytecode-cache-directory"
    if is_directory and re.fullmatch(
        r"atomic_(?:local|overlay|repairs?)", name, re.IGNORECASE
    ):
        return "oracle-metadata", "run-local-atomic-input-directory"
    if is_directory and re.fullmatch(
        r"(?:seq|batch|run|slurm)[_-]logs?", name, re.IGNORECASE
    ):
        return "run-log", "run-log-directory"
    if name in SCRATCH_EXACT:
        return "scratch", "scratch-exact"
    if re.fullmatch(
        r"(?:CSCRATCH\d+|BA_ASCI_N_D(?:ND|\d+)|BAMAT(?:PNT)?|fort\.\d+)", name
    ):
        return "scratch", "cmfgen-transient-output-convention"
    if name in ORACLE_DATA_EXACT:
        return "oracle-data", "oracle-data-exact"
    if re.fullmatch(r"POP[A-Z0-9_]+", name):
        return "oracle-data", "POP*"
    if re.fullmatch(r"[A-Za-z0-9]+(?:OUT|PRRR)", name):
        return "oracle-data", "ion-OUT-or-PRRR"
    if name.endswith("_INFO"):
        return "oracle-metadata", "*_INFO"
    if name in ORACLE_METADATA_EXACT:
        return "oracle-metadata", "oracle-metadata-exact"
    if name in LINK_EXACT or re.fullmatch(
        r"(?:[A-Za-z0-9]+_(?:COL_DATA|F_OSCDAT|F_TO_S)|PHOT[A-Za-z0-9]+_[AB])",
        name,
    ):
        return "oracle-metadata", "atomic-or-reader-link"
    if re.fullmatch(
        r"PROVENANCE(?:[_-][A-Za-z0-9][A-Za-z0-9_.-]*)?\.(?:txt|md|json)",
        name,
        re.IGNORECASE,
    ):
        return "provenance", "provenance-document-convention"
    if re.fullmatch(
        r"(?:[A-Z0-9]+_)*DIFF_MANIFEST\.(?:txt|json|csv)", name
    ):
        return "provenance", "lineage-diff-manifest-convention"
    if re.fullmatch(
        r"MODEL_SPEC\.(?:base|baseline)(?:[_-]reference)?", name, re.IGNORECASE
    ):
        return "provenance", "ancestral-model-spec-reference"
    if re.fullmatch(
        r"RUNTIME(?:_[A-Z0-9]+)*_ESTIMATE\.(?:txt|json|csv)", name
    ):
        return "oracle-metadata", "run-planning-estimate-document"
    if name.endswith(".py") or name.endswith(".sh") or name.startswith("snia_"):
        return "oracle-metadata", "run-input-or-provenance-source"
    if name in RUN_LOG_EXACT:
        return "run-log", "run-log-exact"
    if re.fullmatch(
        r"PHOT(?:_[A-Z0-9]+)*_PRESCAN\.(?:txt|json|csv)", name
    ) or re.fullmatch(
        r"SIGMA(?:_[A-Z0-9]+)*_CHECK\.(?:txt|json|csv)", name
    ) or re.fullmatch(r"PREFLIGHT(?:_[A-Z0-9]+)*\.(?:txt|json|csv)", name):
        return "run-log", "verification-log-document-convention"
    if re.fullmatch(r"OUTGEN(?:\..+)?", name):
        return "run-log", "OUTGEN*"
    if re.fullmatch(r"run_.+\.info", name) or re.fullmatch(r"batch.*\.log", name):
        return "run-log", "run-or-batch-log"
    return "unclassified", "no-rule"


def object_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "other"


def entry_for(path: Path) -> dict[str, Any]:
    stat_result = path.lstat()
    kind = object_type(path)
    role, rule = classify(path)
    hash_excluded = role == "scratch" or kind in {"directory", "other"}
    if hash_excluded:
        digest = None
        if role == "scratch":
            reason = (
                "transient/cache/scratch object; presence and classification are "
                "sealed but content hashing is excluded"
            )
        elif kind == "directory":
            reason = (
                "directory object; presence and role are sealed, while descendants "
                "are outside the immediate-children inventory scope"
            )
        else:
            reason = "unsupported object type; content hashing is excluded"
    elif kind == "file":
        digest = sha256_file(path)
        reason = None
    elif kind == "symlink":
        digest = sha256_link(path)
        reason = "symlink object hash covers link text only; target is never followed"
    else:
        digest = None
        reason = "unsupported object type"
    return {
        "path": path.name,
        "object_type": kind,
        "role": role,
        "classification_rule": rule,
        "size_bytes": stat_result.st_size,
        "sha256": digest,
        "hash_excluded": hash_excluded,
        "hash_exclusion_reason": reason if hash_excluded else None,
        "hash_semantics": reason if kind == "symlink" else "file-bytes",
        "symlink_target": os.readlink(path) if kind == "symlink" else None,
        "mtime_utc_informational_only": iso_mtime(stat_result),
        "mtime_ns_informational_only": stat_result.st_mtime_ns,
    }


def inventory(root: Path) -> list[dict[str, Any]]:
    children = sorted(root.iterdir(), key=lambda item: os.fsencode(item.name))
    return [entry_for(path) for path in children]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_info(path: Path) -> dict[str, Any]:
    lines = read_lines(path)
    if len(lines) < 4:
        raise ContractError(RC_RECORD_SCHEMA, f"{path.name}: INFO has fewer than 4 lines")
    values = lines[2].split()
    labels = lines[3].split()
    expected_labels = ["ND", "RECL", "WORD_SIZE", "UNIT_SIZE", "INT_SIZE", "LIT_END"]
    if labels != expected_labels or len(values) != 6:
        raise ContractError(
            RC_RECORD_SCHEMA,
            f"{path.name}: expected labels {expected_labels}, got {labels}",
        )
    try:
        nd, recl, word_size, unit_size, int_size = map(int, values[:5])
    except ValueError as exc:
        raise ContractError(RC_RECORD_SCHEMA, f"{path.name}: non-integer INFO field") from exc
    if values[5] not in {"T", "F"}:
        raise ContractError(RC_RECORD_SCHEMA, f"{path.name}: invalid LIT_END={values[5]!r}")
    return {
        "info_format_date": lines[0].split("!", 1)[0].strip(),
        "file_format_date": lines[1].split("!", 1)[0].strip(),
        "nd": nd,
        "recl_units": recl,
        "word_size_bytes": word_size,
        "unit_size_bytes": unit_size,
        "integer_size_bytes": int_size,
        "little_endian": values[5] == "T",
        "record_size_bytes": recl * unit_size,
        "declared_record_count": None,
        "declared_units": None,
        "declared_frame": None,
        "metadata_omissions": ["record_count", "units", "frame", "iteration_id"],
        "source_lines": {"values": 3, "labels": 4},
    }


def outgen_ncf(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise ContractError(RC_RECORD_SCHEMA, "OUTGEN missing: cannot derive EDDFACTOR record count")
    for number, line in enumerate(read_lines(path), 1):
        match = re.search(r"Number of frequencies is:\s*(\d+)", line)
        if match:
            return int(match.group(1)), number
    raise ContractError(RC_RECORD_SCHEMA, "OUTGEN: frequency count not found")


def unpack_int(record: bytes, endian: str, offset: int = 0) -> int:
    return struct.unpack_from(endian + "i", record, offset)[0]


def read_record(stream: Any, record_size: int, number: int) -> bytes:
    stream.seek((number - 1) * record_size)
    value = stream.read(record_size)
    if len(value) != record_size:
        raise ContractError(
            RC_RECORD_SCHEMA,
            f"short direct-access record {number}: {len(value)} != {record_size}",
        )
    return value


def compare_edd_jh(
    root: Path, edd: dict[str, Any], jh: dict[str, Any]
) -> dict[str, Any]:
    if edd["nd"] != jh["nd"]:
        return {"status": "MISMATCH", "reason": "ND differs"}
    if edd["little_endian"] != jh["little_endian"]:
        return {"status": "MISMATCH", "reason": "endianness differs"}
    endian = "<" if edd["little_endian"] else ">"
    nd = edd["nd"]
    ncf = edd["derived_record_count"]["frequency_records"]
    edd_size = edd["record_size_bytes"]
    jh_size = jh["record_size_bytes"]
    freq_mismatch = 0
    rsqj_mismatch = 0
    max_rsqj_relative = 0.0
    with (root / "EDDFACTOR").open("rb") as edd_stream, (
        root / "JH_AT_CURRENT_TIME"
    ).open("rb") as jh_stream:
        rv_pointer_record = read_record(edd_stream, edd_size, 2)
        rv_record = unpack_int(rv_pointer_record, endian)
        edd_r_raw = read_record(edd_stream, edd_size, rv_record)[: nd * 8]
        edd_v_raw = read_record(edd_stream, edd_size, rv_record + 1)[: nd * 8]
        jh_rv = read_record(jh_stream, jh_size, jh["jh_header"]["start_record"])
        jh_r_raw = jh_rv[: nd * 8]
        jh_v_raw = jh_rv[nd * 8 : 2 * nd * 8]
        radii = struct.unpack(endian + f"{nd}d", jh_r_raw)
        r_grid_exact = edd_r_raw == jh_r_raw
        v_grid_exact = edd_v_raw == jh_v_raw
        for index in range(ncf):
            edd_record = read_record(edd_stream, edd_size, 15 + index)
            jh_record = read_record(
                jh_stream, jh_size, jh["jh_header"]["start_record"] + 2 + index
            )
            edd_freq = edd_record[nd * 8 : (nd + 1) * 8]
            jh_freq = jh_record[(2 * nd + 1) * 8 : (2 * nd + 2) * 8]
            if edd_freq != jh_freq:
                freq_mismatch += 1
            edd_j = struct.unpack_from(endian + f"{nd}d", edd_record, 0)
            jh_rsqj = struct.unpack_from(endian + f"{nd}d", jh_record, 0)
            for depth in range(nd):
                reconstructed = edd_j[depth] * radii[depth] * radii[depth]
                expected = jh_rsqj[depth]
                scale = max(abs(expected), abs(reconstructed), sys.float_info.min)
                relative = abs(reconstructed - expected) / scale
                max_rsqj_relative = max(max_rsqj_relative, relative)
                if relative > 8.0e-15:
                    rsqj_mismatch += 1
    status = (
        "MATCH"
        if r_grid_exact and v_grid_exact and freq_mismatch == 0 and rsqj_mismatch == 0
        else "MISMATCH"
    )
    return {
        "status": status,
        "comparison_basis": (
            "full content: R/V bytes, every frequency byte, and every "
            "JH.RSQ_J versus EDD.J*R^2 value"
        ),
        "r_grid_bitwise_equal": r_grid_exact,
        "v_grid_bitwise_equal": v_grid_exact,
        "frequency_records_compared": ncf,
        "frequency_bit_mismatches": freq_mismatch,
        "rsqj_values_compared": ncf * nd,
        "rsqj_mismatches_over_8e-15": rsqj_mismatch,
        "max_rsqj_relative_error": max_rsqj_relative,
    }


def record_schemas(root: Path, scan_generation: bool = True) -> dict[str, Any]:
    edd_info = parse_info(root / "EDDFACTOR_INFO")
    jh_info = parse_info(root / "JH_AT_CURRENT_TIME_INFO")
    if edd_info["record_size_bytes"] != edd_info["word_size_bytes"] * (
        edd_info["nd"] + 1
    ):
        raise ContractError(RC_RECORD_SCHEMA, "EDDFACTOR_INFO: RECL != WORD_SIZE*(ND+1)")
    if jh_info["record_size_bytes"] != jh_info["word_size_bytes"] * (
        2 * jh_info["nd"] + 2
    ):
        raise ContractError(
            RC_RECORD_SCHEMA, "JH_AT_CURRENT_TIME_INFO: RECL != WORD_SIZE*(2*ND+2)"
        )
    if edd_info["word_size_bytes"] != 8 or jh_info["word_size_bytes"] != 8:
        raise ContractError(RC_RECORD_SCHEMA, "only CMFGEN 8-byte real records are supported")
    ncf, ncf_line = outgen_ncf(root / "OUTGEN")
    edd_records = 14 + ncf
    edd_expected = edd_records * edd_info["record_size_bytes"]
    edd_actual = (root / "EDDFACTOR").stat().st_size
    if edd_actual != edd_expected:
        raise ContractError(
            RC_RECORD_SCHEMA,
            f"EDDFACTOR size/schema mismatch: {edd_actual} != {edd_expected}",
        )
    endian = "<" if edd_info["little_endian"] else ">"
    with (root / "EDDFACTOR").open("rb") as stream:
        finish_raw = read_record(stream, edd_info["record_size_bytes"], 5)[:8]
    finish_value = struct.unpack(endian + "d", finish_raw)[0]
    if finish_value != 1.0:
        raise ContractError(
            RC_RECORD_SCHEMA, f"EDDFACTOR FINISH_REC=5 is {finish_value!r}, expected 1.0"
        )
    jh_endian = "<" if jh_info["little_endian"] else ">"
    with (root / "JH_AT_CURRENT_TIME").open("rb") as stream:
        header = read_record(stream, jh_info["record_size_bytes"], 3)
    start_record, jh_ncf, jh_nd = struct.unpack_from(jh_endian + "3i", header, 0)
    if (start_record, jh_nd) != (6, jh_info["nd"]):
        raise ContractError(
            RC_RECORD_SCHEMA,
            f"JH record 3 invalid: start={start_record}, ND={jh_nd}",
        )
    if jh_ncf != ncf:
        raise ContractError(
            RC_RECORD_SCHEMA, f"JH NCF {jh_ncf} != OUTGEN NCF {ncf}"
        )
    jh_records = start_record + 1 + jh_ncf
    jh_expected = jh_records * jh_info["record_size_bytes"]
    jh_actual = (root / "JH_AT_CURRENT_TIME").stat().st_size
    if jh_actual != jh_expected:
        raise ContractError(
            RC_RECORD_SCHEMA,
            f"JH_AT_CURRENT_TIME size/schema mismatch: {jh_actual} != {jh_expected}",
        )
    edd_info.update(
        {
            "data_file": "EDDFACTOR",
            "writer_source": {
                "record_constants": "new_main/mod_subs/eddfac_rec_defs_mod.f:8-15",
                "recl_and_info": "new_main/subs/open_rw_eddfactor.f:58,112",
                "payload_write": "new_main/mod_subs/comp_j_blank.f:808",
                "finish_write": "new_main/cmfgen_sub.f:3931",
            },
            "record_layout": "records 1-14 control/grid; records 15.. contain ND J values + FL",
            "derived_record_count": {
                "total_records": edd_records,
                "frequency_records": ncf,
                "derivation": "14 + OUTGEN.Number_of_frequencies",
                "outgen_line": ncf_line,
            },
            "actual_size_bytes": edd_actual,
            "expected_size_bytes": edd_expected,
            "size_matches_schema": True,
            "units": {
                "status": "NOT_DECLARED_BY_INFO",
                "source_derived_frequency": "FL in 1e15 Hz",
                "source_derived_payload": "CMFGEN comoving mean intensity J",
            },
            "frame": {"status": "NOT_DECLARED_BY_INFO", "source_derived": "comoving"},
            "finish_rec": {
                "record_number": 5,
                "value": finish_value,
                "status": "COMPLETE",
                "semantics": "file-complete-only-not-physical-convergence",
            },
        }
    )
    jh_info.update(
        {
            "data_file": "JH_AT_CURRENT_TIME",
            "writer_source": {
                "recl_and_info": "plane/out_jh.f:47-49",
                "header_and_grid": "plane/out_jh.f:65-67,86-88",
                "payload_write": "plane/out_jh.f:94",
            },
            "record_layout": (
                "records 1-5 control; record 6 R,V; record 7 integrated J,H; "
                "records 8.. contain R^2J, R^2H, boundaries, NU"
            ),
            "jh_header": {
                "record_number": 3,
                "start_record": start_record,
                "frequency_records": jh_ncf,
                "nd": jh_nd,
            },
            "derived_record_count": {
                "total_records": jh_records,
                "frequency_records": jh_ncf,
                "derivation": "record3.start_record + 1 + record3.NCF",
            },
            "actual_size_bytes": jh_actual,
            "expected_size_bytes": jh_expected,
            "size_matches_schema": True,
            "units": {
                "status": "NOT_DECLARED_BY_INFO",
                "source_derived_frequency": "NU in 1e15 Hz",
                "source_derived_payload": "R^2J, R^2H and boundary H values",
            },
            "frame": {"status": "NOT_DECLARED_BY_INFO", "source_derived": "comoving"},
            "finish_rec": {
                "status": "NOT_PRESENT_IN_FORMAT",
                "semantics": "no JH FINISH_REC is declared by out_jh.f",
            },
        }
    )
    result: dict[str, Any] = {
        "EDDFACTOR": edd_info,
        "JH_AT_CURRENT_TIME": jh_info,
    }
    if scan_generation:
        result["cross_file_content_check"] = compare_edd_jh(root, edd_info, jh_info)
    return result


def parse_float_tokens(line: str) -> list[float]:
    values: list[float] = []
    for token in FLOAT_RE.findall(line):
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return values


def header_value(lines: Iterable[str], label: str) -> tuple[str | None, int | None]:
    for number, line in enumerate(lines, 1):
        if label in line:
            return line.split(label, 1)[1].strip(), number
    return None, None


def named_vector(lines: list[str], label: str, count: int) -> list[float]:
    for index, line in enumerate(lines):
        if line.strip() == label:
            values: list[float] = []
            for following in lines[index + 1 :]:
                found = parse_float_tokens(following)
                if not found:
                    if values:
                        break
                    continue
                values.extend(found)
                if len(values) >= count:
                    return values[:count]
    return []


def repeated_block_vector(lines: list[str], prefix: str) -> list[float]:
    values: list[float] = []
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            for following in lines[index + 1 :]:
                found = parse_float_tokens(following)
                if found:
                    values.extend(found)
                    break
    return values


def close_vectors(left: list[float], right: list[float], relative: float) -> dict[str, Any]:
    if not left or len(left) != len(right):
        return {
            "status": "MISMATCH",
            "left_count": len(left),
            "right_count": len(right),
            "reason": "missing or unequal vector length",
        }
    mismatches = 0
    maximum = 0.0
    for a, b in zip(left, right):
        scale = max(abs(a), abs(b), sys.float_info.min)
        error = abs(a - b) / scale
        maximum = max(maximum, error)
        if error > relative:
            mismatches += 1
    return {
        "status": "MATCH" if mismatches == 0 else "MISMATCH",
        "values_compared": len(left),
        "relative_tolerance_from_declared_print_precision": relative,
        "mismatches": mismatches,
        "max_relative_error": maximum,
    }


def parse_rvtj(path: Path) -> dict[str, Any]:
    lines = read_lines(path)
    nd_raw, nd_line = header_value(lines, "ND:")
    if nd_raw is None:
        return {"status": "UNREADABLE", "reason": "ND header absent"}
    match = re.search(r"\d+", nd_raw)
    if not match:
        return {"status": "UNREADABLE", "reason": "ND header malformed"}
    nd = int(match.group())
    completion, completion_line = header_value(lines, "Completion of Model:")
    return {
        "status": "PARSED",
        "nd": nd,
        "nd_line": nd_line,
        "completion_token": completion,
        "completion_line": completion_line,
        "radius": named_vector(lines, "Radius (10^10 cm)", nd),
        "velocity": named_vector(lines, "Velocity (km/s)", nd),
        "temperature": named_vector(lines, "Temperature (10^4K)", nd),
        "electron_density": named_vector(lines, "Electron density", nd),
    }


def parse_pop_header(path: Path) -> dict[str, Any]:
    completion = None
    completion_line = None
    nd = None
    nd_line = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if "Completion of Model:" in line:
                completion = line.split("Completion of Model:", 1)[1].strip()
                completion_line = number
            if re.search(r"\bND:\s*\d+", line):
                nd = int(re.search(r"\bND:\s*(\d+)", line).group(1))  # type: ignore[union-attr]
                nd_line = number
            if number >= 12:
                break
    return {
        "completion_token": completion,
        "completion_line": completion_line,
        "nd": nd,
        "nd_line": nd_line,
    }


def parse_obs_pair(root: Path) -> dict[str, Any]:
    obsflux_lines = read_lines(root / "OBSFLUX")
    count = None
    start = None
    for index, line in enumerate(obsflux_lines):
        match = re.search(r"Continuum Frequencies\s*\(\s*(\d+)\s*\)", line)
        if match:
            count = int(match.group(1))
            start = index + 1
            break
    if count is None or start is None:
        return {"status": "MISMATCH", "reason": "OBSFLUX frequency header absent"}
    flux_freq: list[float] = []
    for line in obsflux_lines[start:]:
        found = parse_float_tokens(line)
        if found:
            flux_freq.extend(found)
            if len(flux_freq) >= count:
                break
    standalone: list[float] = []
    for line in read_lines(root / "OBS_FREQ"):
        found = parse_float_tokens(line)
        if len(found) >= 2:
            standalone.append(found[0])
    comparison = close_vectors(flux_freq[: len(standalone)], standalone, 1.1e-6)
    comparison.update(
        {
            "obsflux_declared_frequency_count": count,
            "obsfreq_rows": len(standalone),
            "expected_obsfreq_rows": max(0, count - 1),
        }
    )
    if len(standalone) != max(0, count - 1):
        comparison["status"] = "MISMATCH"
    return comparison


def load_attestation(root: Path) -> dict[str, Any] | None:
    path = root / ATTESTATION
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(RC_OPHYS, f"invalid {ATTESTATION}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(RC_OPHYS, f"{ATTESTATION}: top level must be an object")
    return value


def generation_evidence(root: Path, schemas: dict[str, Any]) -> dict[str, Any]:
    targets: list[str] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        name = child.name
        if (
            name
            in {
                "EDDFACTOR",
                "JH_AT_CURRENT_TIME",
                "RVTJ",
                "OBSFLUX",
                "OBS_FREQ",
                "GENCOOL",
            }
            or re.fullmatch(r"POP[A-Z0-9_]+", name)
            or re.fullmatch(r"[A-Za-z0-9]+PRRR", name)
        ):
            targets.append(name)
    evidence: dict[str, Any] = {
        "target_files": targets,
        "mtime_used_for_verdict": False,
        "content_links": {},
        "unresolved_links": [],
    }
    core = schemas.get("cross_file_content_check", {})
    evidence["content_links"]["EDDFACTOR<->JH_AT_CURRENT_TIME"] = core
    rv = parse_rvtj(root / "RVTJ") if (root / "RVTJ").is_file() else None
    mixed_reasons: list[str] = []
    if core.get("status") == "MISMATCH":
        mixed_reasons.append("EDDFACTOR/JH shared content differs")
    completion_tokens: dict[str, Any] = {}
    if rv:
        completion_tokens["RVTJ"] = rv.get("completion_token")
        pop_checks: dict[str, Any] = {}
        for path in sorted(root.glob("POP*")):
            if not path.is_file():
                continue
            parsed = parse_pop_header(path)
            completion_tokens[path.name] = parsed["completion_token"]
            pop_checks[path.name] = {
                "nd": parsed["nd"],
                "nd_matches_rvtj": parsed["nd"] == rv["nd"],
                "completion_token": parsed["completion_token"],
                "completion_matches_rvtj": parsed["completion_token"]
                == rv["completion_token"],
            }
            if parsed["completion_token"] not in {None, rv["completion_token"]}:
                mixed_reasons.append(f"{path.name}/RVTJ completion tokens differ")
        evidence["content_links"]["RVTJ<->POP*"] = pop_checks
        prrr_checks: dict[str, Any] = {}
        for path in sorted(root.glob("*PRRR")):
            lines = read_lines(path)
            checks = {
                "radius": close_vectors(
                    rv["radius"], repeated_block_vector(lines, "Radius ["), 6.0e-5
                ),
                "temperature": close_vectors(
                    rv["temperature"],
                    repeated_block_vector(lines, "Temperature ["),
                    6.0e-5,
                ),
                "electron_density": close_vectors(
                    rv["electron_density"],
                    repeated_block_vector(lines, "Electron Density"),
                    6.0e-5,
                ),
            }
            prrr_checks[path.name] = checks
            if any(item["status"] == "MISMATCH" for item in checks.values()):
                mixed_reasons.append(f"{path.name}/RVTJ shared state differs")
        evidence["content_links"]["RVTJ<->*PRRR"] = prrr_checks
        if (root / "GENCOOL").is_file():
            lines = read_lines(root / "GENCOOL")
            checks = {
                "radius": close_vectors(
                    rv["radius"], repeated_block_vector(lines, "Radius ["), 6.0e-5
                ),
                "velocity": close_vectors(
                    rv["velocity"], repeated_block_vector(lines, "Velocity ["), 6.0e-5
                ),
                "temperature": close_vectors(
                    rv["temperature"],
                    repeated_block_vector(lines, "Temperature ["),
                    6.0e-5,
                ),
                "electron_density": close_vectors(
                    rv["electron_density"],
                    repeated_block_vector(lines, "Electron Density"),
                    6.0e-5,
                ),
            }
            evidence["content_links"]["RVTJ<->GENCOOL"] = checks
            if any(item["status"] == "MISMATCH" for item in checks.values()):
                mixed_reasons.append("GENCOOL/RVTJ shared state differs")
    if (root / "OBSFLUX").is_file() and (root / "OBS_FREQ").is_file():
        obs_check = parse_obs_pair(root)
        evidence["content_links"]["OBSFLUX<->OBS_FREQ"] = obs_check
        if obs_check.get("status") == "MISMATCH":
            mixed_reasons.append("OBSFLUX/OBS_FREQ grids differ")
    evidence["completion_tokens_from_file_content"] = completion_tokens
    attestation = load_attestation(root)
    if attestation:
        proof = attestation.get("generation_proof", {})
        declared_files = set(proof.get("files", [])) if isinstance(proof, dict) else set()
        if (
            isinstance(proof, dict)
            and not mixed_reasons
            and proof.get("verdict") == "SAME_GENERATION_PROVEN"
            and proof.get("evidence_kind") == "content"
            and proof.get("output_after_last_iteration") is True
            and set(targets).issubset(declared_files)
            and proof.get("iteration_id") not in {None, ""}
        ):
            evidence["verdict"] = "SAME_GENERATION_PROVEN"
            evidence["verdict_basis"] = (
                "machine attestation supplies one content-derived iteration ID for every "
                "target, and all independently checkable shared content agrees"
            )
            return evidence
    if mixed_reasons:
        evidence["verdict"] = "MIXED_GENERATION_PROVEN"
        evidence["verdict_basis"] = "; ".join(sorted(set(mixed_reasons)))
        return evidence
    evidence["unresolved_links"].append(
        "OBSFLUX/OBS_FREQ expose a shared observer grid but no iteration ID or "
        "iteration-specific quantity shared with EDD/JH/RVTJ/POP/PRRR/GENCOOL"
    )
    evidence["unresolved_links"].append(
        "the two *_INFO formats do not declare iteration ID, units, frame, or NCF"
    )
    evidence["verdict"] = "UNDECIDABLE_WITH_CURRENT_EVIDENCE"
    evidence["verdict_basis"] = (
        "available content establishes several internally consistent groups but does not "
        "bind every requested file to one CMFGEN great iteration"
    )
    return evidence


def find_line(lines: list[str], pattern: str) -> tuple[int | None, str | None]:
    regex = re.compile(pattern)
    for number, line in enumerate(lines, 1):
        if regex.search(line):
            return number, line.strip()
    return None, None


def qualification(root: Path, generation: dict[str, Any]) -> dict[str, Any]:
    run_files = sorted(root.glob("run_*.info"))
    run_path = next((path for path in run_files if path.name == "run_jnu4.info"), None)
    if run_path is None and run_files:
        run_path = run_files[-1]
    run_lines = read_lines(run_path) if run_path else []
    out_lines = read_lines(root / "OUTGEN") if (root / "OUTGEN").is_file() else []
    fix_line, fix_text = find_line(run_lines, r"\[FIX_T\]")
    restart_line, restart_text = find_line(run_lines, r"(?:restart record|POINT1:)")
    num_line, num_text = find_line(run_lines, r"\[NUM_ITS\]")
    lambda_line, lambda_text = find_line(run_lines, r"\[DO_LAM_IT\]")
    plan_line, plan_text = find_line(run_lines, r"it67 NaN")
    iterations: list[tuple[int, int]] = []
    corrections: list[tuple[int, float, int]] = []
    for number, line in enumerate(out_lines, 1):
        match = re.search(r"Current great iteration count is\s+(\d+)", line)
        if match:
            iterations.append((int(match.group(1)), number))
        match = re.search(
            r"Maximum % increase.*?([0-9.]+E[+-]\d+).*iteration\s+(\d+)", line
        )
        if match:
            corrections.append((int(match.group(2)), float(match.group(1)), number))
    nonlinear = "UNDECIDABLE_WITH_CURRENT_EVIDENCE"
    nonlinear_reason = "no measured convergence discriminator found"
    if corrections and max(value for _, value, _ in corrections[-3:]) > 1.0:
        nonlinear = "FAIL"
        nonlinear_reason = (
            f"OUTGEN lines {','.join(str(line) for _, _, line in corrections[-3:])}: "
            f"last-three maximum population increases are "
            f"{[value for _, value, _ in corrections[-3:]]}%, above 1%"
        )
    physical = "INELIGIBLE"
    return {
        "CMFGEN_FILE_INTEGRITY": {
            "value": "PASS",
            "reason": "all classified hashed entries and both direct-access size/schema checks completed",
        },
        "CMFGEN_SNAPSHOT_REPLAY": {
            "value": "ELIGIBLE",
            "reason": (
                "EDDFACTOR FINISH_REC=1 and exact record-size checks permit file replay; "
                "this does not authorize cross-file physics gates"
            ),
        },
        "CMFGEN_NONLINEAR_CONVERGENCE": {
            "value": nonlinear,
            "reason": nonlinear_reason,
        },
        "CMFGEN_PHYSICAL_ORACLE": {
            "value": physical,
            "reason": (
                f"nonlinear={nonlinear}; generation={generation['verdict']}; "
                f"FIX_T evidence={run_path.name if run_path else 'missing'}:{fix_line} {fix_text}"
            ),
        },
        "remeasured_run_evidence": {
            "run_file": run_path.name if run_path else None,
            "fix_t": {"line": fix_line, "text": fix_text},
            "restart_record_62": {"line": restart_line, "text": restart_text},
            "num_iterations_4": {"line": num_line, "text": num_text},
            "pure_lambda": {"line": lambda_line, "text": lambda_text},
            "stopped_before_iteration_67_nan": {"line": plan_line, "text": plan_text},
            "outgen_iterations": iterations,
            "outgen_maximum_increases": corrections,
        },
    }


def profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "A2_00_OPHYS_PROFILE.json"


def load_profile(path: Path | None) -> dict[str, Any]:
    selected = path or profile_path()
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(RC_MANIFEST, f"invalid profile {selected}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(RC_MANIFEST, f"profile {selected}: top level must be object")
    return value


def ophys_gaps(
    root: Path, manifest: dict[str, Any], profile: dict[str, Any]
) -> list[str]:
    names = {entry["path"] for entry in manifest["entries"]}
    gaps: list[str] = []
    for name in profile["required_exact_files"]:
        if name not in names:
            gaps.append(f"MISSING_REQUIRED_FILE:{name}")
    for rule in profile["required_patterns"]:
        matches = sorted(name for name in names if fnmatch.fnmatchcase(name, rule["glob"]))
        if len(matches) < int(rule["minimum_count"]):
            gaps.append(
                f"PATTERN_COVERAGE:{rule['glob']}:{len(matches)}<{rule['minimum_count']}"
            )
    for name in sorted(name for name in names if name.endswith("PRRR")):
        partner = name[:-4] + "OUT"
        if partner not in names:
            gaps.append(f"MISSING_MATCHING_OUT:{name}->{partner}")
    attestation = load_attestation(root)
    if attestation is None:
        gaps.append(f"MISSING_ATTESTATION:{ATTESTATION}")
        return gaps
    required_top = profile["attestation_required_fields"]
    for dotted in required_top:
        cursor: Any = attestation
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                gaps.append(f"ATTESTATION_FIELD_MISSING:{dotted}")
                break
            cursor = cursor[part]
    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    revision = attestation.get("code_revision")
    if not isinstance(revision, str) or not revision.strip():
        gaps.append("CODE_REVISION_EMPTY")
    input_files = attestation.get("input_files", {})
    if not isinstance(input_files, dict):
        gaps.append("INPUT_HASH_MAP_INVALID")
    else:
        for name in profile.get("required_input_hashes", []):
            digest = input_files.get(name)
            if not isinstance(digest, str) or not hash_pattern.fullmatch(digest):
                gaps.append(f"INPUT_HASH_INVALID:{name}")
    atomic_data = attestation.get("atomic_data")
    atomic_hashes: list[Any] = []
    if isinstance(atomic_data, dict):
        atomic_hashes = list(atomic_data.values())
    elif isinstance(atomic_data, list):
        atomic_hashes = [
            item.get("sha256") for item in atomic_data if isinstance(item, dict)
        ]
    if not atomic_hashes or any(
        not isinstance(digest, str) or not hash_pattern.fullmatch(digest)
        for digest in atomic_hashes
    ):
        gaps.append("ATOMIC_DATA_HASHES_INVALID")
    convergence = attestation.get("convergence", {})
    if isinstance(convergence, dict):
        thresholds = profile["thresholds"]
        for field in ["jnu_last3_max_fraction", "te_last3_max_fraction", "ion_last3_max_fraction"]:
            values = convergence.get(field, [])
            if not isinstance(values, list) or len(values) != 3 or any(
                not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item > thresholds["last3_max_fraction"]
                for item in values
            ):
                gaps.append(f"CONVERGENCE_THRESHOLD:{field}")
        for field, limit in [
            ("active_population_max_correction_fraction", thresholds["population_max_fraction"]),
            ("max_normalized_heat_residual", thresholds["heat_residual_max"]),
        ]:
            value = convergence.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
                gaps.append(f"CONVERGENCE_THRESHOLD:{field}")
        if convergence.get("nan_count") != 0 or convergence.get("inf_count") != 0:
            gaps.append("NONFINITE_VALUES_PRESENT")
    run = attestation.get("run", {})
    if not isinstance(run, dict) or run.get("fix_t") is not False or run.get("temperature_solved") is not True:
        gaps.append("TEMPERATURE_NOT_SOLVED")
    freezes = attestation.get("freezes", {})
    if not isinstance(freezes, dict) or freezes.get("undisclosed_count") != 0:
        gaps.append("UNDISCLOSED_FREEZE")
    elif isinstance(freezes.get("components"), list):
        for index, component in enumerate(freezes["components"]):
            if not isinstance(component, dict) or any(
                field not in component
                for field in [
                    "name",
                    "population_fraction",
                    "rate_fraction",
                    "opacity_fraction",
                    "emissivity_fraction",
                ]
            ):
                gaps.append(f"FREEZE_DISCLOSURE_INCOMPLETE:{index}")
    coverage = attestation.get("coverage", {})
    stems = coverage.get("matching_ion_stems", []) if isinstance(coverage, dict) else []
    if not isinstance(stems, list) or not stems:
        gaps.append("MATCHING_ION_STEMS_EMPTY")
    else:
        for stem in stems:
            if not isinstance(stem, str) or not stem:
                gaps.append("MATCHING_ION_STEM_INVALID")
                continue
            for suffix in ["PRRR", "OUT"]:
                if stem + suffix not in names:
                    gaps.append(f"MATCHING_ION_FILE_MISSING:{stem + suffix}")
    rate_audit = attestation.get("rate_audit", {})
    if not isinstance(rate_audit, dict) or rate_audit.get("upward_downward_separated") is not True:
        gaps.append("RATE_AUDIT_COMPONENTS_NOT_SEPARATED")
    proof = attestation.get("generation_proof", {})
    generation_targets = {
        name
        for name in names
        if name in {
            "EDDFACTOR",
            "JH_AT_CURRENT_TIME",
            "RVTJ",
            "OBSFLUX",
            "OBS_FREQ",
            "GENCOOL",
        }
        or re.fullmatch(r"POP[A-Z0-9_]+", name)
        or re.fullmatch(r"[A-Za-z0-9]+PRRR", name)
    }
    proof_files = set(proof.get("files", [])) if isinstance(proof, dict) else set()
    if (
        not isinstance(proof, dict)
        or proof.get("verdict") != "SAME_GENERATION_PROVEN"
        or proof.get("evidence_kind") != "content"
        or proof.get("output_after_last_iteration") is not True
        or proof.get("iteration_id") in {None, ""}
        or not generation_targets.issubset(proof_files)
        or manifest.get("generation_consistency", {}).get("verdict")
        != "SAME_GENERATION_PROVEN"
    ):
        gaps.append("GENERATION_NOT_PROVEN")
    for data_name in ["EDDFACTOR", "JH_AT_CURRENT_TIME", "CHI_DATA", "ETA_DATA"]:
        declared = attestation.get("record_schemas", {}).get(data_name, {})
        if not isinstance(declared, dict) or not declared.get("units") or not declared.get("frame"):
            gaps.append(f"UNIT_FRAME_NOT_DECLARED:{data_name}")
    return gaps


def write_manifest(
    root: Path, target: Path, profile_name: str, scan_generation: bool
) -> dict[str, Any]:
    root = root.resolve()
    target = target.resolve()
    if not root.is_dir():
        raise ContractError(RC_INVENTORY, f"not a directory: {root}")
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise ContractError(RC_MANIFEST, "manifest must be outside the sealed oracle directory")
    entries = inventory(root)
    unknown = [entry["path"] for entry in entries if entry["role"] == "unclassified"]
    if unknown:
        raise ContractError(RC_UNCLASSIFIED, f"unclassified entries: {unknown}")
    schemas = record_schemas(root, scan_generation=scan_generation)
    generation = generation_evidence(root, schemas)
    role_counts = {role: 0 for role in sorted(ROLES)}
    for entry in entries:
        role_counts[entry["role"]] += 1
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root_informational_only": str(root),
        "scope": {
            "kind": "immediate-children-only",
            "entry_count": len(entries),
            "directory_descendants": "not traversed; classified directories are explicit exclusions",
        },
        "mtime_policy": "recorded-informational-only-never-compared-for-pass-fail",
        "symlink_policy": "hash-link-text-never-follow-target",
        "requested_profile_at_write": profile_name,
        "role_counts": role_counts,
        "hash_target_paths": [entry["path"] for entry in entries if not entry["hash_excluded"]],
        "hash_exclusions": [
            {"path": entry["path"], "reason": entry["hash_exclusion_reason"]}
            for entry in entries
            if entry["hash_excluded"]
        ],
        "entries": entries,
        "record_schemas": schemas,
        "finish_rec_contract": {
            "value": schemas["EDDFACTOR"]["finish_rec"]["value"],
            "status": "FILE_COMPLETE",
            "is_physical_convergence": False,
            "statement": "FINISH_REC proves completed EDDFACTOR writes, not nonlinear convergence",
        },
        "generation_consistency": generation,
        "eligibility": qualification(root, generation),
    }
    if profile_name == "ophys":
        manifest["ophys_gaps_at_write"] = ophys_gaps(root, manifest, load_profile(None))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(RC_MANIFEST, f"invalid manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ContractError(RC_MANIFEST, f"manifest schema is not {SCHEMA}")
    return value


def check_manifest(
    root: Path, manifest_path: Path, profile_name: str, profile_file: Path | None
) -> tuple[int, list[str]]:
    root = root.resolve()
    manifest = read_manifest(manifest_path.resolve())
    expected = {entry["path"]: entry for entry in manifest.get("entries", [])}
    actual_entries = inventory(root)
    actual = {entry["path"]: entry for entry in actual_entries}
    messages: list[str] = []
    codes: list[int] = []
    for name in sorted(set(expected) - set(actual)):
        messages.append(f"ERROR MISSING_PATH {name}")
        codes.append(RC_INVENTORY)
    for name in sorted(set(actual) - set(expected)):
        if actual[name]["role"] == "unclassified":
            messages.append(f"ERROR UNCLASSIFIED_EXTRA {name}")
            codes.append(RC_UNCLASSIFIED)
        else:
            messages.append(f"ERROR EXTRA_PATH {name} role={actual[name]['role']}")
            codes.append(RC_INVENTORY)
    for name in sorted(set(expected) & set(actual)):
        old = expected[name]
        new = actual[name]
        if new["role"] == "unclassified":
            messages.append(f"ERROR UNCLASSIFIED {name}")
            codes.append(RC_UNCLASSIFIED)
        for field in ["object_type", "role", "symlink_target"]:
            if old.get(field) != new.get(field):
                messages.append(
                    f"ERROR ENTRY_FIELD {name} {field}={new.get(field)!r} expected={old.get(field)!r}"
                )
                codes.append(RC_INVENTORY)
        if old.get("size_bytes") != new.get("size_bytes"):
            messages.append(
                f"ERROR SIZE_MISMATCH {name} {new.get('size_bytes')} != {old.get('size_bytes')}"
            )
            codes.append(RC_SIZE)
        if not old.get("hash_excluded") and old.get("sha256") != new.get("sha256"):
            messages.append(f"ERROR HASH_MISMATCH {name}")
            codes.append(RC_HASH)
        if old.get("mtime_ns_informational_only") != new.get("mtime_ns_informational_only"):
            messages.append(f"INFO MTIME_CHANGED_IGNORED {name}")
    try:
        current_schemas = record_schemas(root, scan_generation=False)
        for name in ["EDDFACTOR", "JH_AT_CURRENT_TIME"]:
            old_schema = manifest["record_schemas"][name]
            new_schema = current_schemas[name]
            for field in ["nd", "record_size_bytes", "expected_size_bytes"]:
                if old_schema.get(field) != new_schema.get(field):
                    messages.append(
                        f"ERROR RECORD_SCHEMA_MISMATCH {name}.{field} "
                        f"{new_schema.get(field)} != {old_schema.get(field)}"
                    )
                    codes.append(RC_RECORD_SCHEMA)
    except ContractError as exc:
        messages.append(f"ERROR RECORD_SCHEMA_FATAL {exc}")
        codes.append(RC_RECORD_SCHEMA)
    if profile_name == "ophys":
        profile = load_profile(profile_file)
        gaps = ophys_gaps(root, {**manifest, "entries": actual_entries}, profile)
        for gap in gaps:
            messages.append(f"ERROR OPHYS_GAP {gap}")
        if gaps:
            codes.append(RC_OPHYS)
    if not codes:
        messages.append(
            f"PASS CMFGEN_ORACLE_CONTRACT entries={len(actual_entries)} "
            f"unclassified={sum(item['role'] == 'unclassified' for item in actual_entries)} "
            f"role_counts={json.dumps(manifest['role_counts'], sort_keys=True, separators=(',', ':'))} "
            f"profile={profile_name} mtime_used=false"
        )
        return RC_OK, messages
    priority = [RC_UNCLASSIFIED, RC_RECORD_SCHEMA, RC_OPHYS, RC_INVENTORY, RC_SIZE, RC_HASH]
    code = next(candidate for candidate in priority if candidate in codes)
    messages.append(f"FAIL CMFGEN_ORACLE_CONTRACT exit_code={code}")
    return code, messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="write/check the fail-closed A2-00 CMFGEN oracle manifest"
    )
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("snapshot", "ophys"), default="snapshot")
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument(
        "--no-generation-scan",
        action="store_true",
        help="write-only test aid; production A2-00 commands must not use this",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "write":
            manifest = write_manifest(
                args.root,
                args.manifest,
                args.profile,
                scan_generation=not args.no_generation_scan,
            )
            print(
                f"WRITE CMFGEN_ORACLE_CONTRACT path={args.manifest} "
                f"entries={manifest['scope']['entry_count']} "
                f"unclassified={manifest['role_counts']['unclassified']} "
                f"role_counts={json.dumps(manifest['role_counts'], sort_keys=True, separators=(',', ':'))} "
                f"generation={manifest['generation_consistency']['verdict']}"
            )
            if args.profile == "ophys" and manifest.get("ophys_gaps_at_write"):
                for gap in manifest["ophys_gaps_at_write"]:
                    print(f"ERROR OPHYS_GAP {gap}")
                return RC_OPHYS
            return RC_OK
        code, messages = check_manifest(
            args.root, args.manifest, args.profile, args.profile_file
        )
        print("\n".join(messages))
        return code
    except ContractError as exc:
        print(f"FATAL CMFGEN_ORACLE_CONTRACT {exc}", file=sys.stderr)
        return exc.code
    except (OSError, ValueError, struct.error) as exc:
        print(f"FATAL CMFGEN_ORACLE_CONTRACT unexpected: {exc}", file=sys.stderr)
        return RC_MANIFEST


if __name__ == "__main__":
    raise SystemExit(main())
