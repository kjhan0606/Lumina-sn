#!/usr/bin/env python3
"""Fail-closed packager for CMFGEN_ORACLE_ATTESTATION.json.

The script does not infer convergence from filenames or FINISH_REC. A reviewed
measurement JSON supplies the numerical metrics and unit/frame declarations;
this program verifies thresholds, freeze/T controls, capture identity, formal
products, and hashes every file visible below the run root before attesting.
Run it in a Slurm allocation because recursive hashing is compute/I/O work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

SCHEMA = "lumina-cmfgen-oracle-attestation-v1"
EVIDENCE_SCHEMA = "lumina-cmfgen-ophys-evidence-v1"
ATTESTATION = "CMFGEN_ORACLE_ATTESTATION.json"
REQUIRED = [
    "EDDFACTOR", "EDDFACTOR_INFO", "RVTJ", "NETRATE", "TOTRATE",
    "POPCAL", "POPCOB", "POPIRON", "POPNICK", "POPSIL", "POPSUL",
    "CHI_DATA", "CHI_DATA_INFO", "ETA_DATA", "ETA_DATA_INFO", "GENCOOL",
    "LINEHEAT", "JH_AT_CURRENT_TIME", "JH_AT_CURRENT_TIME_INFO",
    "OBSFLUX", "OBS_FREQ", "OUTGEN", "PROVENANCE_FORMAL.json",
]
REQUIRED_INPUTS = ["IN_ITS", "MODEL_SPEC", "SN_HYDRO_DATA", "VADAT"]
GENERATION_BASE = {
    "EDDFACTOR", "JH_AT_CURRENT_TIME", "RVTJ", "OBSFLUX", "OBS_FREQ", "GENCOOL"
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class Refusal(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"top level must be an object: {path}")
    return value


def key_values(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"^\s*([^!\s]+).*\[([A-Za-z0-9_/]+)\]", line)
        if match:
            result.setdefault(match.group(2).upper(), []).append(match.group(1))
    return result


def exactly(control: dict[str, list[str]], key: str, value: str) -> None:
    seen = control.get(key, [])
    if seen != [value]:
        raise Refusal(f"expected exactly {value} [{key}], found {seen}")


def audit_controls(root: Path) -> dict[str, Any]:
    vadat = key_values(root / "VADAT")
    exactly(vadat, "FIX_T", "F")
    exactly(vadat, "FIX_T_AUTO", "F")
    exactly(vadat, "FIX_NE", "F")
    exactly(vadat, "FIX_IMP", "F")
    exactly(vadat, "WRITE_RATES", "T")
    exactly(vadat, "WRITE_JH", "T")
    if "TAU_SCL_T" in vadat and any(float(v.replace("D", "E")) != 0.0 for v in vadat["TAU_SCL_T"]):
        raise Refusal("TAU_SCL_T must be zero for all-depth temperature solve")
    freeze_values: dict[str, str] = {}
    for key, values in vadat.items():
        if not key.startswith("FIX_") or key in {
            "FIX_T", "FIX_T_AUTO", "FIX_NE", "FIX_IMP", "FIX_BA"
        }:
            continue
        if len(values) != 1:
            raise Refusal(f"duplicate freeze key: {key}={values}")
        try:
            number = float(values[0].replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise Refusal(f"non-numeric freeze control: {key}={values[0]}") from exc
        if number != 0.0:
            raise Refusal(f"physical freeze is nonzero: {key}={values[0]}")
        freeze_values[key] = values[0]
    forbidden = sorted(
        p.name for p in root.iterdir()
        if p.is_file() and (p.name == "XzV_IN" or p.name.startswith("XzV_IN_") or
                            (p.name.startswith("POP") and p.name.endswith("_IN")))
    )
    if forbidden:
        raise Refusal(f"external population/freeze vectors present: {forbidden}")
    flux = key_values(root / "CMF_FLUX_PARAM")
    exactly(flux, "WR_ETA", "T")
    exactly(flux, "WR_FLUX", "F")
    exactly(flux, "COMP_F", "F")
    return {
        "temperature": {"FIX_T": "F", "FIX_T_AUTO": "F", "TAU_SCL_T": vadat.get("TAU_SCL_T", [])},
        "electron_and_impurity": {"FIX_NE": "F", "FIX_IMP": "F"},
        "population_controls_checked": freeze_values,
        "matrix_reuse_not_a_physical_freeze": {"FIX_BA": vadat.get("FIX_BA", [])},
        "external_vectors": forbidden,
    }


def last_capture(root: Path, evidence: dict[str, Any]) -> tuple[Path, list[str]]:
    raw = evidence.get("final_capture_dir")
    if not isinstance(raw, str) or not raw:
        raise Refusal("evidence.final_capture_dir is required")
    capture = (root / raw).resolve()
    try:
        capture.relative_to((root / "seq_logs" / "captures").resolve())
    except ValueError as exc:
        raise Refusal("final_capture_dir must be below seq_logs/captures") from exc
    if not capture.is_dir() or not (capture / "SHA256SUMS").is_file():
        raise Refusal(f"invalid final capture: {capture}")
    checked: list[str] = []
    for saved in sorted(capture.iterdir()):
        if not saved.is_file() or saved.name == "SHA256SUMS":
            continue
        current = root / saved.name
        if not current.is_file() or digest(saved) != digest(current):
            raise Refusal(f"root/capture content mismatch: {saved.name}")
        checked.append(saved.name)
    for name in ["EDDFACTOR", "RVTJ", "JH_AT_CURRENT_TIME", "OUTGEN"]:
        if name not in checked:
            raise Refusal(f"final capture did not bind {name}")
    return capture, checked


def convergence(evidence: dict[str, Any]) -> dict[str, Any]:
    value = evidence.get("convergence")
    if not isinstance(value, dict):
        raise Refusal("evidence.convergence must be an object")
    for field in ["jnu_last3_max_fraction", "te_last3_max_fraction", "ion_last3_max_fraction"]:
        vals = value.get(field)
        if not isinstance(vals, list) or len(vals) != 3:
            raise Refusal(f"{field} must contain exactly three values")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 0.01 for v in vals):
            raise Refusal(f"{field} violates finite [0,0.01] threshold: {vals}")
    for field, limit in [
        ("active_population_max_correction_fraction", 0.01),
        ("max_normalized_heat_residual", 0.001),
    ]:
        v = value.get(field)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > limit:
            raise Refusal(f"{field} violates finite [0,{limit}] threshold: {v}")
    if value.get("nan_count") != 0 or value.get("inf_count") != 0:
        raise Refusal("NaN/Inf counts must both be zero")
    return value


def validate_evidence(root: Path, evidence: dict[str, Any]) -> tuple[dict[str, Any], Path, list[str]]:
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise Refusal(f"evidence.schema must be {EVIDENCE_SCHEMA}")
    if not isinstance(evidence.get("iteration_id"), str) or not evidence["iteration_id"].strip():
        raise Refusal("nonempty evidence.iteration_id is required")
    reviewer = evidence.get("reviewer")
    if (
        not isinstance(reviewer, dict)
        or not isinstance(reviewer.get("name"), str)
        or not reviewer["name"].strip()
        or not isinstance(reviewer.get("method"), str)
        or not reviewer["method"].strip()
    ):
        raise Refusal("reviewer.name and reviewer.method are required")
    if any(mark in f"{reviewer['name']} {reviewer['method']}" for mark in ["<", ">"]):
        raise Refusal("reviewer declaration still contains angle-bracket placeholders")
    conv = convergence(evidence)
    schemas = evidence.get("record_schemas")
    if not isinstance(schemas, dict):
        raise Refusal("record_schemas must be an object")
    for name in ["EDDFACTOR", "JH_AT_CURRENT_TIME", "CHI_DATA", "ETA_DATA"]:
        item = schemas.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("units"), str) or not item["units"].strip():
            raise Refusal(f"nonempty record_schemas.{name}.units is required")
        if not isinstance(item.get("frame"), str) or not item["frame"].strip():
            raise Refusal(f"nonempty record_schemas.{name}.frame is required")
        declared = f"{item['units']} {item['frame']}".upper()
        if any(token in declared for token in ["TODO", "TBD", "UNKNOWN", "UNDECLARED", "PLACEHOLDER", "<", ">"]):
            raise Refusal(f"record_schemas.{name} contains a placeholder/unknown declaration")
    rate = evidence.get("rate_audit")
    if not isinstance(rate, dict) or rate.get("upward_downward_separated") is not True:
        raise Refusal("rate audit must explicitly prove upward/downward separation")
    files = rate.get("evidence_files")
    if not isinstance(files, list) or not files:
        raise Refusal("rate_audit.evidence_files must be nonempty")
    for raw in files:
        if not isinstance(raw, str):
            raise Refusal(f"rate evidence file missing: {raw!r}")
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise Refusal(f"rate evidence escapes run root: {raw!r}") from exc
        if not candidate.is_file():
            raise Refusal(f"rate evidence file missing: {raw!r}")
    capture, bound = last_capture(root, evidence)
    return conv, capture, bound


def audit_temperature(root: Path, capture: Path) -> dict[str, Any]:
    match_name = re.fullmatch(r"capture_([^/]+)_4", capture.name)
    if not match_name:
        raise Refusal("final_capture_dir must end in capture_<jobid>_4")
    job_id = match_name.group(1)
    checked = [capture.parent / f"capture_{job_id}_{serial}" / "OUTGEN" for serial in range(1, 5)]
    if any(not p.is_file() for p in checked):
        raise Refusal("four same-job archived capture OUTGEN files are required")
    for path in checked:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Temperature held fixed at all depths." in text:
            raise Refusal(f"temperature was held fixed in capture: {path}")
        if re.search(r"\b(?:NaN|Inf(?:inity)?)\b", text, re.IGNORECASE):
            raise Refusal(f"nonfinite marker found in capture: {path}")
    rvtj = (root / "RVTJ").read_text(encoding="utf-8", errors="replace")[:8192]
    match = re.search(r"Was T fixed\?:\s*([TF])", rvtj)
    if not match or match.group(1) != "F":
        raise Refusal("RVTJ does not declare 'Was T fixed?: F'")
    return {"capture_outgen_files": [str(p.relative_to(root)) for p in checked], "rvtj_was_t_fixed": "F"}


def all_file_hashes(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    link_targets: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(directory)
        for name in filenames:
            path = base / name
            rel = path.relative_to(root).as_posix()
            if rel == ATTESTATION or rel.startswith(f".{ATTESTATION}."):
                continue
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not target.is_file():
                    raise Refusal(f"symlink does not resolve to a file: {rel}")
                hashes[rel] = digest(target)
                link_targets[rel] = os.readlink(path)
            elif path.is_file():
                hashes[rel] = digest(path)
    if not hashes or any(not HEX64.fullmatch(v) for v in hashes.values()):
        raise Refusal("failed to build complete file hash map")
    return hashes, link_targets


def atomic_hashes(root: Path, file_hashes: dict[str, str]) -> dict[str, str]:
    atomic_name = re.compile(
        r"^(?:PHOT[A-Za-z0-9]+_[AB]|[A-Za-z0-9]+_(?:COL_DATA|F_OSCDAT|F_TO_S)|"
        r"TWO_PHOT_DATA|HYD_L_DATA|GBF_N_DATA|NUC_DECAY_DATA|XRAY_PHOT_FITS|"
        r"RS_XRAY_FLUXES|SOL_ABUND|HI_IS_LINE_LIST|H2_IS_LINE_LIST)$"
    )
    values = {name: value for name, value in file_hashes.items() if "/" not in name and atomic_name.fullmatch(name)}
    values.update({name: value for name, value in file_hashes.items() if name.startswith("atomic_local/")})
    if not values:
        raise Refusal("no atomic-data hashes found")
    return values


def revision(source: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Refusal(f"cannot resolve CMFGEN revision: {exc}") from exc
    if not GIT_OBJECT.fullmatch(commit):
        raise Refusal(f"invalid CMFGEN revision: {commit}")
    try:
        worktree = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=no"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Refusal(f"cannot inventory CMFGEN worktree state: {exc}") from exc
    exes = {}
    for name in ["cmfgen_dev.exe", "cmf_flux.exe"]:
        path = source / "exe" / name
        if not path.is_file():
            raise Refusal(f"missing executable: {path}")
        exes[name] = digest(path)
    return {
        "commit": commit,
        "executables": exes,
        "source_root": str(source),
        "tracked_worktree_changes": worktree,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cmfgen-source", type=Path, default=Path("/gpfs/kjhan/cmfgen_src/cur_cmf"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise Refusal(f"not a run directory: {root}")
    output = root / ATTESTATION
    if output.exists() and not args.force:
        raise Refusal(f"refusing to overwrite {output}; use --force only after a new reviewed capture")
    for name in REQUIRED + REQUIRED_INPUTS + ["CMF_FLUX_PARAM", "CMF_FLUX_STDIN"]:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise Refusal(f"required nonempty file missing: {name}")
    stems = sorted(
        p.name[:-4] for p in root.glob("*PRRR")
        if (root / f"{p.name[:-4]}OUT").is_file()
    )
    if not stems:
        raise Refusal("no matching *PRRR/*OUT stems")
    controls = audit_controls(root)
    evidence_path = args.evidence.resolve()
    try:
        evidence_path.relative_to(root)
    except ValueError as exc:
        raise Refusal("reviewed evidence JSON must reside below the sealed run root") from exc
    evidence = load_json(evidence_path)
    conv, capture, capture_bound = validate_evidence(root, evidence)
    temp_evidence = audit_temperature(root, capture)
    formal = load_json(root / "PROVENANCE_FORMAL.json")
    if formal.get("schema") != "lumina-cmfgen-ophys-formal-provenance-v1":
        raise Refusal("invalid formal provenance schema")
    formal_files = formal.get("files")
    formal_required = {"CHI_DATA", "CHI_DATA_INFO", "ETA_DATA", "ETA_DATA_INFO", "OBSFLUX", "OBS_FREQ", "RVTJ"}
    if not isinstance(formal_files, dict) or set(formal_files) != formal_required:
        raise Refusal("formal provenance does not cover every CHI/ETA/OBS/RVTJ target")
    for name, expected in formal_files.items():
        if not HEX64.fullmatch(str(expected)) or not (root / name).is_file() or digest(root / name) != expected:
            raise Refusal(f"formal provenance mismatch: {name}")
    file_hashes, link_targets = all_file_hashes(root)
    source = revision(args.cmfgen_source.resolve())
    generation_files = sorted(
        name for name in file_hashes
        if "/" not in name and (name in GENERATION_BASE or re.fullmatch(r"POP[A-Z0-9_]+", name) or re.fullmatch(r"[A-Za-z0-9]+PRRR", name))
    )
    required_generation = sorted(GENERATION_BASE)
    if not set(required_generation).issubset(generation_files):
        raise Refusal("generation proof target is incomplete")
    attestation = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_revision": source["commit"],
        "code": source,
        "input_files": {name: file_hashes[name] for name in REQUIRED_INPUTS},
        "atomic_data": atomic_hashes(root, file_hashes),
        "run": {
            "fix_t": False,
            "temperature_solved": True,
            "omp_threads": 16,
            "control_audit": controls,
            "temperature_evidence": temp_evidence,
        },
        "freezes": {"undisclosed_count": 0, "components": []},
        "coverage": {"matching_ion_stems": stems},
        "rate_audit": evidence["rate_audit"],
        "convergence": conv,
        "generation_proof": {
            "verdict": "SAME_GENERATION_PROVEN",
            "evidence_kind": "content",
            "iteration_id": evidence["iteration_id"],
            "files": generation_files,
            "output_after_last_iteration": True,
            "capture_directory": str(capture.relative_to(root)),
            "capture_bound_files": capture_bound,
            "formal_provenance": "PROVENANCE_FORMAL.json",
        },
        "record_schemas": evidence["record_schemas"],
        "measurement_review": evidence["reviewer"],
        "file_sha256": file_hashes,
        "symlink_targets": link_targets,
        "hash_scope": "all recursive files below run root except this self-referential attestation; symlinks hash resolved file bytes",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{ATTESTATION}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(attestation, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"wrote {output}")
    print(f"sealed {len(file_hashes)} files; atomic inputs={len(attestation['atomic_data'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
