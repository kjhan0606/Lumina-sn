#!/usr/bin/env python3
"""Judge the A2-10 pre-core tau-seed diagnostic against a sealed baseline."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


class ComparisonError(RuntimeError):
    pass


KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
REFERENCE_PREFIXES = (
    "[cmf_fine][SIGNED-MATERIAL-CENSUS]",
    "[cmf_fine][EXACT-MULTIGPU-EPOCH]",
    "[R6][LINE-IDENTITY]",
    "[R6][LINE-COVERAGE]",
)
PRECORE_PREFIX = "[A2-10][PRECORE-TAU-SEED]"
NOBRACKET_PREFIX = "[A2-10][VECTOR-NOBRACKET] count="
INTERIOR_PREFIX = "[A2-10][VECTOR-INTERIOR-SCAN] phase=GEOMETRIC_MID shell="
PAIR_ARTIFACTS = (
    "input/lumina_cuda",
    "input/deck.sha256",
    "input/topion.sha256",
    "input/sigma.sha256",
)
PAIR_CONTROLS = (
    "input/outer_iterations.txt",
    "input/single_total.txt",
    "input/stage4.txt",
    "input/envelope_refinements.txt",
    "input/diagnostic_mode.txt",
)
PRECORE_ENV = 'declare -x LUMINA_A210_PRECORE_TAU_REFRESH="1"'
SHA256_MANIFEST_ROW = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_lines(path: Path) -> list[str]:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe log: {path}")
    return path.read_text(encoding="utf-8", errors="strict").splitlines()


def load_scalar(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe sealed scalar: {path}")
    value = path.read_text(encoding="utf-8", errors="strict").strip()
    if not value or "\n" in value:
        raise ComparisonError(f"invalid sealed scalar: {path}")
    return value


def require_inside(root: Path, path: Path, label: str) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ComparisonError(f"{label} is outside sealed root: {path}")


def normalized_environment(root: Path) -> list[str]:
    path = root / "input/resolved_lumina.exports"
    lines = load_lines(path)
    root_text = str(root.resolve())
    return sorted(line.replace(root_text, "$RUN_ROOT") for line in lines)


def verify_sha256_manifest(manifest: Path, directory: Path) -> int:
    rows = load_lines(manifest)
    if not rows:
        raise ComparisonError(f"empty SHA256 manifest: {manifest}")
    verified = 0
    resolved_directory = directory.resolve()
    for row in rows:
        match = SHA256_MANIFEST_ROW.fullmatch(row)
        if not match:
            raise ComparisonError(f"invalid SHA256 manifest row: {manifest}")
        expected, relative_text = match.groups()
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ComparisonError(f"unsafe SHA256 manifest path: {relative_text}")
        path = directory / relative
        if not path.is_file() or path.is_symlink():
            raise ComparisonError(f"missing or unsafe sealed input: {path}")
        require_inside(resolved_directory, path, "manifest input")
        if digest(path) != expected:
            raise ComparisonError(f"sealed input digest mismatch: {path}")
        verified += 1
    return verified


def validate_sealed_pair(
    baseline_root: Path,
    candidate_root: Path,
    baseline_stderr: Path,
    candidate_stderr: Path,
    baseline_model_rc: Path,
    candidate_model_rc: Path,
) -> dict[str, Any]:
    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    if baseline_root == candidate_root:
        raise ComparisonError("baseline and candidate roots are identical")
    for root, label in ((baseline_root, "baseline"),
                        (candidate_root, "candidate")):
        if not root.is_dir() or root.is_symlink() or not (root / "READY").is_file():
            raise ComparisonError(f"{label} root is not sealed READY: {root}")
    require_inside(baseline_root, baseline_stderr, "baseline stderr")
    require_inside(candidate_root, candidate_stderr, "candidate stderr")
    require_inside(baseline_root, baseline_model_rc, "baseline model rc")
    require_inside(candidate_root, candidate_model_rc, "candidate model rc")

    artifact_hashes: dict[str, str] = {}
    for relative in PAIR_ARTIFACTS:
        baseline_path = baseline_root / relative
        candidate_path = candidate_root / relative
        if (not baseline_path.is_file() or baseline_path.is_symlink() or
                not candidate_path.is_file() or candidate_path.is_symlink()):
            raise ComparisonError(f"missing or unsafe pair artifact: {relative}")
        baseline_hash = digest(baseline_path)
        candidate_hash = digest(candidate_path)
        if baseline_hash != candidate_hash:
            raise ComparisonError(f"sealed pair artifact differs: {relative}")
        artifact_hashes[relative] = baseline_hash

    binary_hash = digest(baseline_root / "input/lumina_cuda")
    for root, label in ((baseline_root, "baseline"),
                        (candidate_root, "candidate")):
        declared = load_scalar(root / "input/binary.sha256").split()[0]
        if declared != binary_hash:
            raise ComparisonError(
                f"{label} declared binary SHA does not match staged binary"
            )

    manifest_counts: dict[str, int] = {}
    for root, label in ((baseline_root, "baseline"),
                        (candidate_root, "candidate")):
        manifest_counts[f"{label}_deck"] = verify_sha256_manifest(
            root / "input/deck.sha256", root / "input/model"
        )
        manifest_counts[f"{label}_topion"] = verify_sha256_manifest(
            root / "input/topion.sha256", root / "input/global_atomic"
        )
        declared_sigma = load_scalar(root / "input/sigma.sha256").split()[0]
        actual_sigma = digest(root / "input/model/cmfgen_sigma_bf.bin")
        if declared_sigma != actual_sigma:
            raise ComparisonError(
                f"{label} declared sigma SHA does not match staged sigma"
            )

    controls: dict[str, str] = {}
    for relative in PAIR_CONTROLS:
        baseline_value = load_scalar(baseline_root / relative)
        candidate_value = load_scalar(candidate_root / relative)
        if baseline_value != candidate_value:
            raise ComparisonError(
                f"sealed pair control differs: {relative} "
                f"{baseline_value!r}!={candidate_value!r}"
            )
        controls[relative] = baseline_value

    baseline_precore = load_scalar(
        baseline_root / "input/precore_tau_refresh.txt"
    )
    candidate_precore = load_scalar(
        candidate_root / "input/precore_tau_refresh.txt"
    )
    if baseline_precore != "0" or candidate_precore != "1":
        raise ComparisonError(
            "sealed pre-core controls are not baseline=0/candidate=1"
        )

    baseline_env = normalized_environment(baseline_root)
    candidate_env = normalized_environment(candidate_root)
    if PRECORE_ENV in baseline_env:
        raise ComparisonError("baseline sealed env unexpectedly enables pre-core seed")
    if candidate_env.count(PRECORE_ENV) != 1:
        raise ComparisonError("candidate sealed env lacks unique pre-core seed")
    candidate_without_precore = [
        line for line in candidate_env if line != PRECORE_ENV
    ]
    if baseline_env != candidate_without_precore:
        difference = list(difflib.unified_diff(
            baseline_env, candidate_without_precore,
            fromfile="baseline_env", tofile="candidate_env_without_precore",
            lineterm="",
        ))
        raise ComparisonError(
            "sealed environments differ beyond pre-core seed: " +
            " | ".join(difference[:8])
        )

    return {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "artifact_sha256": artifact_hashes,
        "verified_manifest_entries": manifest_counts,
        "controls": controls,
        "environment_identity": "ONLY_PRECORE_TAU_REFRESH_DIFFERS",
        "baseline_precore_tau_refresh": 0,
        "candidate_precore_tau_refresh": 1,
    }


def unique_records(lines: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for prefix in REFERENCE_PREFIXES:
        found = [line for line in lines if line.startswith(prefix)]
        if len(found) != 1:
            raise ComparisonError(
                f"expected exactly one {prefix!r}, found {len(found)}"
            )
        result[prefix] = dict(KEY_VALUE.findall(found[0]))
    return result


def exact_differences(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
) -> list[dict[str, str | None]]:
    differences: list[dict[str, str | None]] = []
    for prefix in REFERENCE_PREFIXES:
        lhs = baseline[prefix]
        rhs = candidate[prefix]
        for key in sorted(set(lhs) | set(rhs)):
            if lhs.get(key) != rhs.get(key):
                differences.append({
                    "record": prefix,
                    "field": key,
                    "baseline": lhs.get(key),
                    "candidate": rhs.get(key),
                })
    return differences


def interior_records(lines: list[str]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for line in lines:
        if not line.startswith(INTERIOR_PREFIX):
            continue
        fields = dict(KEY_VALUE.findall(line))
        try:
            shell = int(fields["shell"])
        except (KeyError, ValueError) as exc:
            raise ComparisonError("invalid interior shell record") from exc
        if shell in result:
            raise ComparisonError(f"duplicate interior record for shell {shell}")
        for key in ("T_mid", "res_lo", "res_mid", "res_hi"):
            try:
                value = float(fields[key])
            except (KeyError, ValueError) as exc:
                raise ComparisonError(
                    f"invalid interior {key} for shell {shell}"
                ) from exc
            if not (value == value and abs(value) != float("inf")):
                raise ComparisonError(
                    f"nonfinite interior {key} for shell {shell}"
                )
        result[shell] = fields
    return result


def residual_comparison(
    baseline: dict[int, dict[str, str]],
    candidate: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shell in sorted(baseline):
        if shell not in candidate:
            raise ComparisonError(f"candidate lacks interior shell {shell}")
        row: dict[str, Any] = {"shell": shell}
        for key in ("res_lo", "res_mid", "res_hi"):
            before = float(baseline[shell][key])
            after = float(candidate[shell][key])
            row[f"baseline_{key}"] = before
            row[f"candidate_{key}"] = after
            row[f"candidate_over_baseline_{key}"] = (
                after / before if before != 0.0 else None
            )
        rows.append(row)
    return rows


def parse_model_rc(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe model rc: {path}")
    try:
        return int(path.read_text(encoding="ascii").strip())
    except ValueError as exc:
        raise ComparisonError(f"invalid model rc: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-stderr", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    parser.add_argument("--baseline-model-rc", type=Path, required=True)
    parser.add_argument("--candidate-model-rc", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        pair_seal = validate_sealed_pair(
            args.baseline_root, args.candidate_root,
            args.baseline_stderr, args.candidate_stderr,
            args.baseline_model_rc, args.candidate_model_rc,
        )
        baseline_lines = load_lines(args.baseline_stderr)
        candidate_lines = load_lines(args.candidate_stderr)
        differences = exact_differences(
            unique_records(baseline_lines), unique_records(candidate_lines)
        )
        if differences:
            first = differences[0]
            raise ComparisonError(
                "exact/R6 identity changed "
                f"record={first['record']} field={first['field']}"
            )

        baseline_seed = [
            line for line in baseline_lines if line.startswith(PRECORE_PREFIX)
        ]
        candidate_seed = [
            line for line in candidate_lines if line.startswith(PRECORE_PREFIX)
        ]
        if baseline_seed:
            raise ComparisonError("baseline unexpectedly contains pre-core seed")
        if len(candidate_seed) < 2:
            raise ComparisonError(
                f"candidate pre-core seed count {len(candidate_seed)} < 2"
            )
        for line in candidate_seed:
            fields = dict(KEY_VALUE.findall(line))
            required = {
                "status": "DIAGNOSTIC_AB_ONLY",
                "source": "TRIAL_LTE_IONIZATION",
                "rate_consumer": "MODE3_BETA_JINC",
                "population_tau_fixed_point": "0",
                "public_mutation": "0",
                "floor": "0",
                "cap": "0",
                "clamp": "0",
                "jitter": "0",
                "repair": "0",
            }
            for key, expected in required.items():
                if fields.get(key) != expected:
                    raise ComparisonError(
                        f"invalid pre-core seed field {key}={fields.get(key)}"
                    )

        baseline_model_rc = parse_model_rc(args.baseline_model_rc)
        model_rc = parse_model_rc(args.candidate_model_rc)
        if baseline_model_rc != 1:
            raise ComparisonError(
                f"baseline must be a fail-closed no-bracket run, rc={baseline_model_rc}"
            )
        baseline_no_bracket = [
            line for line in baseline_lines if line.startswith(NOBRACKET_PREFIX)
        ]
        candidate_no_bracket = [
            line for line in candidate_lines if line.startswith(NOBRACKET_PREFIX)
        ]
        if len(baseline_no_bracket) != 1:
            raise ComparisonError(
                "baseline must contain exactly one vector no-bracket summary"
            )
        baseline_interior = interior_records(baseline_lines)
        if sorted(baseline_interior) != [0, 1, 2, 3]:
            raise ComparisonError(
                f"baseline interior shells are {sorted(baseline_interior)}"
            )

        residuals: list[dict[str, Any]] = []
        if model_rc == 0:
            if candidate_no_bracket:
                raise ComparisonError("model rc=0 but no-bracket record remains")
            outcome = "BRACKET_RESTORED_GATE_PASS"
        elif model_rc == 1:
            if len(candidate_no_bracket) != 1:
                raise ComparisonError(
                    "model rc=1 without exactly one no-bracket summary"
                )
            candidate_interior = interior_records(candidate_lines)
            residuals = residual_comparison(baseline_interior, candidate_interior)
            outcome = "NO_BRACKET_PERSISTS"
        else:
            raise ComparisonError(f"unexpected candidate model rc={model_rc}")

        payload = {
            "schema": "LUMINA_A210_PRECORE_TAU_AB_COMPARISON_V2",
            "status": "PASS",
            "outcome": outcome,
            "candidate_model_rc": model_rc,
            "exact_r6_identity": "BIT_EXACT",
            "precore_seed_records": len(candidate_seed),
            "sealed_pair": pair_seal,
            "baseline_stderr": str(args.baseline_stderr.resolve()),
            "baseline_sha256": digest(args.baseline_stderr),
            "candidate_stderr": str(args.candidate_stderr.resolve()),
            "candidate_sha256": digest(args.candidate_stderr),
            "residual_comparison": residuals,
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "repair": 0,
        }
        atomic_write(args.report, payload)
        print(
            "PASS A210_PRECORE_TAU_AB_COMPARISON "
            f"outcome={outcome} exact_r6=BIT_EXACT "
            f"seed_records={len(candidate_seed)} repair=0"
        )
        return 0
    except (ComparisonError, OSError, UnicodeError) as exc:
        atomic_write(args.report, {
            "schema": "LUMINA_A210_PRECORE_TAU_AB_COMPARISON_V2",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_PRECORE_TAU_AB_COMPARISON reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
