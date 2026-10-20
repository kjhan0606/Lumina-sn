#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import sys
import tempfile
from decimal import Decimal
from pathlib import Path


TREE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cmfgen_twin_map_binary64", TREE / "scripts/cmfgen_twin_map.py")
assert SPEC and SPEC.loader
mapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mapper
SPEC.loader.exec_module(mapper)
MODE = "BINARY64_ROUNDTRIP_V1"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_results(evidence: Path) -> tuple[list[dict[str, str]], dict]:
    with (evidence / "result/divergence_map.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((evidence / "result/summary.json").read_text(encoding="utf-8"))
    return rows, summary


def analyze(expectations: Path, run: Path, evidence: Path) -> tuple[list[dict[str, str]], dict]:
    mapper.analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
    return load_results(evidence)


def rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def same_binary64_distinct_decimal(token: str) -> str:
    original = Decimal(token)
    candidate = Decimal.from_float(float(original))
    if candidate == original:
        raise AssertionError("fixture token already is the exact binary64 Decimal expansion")
    return str(candidate)


def unit_checks() -> None:
    check(mapper.equal_temperature_tokens("9696.1", "9696.1000000000004", MODE) is True,
          "roundtrip-equivalent temperatures rejected")
    check(mapper.equal_temperature_tokens("9.6961e3", "9696.100", MODE) is True,
          "notation-equivalent temperatures rejected")
    adjacent = repr(math.nextafter(9696.1, math.inf))
    check(mapper.equal_temperature_tokens("9696.1", adjacent, MODE) is False,
          "adjacent binary64 accepted")
    check(mapper.equal_temperature_tokens("9696.1", "9696.1000000000004", None) is False,
          "legacy Decimal behavior changed")
    for invalid in ("0", "-0", "-1", "nan", "inf", "-inf", "bad", "1e9999", "1e-9999"):
        check(mapper.equal_temperature_tokens(invalid, "9696.1", MODE) is None,
              f"invalid temperature accepted: {invalid}")


def analyze_checks() -> None:
    with tempfile.TemporaryDirectory(prefix="cmfgen-te-binary64-analyze.") as raw:
        base, reference, evidence, run, expectations = mapper._private_fixture(Path(raw))
        del base
        phys = reference / mapper.REFERENCE_NAMES[0]
        original_phys = phys.read_text(encoding="utf-8")
        same_binary_phys = original_phys.replace("0.6 1.0000E+04", "0.6 10000.0000000000004", 1)
        check(same_binary_phys != original_phys, "reference temperature mutation target missing")
        mapper.text_write(phys, same_binary_phys)

        rows, summary = analyze(expectations, run, evidence)
        shell_zero = [row for row in rows if row["shell_id"] == "0"]
        check("temperature_equality" not in summary, "legacy summary changed")
        check(any("T_NOT_TWIN" in row["reason"] for row in shell_zero),
              "legacy exact Decimal mismatch disappeared")

        contract = json.loads(expectations.read_text(encoding="utf-8"))
        contract["temperature_equality"] = MODE
        mapper.json_write(expectations, contract)
        rows, summary = analyze(expectations, run, evidence)
        shell_zero = [row for row in rows if row["shell_id"] == "0"]
        check(summary.get("temperature_equality") == MODE, "selected mode missing from summary")
        check(not any("T_NOT_TWIN" in row["reason"] for row in shell_zero),
              "binary64-equivalent public/reference temperature rejected")

        result_paths = (evidence / "result/divergence_map.csv", evidence / "result/summary.json")
        preserved = tuple(path.read_bytes() for path in result_paths)
        contract["temperature_equality"] = "UNKNOWN"
        mapper.json_write(expectations, contract)
        try:
            analyze(expectations, run, evidence)
        except mapper.Failure as exc:
            check(exc.reason == "TEMPERATURE_EQUALITY_MODE_INVALID", "wrong unknown-mode failure")
        else:
            raise AssertionError("unknown temperature mode accepted")
        check(preserved == tuple(path.read_bytes() for path in result_paths),
              "unknown mode changed existing outputs")

        contract["temperature_equality"] = MODE
        mapper.json_write(expectations, contract)
        adjacent = repr(math.nextafter(10000.0, math.inf))
        mapper.text_write(phys, original_phys.replace("0.6 1.0000E+04", f"0.6 {adjacent}", 1))
        rows, _summary = analyze(expectations, run, evidence)
        shell_zero = [row for row in rows if row["shell_id"] == "0"]
        check(any(row["authority"] == "PUBLIC_NE" and "T_NOT_TWIN" in row["reason"] for row in shell_zero),
              "nextafter public-state mismatch missing")
        check(any(row["authority"] == "SAHA_STAGE" and "T_NOT_TWIN" in row["reason"] for row in shell_zero),
              "nextafter fraction qualification mismatch missing")

        mapper.text_write(phys, original_phys.replace("0.6 1.0000E+04", "0.6 1e-9999", 1))
        rows, _summary = analyze(expectations, run, evidence)
        check(any(row["authority"] == "PUBLIC_NE" and row["shell_id"] == "0" and
                  "INVALID_PUBLIC_T" in row["reason"] for row in rows),
              "underflow-to-zero temperature was not rejected safely")

        mapper.text_write(phys, same_binary_phys)
        nlte_path = run / "work/nlte_levels_final.csv"
        with nlte_path.open(newline="", encoding="utf-8") as stream:
            nlte_rows = list(csv.DictReader(stream))
        original_nion = nlte_rows[0]["n_ion_total"]
        changed_nion = same_binary64_distinct_decimal(original_nion)
        check(mapper.equal_temperature_tokens(original_nion, changed_nion, MODE) is True,
              "test n_ion_total tokens are not binary64-equivalent")
        check(mapper.equal_decimal_tokens(original_nion, changed_nion) is False,
              "test n_ion_total tokens are not Decimal-distinct")
        nlte_rows[0]["n_ion_total"] = changed_nion
        rewrite_csv(nlte_path, nlte_rows)
        rows, summary = analyze(expectations, run, evidence)
        check(any(item.startswith("NLTE_SAHA_STAGE_LINK_MISMATCH:") for item in summary["pending"]),
              "Decimal-exact NLTE/Saha link mismatch was relaxed")
        check(any(row["authority"] == "NLTE_LEVEL_SUM" and
                  "NLTE_SAHA_STAGE_LINK_MISMATCH" in row["reason"] for row in rows),
              "NLTE/Saha link mismatch missing from map")


def main() -> int:
    unit_checks()
    analyze_checks()
    print("CMFGEN_TE_BINARY64_SELFTEST_PASS unit_boundaries=13 analyze_legacy=1 analyze_roundtrip=1 unknown_no_write=1 nextafter=1 underflow=1 nion_decimal_link=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
