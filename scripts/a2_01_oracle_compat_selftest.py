#!/usr/bin/env python3
"""A2-01 regression for the renamed CMFGEN write-order verdict field."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from a2_00_oracle_negative_controls import write_fixture


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    contract = script_dir / "cmfgen_oracle_contract.py"
    with tempfile.TemporaryDirectory(prefix="a2_01_oracle_compat_") as raw:
        root = Path(raw) / "run"
        manifest = Path(raw) / "manifest.json"
        write_fixture(root)
        prrr = root / "Ca2PRRR"
        text = prrr.read_text(encoding="ascii")
        old = "4.0000000000E+08 3.0000000000E+08 2.0000000000E+08 1.0000000000E+08"
        new = "2.0000000000E+08 2.0000000000E+08 1.0000000000E+08 5.0000000000E+07"
        if old not in text:
            print("FAIL fixture electron-density vector not found")
            return 2
        prrr.write_text(text.replace(old, new, 1), encoding="ascii")
        result = subprocess.run(
            [
                sys.executable,
                str(contract),
                "write",
                str(root),
                "--manifest",
                str(manifest),
                "--profile",
                "snapshot",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout, end="")
            print(f"FAIL contract write rc={result.returncode}")
            return 2
        document = json.loads(manifest.read_text(encoding="utf-8"))
        canonical = document.get("write_order_offset_convergence", {})
        offset = canonical.get("write_order_offset", {})
        legacy = document.get("generation_consistency", {})
        old_document = dict(document)
        old_document.pop("write_order_offset_convergence", None)
        old_legacy = dict(legacy)
        for name in [
            "assessment",
            "assessment_basis",
            "write_order_offset",
            "write_order_signature_files",
            "deprecated_alias",
            "canonical_field",
            "legacy_semantics_warning",
        ]:
            old_legacy.pop(name, None)
        old_document["generation_consistency"] = old_legacy
        old_manifest = Path(raw) / "old_v1_manifest.json"
        old_manifest.write_text(json.dumps(old_document, indent=2) + "\n", encoding="utf-8")
        old_check = subprocess.run(
            [
                sys.executable,
                str(contract),
                "check",
                str(root),
                "--manifest",
                str(old_manifest),
                "--profile",
                "snapshot",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        checks = {
            "canonical_assessment": canonical.get("assessment")
            == "WRITE_ORDER_OFFSET_MEASURED",
            "continuous_nonzero": isinstance(offset.get("max_relative_offset"), (int, float))
            and offset["max_relative_offset"] > 0.0,
            "continuous_vector": offset.get("values_compared") == 4,
            "legacy_verdict": legacy.get("verdict") == "MIXED_GENERATION_PROVEN",
            "legacy_deprecated": legacy.get("deprecated_alias") is True,
            "legacy_target": legacy.get("canonical_field")
            == "write_order_offset_convergence",
            "old_v1_manifest_check": old_check.returncode == 0,
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            print(f"FAIL A2_01_ORACLE_COMPAT failures={','.join(failures)}")
            return 2
        print(
            "PASS A2_01_ORACLE_COMPAT canonical=WRITE_ORDER_OFFSET_MEASURED "
            f"max_relative_offset={offset['max_relative_offset']:.17g} "
            "legacy=MIXED_GENERATION_PROVEN deprecated=true"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
