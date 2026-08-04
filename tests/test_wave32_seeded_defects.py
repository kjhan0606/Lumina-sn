#!/usr/bin/env python3
"""R7 writer/consumer and D6 matrix-debit seeded negative controls."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(["make", "-B", "selftest_cmf_chieta_dump",
                    "selftest_wave32_matrix_debit"], cwd=ROOT, check=True)
    tmp = Path(tempfile.mkdtemp(prefix="w32_seeded_", dir="/tmp"))

    bad = tmp / "bad_eta.lcmfce"
    env = os.environ.copy()
    env["W32_SEED_BAD_ETA"] = "1"
    subprocess.run([str(ROOT / "selftest_cmf_chieta_dump"), str(bad)],
                   cwd=ROOT, env=env, check=True)
    selected = subprocess.run(
        ["python3", "scripts/cmf_chieta_check.py", str(bad)],
        cwd=ROOT, text=True, capture_output=True)
    if selected.returncode == 0:
        raise RuntimeError("bad eta passed the R7 selector")

    quarantine_payload = tmp / "sidecar_fail.lcmfce"
    Path(str(quarantine_payload) + ".manifest.json").mkdir()
    sidecar = subprocess.run(
        [str(ROOT / "selftest_cmf_chieta_dump"), str(quarantine_payload)],
        cwd=ROOT)
    if sidecar.returncode == 0 or quarantine_payload.exists() or not Path(
            str(quarantine_payload) + ".quarantine").is_file():
        raise RuntimeError("sidecar failure did not quarantine payload")

    metadata_cases = [
        ("wrong_iter", {"W32_FIXTURE_ITER": "7"}),
        ("wrong_generation", {"W32_FIXTURE_GENERATION": "9"}),
        ("pre_damp", {"W32_FIXTURE_POST_DAMP": "0"}),
    ]
    metadata_rc = {}
    for name, update in metadata_cases:
        payload = tmp / f"{name}.lcmfce"
        case_env = os.environ.copy()
        case_env.update(update)
        subprocess.run([str(ROOT / "selftest_cmf_chieta_dump"), str(payload)],
                       cwd=ROOT, env=case_env, check=True)
        checked = subprocess.run(
            ["python3", "scripts/cmf_chieta_check.py", str(payload)],
            cwd=ROOT, text=True, capture_output=True)
        if checked.returncode == 0:
            raise RuntimeError(f"{name} passed the default consumer contract")
        metadata_rc[name] = checked.returncode

    bypass_payload = tmp / "iter7_generation7.lcmfce"
    bypass_env = os.environ.copy()
    bypass_env.update({"W32_FIXTURE_ITER": "7",
                       "W32_FIXTURE_GENERATION": "7"})
    subprocess.run([str(ROOT / "selftest_cmf_chieta_dump"),
                    str(bypass_payload)], cwd=ROOT, env=bypass_env, check=True)
    bypass_args = ["--expected-iteration", "7",
                   "--expected-field-generation", "7"]
    unauthorized = subprocess.run(
        ["python3", "scripts/cmf_chieta_check.py", str(bypass_payload),
         *bypass_args], cwd=ROOT, text=True, capture_output=True)
    if unauthorized.returncode == 0 or "non-contract expectation" not in (
            unauthorized.stdout + unauthorized.stderr):
        raise RuntimeError("iter=7 expectation bypass was not rejected")
    overridden = subprocess.run(
        ["python3", "scripts/cmf_chieta_check.py", str(bypass_payload),
         *bypass_args, "--non-contract-override"],
        cwd=ROOT, text=True, capture_output=True)
    if (overridden.returncode != 2 or
            not overridden.stdout.startswith("NON-CONTRACT:") or
            "PASS:" in overridden.stdout):
        raise RuntimeError("explicit override was not fail-closed/labeled")

    sys.path.insert(0, str(ROOT / "scripts"))
    from cmf_chieta_check import (  # pylint: disable=import-outside-toplevel
        CheckError, check_artifact)

    api_contract_payload = tmp / "api_contract.lcmfce"
    subprocess.run([str(ROOT / "selftest_cmf_chieta_dump"),
                    str(api_contract_payload)], cwd=ROOT, check=True)
    api_contract = check_artifact(api_contract_payload)
    if api_contract.contract_status != "CONTRACT":
        raise RuntimeError("Python API omitted/mislabeled contract status")
    try:
        check_artifact(bypass_payload, expected_iteration=7,
                       expected_generation=7)
    except CheckError:
        api_unauthorized = "REJECTED"
    else:
        raise RuntimeError("Python API accepted an unauthorized override")
    api_override = check_artifact(
        bypass_payload, expected_iteration=7, expected_generation=7,
        non_contract_override=True)
    if api_override.contract_status != "NON-CONTRACT":
        raise RuntimeError("Python API hid non-contract override status")

    matrix = subprocess.run([str(ROOT / "selftest_wave32_matrix_debit")],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    print("bad_eta_selector_rc=", selected.returncode, sep="")
    print("sidecar_writer_rc=", sidecar.returncode, sep="")
    print("metadata_consumer_rc=", metadata_rc, sep="")
    print("iter7_bypass_rc=", unauthorized.returncode,
          " explicit_override_rc=", overridden.returncode, sep="")
    print("api_contract_status=", api_contract.contract_status,
          " api_unauthorized=", api_unauthorized,
          " api_override_status=", api_override.contract_status, sep="")
    print(matrix.stdout.strip())
    print("PASS seeded defects: eta/iter/generation/post-damp consumer FAIL; "
          "D6 debit ledger FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
