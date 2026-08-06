#!/usr/bin/env python3
"""Verify current, sidecar, and pre-implementation Git-blob hashes agree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation/a2_08"
START = json.loads((VAL / "A2_08_IMPLEMENTATION_START.json").read_text())
ALLOW = VAL / "A2_08_CHANGED_OUTPUT_ALLOWLIST.json"


def main() -> int:
    current = hashlib.sha256(ALLOW.read_bytes()).hexdigest()
    side = (VAL / "A2_08_CHANGED_OUTPUT_ALLOWLIST.sha256").read_text().split()[0]
    env = os.environ.copy()
    env["GIT_OBJECT_DIRECTORY"] = str(ROOT / START["object_directory"])
    env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(ROOT / ".git/objects")
    sealed = subprocess.check_output(
        ("git", "show", f"{START['seal_commit']}:validation/a2_08/"
         "A2_08_CHANGED_OUTPUT_ALLOWLIST.json"), cwd=ROOT, env=env)
    sealed_hash = hashlib.sha256(sealed).hexdigest()
    changed = subprocess.check_output(
        ("git", "diff-tree", "--no-commit-id", "--name-only", "-r",
         START["seal_commit"]), cwd=ROOT, env=env, text=True).splitlines()
    expected_paths = [
        "validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.json",
        "validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.sha256",
    ]
    ok = (current == side == sealed_hash == START["json_sha256"] and
          changed == expected_paths and
          hashlib.sha1(b"blob " + str(len(sealed)).encode() + b"\0" + sealed).hexdigest()
          == START["blob_id"])
    print(f"{'PASS' if ok else 'FAIL'} A2_08_ALLOWLIST_SEAL "
          f"current={current} sidecar={side} sealed={sealed_hash} "
          f"commit={START['seal_commit']} blob={START['blob_id']}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
