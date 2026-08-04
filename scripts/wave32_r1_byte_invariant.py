#!/usr/bin/env python3
"""Wave-3.2 R1 frozen-replay byte invariant (s0 and s8).

Runs only the archived parity59 CPU oracle.  For each cell it compares the
authoritative oracle CSV, pair ion fractions, and every pair-owned II/III
full-level population byte-for-byte between armed COMMIT=0 and unarmed runs.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = Path("/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59")
DEFAULT_MODEL = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bench", type=Path, default=ROOT / "bench_frozen_oracle")
    parser.add_argument("--out", type=Path,
                        help="output directory (default: /tmp/wave32_r1_*)")
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_cell(args, out, shell, armed, super_levels):
    label = (f"{'armed' if armed else 'unarmed'}_super{super_levels}"
             f"_s{shell}")
    dest = out / label
    dest.mkdir(parents=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "LUMINA_FROZEN_ORACLE_ONLY_SHELL": str(shell),
           "LUMINA_SUPER_LEVELS": str(super_levels)}
    if armed:
        env.update({
            "LUMINA_NLTE_ELEMENT_WIDE": "1",
            "LUMINA_NLTE_ELEMENT_WIDE_Z": "26" if shell == 0 else "16,26",
            "LUMINA_NLTE_ELEMENT_WIDE_SHELL": str(shell),
            "LUMINA_NLTE_ELEMENT_WIDE_COMMIT": "0",
            "LUMINA_NLTE_ELEMENT_WIDE_DUMP": "1",
            "LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR": str(dest),
        })
    with (dest / "stdout.txt").open("wb") as stdout, \
         (dest / "stderr.txt").open("wb") as stderr:
        subprocess.run([str(args.bench), str(args.frozen), str(args.model),
                        str(dest)], cwd=ROOT, env=env, stdout=stdout,
                       stderr=stderr, check=True)
    return dest


def main():
    args = arguments()
    if not args.no_build:
        subprocess.run(["make", "bench_frozen_oracle"], cwd=ROOT, check=True)
    for path in (args.frozen, args.model, args.bench):
        if not path.exists():
            raise FileNotFoundError(path)
    out = args.out.resolve() if args.out else Path(
        tempfile.mkdtemp(prefix="wave32_r1_", dir="/tmp"))
    out.mkdir(parents=True, exist_ok=True)
    compared = []
    for super_levels in (0, 1):
        for shell in (0, 8):
            armed = run_cell(args, out, shell, True, super_levels)
            unarmed = run_cell(args, out, shell, False, super_levels)
            zlist = "26" if shell == 0 else "16,26"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/wave3_d8_pair_dump.py"),
                "--frozen", str(args.frozen), "--armed-dir", str(armed),
                "--unarmed-dir", str(unarmed), "--shell", str(shell),
                "--z", zlist], cwd=ROOT, check=True)
            names = [f"lumina_oracle_cell_s{shell}.csv",
                     "pair_ion_fractions.csv", "pair_level_populations.csv"]
            for name in names:
                left, right = armed / name, unarmed / name
                if left.read_bytes() != right.read_bytes():
                    print(f"FAIL super={super_levels} s{shell} {name}: byte diff",
                          file=sys.stderr)
                    return 1
                compared.append((super_levels, shell, name,
                                 left.stat().st_size, sha256(left)))
    summary = out / "byte_invariant_summary.csv"
    with summary.open("w", newline="") as handle:
        handle.write("super_levels,shell,artifact,bytes,sha256,byte_equal\n")
        for super_levels, shell, name, size, digest in compared:
            handle.write(f"{super_levels},{shell},{name},{size},{digest},1\n")
    print(f"PASS: {len(compared)} SUPER_LEVELS={{0,1}} x "
          "armed COMMIT=0/unarmed byte comparisons")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
