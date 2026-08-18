#!/usr/bin/env python3
"""Fail-closed census for bound-free thresholds below the Layer-1 grid.

MC-EVT may name an out-of-grid bound-free lookup ``EXACT_ZERO`` only after
this census proves that no implemented (CMFGEN or Kramers-fallback) positive
threshold lies at or below NLTE_NU_MIN.  A non-zero count is not a numerical
failure: it is the contract signal to reopen SH-GRID.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


H_PLANCK = 6.62607015e-27
EV_TO_ERG = 1.602176634e-12
DEFAULT_NU_MIN = 5.8412785919616062e13
CMFGEN_MAGIC = 0x434D4644
CMFGEN_VERSION = 1
LIGHT_ANGSTROM_HZ = 2.99792458e18


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_cmfgen_has(path: Path) -> tuple[int, int, float, float, bytes]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"short CMFGEN BF header: {path}")
        magic, version, n_levels, n_freq, nu_min, nu_max = struct.unpack(
            "<IIiidd", header
        )
        if magic != CMFGEN_MAGIC or version != CMFGEN_VERSION:
            raise ValueError(
                f"bad CMFGEN BF header magic=0x{magic:08x} version={version}"
            )
        has = handle.read(n_levels)
        if len(has) != n_levels:
            raise ValueError(f"short CMFGEN BF has[] slab: {path}")
    return n_levels, n_freq, nu_min, nu_max, has


def census(
    levels: Iterable[dict[str, str]],
    ionization: Iterable[dict[str, str]],
    has_cmfgen: bytes,
    nu_min: float,
) -> tuple[list[dict[str, object]], Counter[str]]:
    ion_eV = {
        (int(row["atomic_number"]), int(row["ion_number"])):
        float(row["ionization_energy_eV"])
        for row in ionization
    }
    below: list[dict[str, object]] = []
    count: Counter[str] = Counter()

    for global_index, row in enumerate(levels):
        z = int(row["atomic_number"])
        ion = int(row["ion_number"])
        level = int(row["level_number"])
        energy_eV = float(row["energy_eV"])
        ip_eV = ion_eV.get((z, ion))
        if ip_eV is None:
            count["missing_ionization_energy"] += 1
            continue

        threshold_eV = ip_eV - energy_eV
        if not math.isfinite(threshold_eV) or threshold_eV <= 0.0:
            count["nonpositive_threshold"] += 1
            continue

        nu_edge = threshold_eV * EV_TO_ERG / H_PLANCK
        count["positive_threshold"] += 1
        if nu_edge > nu_min:
            count["above_grid_min"] += 1
            continue

        baked = bool(has_cmfgen[global_index])
        entry = {
            "global_index": global_index,
            "atomic_number": z,
            "ion_number": ion,
            "level_number": level,
            "energy_eV": energy_eV,
            "threshold_eV": threshold_eV,
            "nu_edge": nu_edge,
            "wavelength_A": LIGHT_ANGSTROM_HZ / nu_edge,
            "cross_section": "CMFGEN" if baked else "KRAMERS_FALLBACK",
            # Production currently excludes neutral BF unless
            # LUMINA_FIX_BF_NEUTRAL is enabled.
            "default_active": ion >= 1,
        }
        below.append(entry)
        count["below_or_at_grid_min_all"] += 1
        count["below_or_at_grid_min_cmfgen" if baked else
              "below_or_at_grid_min_kramers"] += 1
        if ion >= 1:
            count["below_or_at_grid_min_default_active"] += 1
        else:
            count["below_or_at_grid_min_neutral_option"] += 1

    below.sort(key=lambda item: float(item["nu_edge"]))
    return below, count


def selftest() -> int:
    levels = [
        {"atomic_number": "6", "ion_number": "1", "level_number": "0",
         "energy_eV": "9.8"},
        {"atomic_number": "6", "ion_number": "1", "level_number": "1",
         "energy_eV": "1.0"},
        {"atomic_number": "6", "ion_number": "0", "level_number": "0",
         "energy_eV": "10.8"},
    ]
    ions = [
        {"atomic_number": "6", "ion_number": "1",
         "ionization_energy_eV": "10.0"},
        {"atomic_number": "6", "ion_number": "0",
         "ionization_energy_eV": "11.0"},
    ]
    below, count = census(levels, ions, bytes([1, 0, 0]), DEFAULT_NU_MIN)
    expected = {
        "below_or_at_grid_min_all": 2,
        "below_or_at_grid_min_default_active": 1,
        "below_or_at_grid_min_neutral_option": 1,
        "below_or_at_grid_min_cmfgen": 1,
        "below_or_at_grid_min_kramers": 1,
        "above_grid_min": 1,
    }
    for name, value in expected.items():
        if count[name] != value:
            print(
                f"[MC-EVT][BF-EDGE-CENSUS][SELFTEST][FAIL] "
                f"field={name} got={count[name]} expected={value}",
                file=sys.stderr,
            )
            return 4
    if len(below) != 2 or not below[0]["default_active"]:
        print(
            "[MC-EVT][BF-EDGE-CENSUS][SELFTEST][FAIL] ordering_or_activity",
            file=sys.stderr,
        )
        return 4
    print("[MC-EVT][BF-EDGE-CENSUS][SELFTEST] status=PASS rc=0")
    return 0


def resolve_sigma_path(ref_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    local = ref_dir / "cmfgen_sigma_bf.bin"
    if local.is_file():
        return local
    return ref_dir.parent / "atomic" / "cmfgen_sigma_bf.bin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", type=Path,
                        default=Path(
                            "data/tardis_reference_toy06_19p48d_sivcaiv_active"
                        ))
    parser.add_argument("--sigma", type=Path)
    parser.add_argument("--nu-min", type=float, default=DEFAULT_NU_MIN)
    parser.add_argument("--show", type=int, default=20,
                        help="number of lowest-frequency witnesses to print")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not math.isfinite(args.nu_min) or args.nu_min <= 0.0:
        parser.error("--nu-min must be finite and positive")

    levels_path = args.ref_dir / "levels.csv"
    ions_path = args.ref_dir / "ionization_energies.csv"
    sigma_path = resolve_sigma_path(args.ref_dir, args.sigma)
    try:
        levels = read_csv(levels_path)
        ions = read_csv(ions_path)
        n_levels, n_freq, sigma_nu_min, sigma_nu_max, has = load_cmfgen_has(
            sigma_path
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"[MC-EVT][BF-EDGE-CENSUS][ERROR] {exc}", file=sys.stderr)
        return 2

    if n_levels != len(levels):
        print(
            f"[MC-EVT][BF-EDGE-CENSUS][ERROR] level_order_mismatch "
            f"csv={len(levels)} sigma={n_levels}",
            file=sys.stderr,
        )
        return 2

    below, count = census(levels, ions, has, args.nu_min)
    print(
        f"[MC-EVT][BF-EDGE-CENSUS] ref={args.ref_dir} sigma={sigma_path} "
        f"levels={len(levels)} sigma_bins={n_freq} "
        f"sigma_grid=[{sigma_nu_min:.9g},{sigma_nu_max:.9g}] "
        f"transport_nu_min={args.nu_min:.9g}"
    )
    print(
        "[MC-EVT][BF-EDGE-CENSUS] "
        f"positive={count['positive_threshold']} "
        f"above={count['above_grid_min']} "
        f"below_or_at_all={count['below_or_at_grid_min_all']} "
        f"default_active={count['below_or_at_grid_min_default_active']} "
        f"neutral_option={count['below_or_at_grid_min_neutral_option']} "
        f"cmfgen={count['below_or_at_grid_min_cmfgen']} "
        f"kramers_fallback={count['below_or_at_grid_min_kramers']} "
        f"nonpositive={count['nonpositive_threshold']} "
        f"missing_ip={count['missing_ionization_energy']}"
    )
    for row in below[: max(args.show, 0)]:
        print(
            "[MC-EVT][BF-EDGE-CENSUS][WITNESS] "
            f"global={row['global_index']} Z={row['atomic_number']} "
            f"ion={row['ion_number']} level={row['level_number']} "
            f"threshold_eV={float(row['threshold_eV']):.9g} "
            f"nu_edge={float(row['nu_edge']):.9g} "
            f"lambda_A={float(row['wavelength_A']):.9g} "
            f"sigma={row['cross_section']} "
            f"default_active={int(bool(row['default_active']))}"
        )

    active = count["below_or_at_grid_min_default_active"]
    if active:
        print(
            f"[MC-EVT][BF-EDGE-CENSUS][BLOCKED] active_below_or_at_nu_min={active} "
            "action=REOPEN_SH_GRID rc=3"
        )
        return 3

    print(
        "[MC-EVT][BF-EDGE-CENSUS][PASS] active_below_or_at_nu_min=0 "
        "oog_bf_exact_zero_contract=ELIGIBLE rc=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
