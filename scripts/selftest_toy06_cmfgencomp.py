#!/usr/bin/env python3
"""CPU-only synthetic and production-input self-tests for composition mapping."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from toy06_cmfgen_composition import (
    CompositionError,
    DAY_S,
    FOUR_PI_OVER_THREE,
    KM_CM,
    MSUN_G,
    conservative_map,
    parse_sn_hydro_data,
    read_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
AGE_D = 19.48


def write_values(stream, values):
    for start in range(0, len(values), 6):
        stream.write("  " + "  ".join(f"{value:.8E}" for value in values[start:start + 6]) + "\n")
    stream.write("\n")


def synthetic_hydro(path: Path, extra_block: bool = False) -> None:
    velocity_inner = [1025.0 + 50.0 * index for index in range(700)]
    time_s = AGE_D * DAY_S
    radius_inner = [value * KM_CM * time_s / 1.0e10 for value in velocity_inner]
    density_inner = [1.0e-14 * math.exp(-(value - 1000.0) / 15000.0)
                     for value in velocity_inner]
    arrays = {name: [] for name in ("Si", "S", "Ca", "Fe", "Co", "Ni",
                                    "Ni56", "Co56", "Fe56")}
    for velocity in velocity_inner:
        if velocity < 18000.0:
            values = (0.0, 0.0, 0.0, 0.098, 0.794, 0.108)
        else:
            values = (0.55, 0.35, 0.10, 0.0, 0.0, 0.0)
        for name, value in zip(("Si", "S", "Ca", "Fe", "Co", "Ni"), values):
            arrays[name].append(value)
        arrays["Ni56"].append(values[5])
        arrays["Co56"].append(values[4])
        arrays["Fe56"].append(values[3])
    # Canonical ordering is outer->inner.
    rev = lambda values: list(reversed(values))
    blocks = [
        ("Radius grid (10^10cm)", rev(radius_inner)),
        ("Velocity (km/s)", rev(velocity_inner)),
        ("Density (g/cm^3)", rev(density_inner)),
        ("SIL mass fraction", rev(arrays["Si"])),
        ("SUL mass fraction", rev(arrays["S"])),
        ("CAL mass fraction", rev(arrays["Ca"])),
        ("IRON mass fraction", rev(arrays["Fe"])),
        ("COB mass fraction", rev(arrays["Co"])),
        ("NICK mass fraction", rev(arrays["Ni"])),
        ("NICK 56 mass fraction", rev(arrays["Ni56"])),
        ("COB 56 mass fraction", rev(arrays["Co56"])),
        ("IRON 56 mass fraction", rev(arrays["Fe56"])),
    ]
    with path.open("w", encoding="latin-1") as stream:
        stream.write("! synthetic toy06, truncated v in [1000,36000] km/s, outer->inner\n")
        stream.write("Number of data points: 700\n")
        stream.write("Number of mass fractions: 6\n")
        stream.write("Number of isotopes: 3\n")
        stream.write(f"Time(days) since explosion: {AGE_D}\n\n")
        for label, values in blocks:
            stream.write(label + "\n")
            write_values(stream, values)
        if extra_block:
            stream.write("OXY mass fraction\n")
            write_values(stream, [0.0] * 700)


def synthetic_geometry(path: Path, outer_km_s: float = 36000.0) -> None:
    time_s = AGE_D * DAY_S
    low = 1000.0
    step = (outer_km_s - low) / 50
    with path.open("w", encoding="utf-8") as stream:
        stream.write("shell_id,r_inner,r_outer,v_inner,v_outer\n")
        for shell in range(50):
            vi = low + shell * step
            vo = low + (shell + 1) * step
            stream.write(f"{shell},{vi*KM_CM*time_s:.17g},{vo*KM_CM*time_s:.17g},"
                         f"{vi*KM_CM:.17g},{vo*KM_CM:.17g}\n")


class CompositionMappingTests(unittest.TestCase):
    def test_exact_domain_is_conservative_and_stratified(self):
        with tempfile.TemporaryDirectory(prefix="toy06_comp_selftest_") as temporary:
            root = Path(temporary)
            synthetic_hydro(root / "SN_HYDRO_DATA")
            synthetic_geometry(root / "geometry.csv")
            hydro = parse_sn_hydro_data(root / "SN_HYDRO_DATA")
            geometry = read_geometry(root / "geometry.csv", AGE_D)
            mapped = conservative_map(hydro, geometry)
            self.assertEqual(len(geometry), 50)
            self.assertTrue(all(abs(value - 1.0) < 2.0e-12
                                for value in mapped.volume_coverage))
            self.assertAlmostEqual(sum(mapped.source_species_mass_msun.values()),
                                   sum(mapped.mapped_species_mass_msun.values()), places=12)
            self.assertGreater(mapped.abundances[27][0], 0.79)
            self.assertAlmostEqual(mapped.abundances[14][-1], 0.55, places=12)
            self.assertAlmostEqual(mapped.abundances[16][-1], 0.35, places=12)
            self.assertAlmostEqual(mapped.abundances[20][-1], 0.10, places=12)

    def test_no_outer_extrapolation(self):
        with tempfile.TemporaryDirectory(prefix="toy06_comp_selftest_") as temporary:
            root = Path(temporary)
            synthetic_hydro(root / "SN_HYDRO_DATA")
            synthetic_geometry(root / "geometry.csv", outer_km_s=40300.0)
            hydro = parse_sn_hydro_data(root / "SN_HYDRO_DATA")
            geometry = read_geometry(root / "geometry.csv", AGE_D)
            mapped = conservative_map(hydro, geometry)
            incomplete = [index for index, value in enumerate(mapped.volume_coverage)
                          if value < 1.0 - 2.0e-12]
            self.assertTrue(incomplete)
            self.assertTrue(math.isnan(mapped.abundances[14][-1]))

    def test_extra_canonical_element_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="toy06_comp_selftest_") as temporary:
            path = Path(temporary) / "SN_HYDRO_DATA"
            synthetic_hydro(path, extra_block=True)
            with self.assertRaisesRegex(CompositionError, "mass-fraction block set differs"):
                parse_sn_hydro_data(path)

    @unittest.skipUnless((RUN / "SN_HYDRO_DATA").is_file(), "production CMFGEN truth unavailable")
    def test_production_coordinates_expose_expected_uncovered_shells(self):
        hydro = parse_sn_hydro_data(RUN / "SN_HYDRO_DATA")
        geometry = read_geometry(
            ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos/geometry.csv",
            hydro.age_days)
        mapped = conservative_map(hydro, geometry)
        incomplete = [index for index, value in enumerate(mapped.volume_coverage)
                      if value < 1.0 - 2.0e-12]
        self.assertEqual((hydro.v_edges_km_s[0], hydro.v_edges_km_s[-1]),
                         (1000.0, 36000.0))
        self.assertEqual(incomplete, [44, 45, 46, 47, 48, 49])
        self.assertTrue(math.isnan(mapped.abundances[14][49]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
