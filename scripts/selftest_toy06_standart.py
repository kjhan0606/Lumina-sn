#!/usr/bin/env python3
"""CPU-only in-memory fixtures; does not create the requested deck."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

from standart_toy06_composition import (
    EXPECTED_Z, conservative_map, core_decay_fractions, decay_to_epoch,
    parse_standart_model, read_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/standart_data1/input_models/snia_toy06_1h_lowres.dat"
GEOMETRY = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos/geometry.csv"
TARGET_D = 19.48


class StandartCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = parse_standart_model(MODEL)
        cls.decayed = decay_to_epoch(cls.model, TARGET_D)
        cls.shells = read_geometry(GEOMETRY, TARGET_D)
        cls.mapped = conservative_map(cls.model, cls.decayed, cls.shells)

    def test_canonical_measured_facts(self):
        self.assertEqual(len(self.model.velocity_km_s), 202)
        self.assertEqual((self.model.velocity_km_s[0], self.model.velocity_km_s[-1]),
                         (100.0, 40300.0))
        self.assertEqual((self.model.velocity_edges_km_s[0],
                          self.model.velocity_edges_km_s[-1]), (0.0, 40400.0))
        self.assertEqual({name: self.model.zero_counts[name]
                          for name in ("Ti", "O", "C")},
                         {"Ti": 202, "O": 202, "C": 202})
        self.assertEqual({name: self.model.positive_counts[name]
                          for name in ("Ni", "Co", "Fe", "Ca", "S", "Si")},
                         {"Ni": 62, "Co": 62, "Fe": 62,
                          "Ca": 169, "S": 169, "Si": 169})

    def test_decay_matches_secondary_core_anchor(self):
        got = core_decay_fractions(TARGET_D)
        expected = (0.1083233103068866, 0.7937131695869826,
                    0.09796352010613085)
        self.assertLess(max(abs(a - b) for a, b in zip(got, expected)), 5.0e-12)

    def test_mapping_covers_50_shells_and_conserves_mass(self):
        self.assertEqual(len(self.shells), 50)
        self.assertTrue(all(abs(value - 1.0) <= 2.0e-12
                            for value in self.mapped.volume_coverage))
        self.assertTrue(all(mass > 0.0 for mass in self.mapped.shell_mass_g))
        for s in range(50):
            total = sum(self.mapped.abundances[z][s] for z in EXPECTED_Z)
            self.assertLess(abs(total - 1.0), 2.0e-12)
        for name in self.mapped.mapped_species_msun:
            self.assertLess(abs(self.mapped.mapped_species_msun[name] -
                                self.mapped.source_overlap_species_msun[name]), 2.0e-12)

    def test_explicit_normalization_is_small_and_reportable(self):
        delta = max(abs(value - 1.0) for value in self.decayed.normalization_factors)
        self.assertGreater(delta, 0.0)
        self.assertLess(delta, 1.0e-5)

    def test_layering(self):
        self.assertGreater(self.mapped.abundances[27][0],
                           self.mapped.abundances[28][0])
        self.assertAlmostEqual(self.mapped.abundances[14][-1], 0.55, places=12)
        self.assertAlmostEqual(self.mapped.abundances[16][-1], 0.35, places=12)
        self.assertAlmostEqual(self.mapped.abundances[20][-1], 0.10, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
