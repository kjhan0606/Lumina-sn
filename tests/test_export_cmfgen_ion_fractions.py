import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.export_cmfgen_ion_fractions import export, extract_rows
from scripts.ptf_gated_metrics import (
    BatteryError,
    ScrtempReader,
    read_dimensions,
    read_variables,
)


class ExportCmfgenIonFractionsTest(unittest.TestCase):
    def make_rundir(self, root: Path) -> tuple[Path, np.ndarray]:
        rundir = root / "cmfgen"
        rundir.mkdir()
        (rundir / "MODEL").write_text(
            "2 Number of depth points\n12 Total number of variables\n",
            encoding="utf-8",
        )
        links = [
            (1, "Sk2"), (2, "SkIII"), (3, "SkIV"), (4, "SkV"),
            # Variable 5 is the inferred Si VI terminal closure.
            (6, "S2"), (7, "SIII"), (8, "SIV"), (9, "SV"),
            # Variable 10 is the inferred S VI terminal closure.
        ]
        (rundir / "LEVEL_SL_STEQ_LINKS").write_text(
            "".join(f"{ion} 1 0 1 {index} level-{index}\n" for index, ion in links),
            encoding="utf-8",
        )

        first = np.asarray([
            [8.0, 0.0], [2.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 4.0],
            [3.0, 1.0], [1.0, 3.0], [0.0, 2.0], [0.0, 4.0], [6.0, 10.0],
            [10.0, 20.0], [0.5, 0.6],
        ])
        second = first.copy()
        second[0] = [5.0, 1.0]
        second[1] = [5.0, 2.0]
        states = np.stack((first, second))

        rec_values = 2047
        raw = np.zeros(4 * rec_values, dtype="<f8")
        raw[:2] = [1.0, 2.0]
        for state_index, state in enumerate(states):
            start = (2 + state_index) * rec_values
            raw[start:start + state.size] = state.T.ravel()
        raw.tofile(rundir / "SCRTEMP")
        (rundir / "RVTJ").write_text(
            "Radius (10^10 cm)\n1.0 2.0\n"
            "Electron density\n10.0 20.0\n"
            "Temperature (10^4K)\n0.5 0.6\n"
            "Velocity (km/s)\n2000.0 1000.0\n",
            encoding="utf-8",
        )
        self.write_pop_file(
            rundir / "POPSi", "Si", [10.0, 10.0],
            [("Sk2", [[8.0, 0.0]], [2.0, 3.0]),
             ("SkIII", [[2.0, 3.0]], [0.0, 2.0]),
             ("SkIV", [[0.0, 2.0]], [0.0, 1.0]),
             ("SkV", [[0.0, 1.0]], [0.0, 4.0])],
        )
        self.write_pop_file(
            rundir / "POPS", "S", [10.0, 20.0],
            [("S2", [[3.0, 1.0]], [1.0, 3.0]),
             ("SIII", [[1.0, 3.0]], [0.0, 2.0]),
             ("SIV", [[0.0, 2.0]], [0.0, 4.0]),
             ("SV", [[0.0, 4.0]], [6.0, 10.0])],
        )
        return rundir, states

    @staticmethod
    def write_pop_file(path, element, species_total, ion_blocks):
        def fmt(values):
            return " " + "".join(f"{value:16.7E}" for value in values) + "\n"

        lines = [f" {element}/He abundance:                  1.0\n", fmt(species_total)]
        for ion, populations, next_ion_ground in ion_blocks:
            lines.extend([
                f" Number of {ion} levels:                    {len(populations)}\n",
                f" {ion} Oscillator date: test-vintage\n",
            ])
            lines.append(fmt([value for level in populations for value in level]))
            lines.append(fmt(next_ion_ground))
        path.write_text("".join(lines), encoding="utf-8")

    def test_exports_all_stages_shells_iterations_and_terminal_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rundir, _states = self.make_rundir(root)
            nd, nt = read_dimensions(rundir)
            variables = read_variables(rundir, nt - 2)
            reader = ScrtempReader(rundir / "SCRTEMP", nd, nt)
            velocity = np.asarray([2000.0, 1000.0])

            rows = extract_rows(reader, variables, range(1, 3), velocity)

            self.assertEqual(len(rows), 2 * 2 * 10)
            si_v = [row for row in rows if row["iteration"] == 1
                    and row["element"] == "Si" and row["ion_stage"] == "V"]
            self.assertEqual([row["fraction_of_element"] for row in si_v], [0.0, 0.1])
            self.assertFalse(any(row["includes_terminal_closure"] for row in si_v))
            si_vi = [row for row in rows if row["iteration"] == 1
                     and row["element"] == "Si" and row["ion_stage"] == "VI"]
            self.assertEqual([row["fraction_of_element"] for row in si_vi], [0.0, 0.4])
            self.assertTrue(all(row["includes_terminal_closure"] for row in si_vi))
            self.assertEqual([row["velocity_km_s"] for row in si_v], [2000.0, 1000.0])
            self.assertEqual([row["radius_1e10_cm"] for row in si_v], [1.0, 2.0])
            s_ii = [row for row in rows if row["iteration"] == 1
                    and row["element"] == "S" and row["ion_stage"] == "II"]
            self.assertEqual([row["fraction_of_element"] for row in s_ii], [0.3, 0.05])
            for iteration in (1, 2):
                for element in ("Si", "S"):
                    for depth in (1, 2):
                        stage_rows = [row for row in rows
                                      if row["iteration"] == iteration
                                      and row["element"] == element
                                      and row["depth_index_1based"] == depth]
                        self.assertAlmostEqual(
                            sum(row["fraction_of_element"] for row in stage_rows), 1.0
                        )

    def test_export_binds_hashes_range_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rundir, _states = self.make_rundir(root)
            csv_path = root / "result.csv"
            json_path = root / "result.json"

            metadata = export(rundir, csv_path, json_path, 2, 2, 1)

            self.assertEqual(metadata["iteration_range"], {"first": 2, "last": 2, "count": 1})
            self.assertEqual(metadata["row_count"], 20)
            self.assertEqual(len(metadata["input_sha256"]["SCRTEMP"]), 64)
            self.assertEqual(len(metadata["input_sha256"]["RVTJ"]), 64)
            self.assertEqual(len(metadata["input_sha256"]["POPSi"]), 64)
            self.assertEqual(len(metadata["input_sha256"]["POPS"]), 64)
            self.assertAlmostEqual(
                metadata["population_conservation_and_di_crosscheck"]["Si"]
                ["terminal_di_fraction_max"], 0.4
            )
            self.assertEqual(
                metadata["population_conservation_and_di_crosscheck"]["Si"]
                ["terminal_di_stage"], "VI"
            )
            self.assertAlmostEqual(
                metadata["population_conservation_and_di_crosscheck"]["S"]
                ["terminal_di_fraction_max"], 0.6
            )
            self.assertTrue(metadata["depth_coordinate"]["rvtj_final_state_validation"]["complete"])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["iteration"]) for row in rows}, {2})
            self.assertEqual(json.loads(json_path.read_text())["csv_sha256"], metadata["csv_sha256"])
            with self.assertRaises(FileExistsError):
                export(rundir, csv_path, root / "other.json", 2, 2, 1)

    def test_canonical_cmfgen_species_filenames_and_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rundir, _ = self.make_rundir(root)
            for old, new, symbol, internal in (
                    ("POPSi", "POPSIL", "Si", "SIL"),
                    ("POPS", "POPSUL", "S", "SUL")):
                source = rundir / old
                source.write_text(source.read_text().replace(
                    f"{symbol}/He abundance:", f"{internal}/He abundance:"))
                source.rename(rundir / new)
            metadata = export(rundir, root / "real-names.csv",
                              root / "real-names.json", 2, 2, 1)
            self.assertIn("POPSIL", metadata["input_sha256"])
            self.assertIn("POPSUL", metadata["input_sha256"])

    def test_export_rejects_species_total_or_terminal_di_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rundir, _states = self.make_rundir(root)
            popsi = rundir / "POPSi"
            original = popsi.read_text(encoding="utf-8")
            popsi.write_text(original.replace("1.0000000E+01", "1.1000000E+01", 1),
                             encoding="utf-8")
            with self.assertRaisesRegex(BatteryError, "POP_SPECIES"):
                export(rundir, root / "bad-total.csv", root / "bad-total.json", 2, 2, 1)

            self.write_pop_file(
                popsi, "Si", [10.0, 10.0],
                [("Sk2", [[8.0, 0.0]], [2.0, 3.0]),
                 ("SkIII", [[2.0, 3.0]], [0.0, 2.0]),
                 ("SkIV", [[0.0, 2.0]], [0.0, 1.0]),
                 ("SkV", [[0.0, 1.0]], [0.0, 3.0])],
            )
            with self.assertRaisesRegex(BatteryError, "terminal closure"):
                export(rundir, root / "bad-di.csv", root / "bad-di.json", 2, 2, 1)

    def test_export_rejects_stale_rvtj_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rundir, _states = self.make_rundir(root)
            rvtj = rundir / "RVTJ"
            rvtj.write_text(
                rvtj.read_text(encoding="utf-8").replace(
                    "Electron density\n10.0 20.0", "Electron density\n10.0 22.0"
                ),
                encoding="utf-8",
            )
            csv_path = root / "result.csv"
            json_path = root / "result.json"

            with self.assertRaises(BatteryError):
                export(rundir, csv_path, json_path, 2, 2, 1)

            self.assertFalse(csv_path.exists())
            self.assertFalse(json_path.exists())


if __name__ == "__main__":
    unittest.main()
