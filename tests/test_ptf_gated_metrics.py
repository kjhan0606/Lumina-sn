import unittest

import numpy as np

from scripts.ptf_gated_metrics import (
    BatteryError,
    OutgenRecord,
    Variable,
    gate_metrics,
    ion_fraction_active_correction_summary,
    mapped_solution_pairs,
    population_correction_summary,
    solution_iterations,
    validate_solution,
)


class IterationMappingTest(unittest.TestCase):
    class Reader:
        state_first_it = 40
        last_it = 51

        def has_iteration(self, iteration):
            return self.state_first_it <= iteration <= self.last_it

    def test_restart_outgen_ids_map_to_newest_steq_suffix(self):
        solutions = [np.asarray([[float(index)]]) for index in range(50)]
        outgen = {iteration: OutgenRecord() for iteration in range(41, 51)}

        iterations = solution_iterations(solutions, outgen, self.Reader())
        pairs = mapped_solution_pairs(solutions, iterations)

        self.assertEqual(iterations, list(range(41, 51)))
        self.assertEqual([int(solution[0, 0]) for _, solution in pairs], list(range(40, 50)))

    def test_no_outgen_records_keep_state_range_fallback(self):
        solutions = [np.zeros((1, 1)) for _ in range(3)]

        self.assertEqual(
            solution_iterations(solutions, {}, self.Reader()), [49, 50, 51]
        )


class PopulationCorrectionSummaryTest(unittest.TestCase):
    def setUp(self):
        self.variables = [
            Variable(1, "SiII", "Si", "Si", 1, 1, "ground"),
            Variable(2, "SiIII", "Si", "Si", 1, 2, "ground"),
        ]

    def test_uses_all_population_cells_without_activity_filter(self):
        solution = np.asarray(
            [[0.01, -0.02], [0.03, -0.04], [0.0, 0.0]], dtype=np.float64
        )

        result = population_correction_summary(solution, self.variables)

        self.assertEqual(result["population_cells"], 4)
        self.assertAlmostEqual(result["p95_percent"], 3.85)
        self.assertAlmostEqual(result["max_percent"], 4.0)
        self.assertEqual(
            result["max_owner"],
            {"variable": 2, "ion": "SiIII", "superlevel": 1, "depth": 2},
        )
        self.assertIn("no activity filtering", result["selection"])

    def test_rejects_non_finite_population_corrections(self):
        solution = np.asarray([[np.nan, 0.0], [0.0, 0.0]])
        with self.assertRaises(BatteryError):
            population_correction_summary(solution, self.variables)

    def test_rejects_missing_population_rows(self):
        solution = np.zeros((1, 2), dtype=np.float64)
        with self.assertRaises(BatteryError):
            population_correction_summary(solution, self.variables)

    def test_gate_report_carries_unfiltered_diagnostic_separately(self):
        solution = np.asarray([[0.01, -0.02], [0.03, -0.04]], dtype=np.float64)
        state = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

        result = gate_metrics(
            solution,
            state,
            self.variables,
            {"Si": np.asarray([0, 1], dtype=np.int64)},
            threshold=0.0,
        )

        self.assertAlmostEqual(
            result["all_population_correction"]["p95_percent"], 3.85
        )
        self.assertIn(
            "no activity filtering",
            result["all_population_correction"]["selection"],
        )

    def test_returned_maxch_accepts_source_precision_interval(self):
        solution = np.asarray([[0.9985] + [0.2] * 9 + [-0.01]])
        record = OutgenRecord(
            increase_percent=1.0,
            decrease_percent=99.85,
            returned_maxch_percent=66722.08712577267,
        )

        result = validate_solution(82, solution, {82: record})

        returned = next(check for check in result["checks"] if check["field"] == "returned")
        low, high = returned["printed_solution_interval"]
        self.assertTrue(returned["ok"])
        self.assertLessEqual(low, record.returned_maxch_percent)
        self.assertGreaterEqual(high, record.returned_maxch_percent)

    def test_returned_maxch_rejects_values_outside_source_precision_interval(self):
        solution = np.asarray([[0.9985] + [0.2] * 9 + [-0.01]])
        record = OutgenRecord(
            increase_percent=1.0,
            decrease_percent=99.85,
            returned_maxch_percent=90000.0,
        )

        with self.assertRaises(BatteryError):
            validate_solution(82, solution, {82: record})

    def test_ion_fraction_activity_is_explicitly_a_subset(self):
        variables = [
            Variable(1, "SkII", "Sk", "Si", 1, 1, "Si II ground"),
            Variable(2, "SkIII", "Sk", "Si", 1, 2, "Si III ground"),
            Variable(3, "CaII", "Ca", "Ca", 1, 1, "Ca II trace"),
            Variable(4, "CaIII", "Ca", "Ca", 1, 2, "Ca III ground"),
        ]
        pre_state = np.asarray(
            [
                [9.0, 0.1],
                [1.0, 9.9],
                [1.0e-9, 1.0e-10],
                [1.0 - 1.0e-9, 1.0 - 1.0e-10],
            ]
        )
        solution = np.asarray(
            [
                [0.01, 0.02],
                [0.03, 0.04],
                [1.0, 2.0],
                [0.05, 0.06],
            ]
        )
        grouped = {"Si": np.asarray([0, 1]), "Ca": np.asarray([2, 3])}

        result = ion_fraction_active_correction_summary(
            solution, pre_state, variables, grouped
        )

        self.assertEqual(result["active_cells"], 6)
        self.assertAlmostEqual(result["p95_percent"], 5.75)
        self.assertAlmostEqual(result["max_percent"], 6.0)
        self.assertEqual(result["max_owner"]["ion"], "CaIII")
        self.assertEqual(result["max_owner"]["depth"], 2)
        self.assertFalse(result["contribution_channels_included"])


if __name__ == "__main__":
    unittest.main()
