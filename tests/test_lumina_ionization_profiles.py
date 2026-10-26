import tempfile
from pathlib import Path
import unittest
import numpy as np
from compare_lumina_ionization_profiles import compare, diagnostics


class ProfileTests(unittest.TestCase):
    def test_positive_subnormal_is_not_lost_in_ratio_underflow(self):
        log, summary = diagnostics([1., 1.e-320], [1.e-320, 1.])
        self.assertTrue(np.isfinite(log).all())
        self.assertEqual(summary['active_support_mismatch_count'], 0)
        self.assertGreater(summary['active_log_abs_max_dex'], 319.)

    def test_small_absolute_difference_can_have_large_log_difference(self):
        log, summary = diagnostics([1.e-6, .5], [2.e-6, .51])
        self.assertAlmostEqual(log[0], np.log10(2.))
        self.assertEqual(summary['active_log_above_0p1_dex_count'], 1)
        self.assertEqual(summary['active_log_above_0p1_dex_nodes_zero_based'], [0])

    def test_zero_support_is_not_floored(self):
        log, summary = diagnostics([0., 1.e-7, 0.], [1.e-6, 0., 0.])
        self.assertTrue(np.isnan(log).all())
        self.assertEqual(summary['active_support_mismatch_count'], 2)
        self.assertIsNone(summary['active_log_abs_max_dex'])

    def test_either_side_defines_fraction_active_subset(self):
        _, summary = diagnostics([1.e-9, 1.e-6], [1.e-6, 1.e-9])
        self.assertEqual(summary['known_fraction_active_points'], 2)
        self.assertEqual(summary['active_log_above_0p1_dex_count'], 2)
        for bad in ([-1., 1.], [np.nan, 1.]):
            with self.assertRaises(ValueError):
                diagnostics(bad, [1., 1.])

    def test_existing_comparison_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileExistsError):
                compare(root / 'absent.json', 'bad-hash', root)


if __name__ == '__main__':
    unittest.main()
