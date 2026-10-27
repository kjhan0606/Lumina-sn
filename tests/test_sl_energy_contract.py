import unittest
import numpy as np
from check_lumina_sl_energy_contract import atomic_proxy, cycle_work
import cmfgen_model_continuum as B


class EnergyContractTests(unittest.TestCase):
    def test_fallback_is_not_a_level_energy_zero_shift(self):
        frequency = np.array([1., 2., 3.]) * B.EV / B.H
        physical = cycle_work(frequency, np.ones(3), [1, 1, -1])
        ideal = cycle_work(frequency, np.array([0., 1.25, 2.5 / 3]), [1, 1, -1])
        restored = cycle_work(frequency, np.array([1., 1.25, 2.5 / 3]), [1, 1, -1])
        self.assertAlmostEqual(physical['physical_cycle_defect_eV'], 0.)
        self.assertAlmostEqual(ideal['effective_cycle_defect_eV'], 0.)
        self.assertAlmostEqual(restored['effective_cycle_defect_eV'], 1.)

    def test_electron_excess_is_subtracted_not_added_twice(self):
        ledger = np.zeros((2, 3, 7)); ledger[0, :, :5] = [10., 20., 3., 4., 5.]
        ledger[1, :, :5] = [-1., -2., -3., -4., -5.]
        np.testing.assert_array_equal(atomic_proxy(ledger), [27., 27., 27.])


if __name__ == '__main__':
    unittest.main()
