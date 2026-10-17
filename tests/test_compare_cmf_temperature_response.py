"""온도 대조의 셀 합산만 검사한다. 물리 응답의 수렴 시험이 아니다."""
import unittest
from compare_cmf_temperature_response import parent_profiles


class ParentProfilesTest(unittest.TestCase):
    def rows(self):
        return [dict(parent=39, Z=16, stage=2, nuclei_shell=w,
                     constant_fraction=a, reconstructed_fraction=b, reference_fraction=c,
                     velocity_inner_km_s=lo, velocity_outer_km_s=hi)
                for w,a,b,c,lo,hi in ((1., .2, .4, .3, 12100., 12125.),
                                      (3., .6, .8, .5, 12125., 12150.))]

    def test_nuclei_weighted_not_unweighted(self):
        result, = parent_profiles(self.rows())
        self.assertEqual(result['nuclei_shell'], 4.)
        self.assertAlmostEqual(result['constant_fraction'], .5)
        self.assertAlmostEqual(result['reconstructed_fraction'], .7)
        self.assertAlmostEqual(result['reference_fraction'], .45)
        self.assertAlmostEqual(result['fraction_change'], .2)
        self.assertEqual(result['child_shells'], 2)
        self.assertEqual(result['velocity_inner_km_s'], 12100.)
        self.assertEqual(result['velocity_outer_km_s'], 12150.)

    def test_missing_reference_is_not_zero_or_renormalized(self):
        rows = self.rows();rows[1]['reference_fraction'] = None
        result, = parent_profiles(rows)
        self.assertIsNone(result['reference_fraction'])
        self.assertAlmostEqual(result['constant_fraction'], .5)

    def test_zero_nuclei_has_no_fraction(self):
        rows = self.rows()
        for row in rows:
            row.update(nuclei_shell=0., constant_fraction=None, reconstructed_fraction=None)
        result, = parent_profiles(rows)
        for key in ('constant_fraction', 'reconstructed_fraction', 'reference_fraction', 'fraction_change'):
            self.assertIsNone(result[key])

    def test_elements_stages_and_parents_are_distinct(self):
        rows = self.rows()
        rows += [dict(rows[0], Z=14), dict(rows[0], stage=1), dict(rows[0], parent=40)]
        result = parent_profiles(rows)
        self.assertEqual(len(result), 4)
        self.assertEqual(sum(r['child_shells'] for r in result), len(rows))
        self.assertEqual(sum(r['nuclei_shell'] for r in result), sum(r['nuclei_shell'] for r in rows))


if __name__ == '__main__':
    unittest.main()
