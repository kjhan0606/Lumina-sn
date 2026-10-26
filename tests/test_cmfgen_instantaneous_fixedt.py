"""자체장 fixed-T 제어의 정확한 범위와 fail-closed 시험."""
import unittest
from configure_cmfgen_stageA import set_control
from run_cmfgen_instantaneous_fixedt import configure
from run_cmfgen_lambda_probe import control


def fixture():
    values = dict(FIX_T='T', FIX_T_AUTO='F', FIX_NE='F', DO_DDT='F', DO_NG='F',
        TRT_NON_TE='F', INC_GRID='F', SCL_LN='T', INS_DEP='T', GAMRAY_TRANS='LOCAL',
        MAX_LIN='3.0D0', SCALE_OPT='LOCAL', LAM_VAL='400.0D0', NUM_LAM='2',
        WRITE_RATES='F', WRITE_JH='T', USE_FIXED_J='T', MAX_LAM='1.0D6',
        COMP_BA='F', N_FIX_BA='2', STORE_BA='T', STORE_BA_INV='T', WR_PRT_INV='T')
    return ('\n'.join(f'{v} [{k}]' for k, v in values.items()) + '\n',
            '1 [NUM_ITS]\nT [DO_LAM_IT]\nF [DO_LAM_AUTO]\nF [DO_T_AUTO]\n')


class FixedTTests(unittest.TestCase):
    def test_new_field_matrix_and_finite_caps_without_temperature_release(self):
        v, i = configure(*fixture())
        for k, expected in dict(FIX_T='T', COMP_BA='T', N_FIX_BA='0', MAX_LAM='10.0D0',
            USE_FIXED_J='F', LAM_VAL='400.0D0', NUM_LAM='2', STORE_BA='F').items():
            self.assertEqual(control(v, k), expected)
        self.assertEqual(control(i, 'NUM_ITS'), '4')
        self.assertEqual(control(i, 'DO_LAM_IT'), 'F')
        self.assertEqual(control(i, 'DO_T_AUTO'), 'F')

    def test_changed_physics_or_automatic_release_rejected(self):
        v, i = fixture()
        for k, value in (('FIX_T', 'F'), ('SCL_LN', 'F'), ('GAMRAY_TRANS', 'NL'),
                         ('LAM_VAL', '1.0D100')):
            with self.assertRaises(ValueError):
                configure(set_control(v, k, value), i)
        with self.assertRaises(ValueError):
            configure(v, set_control(i, 'DO_T_AUTO', 'T'))


if __name__ == '__main__':
    unittest.main()
