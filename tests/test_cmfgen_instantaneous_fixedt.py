"""자체장 fixed-T 제어의 정확한 범위와 fail-closed 시험."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import numpy as np
import run_cmfgen_instantaneous_fixedt as R
import continue_cmfgen_instantaneous_fixedt as C
from configure_cmfgen_stageA import set_control
from run_cmfgen_instantaneous_fixedt import ba_policy_violation, configure
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
    def test_explicit_memory_must_match_actual_slurm_allocation(self):
        for memory in (32, 128):
            env = dict(OMP_NUM_THREADS='16', SLURM_CPUS_PER_TASK='16',
                       SLURM_MEM_PER_NODE=str(memory * 1024), SLURM_JOB_ID='123')
            self.assertIn(f'/{memory}GiB/', R.check_resources(memory, env))
            with self.assertRaises(ValueError):
                R.check_resources(128 if memory == 32 else 32, env)
            for key, value in (('OMP_NUM_THREADS', '8'), ('SLURM_CPUS_PER_TASK', '8'),
                               ('SLURM_JOB_ID', ''), ('SLURM_MEM_PER_NODE', '')):
                with self.subTest(memory=memory, key=key), self.assertRaises(ValueError):
                    R.check_resources(memory, dict(env, **{key: value}))

    def test_unapproved_memory_sizes_or_missing_allocation_rejected(self):
        for memory in (True, 64, 256, '128', 128.):
            with self.assertRaises(ValueError):
                R.check_resources(memory, {})
        with self.assertRaises(ValueError):
            R.check_resources(128, {})

    def test_long_runtime_is_explicit_and_preserves_previous_default(self):
        env = dict(OMP_NUM_THREADS='16', SLURM_CPUS_PER_TASK='16',
                   SLURM_MEM_PER_NODE=str(128 * 1024), SLURM_JOB_ID='123')
        self.assertIn('child6600s', R.check_resources(128, env))
        self.assertIn('child42600s', R.check_resources(128, env, seconds=42600))
        for seconds in (True, 0, 6601, 43200):
            with self.assertRaises(ValueError):
                R.check_resources(128, env, seconds=seconds)

    def test_partial_checkpoint_survives_stale_final_RVTJ(self):
        initial = np.array([[1., 1.], [2., 2.], [4., 4.], [100., 100.]])
        updated = initial.copy()
        updated[2] *= 1.01
        reader = SimpleNamespace(state=lambda iteration: initial if iteration == 70 else updated)
        variables = [SimpleNamespace(species='Si', ion='Sk2'), SimpleNamespace(species='S', ion='S2')]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'POINT1').write_text('header\n71 71 1 -1000 F\n')
            with patch.object(R, 'read_dimensions', return_value=(2, 4)), \
                 patch.object(R, 'ScrtempReader', return_value=reader), \
                 patch.object(R, 'read_variables', return_value=variables), \
                 patch.object(R, 'validate_rvtj', side_effect=RuntimeError('stale RVTJ')):
                result = R.state_summary(root, initial)
            self.assertEqual(result['last_completed_record'], 71)
            self.assertEqual(len(result['iterations']), 1)
            self.assertFalse(result['final_RVTJ']['complete'])

    def test_continuation_summary_accepts_new_complete_records_from_iteration74(self):
        initial = np.array([[1., 1.], [2., 2.], [4., 4.], [100., 100.]])
        updated = initial.copy(); updated[0] *= 1.1
        reader = SimpleNamespace(state=lambda iteration: initial if iteration == 74 else updated)
        variables = [SimpleNamespace(species='Si', ion='Sk2'), SimpleNamespace(species='S', ion='S2')]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'POINT1').write_text('header\n75 75 1 -1000 F\n')
            with patch.object(R,'read_dimensions',return_value=(2,4)), \
                 patch.object(R,'ScrtempReader',return_value=reader), \
                 patch.object(R,'read_variables',return_value=variables), \
                 patch.object(R,'validate_rvtj',return_value={'complete':True}):
                result=R.state_summary(root,initial,first=74,max_iterations=16)
            self.assertEqual(result['last_completed_record'],75)
            self.assertEqual(len(result['iterations']),1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'POINT1').write_text('header\n91 91 1 -1000 F\n')
            with patch.object(R,'read_dimensions',return_value=(2,4)), \
                 patch.object(R,'ScrtempReader',return_value=reader), \
                 patch.object(R,'read_variables',return_value=variables):
                with self.assertRaises(ValueError):
                    R.state_summary(root,initial,first=74,max_iterations=16)

    def test_actual_BA_violation_is_not_successful_stabilization(self):
        self.assertIsNone(ba_policy_violation('Current great iteration count is 71\nBA matrix computed\n'))
        self.assertEqual(ba_policy_violation('read BAMATPNT T\nCurrent great iteration count is 71'),
                         'requested_fresh_BA_initial_cache_read')
        self.assertEqual(ba_policy_violation('BA matrix **NOT** computed'),
                         'requested_fresh_BA_not_computed')

    def test_new_field_matrix_and_finite_caps_without_temperature_release(self):
        v, i = configure(*fixture())
        for k, expected in dict(FIX_T='T', COMP_BA='T', N_FIX_BA='0', MAX_LAM='10.0D0',
            USE_FIXED_J='F', LAM_VAL='400.0D0', NUM_LAM='2', STORE_BA='F').items():
            self.assertEqual(control(v, k), expected)
        self.assertEqual(control(i, 'NUM_ITS'), '4')
        self.assertEqual(control(i, 'DO_LAM_IT'), 'F')
        self.assertEqual(control(i, 'DO_T_AUTO'), 'F')

    def test_iteration_extension_is_explicit_bounded_and_keeps_fixed_temperature(self):
        v, i = configure(*fixture(), max_iterations=16)
        self.assertEqual(control(i, 'NUM_ITS'), '16')
        self.assertEqual(control(v, 'FIX_T'), 'T')
        self.assertEqual(control(i, 'DO_T_AUTO'), 'F')
        for value in (True, 0, 33, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure(*fixture(), max_iterations=value)

    def test_continuation_rejects_mixed_final_pointer_generations(self):
        with patch.object(C,'point_records',return_value={'POINT1':90,'POINT2':89}):
            with self.assertRaises(ValueError):
                C.validate_final_generation(Path('/unused'),90)
        with patch.object(C,'point_records',return_value={'POINT1':90,'POINT2':90}):
            self.assertEqual(C.validate_final_generation(Path('/unused'),90),
                             {'POINT1':90,'POINT2':90})

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
