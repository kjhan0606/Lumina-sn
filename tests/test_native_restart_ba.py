"""첫 BA 요청 누락의 실제 native 읽기 재현. 전체 물리 수렴 시험은 아니다."""
import itertools
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import build_cmfgen_instantaneous_reference as B


class RestartSourceTests(unittest.TestCase):
    def test_exact_single_site_and_no_repeat_patch(self):
        self.assertEqual(B.restart_patch('a\n' + B.RESTART_OLD + 'b'), 'a\n' + B.RESTART_NEW + 'b')
        for text in ('unrelated', B.RESTART_OLD * 2, B.RESTART_NEW):
            with self.assertRaises(ValueError):
                B.restart_patch(text)


@unittest.skipUnless(os.environ.get('CMFGEN_SOURCE_ROOT'), '명시 native 소스 필요')
class NativeRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='lumina-native-BA-')
        cls.root = Path(cls.tmp.name)
        cls.native = Path(os.environ['CMFGEN_SOURCE_ROOT'])
        paths = {'new_main/cmfgen_sub.f': B.SEALED['new_main/cmfgen_sub.f'],
            'new_main/subs/read_ba_data_v3.f': '8a7d741918b2b7e7ce4af36d324c0fadd9ae633435db08464fbf7e4aadc62869',
            'new_main/mod_subs/steq_data_mod.f': '8681f10c2adc589b99508c919253b9eb212ad5c01f1c08691409f13a7f1c9c44',
            'tools/set_kind_module.f': '835cfb88d2421b55f9d5c2ecf50d1d2f79fcec693732cf571e6bb7ef10dbe06f',
            **{k: v for k, v in B.SEALED.items() if k.startswith('lib/')}}
        cls.hashes = {str(cls.native / k): h for k, h in paths.items()}
        if any(B.sha(Path(p)) != h for p, h in cls.hashes.items()):
            raise ValueError('native 재현 소스/라이브러리 신원 변경')
        source = (cls.native / 'new_main/cmfgen_sub.f').read_text()
        start = source.index(B.RESTART_OLD)
        end = source.index('\tIF(FLUX_CAL_ONLY)THEN', start)
        block = source[start:end]
        template = (Path(__file__).with_name('native_restart_ba_driver.f90')).read_text()
        cls.binary = {}
        for mode, fragment in (('original', block), ('corrected', B.restart_patch(block))):
            folder = cls.root / mode
            folder.mkdir()
            # 블록 자체는 원 fixed-form이므로 driver도 fixed-form으로 변환한다.
            free_lines = template.splitlines()
            fixed_template = '\n'.join(('      ' + line if not line.startswith('!') else line)
                                       for line in free_lines)
            driver = fixed_template.replace('! RESTART_BLOCK', fragment)
            (folder / 'driver.f').write_text(driver + '\n')
            command = ['gfortran', *B.FLAGS, '-I.', '-J.',
                cls.native / 'tools/set_kind_module.f', cls.native / 'new_main/mod_subs/steq_data_mod.f',
                cls.native / 'new_main/subs/read_ba_data_v3.f', 'driver.f',
                '-L' + str(cls.native / 'lib'), '-lsubs', '-lpgplt', '-lmod',
                cls.native / 'lib/libmy_lapack.a', cls.native / 'lib/libmy_blas.a', '-o', 'test.exe']
            subprocess.run(list(map(str, command)), cwd=folder, check=True,
                           capture_output=True, timeout=60)
            cls.binary[mode] = folder / 'test.exe'

    @classmethod
    def tearDownClass(cls):
        assert all(B.sha(Path(p)) == h for p, h in cls.hashes.items())
        cls.tmp.cleanup()

    def evaluate(self, mode, request, cache, new, initial):
        folder = self.root / (self.id().split('.')[-1] + '-' + mode + '-' +
                              ''.join(map(str, (request, cache, new, initial))))
        folder.mkdir()
        result = subprocess.run([str(self.binary[mode]), *map(str, (request, cache, new, initial))],
            cwd=folder, env=dict(os.environ, SCRATCH_DIR=''), check=True,
            capture_output=True, text=True, timeout=10)
        row = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith('STATE ')]
        self.assertEqual(len(row), 1)
        return [v == 'T' for v in row[0][:5]], float(row[0][5])

    def test_original_requested_fresh_matrix_is_not_applied(self):
        flags, value = self.evaluate('original', 1, 1, 0, 0)
        self.assertFalse(flags[0])
        self.assertTrue(flags[1])
        self.assertEqual(value, 7.)

    def test_corrected_request_cache_missing_and_new_model(self):
        for request, cache, new, initial in itertools.product((0, 1), repeat=4):
            with self.subTest(request=request, cache=cache, new=new, initial=initial):
                flags, value = self.evaluate('corrected', request, cache, new, initial)
                reused = not request and cache and not new
                self.assertEqual(flags[0], not reused)
                self.assertEqual(flags[1], bool(reused))
                self.assertEqual(value, 7. if reused else 999.)
                self.assertEqual(flags[3], not reused)


if __name__ == '__main__':
    unittest.main()
