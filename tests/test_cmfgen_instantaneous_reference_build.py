"""전체 빌드 전 입력 봉인·링크 정의 검사를 확인한다. 물리 수렴 시험은 아니다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_cmfgen_instantaneous_reference as B


class ReferenceBuildTests(unittest.TestCase):
    def test_only_one_actual_function_definition_counts(self):
        symbol = B.SYMBOL
        self.assertEqual(B.definition_count('0001 T ' + symbol + '\n0000 U other_\n'), 1)
        self.assertEqual(B.definition_count('0001 T ' + symbol + '_wrapper\n'), 0)
        self.assertEqual(B.definition_count('0001 T ' + symbol + '\n0002 T ' + symbol), 2)

    def test_existing_output_is_never_touched(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / 'keep'
            marker.write_bytes(b'unchanged')
            with self.assertRaises(FileExistsError):
                B.build(root / 'nonexistent-native', root)
            self.assertEqual(marker.read_bytes(), b'unchanged')

    def test_changed_input_or_module_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'main').write_bytes(b'correct')
            (root / 'mod').mkdir()
            module = root / 'mod/kind.mod'
            module.write_bytes(b'module')
            with patch.object(B, 'SEALED', {'main': B.sha(root / 'main')}):
                before = B.snapshot(root)
                self.assertEqual(set(before), {'main', 'mod/kind.mod'})
                module.unlink()
                module.symlink_to(root / 'main')
                with self.assertRaisesRegex(ValueError, '외부 module'):
                    B.snapshot(root)
                module.unlink()
                module.write_bytes(b'module')
                (root / 'main').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, '입력 변경'):
                    B.snapshot(root)

    def test_output_inside_native_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, '트리 안'):
                B.build(root, root / 'private-build')


if __name__ == '__main__':
    unittest.main()
