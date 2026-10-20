"""서로 다른 격자의 역Milne 주파수 구간 읽기 직접시험."""
from pathlib import Path
import subprocess


def test_log_frequency_band_reader(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / 'bf_log_band_reader'
    subprocess.run([
        'gcc', '-std=gnu11', '-O2', '-I' + str(root / 'src'),
        str(root / 'tests/bf_log_band_reader_selftest.c'),
        str(root / 'src/bf_rate_jnu.c'), '-lm', '-o', str(executable),
    ], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True,
                            capture_output=True, text=True)
    assert 'FREQUENCY_LOG_BAND_DIRECT_TESTS_PASS' in result.stdout
