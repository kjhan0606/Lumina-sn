"""수송 조립기가 BF 방출을 χB로 되돌리지 않는지 확인한다."""
from pathlib import Path
import os
import subprocess


def test_raw_physical_coefficient_assembly(tmp_path):
    root=Path(__file__).resolve().parents[1];binary=tmp_path/'physical_coefficients'
    subprocess.run([os.environ.get('CC','cc'),'-O2','-std=c11','-ffunction-sections',
        '-fdata-sections','-I'+str(root/'src'),str(root/'tests/cmf_physical_coefficients_selftest.c'),
        str(root/'src/lumina_cmfgen.c'),'-Wl,--gc-sections','-lm','-o',str(binary)],
        check=True,timeout=90,capture_output=True)
    result=subprocess.run([str(binary)],check=True,timeout=30,text=True,capture_output=True)
    assert 'synthetic=PASS' in result.stdout
