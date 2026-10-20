"""고정 총수의 직접 보존과 명시적 프로필 손상 거부를 함께 확인한다."""
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def executable(tmp_path_factory):
    root=tmp_path_factory.mktemp('element_density')
    exe=root/'selftest'
    subprocess.run(['gcc','-O2','-std=gnu11','-D_GNU_SOURCE',
        '-ffunction-sections','-fdata-sections','-I'+str(ROOT/'src'),
        '-Wl,--gc-sections','-o',str(exe),str(ROOT/'tests/element_density_selftest.c'),
        str(ROOT/'src/population_contract.c'),str(ROOT/'src/lumina_atomic.c'),'-lm'],
        check=True,capture_output=True,timeout=90)
    return exe


def test_explicit_inventory_and_transaction(executable):
    p=subprocess.run([str(executable)],check=True,capture_output=True,text=True,timeout=10)
    assert '[ELEMENT-DENSITY][PASS]' in p.stdout


VALID='atomic_number,shell_id,N_element_cm3\n14,0,100\n14,1,0\n27,0,200\n27,1,10\n'


def test_explicit_profile_loading_by_identity(executable,tmp_path):
    (tmp_path/'element_number_densities.csv').write_text(VALID)
    subprocess.run([str(executable),str(tmp_path)],check=True,capture_output=True,timeout=10)


@pytest.mark.parametrize('text',[
    None,VALID.replace('27,1,10\n',''),VALID+'14,0,100\n',
    VALID.replace('27,1,10','28,1,10'),VALID.replace('27,1,10','27,2,10'),
    VALID.replace('27,1,10','27,1,-1'),VALID.replace('27,1,10','27,1,nan'),
    VALID.replace('27,1,10','27,1,10garbage'),
    VALID.replace('N_element_cm3','mass_fraction')])
def test_explicit_contract_never_uses_legacy_fallback(executable,tmp_path,text):
    if text is not None:
        (tmp_path/'element_number_densities.csv').write_text(text)
    p=subprocess.run([str(executable),str(tmp_path)],capture_output=True,timeout=10)
    assert p.returncode!=0
