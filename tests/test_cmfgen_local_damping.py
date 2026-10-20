"""실제 LOCAL 갱신 블록의 기본 보존·중간 상태·양수성과 공통 보존을 시험한다."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_cmfgen_local_damping import INSERTION, MARKER, patched, prepare

SOURCE = Path('/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f')


def test_exactly_one_local_call_and_no_original_source_edits():
    original = SOURCE.read_text()
    changed = patched(original)
    assert changed.replace(INSERTION, '') == original
    assert changed.count(INSERTION) == 1
    assert changed.index(INSERTION) < changed.index(MARKER)
    with pytest.raises(ValueError):
        patched(original.replace(MARKER, ''))
    with pytest.raises(ValueError):
        patched(original + MARKER)


def test_existing_output_refused_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match='기존'):
        prepare(Path('not-read'), Path('not-read'), tmp_path)


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    root = tmp_path_factory.mktemp('local_damping')
    stub = root / 'kind.f90'
    stub.write_text('module set_kind_module\ninteger,parameter::ldp=kind(1.d0)\nend module\n')
    helper = SCRIPTS / 'cmfgen_local_step_damping.f90'
    # 실물 소스의 LOCAL 분기 전체를 가져오며 수치식을 시험용으로 재작성하지 않는다.
    original = SOURCE.read_text()
    begin = original.index('\tBIG_LIM=(CHANGE_LIM-1.0_LDP)/CHANGE_LIM')
    header = '''      SUBROUTINE UPDATE(POPS,STEQ,NT,ND)
      USE SET_KIND_MODULE
      IMPLICIT NONE
      INTEGER NT,ND,I,J,LUER
      REAL(KIND=LDP) POPS(NT,ND),STEQ(NT,ND)
      REAL(KIND=LDP) BIG_LIM,LIT_LIM,CHANGE_LIM,MINSCALE
      REAL(KIND=LDP) MAX_dT_COR,T_MIN,T1,T2,T3,SCALE
      CHARACTER*8 SCALE_OPT
      LUER=6
      CHANGE_LIM=1.0D6
      MAX_dT_COR=0.2D0
      T_MIN=0.01D0
      SCALE_OPT='LOCAL'
'''
    driver = root / 'driver.f90'
    driver.write_text('''program actual_local_block
use set_kind_module
implicit none
real(kind=ldp)::p(5,2),c(5,2)
integer::u
open(newunit=u,file='inputs.bin',access='stream',form='unformatted',status='old')
read(u)p,c
close(u)
call update(p,c,5,2)
open(newunit=u,file='result.bin',access='stream',form='unformatted',status='new')
write(u)p,c
close(u)
end program
''')
    executables = {}
    for name, text in (('original', original), ('patched', patched(original))):
        block = text[text.index(original[begin:begin+50]):]
        block = block[:block.index("\tELSE IF(SCALE_OPT(1:4) .EQ. 'NONE')THEN")]
        source = root / (name + '.f')
        source.write_text(header + block + '\n      END IF\n      END\n')
        exe = root / name
        subprocess.run(['gfortran', '-O', '-ffixed-line-length-none', str(stub),
            str(source), str(helper), str(driver), '-o', str(exe)], cwd=root,
            check=True, capture_output=True, timeout=30)
        executables[name] = exe
    def run(out, pre, correction, omega=None, original=False):
        out.mkdir()
        (out/'inputs.bin').write_bytes(pre.tobytes(order='F') + correction.tobytes(order='F'))
        env = dict(os.environ)
        env.pop('LUMINA_LOCAL_STEP_OMEGA', None)
        if omega is not None:
            env['LUMINA_LOCAL_STEP_OMEGA'] = omega
        result = subprocess.run([str(executables['original' if original else 'patched'])],
            cwd=out, env=env, capture_output=True, timeout=10)
        path = out/'result.bin'
        if not path.exists():
            return result, None, None
        # 각 행렬은 Fortran 순서로 연속 저장됐다.
        raw = np.frombuffer(path.read_bytes(), dtype='<f8')
        return result, raw[:10].reshape((5,2),order='F'), raw[10:].reshape((5,2),order='F')
    return run


def fixture_state():
    pre = np.array([[3.,4.],[7.,6.],[2.,3.],[11.,12.],[1.,1.]])
    post = np.array([[4.,5.],[6.,5.],[2.,3.],[10.,11.],[1.,1.]])
    return pre, post, 1-post/pre


@pytest.mark.parametrize('omega', [None, '1'])
def test_default_and_restored_paths_are_byte_identical(kernel,tmp_path,omega):
    pre, post, correction = fixture_state()
    a, reference, raw = kernel(tmp_path/'original',pre,correction,original=True)
    b, result, changed = kernel(tmp_path/'patched',pre,correction,omega)
    assert a.returncode == b.returncode == 0
    assert reference.tobytes() == result.tobytes()
    assert raw.tobytes() == changed.tobytes() == correction.tobytes()
    assert b'LUMINA_LOCAL_DAMPING_V1' not in b.stdout


def test_half_step_is_convex_and_preserves_common_linear_constraints(kernel,tmp_path):
    pre, post, correction = fixture_state()
    a, full, raw = kernel(tmp_path/'full',pre,correction,original=True)
    b, half, changed = kernel(tmp_path/'half',pre,correction,'0.5')
    assert a.returncode == b.returncode == 0
    np.testing.assert_allclose(half,(pre+full)/2,rtol=4e-16,atol=0)
    assert np.all(half > 0)
    np.testing.assert_array_equal(half[-1],pre[-1])
    np.testing.assert_allclose(half[:2].sum(axis=0),pre[:2].sum(axis=0),rtol=1e-15)
    np.testing.assert_allclose(half[1]+2*half[2],half[-2],rtol=1e-15)
    assert changed.tobytes() == raw.tobytes() == correction.tobytes()
    assert b.stdout.count(b'LUMINA_LOCAL_DAMPING_V1') == 2


def test_existing_positivity_scaling_is_applied_before_common_damping(kernel,tmp_path):
    pre, _, correction = fixture_state()
    correction[0,0] = 2.0
    _, full, _ = kernel(tmp_path/'full',pre,correction,original=True)
    result, half, _ = kernel(tmp_path/'half',pre,correction,'0.5')
    assert result.returncode == 0 and np.all(full > 0) and np.all(half > 0)
    np.testing.assert_allclose(half,(pre+full)/2,rtol=4e-16,atol=0)


@pytest.mark.parametrize('value', ['0', '-1', '2', '0.25', 'nan', 'garbage', '1'*40])
def test_unplanned_or_invalid_factor_never_updates_populations(kernel,tmp_path,value):
    pre, _, correction = fixture_state()
    result, state, _ = kernel(tmp_path/value.replace('-','minus'),pre,correction,value)
    assert result.returncode != 0 and state is None
    assert b'LUMINA_LOCAL_STEP_OMEGA_INVALID' in result.stdout
