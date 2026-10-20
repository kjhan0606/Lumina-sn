"""I/nu^3 제조해로 구형 상대론 수송과 유한 주파수 장부를 함께 검증한다."""
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from test_cmfgen_formal_primary import native_link
from analyze_cmfgen_expanding_manufactured import analyze,exact_moments
from run_cmfgen_formal_manufactured import absorption_moments


def test_invariant_solution_reduces_to_static_absorption_as_velocity_vanishes():
    r=np.array([1.,1.1,1.5,1.9,2.])
    np.testing.assert_allclose(exact_moments(r,a=1e-8),absorption_moments(r),atol=1e-7,rtol=0)


@pytest.fixture(scope='module')
def expanding_exe(tmp_path_factory):
    root=tmp_path_factory.mktemp('expanding_fixture');exe=root/'fixture'
    native_link(Path(__file__).resolve().parents[1]/'scripts/formal_primary_expanding.f90',exe)
    return exe


def execute(exe,root,n,nf,nc):
    root.mkdir()
    result=subprocess.run([str(exe),str(n),str(nf),str(nc)],cwd=root,capture_output=True,timeout=30,
        env=dict(os.environ,OMP_NUM_THREADS='1',LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='',LUMINA_WHOLE_MODE=''))
    assert result.returncode==0,result.stderr
    return analyze(root)


def final_j(root):
    with (root/'EXPANDING_TRANSFER.bin').open('rb') as f:
        n,_,_,_=struct.unpack('<4i',f.read(16));f.seek(-8*(1+4*n),2)
        a=np.frombuffer(f.read(),dtype='<f8')
    return a[1:1+n]/a[0]**3


def test_joint_refinement_checks_actual_relative_sphere_energy_and_frequency_end(expanding_exe,tmp_path):
    rows=[]
    for n,nf,nc in ((17,256,32),(33,512,64),(65,1024,128)):
        rows.append(execute(expanding_exe,tmp_path/str(n),n,nf,nc))
    errors=[r['energy']['relative_to_net'] for r in rows]
    assert errors[0]>errors[1]>errors[2] and errors[2]<.01
    coordinates=[r['radial_characteristic_max_absolute_error'] for r in rows]
    assert coordinates[0]>coordinates[1]>coordinates[2]
    assert max(rows[-1]['final_moment_max_absolute_error'].values())<.01
    e=rows[-1]['energy']
    assert abs(e['signed_defect']+e['budgets']['frequency_end_term'])/abs(e['net_boundary'])>.1


def test_frequency_cauchy_convergence_does_not_claim_the_spatial_error_disappears(expanding_exe,tmp_path):
    values=[];errors=[]
    for nf in (512,1024,2048,4096):
        root=tmp_path/str(nf);row=execute(expanding_exe,root,65,nf,128)
        values.append(final_j(root));errors.append(row['final_moment_max_absolute_error']['J'])
    differences=[np.max(abs(y-x)) for x,y in zip(values,values[1:])]
    assert differences[0]>differences[1]>differences[2]
    # 주파수 오차가 공간 오차와 상쇄될 수 있다. 이 증거를 숨기거나 단독 주파수 PASS로 부르지 않는다.
    assert errors[-1]>errors[0]
