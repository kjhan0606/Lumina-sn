"""수치 경로만 바꾸고 실제 angular source·blue·H interface를 보존한다."""
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_formal_capture import formal_state
from prepare_cmfgen_formal_primary import primary_caller, primary_main
from run_cmfgen_formal_manufactured import absorption_moments, simpson_regular, simpson_nonuniform, run as manufactured_run

BASE=Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST')
BUILD=Path(os.environ.get('CMFGEN_FORMAL_PRIMARY_TEST_BUILD',BASE/'formal_primary_v11'))
PARENT=BASE/'whole_accel_v1'
NATIVE=Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
ORIGINAL=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428716')


def native_link(source,exe,extra_objects=()):
    common=[*extra_objects,BUILD/'cmfgen_formal_primary.o']
    if (BUILD/'cmfgen_positive_sc.o').exists():common += [BUILD/'cmfgen_positive_sc.o']
    if (BUILD/'cmfgen_formal_primary_whole.o').exists():common += [BUILD/'cmfgen_formal_primary_whole.o']
    common += [PARENT/(k+'.o') for k in ('cmfgen_mom_acceleration','lumina_affine_boundary','lumina_mom_diag','lumina_formal_diag','lumina_whole_monitor')]
    common += [BASE/'formal_v2/ext_rel_grid_v4.o',BASE/'capture_first_v1/mod_jrel_v9.o']
    subprocess.run(['gfortran','-O','-fopenmp','-ffree-line-length-none','-I'+str(BUILD),'-I'+str(PARENT),
        '-I'+str(BASE/'formal_v2'),'-I'+str(BASE/'capture_first_v1'),'-I'+str(NATIVE/'mod'),str(source),*map(str,common),
        '-L'+str(BUILD),'-L'+str(NATIVE/'lib'),'-lsubs','-lpgplt','-lmod',str(NATIVE/'lib/libmy_lapack.a'),str(NATIVE/'lib/libmy_blas.a'),'-o',str(exe)],check=True,capture_output=True,timeout=90)


@pytest.fixture(scope='module')
def packet_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('formal_primary_packet');exe=root/'packet'
    native_link(Path(__file__).resolve().parents[1]/'scripts/replay_cmfgen_formal_primary.f90',exe)
    return exe


def test_same_packet_source_fixed_point_and_H_without_double_gamma(packet_fixture,tmp_path):
    result=subprocess.run([str(packet_fixture),str(ORIGINAL)],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE=''))
    assert result.returncode==0,result.stderr
    assert (tmp_path/'PRIMARY_GEOMETRY.bin').read_bytes()==(ORIGINAL/'FORMAL_EVENT_geometry.bin').read_bytes()
    if json.loads((BUILD/'formal-primary-build.json').read_text()).get('source_interpolation') is None:
        assert (tmp_path/'PRIMARY_FIRST_RAYS.bin').read_bytes()==(ORIGINAL/'FORMAL_EVENT_rays_out.bin').read_bytes()
    assert (tmp_path/'PRIMARY_BLUE_FINAL.bin').read_bytes()==(ORIGINAL/'FORMAL_EVENT_rays_in.bin').read_bytes()
    with (tmp_path/'PRIMARY_REPLAY.csv').open() as f:rows=list(csv.DictReader(f))
    assert 2<=len(rows)<=20
    assert all(float(rows[-1][key])<1e-8 for key in ('J_residual','source_residual','H_change_over_J'))
    assert float(rows[-1]['angular_support_min'])>0
    inventory=json.loads((BASE/'formal_v2/formal-build.json').read_text())['ray_inventory']
    state=formal_state(tmp_path/'PRIMARY_FORMAL_FINAL.bin',inventory)
    r=state['args']['R'];h=state['ray']['HNU_STORE']['values'];n=len(r)
    lib=ctypes.CDLL(str(BASE/'capture_first_v1/analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    expected=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(r*r*h),expected)
    actual=np.fromfile(tmp_path/'PRIMARY_H_INTERFACE.bin',dtype='<f8')
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=0)
    gamma=state['ray']['GAM_REL_STORE']['values']
    wrong=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(gamma*r*r*h),wrong)
    assert np.max(abs(wrong-actual))/np.max(abs(actual))>1e-4


@pytest.mark.parametrize('mode,reason',[('--wrong-source','actual emissivity'),('--wrong-blue','blue ray changed')])
def test_missing_source_update_or_wrong_blue_is_rejected(packet_fixture,tmp_path,mode,reason):
    result=subprocess.run([str(packet_fixture),str(ORIGINAL),mode],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE=''))
    assert result.returncode==95,result.stderr
    assert reason in (tmp_path/'FORMAL_PRIMARY_FAILURE.txt').read_text()


def test_private_dispatch_does_not_toggle_physical_relativistic_flags():
    original=(PARENT/'comp_j_blank.f').read_text();changed=primary_caller(original)
    assert changed.count('CALL MOM_JREL_V9')==original.count('CALL MOM_JREL_V9')
    assert 'CALL GET_RSQH_REL(RSQHNU,R,V,FL,ND)' in changed
    for assignment in ('USE_J_REL=.FALSE.','USE_FORMAL_REL=.FALSE.','INCL_REL_TERMS=.FALSE.'):
        assert assignment not in changed
    main=primary_main((PARENT/'cmfgen_sub.f').read_text())
    assert 'SOB(1:ND)=R(1:ND)*R(1:ND)*FP_NODE_H(1:ND)' in main
    assert 'IF(FP_ENABLED())H_MOM(1:ND)=FP_NODE_H(1:ND)' in main
    assert 'IF(FP_ENABLED())K_MOM(1:ND)=FP_NODE_K(1:ND)' in changed
    assert main.index('CALL FP_WHOLE_FINISH()')<main.index('CALL SOLVE_FOR_POPS(')


def test_manufactured_continuous_sphere_reference_and_energy_quadrature():
    exact = absorption_moments(np.array([2.]))[0]
    c = np.sqrt(.75)
    expected_j = .5 - (1 - np.exp(-4*c)) / 8
    expected_h = .25 - (1 - (1+4*c)*np.exp(-4*c)) / 32
    np.testing.assert_allclose(exact[:2], [expected_j, expected_h], rtol=1e-12, atol=0)
    x = np.linspace(1., 2., 17)
    assert abs(simpson_regular(x, x*x) - 7/3) < 1e-14
    x=np.linspace(0,1,17)**2
    assert abs(simpson_nonuniform(x, x*x)-1/3) < 1e-14


def test_approximate_lambda_solve_pivots_and_rejects_singular_matrix(tmp_path):
    source = tmp_path/'preconditioner.f90'
    source.write_text('''program preconditioner
use lumina_formal_primary
implicit none
real(ldp) a(3,3),b(3),expected(3),ee,e0,psi,tau
integer info,i
a=reshape([0._ldp,2._ldp,1._ldp,1._ldp,3._ldp,1._ldp,2._ldp,1._ldp,4._ldp],[3,3])
expected=[1._ldp,2._ldp,3._ldp];b=matmul(a,expected)
call fp_solve_preconditioner(3,a,b,info)
if(info/=0.or.maxval(abs(b-expected))>1.e-12_ldp)stop 1
a=0;b=1;call fp_solve_preconditioner(3,a,b,info)
if(info==0)stop 2
do i=-12,3
 tau=10._ldp**i
 call fp_sc_weights(tau,ee,e0,psi)
 if(psi<0.or.e0-psi<0.or.ee<0)stop 3
 if(abs(ee+e0-1)>4*epsilon(1._ldp))stop 4
end do
end program
''')
    exe=tmp_path/'preconditioner'
    native_link(source,exe)
    result=subprocess.run([str(exe)],capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr


def test_manufactured_fixed_point_is_not_mistaken_for_physical_conservation(tmp_path,monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','1')
    source=Path(__file__).resolve().parents[1]/'scripts/formal_primary_manufactured.f90'
    exe=tmp_path/'manufactured.exe';native_link(source,exe)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    exe.with_suffix('.build.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        exe_sha256=sha(exe),transport_build_manifest=str(BUILD/'formal-primary-build.json'))))
    out=tmp_path/'evidence'
    manufactured_run(exe,out,[65,129,257],129,'cosine')
    d=json.loads((out/'manufactured-report.json').read_text())
    assert d['all_cases_executed'] and d['all_native_calls_returned_zero']
    assert d['physical_equivalence']=='NOT_JUDGED'
    for case in d['cases']:
        m=case['metrics']
        if case['mode']!='redshift':
            assert m['calls']<=20
            assert max(m['j_residual'],m['source_residual'],m['h_change_over_j'])<1e-8
    for mode in ('scatter','thick'):
        rows=[c['metrics'] for c in d['cases'] if c['mode']==mode]
        errors=[c['energy_defect_relative'] for c in rows]
        assert errors[0]>errors[1]>errors[2]
        assert errors[2]<.01
    # 원래 거친 격자의 수렴을 에너지 보존 통과로 바꿀 수 없다.
    thick_coarse=next(c for c in d['cases'] if c['mode']=='thick' and c['n']==65)
    assert thick_coarse['metrics']['energy_defect_relative']>.01
    angular=[c['metrics']['J_max_absolute_error'] for c in d['cases'] if c['refinement']=='angular']
    assert angular[0]>angular[1]>angular[2]
    fixed=[next(c for c in d['cases'] if c['mode']=='redshift' and c['n']==65 and c['nfreq']==nf)['metrics']['max_absolute_error'] for nf in (512,1024,2048)]
    assert 1.9<fixed[0]/fixed[1]<2.1 and 1.9<fixed[1]/fixed[2]<2.1
