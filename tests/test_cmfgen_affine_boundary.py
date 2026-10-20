from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_cmfgen_affine_repair import repair_formal,repair_moment

PARENT=Path("/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST")
BUILD=PARENT/"affine_v1"


@pytest.fixture(scope="module")
def native_boundary_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp("affine_boundary")
    source=root/"boundary_test.f90"
    source.write_text("""program boundary_test
use lumina_affine_boundary
use mod_ray_mom_store,only:nplus_ob,nmin_ob
implicit none
integer k
character(len=40) mode
call get_command_argument(1,mode)
allocate(jqw_p(3,3),jqw_m(3,3),hqw_p(3,3),hqw_m(3,3),nqw_p(3,3),nqw_m(3,3))
jqw_p=1._ldp/6; jqw_m=jqw_p
hqw_p=jqw_p/2; hqw_m=-hqw_p
nqw_p=jqw_p/8; nqw_m=-nqw_p
do k=1,3
 allocate(ray(k)%i_p(1),ray(k)%i_m(1),ray(k)%lnk(3))
 ray(k)%i_p=2._ldp;ray(k)%i_m=1._ldp;ray(k)%lnk=1
 if(mode=='--zero-outgoing')ray(k)%i_p=0._ldp
end do
nplus_ob=huge(1._ldp);nmin_ob=-huge(1._ldp)
if(mode=='--bad-start')call lumina_affine_from_formal(20._ldp,.false.,3,3)
call lumina_affine_from_formal(20._ldp,.true.,3,3)
if(abs(hbc-.5_ldp)>1.e-15_ldp.or.abs(nbc-.125_ldp)>1.e-15_ldp)stop 10
if(abs(affine_hinc-.5_ldp)>1.e-15_ldp.or.affine_hinc_prev/=0)stop 11
if(mode=='--blueward')call lumina_affine_from_formal(21._ldp,.false.,3,3)
do k=1,3
 ray(k)%i_m=2._ldp
end do
call lumina_affine_from_formal(20._ldp,.false.,3,3)
if(abs(affine_hinc-1._ldp)>1.e-15_ldp.or.affine_hinc_prev/=0)stop 12
do k=1,3
 ray(k)%i_m=3._ldp
end do
call lumina_affine_from_formal(19._ldp,.false.,3,3)
if(abs(affine_hinc-1.5_ldp)>1.e-15_ldp.or.abs(affine_hinc_prev-1._ldp)>1.e-15_ldp)stop 13
if(abs(affine_ninc_prev-.25_ldp)>1.e-15_ldp)stop 14
do k=1,3
 ray(k)%i_m=4._ldp
end do
call lumina_affine_from_formal(19._ldp,.false.,3,3)
if(abs(affine_hinc-2._ldp)>1.e-15_ldp.or.abs(affine_hinc_prev-1._ldp)>1.e-15_ldp)stop 15
do k=1,3
 ray(k)%i_m=0._ldp
end do
call lumina_affine_from_formal(18._ldp,.true.,3,3)
if(affine_hinc/=0.or.affine_ninc/=0.or.affine_hinc_prev/=0.or.affine_ninc_prev/=0)stop 16
end program
""")
    subprocess.run(["gfortran","-O","-fopenmp","-ffree-line-length-none","-I"+str(BUILD),
        "-I"+str(PARENT/"capture_first_v1"),"-I/gpfs/kjhan/cmfgen_src/cur_cmf/mod",str(source),
        str(BUILD/"lumina_affine_boundary.o"),"-L"+str(PARENT/"capture_first_v1"),
        "-L/gpfs/kjhan/cmfgen_src/cur_cmf/lib","-lsubs","-lmod","-o",str(root/"boundary_test")],
        cwd=root,check=True,capture_output=True,timeout=60)
    return root/"boundary_test"


def test_native_boundary_same_frequency_previous_and_first_frequency(native_boundary_fixture):
    subprocess.run([str(native_boundary_fixture)],check=True,capture_output=True,timeout=20)


@pytest.mark.parametrize("mode",["--bad-start","--blueward","--zero-outgoing"])
def test_native_boundary_bad_history_and_support_rejected(native_boundary_fixture,mode):
    result=subprocess.run([str(native_boundary_fixture),mode],capture_output=True,timeout=20)
    assert result.returncode==89


def test_private_repair_changes_boundary_and_return_not_other_physics():
    mom=(PARENT/"capture_first_v1/mom_jrel_v9.f").read_text()
    changed=repair_moment(mom)
    for text in ("CALL THOMAS","CALL DERIVCHI","IF(MAX_ER .LT. 1.0E-08_LDP)","IF(INCL_ADVEC_TERMS)"):
        assert changed.count(text)==mom.count(text)
    assert "(AFFINE_NINC-AFFINE_NINC_PREV)" in changed
    assert "HFLUX_AT_OB=HBC*JNU_SM(1)-AFFINE_HINC" in changed
    formal=repair_formal((PARENT/"formal_v2/cmf_formal_rel_v4.f").read_text())
    assert "IF(HBC .LT. 0.0_LDP)" not in formal
    assert "LUMINA_AFFINE_FROM_FORMAL(FREQ,INITIALIZE,NP_LIMIT,ND)" in formal


def test_spatial_affine_correction_keeps_first_internal_flux_conservative():
    source=(PARENT/"capture_first_v1/mom_jrel_v9.f").read_text()
    fixed=repair_moment(source,spatial=True)
    assert "VdHdR_TERM(1)=BETA(1)*HBC" in fixed
    assert fixed.index("VdHdR_TERM(1)=BETA(1)*HBC")>fixed.index("CALL NORDTAU(DTAU_J")
    assert "XM(2)=XM(2)-LUM_AFFINE_FLUX_SOURCE" in fixed
    assert "XM(1)=XM(1)+LUM_AFFINE_GRAD_SOURCE" in fixed
    assert fixed.index("GAM_RSQHNU(1)=GAM_RSQHNU(1)+LUM_AFFINE_FLUX_SOURCE")<fixed.index("IF(LUM_SELECTED .AND. (H_CHK_OPTION")
    assert fixed.count("CALL THOMAS")==source.count("CALL THOMAS")
