"""물리 source와 보간 오차를 혼합하지 않고 실제 광선/전체 coverage를 보존한다."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cmfgen_ray_budget import prepare,link
from replay_cmfgen_ray_budget import run as replay
from run_cmfgen_formal_primary import read_cell_energy,validate_cell_energy
from run_cmfgen_ray_budget_manufactured import analyze

BASE=Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST')
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


def write_cell(path,count=2,native=None):
    r=np.array([3.,2.,1.]);native=np.zeros(3) if native is None else native
    values=np.r_[r,[2.,1.,0.,0.],np.zeros(10),np.zeros(3),native,np.zeros(8)]
    path.write_bytes(b'RAYENERGYV1     '+struct.pack('<4i',3,count,count,8)+values.astype('<f8').tobytes())


def test_cell_reader_rejects_partial_or_wrong_frequency_coverage(tmp_path):
    path=tmp_path/'energy';write_cell(path)
    assert read_cell_energy(path,3,2)['count']==2
    with pytest.raises(ValueError,match='coverage'):
        read_cell_energy(path,3,197291)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError,match='부분'):
        read_cell_energy(path,3,2)


def test_actual_common_field_native_flux_consumer_is_required(tmp_path):
    path=tmp_path/'energy';r=np.array([3.,2.,1.])
    dtype=np.dtype([('weight','f8'),('field','f8',(8,3))])
    data=np.zeros(2,dtype=dtype);data['weight']=1;data['field'][:,0,:]=1;data['field'][:,1,:]=.1
    geometry=np.array([r,np.zeros(3),np.zeros(3),np.ones(3)])
    write_cell(path,native=.2*r*r)
    assert validate_cell_energy(geometry,data,path)['physical_energy_pass']=='NOT_JUDGED'
    write_cell(path,native=.21*r*r)
    with pytest.raises(ValueError,match='소비자'):
        validate_cell_energy(geometry,data,path)


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    explicit=os.environ.get('CMFGEN_RAY_BUDGET_TEST_BUILD')
    if explicit:return Path(explicit)
    root=tmp_path_factory.mktemp('ray_budget_compile')/'build'
    prepare(BASE/'formal_primary_v11',root)
    return root


@pytest.fixture(scope='module')
def manufactured(build,tmp_path_factory):
    root=tmp_path_factory.mktemp('ray_budget_exe');exe=root/'manufactured'
    link(SCRIPTS/'formal_primary_cell_budget.f90',exe,build)
    return exe


@pytest.mark.parametrize('mode',['absorb','thermal','extended'])
def test_spherical_physical_volume_and_compact_accumulator(manufactured,tmp_path,mode):
    env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_CELL_BUDGET='1',
        LUMINA_FORMAL_PRIMARY_WHOLE='',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='')
    p=subprocess.run([str(manufactured),mode,'17','32','32'],cwd=tmp_path,env=env,capture_output=True,timeout=30)
    assert p.returncode==0,p.stdout+p.stderr
    record=read_cell_energy(tmp_path/'FORMAL_CELL_ENERGY.bin',17,1)
    data=analyze(tmp_path/'RAY_CELL_BUDGET.bin');best=data['orders']['8']
    assert best['endpoint_reconstruction_max_relative']<1e-12
    assert best['numerical_identity_L1_defect']<1e-8
    assert record['ray_flux'][0]-record['ray_flux'][-1]==pytest.approx(best['boundary']*1e15,rel=2e-13)
    assert record['budget'][:,0].sum()==pytest.approx(best['physical_source']*1e15,rel=2e-13)
    if mode=='thermal':
        assert abs(best['physical_continuous_signed_defect'])<1e-12
        assert abs(data['native_minus_nodal_trapezoid'])>1e3


def test_actual_packet_is_byte_identical_with_material_capture_and_after_integration(build,tmp_path):
    out=tmp_path/'replay'
    replay(build,Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428716'),out)
    report=json.loads((out/'packet-cell.json').read_text())
    assert all(report['baseline_and_observer_byte_equal'].values())
    assert all(report['after_observer_byte_equal'].values())
    assert report['orders']['8']['endpoint_reconstruction_max_relative']<1e-12
    assert report['orders']['8']['numerical_identity_L1_over_largest_cell_term']<1e-7
