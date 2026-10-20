"""동일 물질/경계 계수를 실제 셀 solver에 연결하는 작은 후보 직접시험."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_lumina_cell_candidate as D


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    out=tmp_path_factory.mktemp('cell-candidate')/'candidate.so'
    subprocess.run([os.environ.get('CC','cc'),'-O2','-std=c11','-fPIC','-shared',
        '-fvisibility=hidden','-ffunction-sections','-fdata-sections','-fopenmp',
        '-Wl,--gc-sections','-Wl,-z,defs','-I'+str(ROOT/'src'),'-o',str(out),
        *[str(ROOT/'src'/name) for name in ('lumina_cell_candidate.c','lumina_cmfgen.c',
          'cmf_exact_sliding.c','cmf_error_envelope.c')],'-lm'],
        check=True,timeout=90,capture_output=True)
    return out


def fixture(tmp_path,rel=False):
    cells=tmp_path/'cells';seed=tmp_path/'seed';s=tmp_path/'sources';b=tmp_path/'boundary'
    for folder in (cells,seed,s,b):folder.mkdir()
    model=tmp_path/'MODEL'
    model.write_text(('T' if rel else 'F')+' [USE_J_REL]\n'+
        ('T' if rel else 'F')+' [INCL_REL]\nF [INCL_ADV_TRANS]\nT [COH_ES]\n')
    epoch=1e6;columns=np.array([[20,1,0],[20,2,-1]],dtype='<i4')
    physical=np.array([[2e14,3e14,1e4,1e8,1e-13,1.],
                       [3e14,4e14,1e4,1e8,1e-13,1.]],dtype='<f8')
    native=np.array([[4e14,1e4,1e8,1e-13],[3e14,1e4,1e8,1e-13],
                     [2e14,1e4,1e8,1e-13]],dtype='<f8')
    mat=cells/'material_cells.bin';node=seed/'native_material_nodes.bin'
    for path,tag,p in ((mat,b'LUMCEL01',physical),(node,b'LUMNOD01',native)):
        path.write_bytes(struct.pack('<8sIIId',tag,len(p),2,2,epoch)+columns.tobytes()+
            p.tobytes()+np.tile(np.array([4e7,3e7],dtype='<f8'),(len(p),1)).tobytes())
    (cells/'material-cells.json').write_text(json.dumps(dict(
        output_sha256={'material_cells.bin':D.sha(mat)},boundary_method='DIFFUSION',
        inner_gas_temperature_K=1e4,inner_luminosity_erg_s=1e36)))
    nu=1e14*np.exp((np.arange(32)+.5)*.001)
    for folder,path,ns in ((s,mat,2),(b,node,3)):
        data=np.empty((7,ns,len(nu)));data[:]=0
        data[0]=1e-15;data[1]=1e-22;data[2]=1e-17;data[3]=1e-24;data[6]=1e-16
        if folder==s:data[4,1,16]=-2e-15
        else:data[1,-1,:]=2e-22 # 실제 최내곽 점 선택을 검사한다.
        payload=(struct.pack('<8sII',b'LMPHYS01',ns,len(nu))+nu.astype('<f8').tobytes()+
            np.full(ns,1e4,dtype='<f8').tobytes()+np.full(ns,1e8,dtype='<f8').tobytes()+data.astype('<f8').tobytes())
        (folder/'sources.bin').write_bytes(payload)
        (folder/'sources.json').write_text(json.dumps(dict(schema='lumina-model-local-sources-v1',
            cells=ns,frequencies=len(nu),full_levels=2,
            input_sha256={str(path.resolve()):D.sha(path),str(model.resolve()):D.sha(model)},
            output_sha256={'sources.bin':D.sha(folder/'sources.bin')})))
    return dict(cells=cells,sources=s,seed=seed,boundary_sources=b,model=model)


def rehash(folder):
    path=folder/'sources.json';report=json.loads(path.read_text())
    report['output_sha256']['sources.bin']=D.sha(folder/'sources.bin')
    path.write_text(json.dumps(report))


def test_same_material_raw_source_candidate_reaches_solver(tmp_path,library):
    args=fixture(tmp_path);prepared=D.prepare(**args)
    before={p:D.sha(p) for p in (args['sources']/'sources.bin',args['cells']/'material_cells.bin')}
    assert prepared['report']['execution_eligible']
    assert prepared['report']['memory']['known_allocation_bound_bytes']<32*1024**3
    j,error,report=D.execute_candidate(prepared,library)
    assert j.shape==error.shape==(2,32)
    assert np.isfinite(j).all() and np.all(j>0) and np.all(error>=0)
    assert report['componentwise_verified'] and 0<report['actual_K_norm_upper']<1
    assert report['iterations']>1
    assert report['candidate_solver_calls']==1 and not report['authoritative_field_role']
    assert prepared['report']['boundary_sample'].startswith('original innermost')
    assert all(D.sha(p)==identity for p,identity in before.items())


def test_native_frame_cannot_start_candidate_or_load_library(tmp_path,monkeypatch):
    prepared=D.prepare(**fixture(tmp_path,rel=True))
    assert not prepared['report']['execution_eligible']
    def forbidden(*args):raise AssertionError('프레임 거부 전에 kernel을 불러서는 안 됨')
    monkeypatch.setattr(D.ct,'CDLL',forbidden)
    with pytest.raises(ValueError,match='FRAME_NOT_IMPLEMENTED'):
        D.execute_candidate(prepared,tmp_path/'not-loaded.so')
    assert prepared['report']['authoritative_fields_executed']==0


@pytest.mark.parametrize('kind',['hash','temperature','frequency','negative_eta'])
def test_source_mismatch_is_not_repaired(tmp_path,kind):
    args=fixture(tmp_path);path=args['sources']/'sources.bin'
    if kind=='hash':
        value=bytearray(path.read_bytes());value[-1]^=1;path.write_bytes(value)
    else:
        data=np.memmap(path,mode='r+',dtype='<f8',offset=16)
        if kind=='temperature':data[32]=12000
        elif kind=='frequency':data[4]*=1+1e-8
        else:data[32+4+2*32]=-1e-22
        data.flush();del data;rehash(args['sources'])
    with pytest.raises(ValueError):D.prepare(**args)


def test_cell_gap_is_rejected(tmp_path):
    args=fixture(tmp_path);path=args['cells']/'material_cells.bin'
    data=np.memmap(path,mode='r+',dtype='<f8',offset=28+24)
    data[6]+=1e9;data.flush();del data
    with pytest.raises(ValueError,match='경계'):D.prepare(**args)


def test_boundary_frequency_is_not_silently_interpolated(tmp_path):
    args=fixture(tmp_path);path=args['boundary_sources']/'sources.bin'
    data=np.memmap(path,mode='r+',dtype='<f8',offset=16);data[:32]*=1.001
    data.flush();del data;rehash(args['boundary_sources'])
    with pytest.raises(ValueError,match='주파수 불일치'):D.prepare(**args)


def test_known_allocation_budget_cannot_be_ignored(tmp_path):
    prepared=D.prepare(**fixture(tmp_path),memory_gib=1e-9)
    assert not prepared['report']['execution_eligible']
    assert 'MEMORY_BUDGET_EXCEEDED' in prepared['report']['reasons']


def test_cli_frame_rejection_writes_reason_but_no_field(tmp_path):
    args=fixture(tmp_path,rel=True);out=tmp_path/'output'
    command=[sys.executable,str(ROOT/'scripts/run_lumina_cell_candidate.py')]
    for key,path in args.items():command+=['--'+key.replace('_','-'),str(path)]
    result=subprocess.run(command+['--out',str(out),'--execute-candidate',
        '--library',str(tmp_path/'not-loaded.so')],text=True,capture_output=True,timeout=30)
    assert result.returncode==2
    report=json.loads((out/'preflight.json').read_text())
    assert 'FRAME_NOT_IMPLEMENTED' in report['failure']
    assert report['authoritative_fields_executed']==0
    assert not (out/'candidate_field.npz').exists()
