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
          'cmf_exact_sliding.c','cmf_error_envelope.c','cmf_relativistic_characteristic.c')],'-lm'],
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


def verified_rel_fixture(tmp_path):
    args=fixture(tmp_path,rel=True);model=args['model']
    model.write_text('T [USE_J_REL]\nT [INCL_REL]\nT [INCL_ADV_TRANS]\nT [COH_ES]\nF [INCL_DJDT]\n')
    refresh_model_identity(args)
    (tmp_path/'RVTJ').write_text(' Radius (10^10 cm)\n 40000 30000 20000\n\n'
        ' Velocity (km/s)\n 4000 3000 2000\n\n dlnV/dlnr-1\n 0 0 0\n')
    return args


def refresh_model_identity(args):
    for key in ('sources','boundary_sources'):
        path=args[key]/'sources.json';report=json.loads(path.read_text())
        report['input_sha256'][str(args['model'].resolve())]=D.sha(args['model'])
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


def test_verified_full_sr_frame_reaches_same_raw_candidate_core(tmp_path,library):
    prepared=D.prepare(**verified_rel_fixture(tmp_path))
    assert prepared['report']['execution_eligible']
    assert prepared['report']['implemented_frame']==D.REL_FRAME
    assert prepared['report']['velocity_contract']['sigma_exact_zero']
    assert prepared['report']['boundary_intensity']=='B_nu(T_inner)+mu_cmf*dBdtau'
    memory=prepared['report']['memory']['parts']
    assert memory['previous_angular_samples']==2*2*32*8
    j,error,report=D.execute_candidate(prepared,library)
    assert report['frame']==D.REL_FRAME and report['componentwise_verified']
    assert 0<report['actual_K_norm_upper']<1
    assert np.isfinite(j).all() and np.all(j>0) and np.all(error>=0)
    assert not report['authoritative_field_role']


@pytest.mark.parametrize('flag',['T','F','invalid'])
def test_outer_extension_is_not_silently_replaced_by_vacuum(tmp_path,monkeypatch,flag):
    args=verified_rel_fixture(tmp_path)
    args['model'].write_text(args['model'].read_text()+f'{flag} [THK_CONT]\nT [THK_LINE]\n')
    refresh_model_identity(args)
    if flag=='invalid':
        with pytest.raises(ValueError,match='THK_CONT'):
            D.prepare(**args)
        return
    prepared=D.prepare(**args)
    assert prepared['report']['outer_boundary_requested']==flag
    assert prepared['report']['model_flags']['THK_LINE']=='T'
    assert prepared['report']['outer_boundary_implemented'].startswith('vacuum')
    assert prepared['report']['execution_eligible']==(flag=='F')
    if flag=='T':
        def forbidden(*args):raise AssertionError('미지원 외부경계에서 C 실행 금지')
        monkeypatch.setattr(D.ct,'CDLL',forbidden)
        with pytest.raises(ValueError,match='OUTER_BOUNDARY_NOT_IMPLEMENTED'):
            D.execute_candidate(prepared,tmp_path/'not-loaded.so')
def test_failed_iteration_preserves_actual_partial_arrays(tmp_path,library):
    prepared=D.prepare(**verified_rel_fixture(tmp_path))
    with pytest.raises(D.CandidateFailure) as error:
        D.execute_candidate(prepared,library,iteration_cap=2,tolerance=1e-30)
    assert error.value.status==5
    assert error.value.partial['J'].shape==(2,32)
    assert np.any(error.value.partial['J']>0)
    assert error.value.partial['counts'][0]==2


@pytest.mark.parametrize('kind',['time','sigma','velocity','radius'])
def test_unverified_relativistic_physics_cannot_load_kernel(tmp_path,monkeypatch,kind):
    args=verified_rel_fixture(tmp_path);rvtj=tmp_path/'RVTJ'
    if kind=='time':
        args['model'].write_text(args['model'].read_text().replace('F [INCL_DJDT]','T [INCL_DJDT]'))
        refresh_model_identity(args)
    elif kind=='sigma':rvtj.write_text(rvtj.read_text().replace('0 0 0','0 0.1 0'))
    elif kind=='velocity':rvtj.write_text(rvtj.read_text().replace('4000 3000 2000','4001 3000 2000'))
    else:rvtj.write_text(rvtj.read_text().replace('40000 30000 20000','40001 30000 20000'))
    prepared=D.prepare(**args)
    assert not prepared['report']['execution_eligible']
    def forbidden(*args):raise AssertionError('지원하지 않는 물리 실행 금지')
    monkeypatch.setattr(D.ct,'CDLL',forbidden)
    with pytest.raises(ValueError,match='FRAME_NOT_IMPLEMENTED'):
        D.execute_candidate(prepared,tmp_path/'not-loaded.so')


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


def test_ray_batch_change_preserves_full_sr_result_bitwise(tmp_path,library):
    args=verified_rel_fixture(tmp_path)
    outputs=[]
    # 실제 작은 fixture에서 광선1개/5개/16개로 나눠 합산 순서 보존을 검증한다.
    one_ray=2*3*32*8
    for gib in (one_ray/1024**3,5*one_ray/1024**3,8,24):
        prepared=D.prepare(**args,memory_gib=64,ray_buffer_gib=gib)
        j,error,report=D.execute_candidate(prepared,library)
        assert report['ray_buffer_slots']==prepared['report']['memory']['ray_batch']*3
        outputs.append((j,error,report))
    for j,error,report in outputs[1:]:
        np.testing.assert_array_equal(j,outputs[0][0])
        np.testing.assert_array_equal(error,outputs[0][1])
        for key in ('iterations','relative_change','actual_K_norm_upper','residual_upper','error_upper_max'):
            assert report[key]==outputs[0][2][key]


def test_full_grid_ray_memory_inventory_selects_5_and_16(tmp_path):
    args=verified_rel_fixture(tmp_path)
    material=D.read_material(args['cells']/'material_cells.bin',b'LUMCEL01')
    native=D.read_material(args['seed']/'native_material_nodes.bin',b'LUMNOD01')
    old=D.memory_inventory_sizes(material,89,1115020,1.1267110173453303e-5,native,0,0,True)
    new=D.memory_inventory_sizes(material,89,1115020,1.1267110173453303e-5,native,0,0,True,24)
    assert old['ray_batch']==5 and new['ray_batch']==16
    assert new['parts']['directional_ray_buffers']==2*16*90*1115020*8
    assert new['known_allocation_bound_bytes']>old['known_allocation_bound_bytes']


@pytest.mark.parametrize('value',[0,-1,25,float('inf'),float('nan')])
def test_invalid_ray_budget_rejected_before_kernel(tmp_path,value):
    with pytest.raises(ValueError,match='버퍼'):
        D.prepare(**verified_rel_fixture(tmp_path),ray_buffer_gib=value)


def test_unapproved_memory_extent_and_stationary_buffer_rejected(tmp_path):
    args=fixture(tmp_path)
    with pytest.raises(ValueError,match='64GiB'):D.prepare(**args,memory_gib=65)
    with pytest.raises(ValueError,match='full-SR'):D.prepare(**args,ray_buffer_gib=24)


def test_original_radius_decimal_serialization_not_float_equality(tmp_path):
    binary=np.array([6054883681828956.,6047148868754893.])
    before=binary.copy()
    path=tmp_path/'RVTJ'
    path.write_text('Radius (10^10 cm)\n6.0548836818E+05 6.0471488688E+05\nVelocity (km/s)\n')
    result=D.radius_serialization_contract(path,binary)
    assert result['matches'] and result['serialized_exact'] and not result['binary_exact']
    assert result['max_relative_difference']<1e-10
    np.testing.assert_array_equal(before,binary)
    # 마지막 출력 자리 한 칸만 달라도 허용하지 않는다.
    path.write_text(path.read_text().replace('6.0548836818E+05','6.0548836819E+05'))
    assert not D.radius_serialization_contract(path,binary)['matches']


def test_radius_serialization_does_not_accept_lower_precision(tmp_path):
    binary=np.array([6054883681828956.])
    path=tmp_path/'RVTJ'
    path.write_text('Radius (10^10 cm)\n6.05488368E+05\nVelocity (km/s)\n')
    assert not D.radius_serialization_contract(path,binary)['matches']


def test_frame_uses_unchanged_binary_geometry_for_velocity(tmp_path):
    args=verified_rel_fixture(tmp_path)
    native=D.read_material(args['seed']/'native_material_nodes.bin',b'LUMNOD01')
    physical=np.array(native['physical']);physical[1,0]+=4000.
    native={**native,'physical':physical}
    path=tmp_path/'RVTJ'
    # 이 차이는 원래 .10E 반경 출력에 완전히 포함된다.
    path.write_text('Radius (10^10 cm)\n4.0000000000E+04 3.0000000000E+04 2.0000000000E+04\n'
        'Velocity (km/s)\n4000 3000 2000\n\ndlnV/dlnr-1\n0 0 0\n')
    flags,rel,velocity,reasons=D.frame_contract(args['model'],native,native['epoch'])
    assert rel and not reasons
    assert velocity['radius_serialization']['serialized_exact']
    assert velocity['max_relative_v_vs_r_over_t']>0
    assert native['physical'][1,0]==3e14+4000.


def test_declared_symlink_identity_preserves_hash_and_target(tmp_path):
    target=tmp_path/'actual.dat';target.write_text('same sealed input')
    link=tmp_path/'TWO_PHOT_DATA';link.symlink_to(target)
    report=dict(input_sha256={str(link):D.sha(link)})
    D.identity(report,link)
    D.identity(report,target)  # 선언된 링크의 동일 실제 파일만 허용한다.
    other=tmp_path/'unrelated.dat';other.write_bytes(target.read_bytes())
    with pytest.raises(ValueError):D.identity(report,other)
    target.write_text('changed')
    with pytest.raises(ValueError):D.identity(report,link)


def test_conflicting_declared_and_canonical_hashes_are_rejected(tmp_path):
    target=tmp_path/'actual.dat';target.write_text('sealed')
    link=tmp_path/'alias';link.symlink_to(target)
    report=dict(input_sha256={str(link):D.sha(link),str(target):'0'*64})
    with pytest.raises(ValueError):D.identity(report,link)


@pytest.mark.parametrize('scattering',[None,0.,1e-14])
def test_checkpointed_full_sr_is_bitwise_identical_to_existing_path(tmp_path,library,scattering):
    args=verified_rel_fixture(tmp_path)
    if scattering is not None:
        for key,ns in (('sources',2),('boundary_sources',3)):
            component=np.memmap(args[key]/'sources.bin',mode='r+',dtype='<f8',
                offset=16+8*(32+2*ns),shape=(7,ns,32))
            component[6]=scattering;component.flush();del component;rehash(args[key])
    prepared=D.prepare(**args)
    j0,e0,r0=D.execute_candidate(prepared,library)
    folder=tmp_path/'checkpoint'
    j,e,r=D.execute_candidate(prepared,library,checkpoint_dir=folder)
    np.testing.assert_array_equal(j,j0);np.testing.assert_array_equal(e,e0)
    assert j.tobytes()==j0.tobytes() and e.tobytes()==e0.tobytes()
    for key in ('iterations','relative_change','actual_K_norm_upper','residual_upper','error_upper_max'):
        assert r[key]==r0[key]
    state=json.loads((folder/'checkpoint.json').read_text())
    phases=[ev['phase'] for ev in state['events']]
    assert phases.index('K_NORM_STARTED')>max(i for i,v in enumerate(phases) if v=='ITERATION_COMPLETED')
    assert phases[-1]=='ERROR_ENVELOPE_COMPLETED'
    assert state['completed_iterations']==r['iterations']
    assert not state['accepted_physical_field'] and not state['componentwise_verified']
    assert len(list(folder.glob('J-*.npy')))==2
    assert r['snapshot_seconds_total']>=0 and r['resumed_iterations']==0
    baseline=os.environ.get('LUMINA_BASELINE_LIBRARY')
    if baseline:
        assert D.sha(baseline)==os.environ['LUMINA_BASELINE_LIBRARY_SHA256']
        jb,eb,rb=D.execute_candidate(prepared,baseline)
        np.testing.assert_array_equal(j,jb);np.testing.assert_array_equal(e,eb)
        assert j.tobytes()==jb.tobytes() and e.tobytes()==eb.tobytes()
        for key in ('iterations','relative_change','actual_K_norm_upper','residual_upper','error_upper_max'):
            assert r[key]==rb[key]


@pytest.mark.parametrize('kill_phase',[1,2])
def test_sigkill_checkpoint_resume_preserves_trajectory_and_total_iterations(tmp_path,library,kill_phase):
    args=verified_rel_fixture(tmp_path);prepared=D.prepare(**args)
    j0,e0,r0=D.execute_candidate(prepared,library)
    folder=tmp_path/'interrupted'
    code='''
import json,os,signal,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run_lumina_cell_candidate as D
from lumina_field_checkpoint import FieldCheckpoint
args={k:Path(v) for k,v in json.loads(sys.argv[2]).items()}
phase=int(sys.argv[5]);original=FieldCheckpoint.record
def record(self,p,n,*rest):
    original(self,p,n,*rest)
    if p==phase and (phase==2 or n==2):os.kill(os.getpid(),signal.SIGKILL)
FieldCheckpoint.record=record
D.execute_candidate(D.prepare(**args),sys.argv[3],checkpoint_dir=Path(sys.argv[4]))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts'),
        json.dumps({k:str(v) for k,v in args.items()}),str(library),str(folder),str(kill_phase)],
        capture_output=True,timeout=30)
    assert result.returncode==-9,result.stderr.decode()
    saved=json.loads((folder/'checkpoint.json').read_text())
    assert saved['completed_iterations']==(2 if kill_phase==1 else r0['iterations'])
    j,e,r=D.execute_candidate(prepared,library,checkpoint_dir=tmp_path/'resumed',resume_checkpoint=folder)
    np.testing.assert_array_equal(j,j0);np.testing.assert_array_equal(e,e0)
    assert j.tobytes()==j0.tobytes() and e.tobytes()==e0.tobytes()
    assert r['iterations']==r0['iterations'] and r['resumed_iterations']==saved['completed_iterations']
    state=json.loads((tmp_path/'resumed/checkpoint.json').read_text())
    assert state['resume_origin']['directory']==str(folder)
    if kill_phase==2:
        assert not any(ev['phase']=='ITERATION_COMPLETED' for ev in state['events'])


def test_checkpoint_does_not_reset_exhausted_iteration_budget(tmp_path,library):
    prepared=D.prepare(**verified_rel_fixture(tmp_path));folder=tmp_path/'checkpoint'
    with pytest.raises(D.CandidateFailure) as exc:
        D.execute_candidate(prepared,library,iteration_cap=2,tolerance=1e-30,checkpoint_dir=folder)
    assert exc.value.status==5 and exc.value.partial['counts'][0]==2
    with pytest.raises(ValueError,match='상한 소진'):
        D.execute_candidate(prepared,library,iteration_cap=2,tolerance=1e-30,
            checkpoint_dir=tmp_path/'resumed',resume_checkpoint=folder)


@pytest.mark.parametrize('fault',['J','metadata','contract'])
def test_checkpoint_changes_cannot_resume(tmp_path,library,fault):
    prepared=D.prepare(**verified_rel_fixture(tmp_path));folder=tmp_path/'checkpoint'
    D.execute_candidate(prepared,library,checkpoint_dir=folder)
    path=folder/'checkpoint.json';state=json.loads(path.read_text())
    if fault=='J':(folder/state['J']['filename']).write_bytes(b'incomplete')
    elif fault=='metadata':
        state['completed_iterations']=1;path.write_text(json.dumps(state))
    else:prepared['report']['input_sha256']['changed-input']='0'*64
    with pytest.raises(ValueError):
        D.execute_candidate(prepared,library,checkpoint_dir=tmp_path/'resumed',resume_checkpoint=folder)


def test_failed_checkpoint_commit_keeps_last_valid_J(tmp_path,library,monkeypatch):
    import lumina_field_checkpoint as C
    prepared=D.prepare(**verified_rel_fixture(tmp_path));folder=tmp_path/'checkpoint'
    original=C.os.replace
    def replace(source,target):
        if Path(target).name=='checkpoint.json' and (folder/'J-000003.npy').exists():
            raise OSError('저장 실패 주입')
        return original(source,target)
    monkeypatch.setattr(C.os,'replace',replace)
    with pytest.raises(OSError,match='저장 실패 주입'):
        D.execute_candidate(prepared,library,checkpoint_dir=folder)
    state=json.loads((folder/'checkpoint.json').read_text())
    assert state['completed_iterations']==2
    j,n,relative,absolute,origin=C.read_checkpoint(folder,state['contract'])
    assert n==2 and np.isfinite(j).all()


def test_checkpoint_cli_reuses_completed_J_and_produces_verified_field(tmp_path,library):
    args=verified_rel_fixture(tmp_path);before={p:D.sha(p) for p in
        (args['sources']/'sources.bin',args['boundary_sources']/'sources.bin')}
    command=[sys.executable,'-B',str(ROOT/'scripts/run_lumina_cell_candidate.py')]
    for key,value in args.items():command+=['--'+key.replace('_','-'),str(value)]
    command+=['--library',str(library),'--execute-candidate','--checkpoint']
    reports=[];fields=[]
    for label in ('first','resumed'):
        out=tmp_path/label;cmd=command+['--out',str(out)]
        if label=='resumed':cmd+=['--resume-checkpoint',str(tmp_path/'first/checkpoint')]
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        assert run.returncode==0,(run.stdout,run.stderr)
        report=json.loads((out/'preflight.json').read_text());reports.append(report)
        assert report['status']=='CANDIDATE_COMPLETED' and report['candidate']['componentwise_verified']
        with np.load(out/'candidate_field.npz') as data:
            fields.append({key:np.array(data[key]) for key in data.files})
    for key in fields[0]:
        np.testing.assert_array_equal(fields[0][key],fields[1][key])
        assert fields[0][key].tobytes()==fields[1][key].tobytes()
    assert reports[1]['candidate']['resumed_iterations']==reports[0]['candidate']['iterations']
    assert all(D.sha(path)==expected for path,expected in before.items())
