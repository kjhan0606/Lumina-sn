"""예산 계상/실패 보존의 제어 시험. 여기서 실제 물리장을 실행하지 않는다."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lumina_bridge_field0 as R
from run_lumina_cell_candidate import prepare as real_prepare
from test_lumina_cell_candidate import verified_rel_fixture


def config(tmp_path,monkeypatch):
    args=verified_rel_fixture(tmp_path);monkeypatch.setenv('OMP_NUM_THREADS','16')
    plan=dict(initial_material=str(tmp_path),initial_record_storage_sha256='a'*64,
        reference_field_sha256='b'*64,resources_per_run=dict(CPU=16,OMP=16,memory_GiB=32,
            GPU=0,parent_seconds=2700,child_seconds=2400,node_pinned=False,retry=0))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));identity=R.D.sha(path)
    (args['seed']/'material-seed.json').write_text(json.dumps(dict(plan_sha256=identity,
        source_record=70,record_storage_sha256='a'*64,
        input_sha256={str(tmp_path/'RVTJ'):R.D.sha(tmp_path/'RVTJ')})))
    grid=tmp_path/'frequency.npy';nu=1e14*np.exp(np.arange(32)*.001);np.save(grid,nu)
    (tmp_path/'grid.json').write_text(json.dumps(dict(dlognu=.001,reference_field_sha256='b'*64)))
    lib=tmp_path/'library.so';lib.write_bytes(b'control-test-placeholder-not-loaded')
    c=dict(plan=str(path),plan_sha256=identity,seed=str(args['seed']),cells=str(args['cells']),
        grid=str(grid),grid_sha256=R.D.sha(grid),library=str(lib),library_sha256=R.D.sha(lib),
        full_population='not-used',native_full_population='not-used',phot_library='not-used',
        kernel_library='not-used')
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c))
    monkeypatch.setattr(R.D,'prepare',lambda *a:dict(sources=dict(nu=nu),report=dict(execution_eligible=True,
        implemented_frame=R.D.REL_FRAME,reasons=[])))
    return cfg


@pytest.mark.parametrize('failure_at',[None,1,3])
def test_one_reservation_and_no_hidden_retries(tmp_path,monkeypatch,failure_at):
    cfg=config(tmp_path,monkeypatch);out=tmp_path/'run';calls=[]
    def command(command,**kwargs):
        calls.append(command)
        if len(calls)==failure_at:raise subprocess.CalledProcessError(5,command)
        if len(calls)==3:
            field=out/'field0';field.mkdir()
            (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        assert 0<kwargs['timeout']<=2400
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    if failure_at is None:assert R.execute(cfg,R.D.sha(cfg),out)==0
    else:
        with pytest.raises(subprocess.CalledProcessError):R.execute(cfg,R.D.sha(cfg),out)
    result=json.loads((out/'execution.json').read_text())
    assert result['physical_field_executions']==(0 if failure_at==1 else 1)
    assert result['reserved_run_number']==1 and result['automatic_retry']==0
    assert result['matter_updates']==result['native_executions']==0
    assert len(calls)==(failure_at or 3)
    assert result['status']==('FIELD0_COMPLETED' if failure_at is None else 'FAILED_PRESERVED')
    with pytest.raises(ValueError,match='보존'):R.execute(cfg,R.D.sha(cfg),out)


def test_config_identity_failure_cannot_create_or_execute(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch);out=tmp_path/'run'
    with pytest.raises(ValueError,match='SHA256'):R.execute(cfg,'0'*64,out)
    assert not out.exists()


@pytest.mark.parametrize('fault',[None,'report_hash','input_hash','grid','binary'])
def test_reuse_keeps_budget_and_revalidates_complete_inputs(tmp_path,monkeypatch,fault):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    # 기존 작은 fixture의 두 완성 계수를 봉인된 입력으로 지정한다.
    extra=tmp_path/'producer-input';extra.write_bytes(b'fixed')
    atomic=Path(c['cells'])/'atomic';atomic.mkdir()
    (atomic/'levels.csv').write_text('fixture')
    for key in ('full_population','native_full_population','phot_library','kernel_library'):
        c[key]=str(extra)
    nu=np.fromfile(tmp_path/'sources/sources.bin',dtype='<f8',offset=16,count=32)
    np.save(c['grid'],nu);c['grid_sha256']=R.D.sha(c['grid'])
    monkeypatch.setattr(R.D,'prepare',real_prepare)
    c['reserved_run_number']=3;c['reuse_sources']={}
    for label,folder in (('cell_sources',tmp_path/'sources'),('boundary_sources',tmp_path/'boundary')):
        path=folder/'sources.json';report=json.loads(path.read_text())
        for source in (extra,atomic/'levels.csv'):
            report['input_sha256'][str(source.resolve())]=R.D.sha(source)
        path.write_text(json.dumps(report))
        c['reuse_sources'][label]=dict(directory=str(folder),report_sha256=R.D.sha(path))
    if fault=='report_hash':
        c['reuse_sources']['cell_sources']['report_sha256']='0'*64
    if fault=='input_hash':extra.write_bytes(b'changed')
    if fault=='binary':
        with (tmp_path/'sources/sources.bin').open('ab') as stream:stream.write(b'changed')
    if fault=='grid':
        monkeypatch.setattr(R.D,'prepare',lambda *a:dict(sources=dict(nu=np.array([1.,2.]))))
    cfg.write_text(json.dumps(c));out=tmp_path/'run';calls=[]
    def command(cmd,**kwargs):
        calls.append(cmd)
        field=out/'field0';field.mkdir()
        (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    if fault:
        with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),out)
        assert not calls
    else:
        assert R.execute(cfg,R.D.sha(cfg),out)==0
        assert len(calls)==1  # 생산을 반복하지 않고 실제 수송만 실행한다.
        result=json.loads((out/'execution.json').read_text())
        assert result['reserved_run_number']==3 and result['physical_field_executions']==1
        assert [s['state'] for s in result['steps']]==['REUSED_COMPLETED']*2+['COMPLETED']
        assert str(tmp_path/'sources') in calls[0] and str(tmp_path/'boundary') in calls[0]


@pytest.mark.parametrize('failure',[False,True])
def test_shared_production_is_one_child_with_unchanged_field_count(tmp_path,monkeypatch,failure):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    c['share_source_sigma']=True;c['reserved_run_number']=3;cfg.write_text(json.dumps(c))
    out=tmp_path/'paired_run';calls=[]
    def command(cmd,**kwargs):
        calls.append(cmd)
        assert 0<kwargs['timeout']<=2400
        if len(calls)==1:
            assert '--peer-material' in cmd and '--peer-C-full-population' in cmd
            if failure:raise subprocess.CalledProcessError(5,cmd)
        else:
            field=out/'field0';field.mkdir()
            (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):R.execute(cfg,R.D.sha(cfg),out)
    else:assert R.execute(cfg,R.D.sha(cfg),out)==0
    result=json.loads((out/'execution.json').read_text())
    assert result['reserved_run_number']==3
    assert result['physical_field_executions']==(0 if failure else 1)
    assert [step['step'] for step in result['steps']]==(['paired_sources'] if failure else
        ['paired_sources','field0'])


def amended_config(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    template=Path(__file__).resolve().parents[1]/'docs/LUMINA_FIELD0_RESOURCE_AMENDMENT_2026-10-20.json'
    amendment=json.loads(template.read_text())
    amendment['original_plan_sha256']=c['plan_sha256']
    path=tmp_path/'amendment.json';path.write_text(json.dumps(amendment))
    c.update(resource_amendment=str(path),resource_amendment_sha256=R.D.sha(path),
             reserved_run_number=3,share_source_sigma=True)
    cfg.write_text(json.dumps(c))
    return cfg,path


def test_approved_last_reservation_forwards_shared_budget_and_buffer(tmp_path,monkeypatch):
    cfg,path=amended_config(tmp_path,monkeypatch);out=tmp_path/'run';calls=[];budgets=[]
    def command(cmd,**kwargs):
        calls.append(cmd);budgets.append(kwargs['timeout'])
        if len(calls)==2:
            assert cmd[cmd.index('--memory-GiB')+1]=='64'
            assert cmd[cmd.index('--ray-buffer-GiB')+1]=='24'
            field=out/'field0';field.mkdir()
            (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    assert R.execute(cfg,R.D.sha(cfg),out)==0
    result=json.loads((out/'execution.json').read_text())
    assert 0<budgets[1]<budgets[0]<=10200
    assert result['resources']['memory_GiB']==64
    assert result['reserved_run_number']==3 and result['physical_field_executions']==1
    assert result['resource_amendment_sha256']==R.D.sha(path)
    assert result['ray_buffer_GiB']==24


@pytest.mark.parametrize('fault',['hash','generation','reservation','memory','time','retry','buffer','total'])
def test_resource_amendment_cannot_expand_approval(tmp_path,monkeypatch,fault):
    cfg,path=amended_config(tmp_path,monkeypatch)
    c=json.loads(cfg.read_text());a=json.loads(path.read_text())
    if fault=='hash':c['resource_amendment_sha256']='0'*64
    if fault=='generation':a['original_plan_sha256']='0'*64
    if fault=='reservation':c['reserved_run_number']=4
    if fault=='memory':a['resources_per_run']['memory_GiB']=65
    if fault=='time':a['resources_per_run']['child_seconds']=10800
    if fault=='retry':a['resources_per_run']['retry']=1
    if fault=='buffer':a['ray_buffer_GiB']=25
    if fault=='total':a['maximum_parent_reservation_seconds']=20000
    path.write_text(json.dumps(a))
    if fault!='hash':c['resource_amendment_sha256']=R.D.sha(path)
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def reuse_amended_config(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    extra=tmp_path/'producer-input';extra.write_text('fixed')
    alias=tmp_path/'TWO_PHOT_DATA';alias.symlink_to(extra)
    atomic=Path(c['cells'])/'atomic';atomic.mkdir();(atomic/'levels.csv').write_text('fixture')
    for key in ('full_population','native_full_population','phot_library','kernel_library'):
        c[key]=str(extra)
    reuse={}
    for label,old in (('cell_sources','sources'),('boundary_sources','boundary')):
        folder=tmp_path/label;(tmp_path/old).rename(folder)
        path=folder/'sources.json';report=json.loads(path.read_text())
        for source in (extra,alias,atomic/'levels.csv'):
            report['input_sha256'][str(source)]=R.D.sha(source)
        path.write_text(json.dumps(report))
        reuse[label]=dict(directory=str(folder),report_sha256=R.D.sha(path))
    old=tmp_path/'execution.json'
    old.write_text(json.dumps(dict(slurm_job_id='1200155',status='FAILED_PRESERVED',
        physical_field_executions=0,plan_sha256=c['plan_sha256'],
        steps=[dict(step='paired_sources',state='COMPLETED')])))
    template=Path(__file__).resolve().parents[1]/'docs/LUMINA_FIELD0_REUSE_AMENDMENT_2026-10-20.json'
    a=json.loads(template.read_text())
    a.update(original_plan_sha256=c['plan_sha256'],previous_execution=str(old),
             previous_execution_sha256=R.D.sha(old),approved_reuse_sources=reuse)
    amendment=tmp_path/'reuse-amendment.json';amendment.write_text(json.dumps(a))
    c.update(reserved_run_number=4,resource_amendment=str(amendment),
             resource_amendment_sha256=R.D.sha(amendment),reuse_sources=reuse)
    cfg.write_text(json.dumps(c))
    return cfg


@pytest.mark.parametrize('reservation',[4,5])
def test_approved_reuse_reservation_only_calls_transport(tmp_path,monkeypatch,reservation):
    cfg=reuse_amended_config(tmp_path,monkeypatch);out=tmp_path/'run';calls=[]
    if reservation==5:upgrade_fifth(cfg,tmp_path)
    def command(cmd,**kwargs):
        calls.append(cmd)
        assert Path(cmd[1]).name=='run_lumina_cell_candidate.py'
        assert 0<kwargs['timeout']<=10200
        field=out/'field0';field.mkdir()
        (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    assert R.execute(cfg,R.D.sha(cfg),out)==0
    result=json.loads((out/'execution.json').read_text())
    assert len(calls)==1 and result['physical_field_executions']==1
    assert result['reserved_run_number']==reservation
    assert [s['state'] for s in result['steps']]==['REUSED_COMPLETED']*2+['COMPLETED']


@pytest.mark.parametrize('fault',['missing','regenerate','fifth','history','different_source'])
def test_reuse_amendment_never_falls_back_to_production(tmp_path,monkeypatch,fault):
    cfg=reuse_amended_config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    if fault=='missing':c['reuse_sources'].pop('boundary_sources')
    if fault=='regenerate':c['share_source_sigma']=True
    if fault=='fifth':c['reserved_run_number']=5
    if fault=='history':(tmp_path/'execution.json').write_text('{}')
    if fault=='different_source':c['reuse_sources']['cell_sources']['directory']=str(tmp_path/'other')
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_actual_frame_checked_before_any_source_production(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch)
    rvtj=tmp_path/'RVTJ';rvtj.write_text(rvtj.read_text().replace('4000 3000 2000','4001 3000 2000'))
    seed=tmp_path/'seed/material-seed.json';report=json.loads(seed.read_text())
    report['input_sha256'][str(rvtj)]=R.D.sha(rvtj);seed.write_text(json.dumps(report))
    def forbidden(*args,**kwargs):raise AssertionError('잘못된 frame으로 계수 생산 금지')
    monkeypatch.setattr(R.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='생산 전'):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    result=json.loads((tmp_path/'run/execution.json').read_text())
    assert result['physical_field_executions']==0 and result['steps']==[]


def upgrade_fifth(cfg,tmp_path):
    c=json.loads(cfg.read_text());path=Path(c['resource_amendment']);a=json.loads(path.read_text())
    failed=tmp_path/'fourth-failed.json'
    failed.write_text(json.dumps(dict(slurm_job_id='1200286',reserved_run_number=4,
        status='FAILED_PRESERVED',physical_field_executions=0,plan_sha256=c['plan_sha256'])))
    a.update(schema='lumina-field0-reuse-amendment-v2',reserved_run_number=5,maximum_runs=5,
        previous_reservations=[1199911,1199926,1200155,1200286],maximum_parent_reservation_seconds=37800,
        previous_failed_execution=str(failed),previous_failed_execution_sha256=R.D.sha(failed))
    path.write_text(json.dumps(a));c.update(reserved_run_number=5,resource_amendment_sha256=R.D.sha(path))
    cfg.write_text(json.dumps(c))


@pytest.mark.parametrize('fault',['sixth','failed_hash','failed_field','missing_source'])
def test_fifth_approval_does_not_allow_more_runs_or_hide_failed_history(tmp_path,monkeypatch,fault):
    cfg=reuse_amended_config(tmp_path,monkeypatch);upgrade_fifth(cfg,tmp_path)
    c=json.loads(cfg.read_text())
    if fault=='sixth':c['reserved_run_number']=6
    if fault=='missing_source':c['reuse_sources'].pop('cell_sources')
    if fault in ('failed_hash','failed_field'):
        failed=tmp_path/'fourth-failed.json';f=json.loads(failed.read_text())
        f['physical_field_executions']=1;failed.write_text(json.dumps(f))
        if fault=='failed_field':
            path=Path(c['resource_amendment']);a=json.loads(path.read_text())
            a['previous_failed_execution_sha256']=R.D.sha(failed)
            path.write_text(json.dumps(a));c['resource_amendment_sha256']=R.D.sha(path)
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('resume',[False,True])
@pytest.mark.parametrize('timeout',[False,True])
def test_checkpoint_flags_and_failure_evidence_keep_same_budget(tmp_path,monkeypatch,resume,timeout):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text());c['checkpoint']=True
    origin=tmp_path/'old-checkpoint';origin.mkdir()
    (origin/'checkpoint.json').write_text('sealed old state')
    if resume:c['resume_checkpoint']=dict(directory=str(origin),metadata_sha256=R.D.sha(origin/'checkpoint.json'))
    cfg.write_text(json.dumps(c));out=tmp_path/'run';calls=[]
    def command(cmd,**kwargs):
        calls.append(cmd)
        assert 0<kwargs['timeout']<=2400
        if len(calls)==3:
            assert '--checkpoint' in cmd
            assert ('--resume-checkpoint' in cmd)==resume
            if resume:assert cmd[cmd.index('--resume-checkpoint')+1]==str(origin)
            folder=out/'field0/checkpoint';folder.mkdir(parents=True)
            (folder/'checkpoint.json').write_text('preserved unverified state')
            if timeout:raise subprocess.TimeoutExpired(cmd,kwargs['timeout'])
            (out/'field0/preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):R.execute(cfg,R.D.sha(cfg),out)
    else:assert R.execute(cfg,R.D.sha(cfg),out)==0
    state=json.loads((out/'execution.json').read_text())
    assert len(calls)==3 and state['physical_field_executions']==1
    assert state['automatic_retry']==0 and state['reserved_run_number']==1
    assert state['checkpoint']['metadata_sha256']==R.D.sha(out/'field0/checkpoint/checkpoint.json')
    assert state['checkpoint']['resume_origin']==c.get('resume_checkpoint')
    assert (origin/'checkpoint.json').read_text()=='sealed old state'
    assert state['status']==('FAILED_PRESERVED' if timeout else 'FIELD0_COMPLETED')


@pytest.mark.parametrize('fault',['disabled','hash','format','not_bool','sixth'])
def test_checkpoint_cannot_bypass_identity_or_reservation(tmp_path,monkeypatch,fault):
    cfg=config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    origin=tmp_path/'old-checkpoint';origin.mkdir();(origin/'checkpoint.json').write_text('sealed')
    c.update(checkpoint=True,resume_checkpoint=dict(directory=str(origin),metadata_sha256=R.D.sha(origin/'checkpoint.json')))
    if fault=='disabled':c['checkpoint']=False
    if fault=='hash':c['resume_checkpoint']['metadata_sha256']='0'*64
    if fault=='format':c['resume_checkpoint']=str(origin)
    if fault=='not_bool':c['checkpoint']='yes'
    if fault=='sixth':c['reserved_run_number']=6
    cfg.write_text(json.dumps(c))
    monkeypatch.setattr(R.subprocess,'run',lambda *a,**k:pytest.fail('부적합 입력 실행 금지'))
    with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def checkpoint_approved_config(tmp_path,monkeypatch):
    cfg=reuse_amended_config(tmp_path,monkeypatch);upgrade_fifth(cfg,tmp_path)
    c=json.loads(cfg.read_text());path=Path(c['resource_amendment']);a=json.loads(path.read_text())
    previous=tmp_path/'fifth-timeout.json'
    previous.write_text(json.dumps(dict(slurm_job_id='1200364',reserved_run_number=5,
        status='FAILED_PRESERVED',physical_field_executions=1,plan_sha256=c['plan_sha256'],
        steps=[dict(step='field0',state='FAILED')])) )
    a.update(schema='lumina-field0-checkpoint-amendment-v1',reserved_run_number=6,maximum_runs=6,
        previous_reservations=[1199911,1199926,1200155,1200286,1200364],
        maximum_parent_reservation_seconds=48600,required_checkpoint=True,
        approved_library_sha256=c['library_sha256'],previous_transport_execution=str(previous),
        previous_transport_execution_sha256=R.D.sha(previous))
    path.write_text(json.dumps(a));c.update(reserved_run_number=6,checkpoint=True,
        resource_amendment_sha256=R.D.sha(path));cfg.write_text(json.dumps(c))
    return cfg


def test_sixth_approval_calls_only_checkpointed_transport(tmp_path,monkeypatch):
    cfg=checkpoint_approved_config(tmp_path,monkeypatch);calls=[];out=tmp_path/'run'
    def command(cmd,**kwargs):
        calls.append(cmd)
        assert Path(cmd[1]).name=='run_lumina_cell_candidate.py' and '--checkpoint' in cmd
        assert '--resume-checkpoint' not in cmd and 0<kwargs['timeout']<=10200
        field=out/'field0';field.mkdir()
        (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    assert R.execute(cfg,R.D.sha(cfg),out)==0
    state=json.loads((out/'execution.json').read_text())
    assert len(calls)==1 and state['reserved_run_number']==6
    assert state['physical_field_executions']==1 and state['automatic_retry']==0
    assert state['resources']['memory_GiB']==64 and state['resources']['parent_seconds']==10800
    assert [s['state'] for s in state['steps']]==['REUSED_COMPLETED']*2+['COMPLETED']


@pytest.mark.parametrize('fault',['seventh','checkpoint','resume','library','history_hash',
                                 'history_count','time','total'])
def test_sixth_approval_cannot_expand_or_reset_history(tmp_path,monkeypatch,fault):
    cfg=checkpoint_approved_config(tmp_path,monkeypatch);c=json.loads(cfg.read_text())
    path=Path(c['resource_amendment']);a=json.loads(path.read_text())
    if fault=='seventh':c['reserved_run_number']=7
    if fault=='checkpoint':c['checkpoint']=False
    if fault=='resume':c['resume_checkpoint']={}
    if fault=='library':c['library_sha256']='0'*64
    if fault=='time':a['resources_per_run']['child_seconds']=10201
    if fault=='total':a['maximum_parent_reservation_seconds']=59400
    if fault in ('history_hash','history_count'):
        previous=Path(a['previous_transport_execution']);state=json.loads(previous.read_text())
        state['physical_field_executions']=0;previous.write_text(json.dumps(state))
        if fault=='history_count':a['previous_transport_execution_sha256']=R.D.sha(previous)
    path.write_text(json.dumps(a));c['resource_amendment_sha256']=R.D.sha(path)
    cfg.write_text(json.dumps(c))
    monkeypatch.setattr(R.subprocess,'run',lambda *a,**k:pytest.fail('잘못된 추가 예약 실행 금지'))
    with pytest.raises(ValueError):R.execute(cfg,R.D.sha(cfg),tmp_path/'run')
    assert not (tmp_path/'run').exists()
