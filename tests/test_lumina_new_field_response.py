"""새 장의 소비 계약·좌표별 비교 시험. 실제 BF/SE/RT 생산은 없다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_new_field_response as N
from test_lumina_resident_feedback import scenario
import lumina_resident_feedback as F


def test_thermal_producer_identity_is_available_for_zero_temperature_evaluations():
    hashes=N.thermal_implementation_hashes()
    assert tuple(hashes)==N.completed_seed.THERMAL_PRODUCERS
    assert all(hashes[name]==N.B.sha(Path(N.__file__).with_name(name))
               for name in N.completed_seed.THERMAL_PRODUCERS)


def test_response_libraries_come_from_actual_new_coefficient_producer(tmp_path):
    paths=[tmp_path/'new-run/kernels'/name for name in ('cmfgen_phot_smoothing.so','model_sources.so')]
    paths[0].parent.mkdir(parents=True)
    for p in paths:p.write_bytes(p.name.encode())
    report=dict(input_sha256={str(p):N.B.sha(p) for p in paths})
    assert N.source_atomic_libraries(report)==tuple(paths)
    paths[1].write_bytes(b'changed')
    with pytest.raises(ValueError,match='라이브러리 변경'):N.source_atomic_libraries(report)


@pytest.mark.parametrize('fault',['missing','duplicate'])
def test_response_library_selection_never_falls_back_to_an_old_global_path(tmp_path,fault):
    paths=[tmp_path/name for name in ('cmfgen_phot_smoothing.so','model_sources.so')]
    for p in paths:p.write_bytes(b'fixture')
    inputs={str(p):N.B.sha(p) for p in paths}
    if fault=='missing':inputs.pop(str(paths[1]))
    else:inputs[str(tmp_path/'old/model_sources.so')]='0'*64
    with pytest.raises(ValueError,match='누락/중복'):N.source_atomic_libraries(dict(input_sha256=inputs))


def test_response_library_preserves_the_producer_identity_for_symlink_paths(tmp_path):
    paths=[tmp_path/name for name in ('cmfgen_phot_smoothing.so','model_sources.so')]
    target=tmp_path/'compiled-photo-target.so';target.write_bytes(b'phot')
    paths[0].symlink_to(target);paths[1].write_bytes(b'kernel')
    report=dict(input_sha256={str(p):N.B.sha(p) for p in paths})
    assert N.source_atomic_libraries(report)==tuple(paths)
    assert all(str(p) in report['input_sha256'] for p in N.source_atomic_libraries(report))


@pytest.mark.parametrize('count',[3,790493-718])
def test_response_line_roster_matches_produced_addresses_not_the_old_hardcoded_count(tmp_path,count):
    lines=dict(lower=np.arange(count,dtype='i4'),upper=np.arange(count,dtype='i4')+1,
               nu0=np.full(count,1.e15))
    path=tmp_path/'line_catalog.npz';np.savez(path,**lines)
    report=dict(model_ions=27,full_levels=24542,routes=24667,lines=count,
                output_sha256={'line_catalog.npz':N.B.sha(path)})
    result=N.validate_source_line_roster(tmp_path,report,lines,[None]*27,[None]*24542,[None]*24667)
    assert result==(path,N.B.sha(path))
    lines['upper'][0]+=1
    with pytest.raises(ValueError,match='주소 또는 원자 계수'):
        N.validate_source_line_roster(tmp_path,report,lines,[None]*27,[None]*24542,[None]*24667)


@pytest.mark.parametrize('fault',['model','count','hash'])
def test_response_roster_still_rejects_incomplete_models_or_changed_source(tmp_path,fault):
    lines=dict(lower=np.array([0]),upper=np.array([1]));path=tmp_path/'line_catalog.npz';np.savez(path,**lines)
    report=dict(model_ions=27,full_levels=24542,routes=24667,lines=1,
                output_sha256={'line_catalog.npz':N.B.sha(path)})
    if fault=='model':report['full_levels']-=1
    elif fault=='count':report['lines']=2
    else:report['output_sha256']['line_catalog.npz']='0'*64
    with pytest.raises(ValueError):
        N.validate_source_line_roster(tmp_path,report,lines,[None]*27,[None]*24542,[None]*24667)


@pytest.fixture
def manual_resources(monkeypatch):
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'manual-lageunha')
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    monkeypatch.setattr(N.socket, 'gethostname', lambda: 'LagEunha')
    monkeypatch.setattr(N.os, 'sched_getaffinity', lambda _: set(range(16, 32)))
    monkeypatch.setattr(N.resource, 'getrlimit', lambda _: (32*1024**3, N.resource.RLIM_INFINITY))


def test_manual_resources_use_real_limits_not_fake_slurm(manual_resources):
    result = N.execution_resources(13800)
    assert result['executor'] == 'manual-lageunha'
    assert result['cpu_affinity'] == list(range(16, 32))
    assert result['address_space_limit_bytes'] == 32*1024**3
    assert 'job_id' not in result


@pytest.mark.parametrize('fault', ['host', 'slurm_job', 'affinity', 'unlimited', 'too_large', 'threads', 'time', 'mode'])
def test_manual_resource_mismatch_is_rejected(manual_resources, monkeypatch, fault):
    seconds = 13800
    if fault == 'host': monkeypatch.setattr(N.socket, 'gethostname', lambda: 'syntax')
    elif fault == 'slurm_job': monkeypatch.setenv('SLURM_JOB_ID', '123')
    elif fault == 'affinity': monkeypatch.setattr(N.os, 'sched_getaffinity', lambda _: set(range(17)))
    elif fault in ('unlimited', 'too_large'):
        limit = N.resource.RLIM_INFINITY if fault == 'unlimited' else 33*1024**3
        monkeypatch.setattr(N.resource, 'getrlimit', lambda _: (limit, limit))
    elif fault == 'threads': monkeypatch.setenv('OMP_NUM_THREADS', '32')
    elif fault == 'time': seconds = 42601
    else: monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'other')
    with pytest.raises(ValueError): N.execution_resources(seconds)


def test_existing_slurm_resources_remain_supported(monkeypatch):
    monkeypatch.delenv('LUMINA_RESPONSE_EXECUTOR', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    assert N.execution_resources(13800)['job_id'] == '123'
    with pytest.raises(ValueError): N.execution_resources(42600)
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')
    with pytest.raises(ValueError): N.execution_resources(13800)


@pytest.mark.parametrize('hours,seconds,allocation,shorter',[
    (30,106200,'1-06:00:00','1-05:00:00'),(48,171000,'2-00:00:00','1-23:00:00')])
def test_long_thermal_requires_real_slurm_time_and_temperature_evaluations(monkeypatch,hours,seconds,allocation,shorter):
    from types import SimpleNamespace
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'slurm')
    monkeypatch.setenv('LUMINA_RESPONSE_LONG_THERMAL', '1')
    monkeypatch.setenv('LUMINA_RESPONSE_THERMAL_HOURS',str(hours))
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    shown={'value': 'JobId=456 JobState=RUNNING TimeLimit='+allocation+' NumCPUs=16'}
    def show(command, **kwargs):
        assert command == ['scontrol', 'show', 'job', '456', '-o']
        assert kwargs['check'] and kwargs['timeout'] == 10
        return SimpleNamespace(stdout=shown['value'])
    monkeypatch.setattr(N.subprocess, 'run', show)
    result=N.execution_resources(seconds, 12)
    assert result['long_thermal'] and result['seconds']==seconds
    assert result['allocated_seconds']==hours*3600 and result['thermal_hours_limit']==hours
    with pytest.raises(ValueError, match='온도 평가'):
        N.execution_resources(seconds, 0)
    with pytest.raises(ValueError, match='시간 할당 부족'):
        shown['value']='JobId=456 JobState=RUNNING TimeLimit='+shorter+' NumCPUs=16'
        N.execution_resources(seconds, 12)
    shown['value']='JobId=456 JobState=RUNNING TimeLimit='+allocation+' NumCPUs=16'
    with pytest.raises(ValueError): N.execution_resources(seconds+1, 12)
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'manual-lageunha')
    with pytest.raises(ValueError, match='Slurm 경로'):
        N.execution_resources(seconds, 12)


@pytest.mark.parametrize('reported', ['JobId=456 TimeLimit=UNLIMITED',
    'JobId=456 TimeLimit=1-06:00', 'JobId=456'])
def test_long_thermal_rejects_missing_or_ambiguous_allocation(monkeypatch, reported):
    from types import SimpleNamespace
    monkeypatch.setattr(N.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=reported))
    with pytest.raises(ValueError, match='할당 시간 확인 실패'):
        N.slurm_allocation_seconds('456')


@pytest.mark.parametrize('fault',['budget','no_secant','result','thermal','box','projection','prefix','time'])
def test_secant_root_continuation_requires_one_sealed_48_hour_mode(fault):
    from types import SimpleNamespace
    values=dict(seconds=171000,temperature_evaluations=48,restart_secant_root=True,
        secant_after_accept=True,resume_result='accepted-result',resume_result_sha='result-sha',
        resume_thermal_result='accepted-thermal-result',resume_thermal_result_sha='thermal-sha')
    if fault=='budget':values['temperature_evaluations']=49
    elif fault=='no_secant':values['secant_after_accept']=False
    elif fault=='result':values['resume_result']=None
    elif fault=='thermal':values['resume_thermal_result_sha']=None
    elif fault=='box':values['restart_box_root']=True
    elif fault=='projection':values['charge_projection']=True
    elif fault=='prefix':values['thermal_replay_prefix']=4
    else:values['seconds']=171001
    with pytest.raises(ValueError,match='secant 근 연속탐색'):
        N.run(SimpleNamespace(**values))


def test_approved_manual_duration_keeps_cpu_memory_and_host_limits(manual_resources):
    result=N.execution_resources(42600)
    assert result['seconds']==42600 and result['CPUs']==16
    assert result['memory_limit_bytes']==32*1024**3


def test_temperature_cache_memory_is_reserved_without_claiming_measured_full_RSS():
    from lumina_temperature_response_cache import extra_memory_reserve
    args = (90, 1115020, 24667, 790493, 24542)
    base = N.memory_inventory(*args)
    cached = N.memory_inventory(*args, temperature_cache=True)
    difference = cached['known_allocation_and_reserve_bytes']-base['known_allocation_and_reserve_bytes']
    assert difference == extra_memory_reserve(24667, 90, 24542)
    assert cached['known_allocation_and_reserve_bytes'] < 32*1024**3
    assert cached['whole_process_peak_RSS_verified'] is False


@pytest.mark.parametrize('limit', [4, 13, True])
def test_temperature_mode_rejects_bad_budget_before_creating_output(manual_resources, limit):
    from types import SimpleNamespace
    # out 등 실행 인수가 없어도 온도 예산 검증에서 먼저 종료돼야 한다.
    with pytest.raises(ValueError, match='온도 보정'):
        N.run(SimpleNamespace(seconds=13800, temperature_evaluations=limit))


@pytest.mark.parametrize('fault',['budget','resume','hash','replay','prefix','projection','secant','duration'])
def test_direction_request_refuses_mixed_modes_before_output(fault):
    from types import SimpleNamespace
    values=dict(seconds=42600,temperature_evaluations=4,temperature_direction_only=True,
        resume_result='completed',resume_result_sha='pinned')
    if fault=='budget':values['temperature_evaluations']=5
    elif fault=='resume':values['resume_result']=None
    elif fault=='hash':values['resume_result_sha']=None
    elif fault=='replay':values['resume_thermal_result']='old-trials'
    elif fault=='prefix':values['thermal_replay_prefix']=1
    elif fault=='projection':values['charge_projection']=True
    elif fault=='secant':values['secant_after_accept']=True
    else:values['seconds']=42601
    with pytest.raises(ValueError,match='온도방향 전용'):
        N.run(SimpleNamespace(**values))


@pytest.mark.parametrize('mode,seconds',[('manual-lageunha',0),('slurm',108000)])
def test_direction_request_does_not_expand_executor_or_allocation(monkeypatch,mode,seconds):
    from types import SimpleNamespace
    monkeypatch.setattr(N,'execution_resources',lambda *a:dict(executor=mode,allocated_seconds=seconds))
    with pytest.raises(ValueError,match='grammar Slurm'):
        N.run(SimpleNamespace(seconds=42600,temperature_evaluations=4,temperature_direction_only=True,
                              resume_result='completed',resume_result_sha='pinned'))


def test_direction_output_cannot_be_published_as_accepted_material(tmp_path):
    from prepare_lumina_accepted_material import prepare
    path=tmp_path/'result.json'; out=tmp_path/'must-not-exist'
    path.write_text(json.dumps(dict(
        state='COMPLETE_CURRENT_FIELD_THERMAL_DIRECTION_NOT_ACCEPTED_MATERIAL',
        thermal_energy_policy='physical_stored',material_published=False)))
    with pytest.raises(ValueError):
        prepare(path,N.B.sha(path),tmp_path/'seed',tmp_path/'material',out)
    assert not out.exists()


@pytest.mark.parametrize('mode,seconds,allocation,accepted',[
    ('direction',21000,'06:00:00',True),('direction',42600,'12:00:00',True),
    ('direction',21000,'12:00:00',False),('direction',21000,'05:59:00',False),
    ('refine',10200,'03:00:00',True),('refine',42600,'12:00:00',True),
    ('refine',21000,'06:00:00',True),('refine',21000,'03:00:00',False),
    ('refine',10200,'12:00:00',False),('refine',10200,'02:59:00',False),
    ('coupled30',106200,'1-06:00:00',True),('coupled48',171000,'2-00:00:00',True),
    ('secant48',171000,'2-00:00:00',True),('secant48',171000,'1-06:00:00',False),
    ('coupled48',171000,'1-06:00:00',False),('coupled30',171000,'2-00:00:00',False)])
def test_submission_wrapper_checks_actual_direction_allocation_before_archive(tmp_path,mode,seconds,allocation,accepted):
    import os
    import subprocess
    script=Path(N.__file__).with_name('submit_lumina_new_field_response.slurm')
    binaries=tmp_path/'bin'; binaries.mkdir(); parent=tmp_path/'parent'; parent.mkdir()
    control=binaries/'scontrol'
    control.write_text('#!/bin/bash\nprintf "%s\\n" "JobId=123 TimeLimit='+allocation+'"\n')
    control.chmod(0o700)
    archive=binaries/'git'
    archive.write_text('#!/bin/bash\nprintf "%s\\n" REACHED_ARCHIVE >&2\nexit 97\n')
    archive.chmod(0o700)
    startup=tmp_path/'shell-env'; startup.write_text('module() { :; }\n')
    environment={**os.environ,'PATH':str(binaries)+os.pathsep+os.environ['PATH'],
        'BASH_ENV':str(startup),'SLURM_JOB_ID':'123','LUMINA_RESPONSE_EXECUTOR':'slurm',
        'LUMINA_RESPONSE_LONG_THERMAL':'1','LUMINA_RESPONSE_SECONDS':str(seconds),
        'LUMINA_RESPONSE_THERMAL_HOURS':'48' if mode in ('coupled48','secant48') else '30'}
    evaluations='4' if mode=='direction' else ('48' if mode=='secant48' else '12')
    arguments=['bash',str(script),str(parent),'0'*40,'--temperature-evaluations',evaluations]
    if mode in ('direction','refine'):
        arguments.append('--temperature-direction-only' if mode=='direction' else '--refine-temperature-direction')
    if mode=='secant48': arguments.extend(['--restart-secant-root','--secant-after-accept'])
    result=subprocess.run(arguments,env=environment,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 # 승인된 경우에도 mock archive에서 멈추며 실제 빌드/물리0회다.
    assert ('REACHED_ARCHIVE' in result.stderr) is accepted
    assert (parent/'source').exists() is accepted


def test_secant_slurm_preflight_runs_from_extracted_project_root():
    script=Path(N.__file__).with_name('submit_lumina_new_field_response.slurm')
    text=script.read_text()
    project_created=text.index('project="$parent/source"')
    change_directory=text.index('cd "$project"',project_created)
    preflight=text.index('if [[ " $* " = *" --restart-secant-root "* ]]; then',project_created)
    pytest_invocation=text.index('/home/kjhan/miniconda3/bin/python3 -m pytest',preflight)
    compile_start=text.index('core=(lumina_plasma',preflight)
    assert project_created < change_directory < preflight < pytest_invocation < compile_start
    preflight_command=text[preflight:compile_start]
    for name in ('LUMINA_RESPONSE_EXECUTOR','LUMINA_RESPONSE_LONG_THERMAL',
                 'LUMINA_RESPONSE_SECONDS','LUMINA_RESPONSE_THERMAL_HOURS'):
        assert f'-u {name}' in preflight_command


@pytest.mark.parametrize('path,digest,limit', [('result.json', None, 5),
    (None, 'sha', 5), ('result.json', 'sha', 0)])
def test_resume_requires_pin_pair_and_explicit_temperature_mode(manual_resources, path, digest, limit):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='완료 계수 재개'):
        N.run(SimpleNamespace(seconds=13800, temperature_evaluations=limit,
                              resume_result=path, resume_result_sha=digest))


@pytest.mark.parametrize('path,digest,seed', [('result.json',None,True),
    (None,'sha',True),('result.json','sha',False)])
def test_thermal_resume_requires_pin_pair_and_seed(manual_resources,path,digest,seed):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='열평가 재개'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,
            resume_thermal_result=path,resume_thermal_result_sha=digest,
            resume_result='seed' if seed else None,resume_result_sha='seed-sha' if seed else None))


def test_temperature_correction_rejects_invalid_explicit_equation_before_output(manual_resources):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='열방정식 정책'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,thermal_policy='unknown'))


@pytest.mark.parametrize('policy', N.thermal_replay.ENERGY_POLICIES)
def test_fixed_temperature_continuation_inherits_the_accepted_material_equation(policy):
    manifest = dict(thermal_energy_policy=policy)
    assert N.continued_energy_policy(manifest, None) == policy
    assert N.continued_energy_policy(manifest, None, required=True) == policy
    assert N.continued_energy_policy(manifest, policy) == policy
    assert N.continued_energy_policy({}, policy) == policy


def test_continuation_does_not_silently_change_a_known_thermal_equation():
    with pytest.raises(ValueError, match='상위 수용 물질'):
        N.continued_energy_policy(dict(thermal_energy_policy='native_instantaneous_scenario'), 'physical_stored')
    with pytest.raises(ValueError, match='정책'):
        N.continued_energy_policy(dict(thermal_energy_policy='unknown'), None)
    assert N.continued_energy_policy({}, None) is None
    with pytest.raises(ValueError, match='상위 물질'):
        N.continued_energy_policy({}, None, required=True)


@pytest.mark.parametrize('resume,prefix,limit',[(False,None,5),(True,4,4),(True,None,5)])
def test_projection_needs_whole_completed_source_and_bounded_ne_budget(manual_resources,resume,prefix,limit):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='고정 T 전하'):
        N.run(SimpleNamespace(seconds=1200,temperature_evaluations=limit,charge_projection=True,
            resume_result='seed' if resume else None,resume_result_sha='seed-sha' if resume else None,
            resume_thermal_result='report' if resume else None,resume_thermal_result_sha='sha' if resume else None,
            thermal_replay_prefix=prefix,thermal_policy='native_instantaneous_scenario'))


@pytest.mark.parametrize('prefix',[0,5,True])
def test_invalid_thermal_probe_prefix_is_rejected_before_output(manual_resources,prefix):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='초기 열섭동'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,thermal_policy='physical_stored',
            resume_result='seed',resume_result_sha='sha',resume_thermal_result='report',
            resume_thermal_result_sha='sha',thermal_replay_prefix=prefix))


def test_saved_direction_keeps_exact_linear_system_and_never_marks_candidate_accepted(tmp_path):
    data=dict(temperature=np.array([9000.,10000.]),ne=np.array([20.,30.]))
    metadata=dict(root='evaluation-1',response_sha256='response',balance_sha256='balance')
    matrix=np.diag([1.,2.,3.,4.]);residual=np.array([.01,-.01,.02,-.02])
    delta=np.linalg.solve(matrix,-residual);charge_scale=data['ne'];thermal_scale=np.array([1.,2.])
    record=N.preserve_thermal_direction(tmp_path,1,(data,{},metadata),matrix,residual,delta,
        charge_scale,thermal_scale,field={'J':'F2'},source_identity={'material':'accepted'},
        probes=[{'root':'evaluation-'+str(i)} for i in range(2,6)],
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    root=tmp_path/'thermal-direction-1';path=root/'linearization.npz'
    assert record['linearization_sha256']==N.B.sha(path)
    assert json.loads((root/'result.json').read_text())==record
    with np.load(path) as saved:
        for key,value in dict(temperature=data['temperature'],ne=data['ne'],jacobian=matrix,
            scaled_residual=residual,log_direction=delta,charge_scale=charge_scale,
            thermal_scale=thermal_scale).items():np.testing.assert_array_equal(saved[key],value)
    assert record['base_response']==metadata and len(record['probes'])==4
    assert record['linearization_kind']=='centered' and record['centered_log_step']==1.e-4
    assert record['charge_tolerance']==1.e-10 and record['thermal_tolerance']==1.e-3
    assert record['state']=='SAVED_DIRECTION_NOT_ACCEPTED_CANDIDATE'
    assert not record['joint_convergence'] and record['new_material_evaluations']==0
    with pytest.raises(FileExistsError):
        N.preserve_thermal_direction(tmp_path,1,(data,{},metadata),matrix,residual,delta,
            charge_scale,thermal_scale,field={},source_identity={},probes=[])
    secant=N.preserve_thermal_direction(tmp_path,2,(data,{},metadata),matrix,residual,delta,
        charge_scale,thermal_scale,field={'J':'F2'},source_identity={'material':'accepted'},
        probes=[],linearization_kind='secant')
    assert secant['linearization_kind']=='secant' and secant['centered_log_step'] is None
    assert secant['probes']==[] and secant['base_response']==metadata
    with pytest.raises(ValueError,match='선형화'):
        N.preserve_thermal_direction(tmp_path,3,(data,{},metadata),matrix,residual,delta,
            charge_scale,thermal_scale,field={},source_identity={},probes=[{}],
            linearization_kind='secant')


def test_trapezoid_preserves_original_nonuniform_grid():
    nu = np.array([1., 2., 4., 9.]); saved = nu.copy()
    w = N.trapezoid_weights(nu)
    np.testing.assert_array_equal(nu, saved)
    assert w.sum() == 8.
    assert np.dot(w, 2.*nu+3.) == 104.


@pytest.mark.parametrize('nu', [[1.], [1., 1.], [0., 1.], [2., 1.], [1., np.nan]])
def test_bad_frequency_contract_rejected(nu):
    with pytest.raises(ValueError): N.trapezoid_weights(nu)


def test_ion_comparison_has_all_nodes_and_preserves_absolute_values():
    columns = np.array([[14, 1, 0], [14, 1, 1], [14, 2, 0], [16, 1, 0], [16, 2, 0]])
    old = np.array([[1., 2.], [1., 2.], [8., 6.], [2., 3.], [8., 7.]])
    candidate = old.copy(); candidate[0] += 1.; candidate[2] -= 1.
    summary, rows = N.comparison(columns, old, old, candidate, np.array([20., 10.]))
    assert len(summary) == 4 and len(rows) == 8
    si = next(v for v in summary if v['Z'] == 14 and v['charge'] == 1)
    assert si['max_absolute_fraction_difference'] == pytest.approx(.1)
    assert si['fraction_active_shells'] == 2
    assert si['fraction_active_max_absolute_difference'] == pytest.approx(.1)
    assert si['fraction_active_log10_ratio_max_dex'] == pytest.approx(np.log10(1.5))
    assert si['fraction_active_support_mismatch_shells'] == 0
    assert si['acceptance'] == 'DIAGNOSTIC_ONLY_NOT_FULL_V2_ACCEPTANCE'
    r = next(v for v in rows if v['Z'] == 14 and v['charge'] == 1 and v['node'] == 0)
    assert r['CMFGEN70_fraction'] == .2 and r['new_J_response_fraction'] == .3
    assert r['fraction_active'] is True
    assert r['absolute_log10_ratio_dex'] == pytest.approx(np.log10(1.5))
    assert r['radius_cm'] == 20.


def test_ion_comparison_reports_active_support_mismatch_without_flooring():
    columns = np.array([[14, 1, 0], [14, 2, 0]])
    reference = np.array([[0.], [1.]])
    candidate = np.array([[.2], [.8]])
    summary, rows = N.comparison(columns, reference, reference, candidate, np.array([10.]))
    si2 = next(v for v in summary if v['Z'] == 14 and v['charge'] == 1)
    row = next(v for v in rows if v['charge'] == 1)
    assert si2['fraction_active_shells'] == 1
    assert si2['fraction_active_support_mismatch_shells'] == 1
    assert si2['fraction_active_log10_ratio_max_dex'] is None
    assert row['absolute_log10_ratio_dex'] is None
    assert row['fraction_support_mismatch'] is True


def test_followup_memory_inventory_is_separate_from_coefficient_inventory():
    memory = N.memory_inventory(90, 1115020, 24667, 790493, 24542)
    assert 10*1024**3 < memory['known_allocation_and_reserve_bytes'] < 32*1024**3
    assert not memory['whole_process_peak_RSS_verified']


def connect(scenario, monkeypatch):
    args, calls, behavior = scenario
    source = F.open_prepared(None, None)[1]
    source.radius = np.array([3., 2., 1.]); source.frequency = np.arange(1., 9.)
    monkeypatch.setattr(N, 'open_prepared', F.open_prepared)
    return args, behavior


def test_actual_inputfield_accepts_only_last_complete_new_J(scenario, monkeypatch):
    args, behavior = connect(scenario, monkeypatch)
    record = F.run(**args); feedback = args['out']/'feedback.json'
    with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)) as (_, source, field, last):
        assert last['sweep'] == 2 and source.ns == 3
        np.testing.assert_array_equal(field.values, np.full((3, 8), 2.))
    assert field.closed


def test_completed_source_residual_pass_can_feed_response_without_relabelling_output_as_tested_input(scenario,monkeypatch):
    args,behavior=connect(scenario,monkeypatch)
    value=1.+F.SOURCE_FIXED_POINT_RTOL/4
    behavior['outputs']=[1.,value]
    result=F.run(**args);feedback=args['out']/'feedback.json'
    assert result['status']=='FIXED_MATERIAL_SCATTERING_CLOSED'
    assert result['sweeps'][-1]['state']=='FIXED_MATERIAL_SOURCE_RESIDUAL_PASS'
    assert Path(result['accepted_scattering_field']['path']).parent.name=='F-1'
    with N.open_field(args['accepted_input'],args['accepted_input_sha256'],feedback,N.B.sha(feedback)) as (_,_,field,last):
        assert last['sweep']==2 and Path(last['J']['path']).parent.name=='F-2'
        np.testing.assert_array_equal(field.values,np.full((3,8),value))


def test_partial_second_uses_completed_first_not_partial_J(scenario, monkeypatch):
    args, behavior = connect(scenario, monkeypatch); behavior['fail'] = 'partial'
    F.run(**args); feedback = args['out']/'feedback.json'
    with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)) as (_, _, field, last):
        assert last['sweep'] == 1
        np.testing.assert_array_equal(field.values, np.ones((3, 8)))


@pytest.mark.parametrize('fault', ['J', 'moments', 'record', 'material', 'no_complete'])
def test_changed_or_incomplete_field_is_never_used(scenario, monkeypatch, fault):
    args, behavior = connect(scenario, monkeypatch)
    F.run(**args); feedback = args['out']/'feedback.json'; data = json.loads(feedback.read_text())
    if fault in ('J', 'moments'):
        path = Path(data['last_complete'][fault]['path']); path.write_bytes(b'changed')
    elif fault == 'record':
        (args['out']/'F-2/resident-field.json').write_text('{}')
    elif fault == 'material': data['source_identity'] = dict(material='old70')
    else: data['last_complete'] = None
    feedback.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)):
            pass
