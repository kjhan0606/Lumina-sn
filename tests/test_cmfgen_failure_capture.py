"""승인된 실패 포획의 실행 계약·종료 분류 시험. 새 물리 계산은 하지 않는다."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cmfgen_mom_capture import capture_target, diagnostic_module
import run_cmfgen_mom_capture as R
from run_cmfgen_lambda_probe import sha


def args():
    return SimpleNamespace(capture_build=Path('/unused/build'),capture_build_sha256='a'*64,
        capture_failure_inputs=Path('/unused/input'),capture_failure_inputs_sha256='b'*64,
        memory_gib=128,seconds=13680,max_iterations=16)


ENV=dict(OMP_NUM_THREADS='16',SLURM_CPUS_PER_TASK='16',SLURM_MEM_PER_NODE='131072',SLURM_JOB_ID='1')


def test_capture_resource_profile_does_not_change_ordinary_continuation():
    assert R.failure_profile(SimpleNamespace(),{}) is False
    assert R.failure_profile(args(),ENV) is True


@pytest.mark.parametrize('field,value', [('seconds',42600),('seconds',13680.),
    ('memory_gib',32),('max_iterations',4),('capture_build_sha256',None)])
def test_capture_wrong_time_memory_controls_or_partial_args_rejected(field,value):
    a=args();setattr(a,field,value)
    with pytest.raises(ValueError):R.failure_profile(a,ENV)


@pytest.mark.parametrize('field,value', [('OMP_NUM_THREADS','8'),('SLURM_CPUS_PER_TASK','8'),
    ('SLURM_MEM_PER_NODE','32768'),('SLURM_JOB_ID','')])
def test_capture_requires_actual_approved_slurm_allocation(field,value):
    with pytest.raises(ValueError):R.failure_profile(args(),dict(ENV,**{field:value}))


def build_fixture(tmp_path):
    parent=tmp_path/'parent';build=tmp_path/'capture';parent.mkdir();build.mkdir()
    (parent/'parent-build-identity.json').write_text(json.dumps(dict(
        reference_build_sha256='c'*64,reference_executable_sha256='d'*64)))
    source=tmp_path/'source.o';source.write_bytes(b'sealed object')
    executable=build/'cmfgen_capture.exe';executable.write_bytes(b'executable')
    inventory=build/'state_inventory.json';inventory.write_text('{}')
    data=dict(parent=str(parent),parent_manifest_sha256=sha(parent/'parent-build-identity.json'),
        capture_target=R.FAILURE_TARGET.copy(),first_repair=False,rebuilt_unmodified_sha256='d'*64,
        input_sha256={str(source):sha(source)},output_sha256={str(p):sha(p) for p in (executable,inventory)})
    report=build/'capture-build.json';report.write_text(json.dumps(data))
    return build,source,report,data


def test_current_parent_capture_build_is_hash_checked_and_read_only(tmp_path):
    build,source,report,_=build_fixture(tmp_path);before={p:sha(p) for p in (source,report)}
    result=R.failure_build(build,sha(report),'d'*64,'c'*64)
    assert result['executable']==str(build/'cmfgen_capture.exe')
    assert all(sha(p)==h for p,h in before.items())
    source.write_bytes(b'changed object')
    with pytest.raises(ValueError,match='입력/출력'):R.failure_build(build,sha(report),'d'*64,'c'*64)


def test_blue_entry_build_requires_its_own_exact_frequency_and_mode(tmp_path):
    build,_,report,data=build_fixture(tmp_path)
    data.update(capture_target=R.BLUE_ENTRY_TARGET.copy(),entry_only=True)
    report.write_text(json.dumps(data))
    result=R.failure_build(build,sha(report),'d'*64,'c'*64)
    assert result['entry_only'] is True and result['target']==R.BLUE_ENTRY_TARGET
    data['capture_target']=R.FAILURE_TARGET.copy();report.write_text(json.dumps(data))
    with pytest.raises(ValueError):R.failure_build(build,sha(report),'d'*64,'c'*64)


def test_custom_entry_target_is_paired_and_exactly_bound_to_build(tmp_path):
    target=R.capture_target_override(9.39898883,78)
    assert target==dict(frequency_native=9.39898883,outer_iteration=78,
                        frequency_absolute_tolerance=5.e-9)
    for frequency,outer in ((9.39898883,None),(None,78),(float('nan'),78),(0.,78),(9.3,0)):
        with pytest.raises(ValueError):R.capture_target_override(frequency,outer)
    build,_,report,data=build_fixture(tmp_path)
    data.update(capture_target=target,entry_only=True)
    report.write_text(json.dumps(data))
    result=R.failure_build(build,sha(report),'d'*64,'c'*64,expected_target=target)
    assert result['target']==target
    with pytest.raises(ValueError):
        R.failure_build(build,sha(report),'d'*64,'c'*64,
                        expected_target=R.capture_target_override(9.39898884,78))


def test_custom_entry_target_binds_frequency_and_outer_in_generated_selector():
    target=capture_target(target_frequency=9.39898883,target_outer=78)
    source=diagnostic_module(target_frequency=target['frequency_native'],
                             target_outer=target['outer_iteration'])
    exact=format(target['frequency_native'],'.17e')
    assert f"(LUMINA_OUTER==78 .AND. ABS(FREQ-{exact}_LDP)<5.0E-9_LDP)" in source
    assert "IF(MODE=='CAPTURE' .AND. LUMINA_OUTER>=78)THEN" in source


@pytest.mark.parametrize('bad',['hash','parent','target','inventory'])
def test_capture_build_wrong_provenance_or_scope_rejected(tmp_path,bad):
    build,source,report,data=build_fixture(tmp_path)
    parent='e'*64 if bad=='parent' else 'd'*64
    if bad=='target':data['capture_target']=dict(R.FAILURE_TARGET,outer_iteration=79)
    if bad=='inventory':data['output_sha256'].pop(str(build/'state_inventory.json'))
    report.write_text(json.dumps(data))
    expected='0'*64 if bad=='hash' else sha(report)
    with pytest.raises(ValueError):R.failure_build(build,expected,parent,'c'*64)


def test_original_failed_inputs_and_seed_identity_are_pinned(tmp_path):
    seed=tmp_path/'seed';seed.mkdir();(seed/'result.json').write_text('parent')
    ion=tmp_path/'ions.json';ion.write_text('reference')
    a=SimpleNamespace(seed_run=seed,ion_reference=ion,build_report_sha256='a'*64)
    identity=dict(record=70)
    data=dict(first_iteration=74,maximum_additional_iterations=16,source_70_identity=identity,
        executable_sha256='b'*64,build_report_sha256='a'*64,
        parent_cmfg_result_sha256=sha(seed/'result.json'),source_iteration_reference_sha256=sha(ion))
    path=tmp_path/'failed-inputs.json';path.write_text(json.dumps(data))
    assert R.failure_inputs(path,sha(path),a,identity,'b'*64)==data
    with pytest.raises(ValueError):R.failure_inputs(path,'0'*64,a,identity,'b'*64)
    with pytest.raises(ValueError):R.failure_inputs(path,sha(path),a,dict(record=71),'b'*64)


def evidence_fixture(tmp_path,monkeypatch,outer=78,frequency=6.10732968):
    import analyze_cmfgen_mom_capture as A
    p=tmp_path/'MOM_CAPTURE.bin'
    p.write_bytes(b'LUMINAMOMCAPV1  '+struct.pack('=4i6i',8,64,32,32,outer,123,2,0,1,90))
    (tmp_path/'MOM_TRACE.bin').write_bytes(b'fixture')
    inventory=tmp_path/'inventory.json';inventory.write_text('{}')
    saved=dict(phase=(outer,123,2,0,1),args=dict(FREQ=frequency,INNER_BND_METH='DIFFUSION'))
    calls=[dict(phase=saved['phase'],frequency=frequency)]
    rows=[dict(iteration=i,MAX_ER=.589) for i in range(1,101)]
    # 완전한 Fortran 상태 I/O와 trace 산술은 기존 저장·재생 시험에서 검증한다.
    monkeypatch.setattr(A,'capture',lambda *a:saved)
    monkeypatch.setattr(A,'trace',lambda *a:calls)
    monkeypatch.setattr(A,'summary',lambda *a:rows)
    return dict(inventory=str(inventory)),saved,calls,rows


def test_expected_stop93_is_capture_not_physical_convergence(tmp_path,monkeypatch):
    context,_,_,_=evidence_fixture(tmp_path,monkeypatch)
    result=R.failure_evidence(tmp_path,context,93)
    assert result['status']=='CAPTURED_NOT_PHYSICAL_CONVERGENCE'
    assert result['last_iteration']['iteration']==100
    assert result['phase']['outer_iteration']==78


def test_blue_entry_requires_stop96_and_exact_previous_frequency(tmp_path,monkeypatch):
    context,saved,_,_=evidence_fixture(tmp_path,monkeypatch)
    context.update(entry_only=True,target=R.BLUE_ENTRY_TARGET.copy())
    saved['args']['FREQ']=R.BLUE_ENTRY_TARGET['frequency_native']
    result=R.failure_evidence(tmp_path,context,96)
    assert result['status']=='BLUE_ENTRY_NOT_PHYSICAL_CONVERGENCE'
    with pytest.raises(ValueError):R.failure_evidence(tmp_path,context,93)
    saved['args']['FREQ']=R.FAILURE_TARGET['frequency_native']
    with pytest.raises(ValueError):R.failure_evidence(tmp_path,context,96)


def test_custom_entry_capture_uses_its_requested_outer_and_frequency(tmp_path,monkeypatch):
    target=R.capture_target_override(9.39898883,81)
    context,_,_,_=evidence_fixture(tmp_path,monkeypatch,outer=81,frequency=9.398988832)
    context.update(entry_only=True,target=target)
    result=R.failure_evidence(tmp_path,context,96)
    assert result['phase']['outer_iteration']==81
    assert result['frequency_native']==9.398988832
    context['target']=R.capture_target_override(9.39898884,81)
    with pytest.raises(ValueError):R.failure_evidence(tmp_path,context,96)


@pytest.mark.parametrize('bad',['returncode','phase','frequency','incomplete','converged'])
def test_capture_cannot_be_claimed_without_complete_matching_failed_call(tmp_path,monkeypatch,bad):
    context,saved,calls,rows=evidence_fixture(tmp_path,monkeypatch)
    code=0 if bad=='returncode' else 93
    if bad=='phase':calls[0]['phase']=(77,123,2,0,1)
    if bad=='frequency':saved['args']['FREQ']=6.4
    if bad=='incomplete':rows.pop()
    if bad=='converged':rows[-1]['MAX_ER']=1.e-10
    with pytest.raises(ValueError):R.failure_evidence(tmp_path,context,code)


def test_capture_submission_has_one_bounded_cpu_job_and_no_requeue():
    script=Path(__file__).resolve().parents[1]/'scripts/submit_cmfgen_moment_failure.slurm'
    text=script.read_text()
    for directive in ('--cpus-per-task=16','--mem=128G','--time=04:00:00','--gres=gpu:0',
                      '--no-requeue','--exclude=grammar072,grammar078,grammar080'):
        assert directive in text
    assert '--seconds 13680 --max-iterations 16' in text
    assert 'timeout 14000s' in text
    assert '--target-frequency "$target_frequency" --target-outer "$target_outer"' in text
    assert '--capture-target-frequency "$target_frequency" --capture-target-outer "$target_outer"' in text
    assert 'python3 -m pytest tests/test_cmfgen_failure_capture.py -q' in text
