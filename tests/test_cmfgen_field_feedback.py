"""새 세대 교대 한 번의 물리 제어·공유 시간한도·원본 보존을 확인한다."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_field_feedback as feedback
from run_cmfgen_lambda_probe import control


def fixture():
    values = dict(FIX_T='T', FIX_T_AUTO='F', FIX_NE='F', DO_DDT='F', DO_NG='F',
        TRT_NON_TE='F', USE_FIXED_J='F', INC_GRID='F', WRITE_RATES='T', WRITE_JH='T',
        SCALE_OPT='MAJOR', MAX_LAM='10.0D0')
    v = ''.join(f'{value} [{key}]\n' for key, value in values.items())
    i = '19 [NUM_ITS]\nF [DO_LAM_IT]\nF [DO_LAM_AUTO]\nF [DO_T_AUTO]\n'
    return values, v, i


@pytest.mark.parametrize('fixed', [False, True])
def test_only_registered_controls_change_and_originals_are_not_modified(fixed):
    original, vadat, in_its = fixture()
    v, i = feedback.controls(vadat, in_its, fixed)
    changes = dict(USE_FIXED_J='T' if fixed else 'F', SCALE_OPT='LOCAL', MAX_LAM='2.0D0', WRITE_RATES='F')
    for key, old in original.items():
        assert control(v, key) == changes.get(key, old)
    assert control(v, 'LAM_SCALE_OPT') == 'NONE'
    assert control(i, 'NUM_ITS') == '1'
    assert control(i, 'DO_LAM_IT') == 'T'
    assert control(i, 'DO_LAM_AUTO') == control(i, 'DO_T_AUTO') == 'F'
    assert fixture()[1:] == (vadat, in_its)


@pytest.mark.parametrize('key,old,new', [('FIX_T','T','F'), ('FIX_NE','F','T'),
    ('DO_DDT','F','T'), ('TRT_NON_TE','F','T'), ('INC_GRID','F','T')])
def test_physical_contract_changes_are_refused(key, old, new):
    _, v, i = fixture()
    with pytest.raises(ValueError):
        feedback.controls(v.replace(f'{old} [{key}]', f'{new} [{key}]'), i, False)


def test_expired_shared_deadline_never_starts_a_new_process(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback.subprocess, 'Popen', lambda *a, **k: pytest.fail('기한 밖 자식 시작'))
    result = feedback.child(tmp_path, tmp_path, Path('not-called'), {}, 0)
    assert result['returncode'] is None
    assert result['stop_reason'] == 'shared_child_budget_2400s'


def test_completed_model_state_is_copied_with_completed_populations():
    assert 'CUR_MODEL_DATA' in feedback.RESTART_OVERLAY
    assert 'SCRTEMP' in feedback.RESTART_OVERLAY


def test_selected_full_step_cap_does_not_change_physical_controls():
    _, vadat, in_its = fixture()
    v, i = feedback.controls(vadat, in_its, True, '1.0D6')
    old_v, old_i = feedback.controls(vadat, in_its, True)
    assert feedback.set_control(v, 'MAX_LAM', '2.0D0') == old_v
    assert i == old_i
    with pytest.raises(ValueError, match='상한'):
        feedback.controls(vadat, in_its, True, '1.0D9')


def test_clone_preserves_selected_parent_record_not_hardcoded63(tmp_path, monkeypatch):
    original, source, target, native = (tmp_path/n for n in ('original','source','clone','native'))
    original.mkdir();source.mkdir()
    _, v, i = fixture()
    (original/'VADAT').write_text(v);(original/'IN_ITS').write_text(i)
    for name in feedback.RESTART_OVERLAY:
        content = 'point\n64 64 1 -1000 F\n' if name in ('POINT1','POINT2') else 'state64B\n'
        (source/name).write_text(content)
    def fake_stage(*args, **kwargs):
        assert kwargs == dict(write_record=False)
        target.mkdir();(target/'seq_logs').mkdir()
        return dict(binary=dict(path='native',sha256='hash'),linked_inputs={})
    monkeypatch.setattr(feedback, 'stage', fake_stage)
    report = feedback.clone_material(original, source, target, native, True, 64, '1.0D6')
    assert report['input_state_record'] == 64 and report['next_native_iteration'] == 65
    assert report['numerical_controls']['MAX_LAM'] == 1e6
    assert feedback.point_records(target) == dict(POINT1=64,POINT2=64)
    assert (target/'CUR_MODEL_DATA').read_bytes() == (source/'CUR_MODEL_DATA').read_bytes()


def test_failed_feedback_material_is_not_used_as_next_field_input(tmp_path):
    source = tmp_path/'material';(source/'seq_logs').mkdir(parents=True)
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'seq_logs/feedback-inputs.json').write_text(json.dumps(dict(original='/original')))
    (tmp_path/'seq_logs/feedback-runtime.json').write_text(json.dumps(dict(material_complete=False)))
    (source/'seq_logs/feedback-stage.json').write_text('{}')
    with pytest.raises(ValueError, match='정상 완료'):
        feedback.verify_material(source, Path('/original'), 65)


def test_full_step_evidence_preserves_existing_AB_precision_contract():
    import numpy as np
    from cmfgen_step_precision import applied_full_step
    from run_cmfgen_trace_cap_ab import full_step_check
    pre=np.array([[1e-46,1.],[1e6,2e6]])
    sol=np.array([[-22652.,.5],[-.1,0.]])
    for factor in (1.,.001):
        after=pre*(1-factor*sol)
        assert applied_full_step(pre,after,sol)==full_step_check(pre,after,sol,sol)['actual_full_step_within_print_precision']
    assert applied_full_step(pre,pre*(1-sol),sol)
    assert not applied_full_step(pre,pre*(1-.001*sol),sol)
    with pytest.raises(ValueError):applied_full_step(pre,pre,sol[:1])
    assert not applied_full_step(pre,pre,np.full_like(sol,np.nan))


def test_completed66_with_changed_full_step_premise_cannot_start67(tmp_path):
    source=tmp_path/'material';(source/'seq_logs').mkdir(parents=True)
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'seq_logs/feedback-inputs.json').write_text(json.dumps(dict(original='/original')))
    (tmp_path/'seq_logs/feedback-runtime.json').write_text(json.dumps(dict(material_complete=True,
        field_complete=True,source_unchanged=True,material=dict(returncode=0,stop_reason=None),
        material_record=66,radiation_produced_from_record=65,next_outer_eligible=False)))
    (source/'seq_logs/feedback-stage.json').write_text(json.dumps(dict(input_state_record=65,
        next_native_iteration=66,USE_FIXED_J=True)))
    for name in ('POINT1','POINT2'):(source/name).write_text('point\n66 66 1 -1000 F\n')
    with pytest.raises(ValueError,match='전체 갱신 적용 전제'):
        feedback.verify_material(source,Path('/original'),66)


def test_existing_run_is_refused_before_any_source_reads(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    with pytest.raises(ValueError, match='기존'):
        feedback.run(Path('not-read'), Path('not-read'), tmp_path, Path('not-read'), Path('not-read'))


@pytest.mark.parametrize('returncode,complete', [(1, True), (0, False)])
def test_failed_or_partial_material63_is_not_a_restart(tmp_path, returncode, complete):
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'seq_logs/fixed-field-inputs.json').write_text(json.dumps(dict(seed='/original')))
    report = dict(returncode=returncode, four_native_updates_complete=complete,
        fixed_J_unchanged=True, seed_unchanged=True, T_unchanged=True, last_completed_record=63)
    (tmp_path/'seq_logs/fixed-field-runtime.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='정상 완료'):
        feedback.verify_material(tmp_path, Path('/original'))
