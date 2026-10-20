"""실제 반복 실패와 실제 H 실패를 같은 원식에서 구분한다."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_affine_native_event import analyze

BUILD=Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST')


def test_actual_H_limit_is_not_repaired_by_a_more_accurate_same_equation_root(tmp_path):
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428716')
    replay=Path('/gpfs/kjhan/cmfgen_runs/lumina_affine_66069_replay_20261019-Pyi0PH')
    out=tmp_path/'h_response.json';analyze(run,replay,BUILD,out,blue_response=True)
    report=json.loads(out.read_text())
    assert all(report['replay_byte_equal'].values())
    assert report['event_kind']==2 and report['event_iteration']==1
    assert report['native_linear_mp80_relative']<1e-12
    for root in report['root_checks']:
        assert root['positive'] and root['residual']<1e-10
        assert root['same_mp80_branches'] and root['mp80_relative']<1e-10
        assert root['H_MAX_VAL_edges_1based']==[36]
        assert root['angular_support']['minimum']<-1
    cases={r['case']:r for r in report['blue_history_response']['root_checks']}
    for case in ('blue_J_only','blue_H_only'):
        assert not cases[case]['H_MAX_VAL_edges_1based']
        assert cases[case]['angular_support']['minimum']<0
    assert cases['blue_J_H']['angular_support']['minimum']>0
    assert cases['blue_J_H']['positive'] and cases['blue_J_H']['residual']<1e-10
    assert cases['blue_J_H']['same_mp80_branches']


def test_original_100_iteration_event_still_has_a_positive_admissible_root(tmp_path):
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428713')
    replay=Path('/gpfs/kjhan/cmfgen_runs/lumina_affine_5691_replay_20261019-lSVfbZ')
    out=tmp_path/'cap.json';analyze(run,replay,BUILD,out)
    report=json.loads(out.read_text())
    assert all(report['replay_byte_equal'].values())
    assert report['event_kind']==3 and report['event_iteration']==100
    assert report['native_change_last10'][-1]['max_change']>1e-8
    for root in report['root_checks']:
        assert root['positive'] and root['residual']<1e-10
        assert root['same_mp80_branches'] and root['mp80_relative']<1e-10
        assert not root['H_MAX_VAL_edges_1based']
