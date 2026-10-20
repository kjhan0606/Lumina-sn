"""한 주파수 공동 반응의 실제 blue 고정과 가짜 물리 합격 방지를 검증한다."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_mom_capture import first_event
from run_cmfgen_coupled_packet import matrix_packet,run
from analyze_cmfgen_coupled_terms import analyze as analyze_terms

BASE=Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST')
BUILD=BASE/'coupled_v3'
ORIGINAL=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428716')
REPLAY=Path('/gpfs/kjhan/cmfgen_runs/lumina_affine_66069_replay_20261019-Pyi0PH')


def test_first_packet_preserves_actual_formal_rays_and_native_matrix(tmp_path):
    process=subprocess.run([str(BUILD/'coupled_packet.exe'),str(ORIGINAL)],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='CAPTURE_FIRST',LUMINA_WHOLE_MODE='',LUMINA_FORMAL_CAPTURE_MODE=''))
    assert process.returncode==96,process.stderr
    for current,old in (('TRIAL_GEOMETRY.bin','FORMAL_EVENT_geometry.bin'),('TRIAL_RAYS_OUT.bin','FORMAL_EVENT_rays_out.bin'),('TRIAL_BLUE_AFTER.bin','FORMAL_EVENT_rays_in.bin')):
        assert (tmp_path/current).read_bytes()==(ORIGINAL/old).read_bytes()
    packet=matrix_packet(tmp_path/'TRIAL_MATRIX.bin');event=first_event(ORIGINAL/'MOM_FAULT.bin')
    den=np.maximum(abs(packet['matrix']),abs(event['matrix']))
    assert np.max(np.divide(abs(packet['matrix']-event['matrix']),den,out=np.zeros_like(den),where=den>0))<1e-10
    assert not (tmp_path/'MOM_FAULT.bin').exists()


@pytest.mark.parametrize('change',['short','extra','tag'])
def test_trial_matrix_has_strict_record_length(tmp_path,change):
    path=REPLAY/'coupled_actual_blue_v2/trial_00/TRIAL_MATRIX.bin'
    raw=path.read_bytes();bad=tmp_path/'matrix.bin'
    bad.write_bytes(raw[:-1] if change=='short' else raw+b'X' if change=='extra' else b'X'+raw[1:])
    with pytest.raises(ValueError):matrix_packet(bad)


def test_joint_fixed_point_is_not_accepted_as_physical_solution(tmp_path):
    out=tmp_path/'coupled'
    run(ORIGINAL,BUILD,out,max_seconds=120,max_trials=20)
    report=json.loads((out/'coupled-response.json').read_text())
    assert report['joint_fixed_point'] and len(report['trials'])==3
    last=report['trials'][-1]
    assert last['source_eta_lag']<1e-8 and last['J_change']<1e-8 and last['Q_change']<1e-8
    assert last['formal_MOM_J_difference']>.49
    assert last['H_MAX_VAL_edges_1based']==[36] and last['angular_support']['minimum']<-1
    assert all(r['blue_rays_equal'] and r['moment_blue_equal'] and r['moment_linear_solves']<=100 for r in report['trials'])
    assert report['production_physics']=='NOT_ACCEPTED' and report['final_goal']=='NOT_JUDGED'
    for i in (1,2):
        actual=np.fromfile(out/('trial_%02d'%i)/'SOURCE_J.bin',dtype='<f8')
        previous=json.loads((out/('trial_%02d'%(i-1))/'trial-result.json').read_text())['nonaccepted_J']
        np.testing.assert_array_equal(actual,previous)


def test_primitive_equations_are_checked_before_interpreting_angular_sources(tmp_path):
    output=tmp_path/'terms.json'
    analyze_terms(ORIGINAL,REPLAY/'coupled_actual_blue_v2/trial_02',BASE/'whole_accel_v1',output)
    report=json.loads(output.read_text())
    assert report['summary']['primitive_matrix_identity']<1e-12
    assert report['summary']['zeroth_frequency_current_ray_vs_moment_max']<.001
    depth36=next(r for r in report['samples'] if r['depth_1based']==36)
    assert depth36['zeroth_blue_ray']>2*depth36['zeroth_blue_moment']
    assert depth36['first_blue_moment']>2*depth36['first_blue_ray']
