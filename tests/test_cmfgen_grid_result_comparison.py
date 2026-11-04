"""완료된 같은-state 포획만 후처리에 연결한다. 실제 solver 실행 없음."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_cmfgen_ir_grid_impact as C
import run_cmfgen_fixedj_grid_bundle as B


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path):
    base=tmp_path/'baseline/native';base.mkdir(parents=True)
    (base/'NETRATE').write_text('baseline netrate')
    (base/'NATIVE_HEAT_ROW.bin').write_text('baseline heat')
    reference=tmp_path/'comparison.json'
    write_json(reference,dict(fixed_J_source_sha256='same J',input_sha256={
        'accepted_Lumina_response':dict(sha256='same response'),
        'native_executable':dict(sha256='same binary'),
        'native_heat_row':dict(sha256=C.sha(base/'NATIVE_HEAT_ROW.bin'))}))
    root=tmp_path/'bundle';attempts=[]
    for name,h in [('h05',.5),('h025',.25)]:
        native=root/name/'native';native.mkdir(parents=True)
        inputs={}
        for filename in ['VADAT','MODEL_SPEC','NATIVE_HEAT_INPUT.bin','EDDFACTOR','EDDFACTOR_INFO']:
            (native/filename).write_text(filename+str(h));inputs[filename]=C.sha(native/filename)
        outputs={}
        for filename in ['NETRATE','NATIVE_HEAT_ROW.bin']:
            (native/filename).write_text(filename+str(h));outputs[filename]=dict(sha256=C.sha(native/filename))
        write_json(native/'seq_logs/fixedj-heat-runtime.json',dict(status='CAPTURED',
            srun_exit_code=96,captured_state_exact=True,fixed_j_sha256='same J',
            accepted_response_sha256='same response',build_sha256='same binary',outputs=outputs))
        attempts.append(dict(root=name,frac_dop=h,input_sha256=inputs))
    write_json(root/'bundle-inputs.json',dict(attempts=attempts,baseline=str(base.parent)))
    return root,reference,C.sha(base/'NETRATE')


def call(bundle,analyzer):
    root,reference,digest=bundle
    return C.compare_grid_bundle(Path('sources'),Path('response'),root,reference,
                                 baseline_netrate_sha256=digest,analyzer=analyzer)


def test_complete_pair_uses_sealed_hashes_and_retains_unmet_goal(bundle):
    calls=[]
    def analyzer(source,response,**kwargs):
        calls.append(kwargs)
        assert C.sha(kwargs['netrate'])==kwargs['expected_netrate_sha256']
        assert C.sha(kwargs['native_heat'])==kwargs['expected_native_heat_sha256']
        for z,stage in [(14,2),(14,3),(16,2),(16,3)]:
            kwargs['native_vector_sink'][z,stage]=dict(columns=np.array([[z,stage-1,0],[z,stage-1,1]]),
                native=np.full((2,90),float(len(calls))),gross=np.ones(90))
        return dict(final_goal='NOT_MET',native_heat_comparison=dict(stored_policy_candidate=dict(
            points=[dict(native_steq_t_cgs=float(len(calls))) for _ in range(90)])))
    report=call(bundle,analyzer)
    assert len(calls)==3
    assert [c['frac_dop'] for c in report['cases']]==[1.,.5,.25]
    assert report['final_goal']=='NOT_MET'
    assert len(report['successive_grid_differences'])==2
    for item in report['successive_grid_differences']:
        assert item['native_heat_max_abs_change']==pytest.approx(1.)
        assert item['native_BB_blocks']['Z14_stage2']['max_L1']==pytest.approx(2.)


@pytest.mark.parametrize('fault',['incomplete','failed','different_J','changed_output'])
def test_partial_or_different_capture_never_reaches_analyzer(bundle,fault):
    root,_,_=bundle;path=root/'h025/native/seq_logs/fixedj-heat-runtime.json'
    record=json.loads(path.read_text())
    if fault=='incomplete':
        record['status']='RUNNING'
    elif fault=='failed':record['status']='FAILED'
    elif fault=='different_J':record['fixed_j_sha256']='another J'
    elif fault=='changed_output':(root/'h025/native/NETRATE').write_text('changed')
    write_json(path,record)
    called=[]
    if fault in ('different_J','changed_output'):
        with pytest.raises(ValueError):call(bundle,lambda *a,**k:called.append(1))
    else:
        report=call(bundle,lambda *a,**k:called.append(1))
        assert report['status']=='CAPTURE_NOT_COMPLETE'
    assert called==[]


@pytest.mark.parametrize('state',['missing','publishing'])
def test_absent_or_partially_written_runtime_is_not_solver_failure(bundle,state):
    root,_,_=bundle;path=root/'h025/native/seq_logs/fixedj-heat-runtime.json'
    if state=='missing':path.unlink()
    else:path.write_text('{"status":')
    called=[]
    report=call(bundle,lambda *a,**k:called.append(1))
    assert report['status']==('WAITING_FOR_BOTH_CAPTURES' if state=='missing' else 'WAITING_FOR_RUNTIME_PUBLICATION')
    assert report['final_goal']=='NOT_MET'
    assert not called


@pytest.fixture
def bridged_bundle(bundle,monkeypatch):
    root,_,_=bundle;ref=root.parent/'old_h05';native=ref/'native';native.mkdir(parents=True)
    (native/'VADAT').write_text('0.5D0 [FRAC_DOP]\n')
    outputs={}
    for name in B.capture.OUTPUT_NAMES:
        current=root/'h05/native'/name
        if not current.exists():current.write_text('same physical output '+name)
        (native/name).write_bytes(current.read_bytes());outputs[name]=dict(sha256=C.sha(current))
    write_json(native/'seq_logs/fixedj-heat-runtime.json',dict(status='CAPTURED',
        srun_exit_code=96,captured_state_exact=True,build_sha256='same binary',
        fixed_j_sha256='same J',accepted_response_sha256='same response',outputs=outputs))
    plan_path=root/'bundle-inputs.json';plan=json.loads(plan_path.read_text())
    reference=B.completed_bridge_reference(ref);reference['replacement_build']=str(root.parent/'replacement')
    plan.update(capture_build_sha256='observer-only binary',observer_build_bridge=reference)
    write_json(plan_path,plan)
    for name in ('h05','h025'):
        path=root/name/'native/seq_logs/fixedj-heat-runtime.json';r=json.loads(path.read_text())
        r['build_sha256']='observer-only binary'
        if name=='h05':r['outputs']=outputs
        write_json(path,r)
    monkeypatch.setattr(B,'observer_only_build',lambda *args:dict(output_sha256={
        'cmfgen_fixedj_heat.exe':'observer-only binary'}))
    return bundle


def test_changed_binary_without_bridge_never_reaches_analyzer(bundle):
    root,_,_=bundle;p=root/'bundle-inputs.json';plan=json.loads(p.read_text())
    plan['capture_build_sha256']='unproven changed binary';write_json(p,plan)
    called=[]
    with pytest.raises(ValueError):call(bundle,lambda *a,**k:called.append(1))
    assert not called


def test_observer_only_binary_with_exact_bridge_is_compared_not_accepted_as_final(bridged_bundle):
    calls=[]
    def analyzer(*args,**kwargs):
        calls.append(1)
        for z,stage in [(14,2),(14,3),(16,2),(16,3)]:
            kwargs['native_vector_sink'][z,stage]=dict(columns=np.array([[z,stage-1,0]]),
                native=np.zeros((1,90)),gross=np.ones(90))
        return dict(native_heat_comparison=dict(stored_policy_candidate=dict(
            points=[dict(native_steq_t_cgs=0.) for _ in range(90)])))
    r=call(bridged_bundle,analyzer)
    assert len(calls)==3
    assert r['observer_build_bridge']['status']=='EXACT_PHYSICAL_OUTPUT_REPRODUCTION'
    assert r['final_goal']=='NOT_MET'


@pytest.mark.parametrize('fault',['reference_output','new_h05_output','second_binary','different_proof'])
def test_invalid_observer_bridge_never_reaches_analyzer(bridged_bundle,monkeypatch,fault):
    root,_,_=bridged_bundle;called=[]
    if fault=='reference_output':(root.parent/'old_h05/native/NETRATE').write_text('changed')
    if fault=='new_h05_output':
        p=root/'h05/native/TOTRATE';p.write_text('changed')
        manifest=root/'h05/native/seq_logs/fixedj-heat-runtime.json'
        r=json.loads(manifest.read_text());r['outputs']['TOTRATE']['sha256']=C.sha(p);write_json(manifest,r)
    if fault=='second_binary':
        p=root/'h025/native/seq_logs/fixedj-heat-runtime.json';r=json.loads(p.read_text())
        r['build_sha256']='different binary';write_json(p,r)
    if fault=='different_proof':
        monkeypatch.setattr(B,'observer_only_build',lambda *args:dict(output_sha256={
            'cmfgen_fixedj_heat.exe':'not the capture binary'}))
    with pytest.raises(ValueError):call(bridged_bundle,lambda *a,**k:called.append(1))
    assert not called
