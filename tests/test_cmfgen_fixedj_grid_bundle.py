"""승인된 두 포획의 순서·실패 중단·중복 실행 방지. 실제 CMFGEN 실행 없음."""
import json
from pathlib import Path
import sys
import struct
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_fixedj_grid_bundle as B
from prepare_cmfgen_fixedj_heat import GUARD_POINT,FIXED_SKIP,FULL_POINT,FINISH_POINT,OBSERVER_CALL,OBSERVER_OUTPUT


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    for k,v in [('SLURM_CPUS_PER_TASK','16'),('SLURM_MEM_PER_NODE','32768'),('SLURM_JOB_ID','test')]:
        monkeypatch.setenv(k,v)
    attempts=[]
    for name,h in [('h05',.5),('h025',.25)]:
        native=tmp_path/name/'native';native.mkdir(parents=True)
        (native/'VADAT').write_text('synthetic')
        attempts.append(dict(root=name,frac_dop=h,input_sha256={'VADAT':B.capture.sha(native/'VADAT')}))
    record=dict(schema='lumina-fixedj-grid-bundle-v1',resources=dict(cpus=16,memory_MiB=32768,
        wall_seconds=21600,captures=2,retries=0,gpus=0),attempts=attempts,source_sha256={})
    (tmp_path/'bundle-inputs.json').write_text(json.dumps(record))
    return tmp_path


def test_two_captures_are_sequential_and_not_a_final_physics_pass(bundle):
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0)==0
    assert called==['h05','h025']
    result=json.loads((bundle/'bundle-runtime.json').read_text())
    assert result['final_goal']=='NOT_MET'
    assert result['status']=='CAPTURED_BOTH_NOT_PHYSICAL_EQUIVALENCE'
    with pytest.raises(FileExistsError):B.run_bundle(bundle,lambda p:0)


def test_first_failure_never_starts_second(bundle):
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 1)==1
    assert called==['h05']


def test_changed_second_input_is_not_consumed(bundle):
    called=[]
    (bundle/'h025/native/VADAT').write_text('changed')
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0)==1
    assert called==['h05']


def test_insufficient_remaining_wall_time_preserves_first_only(bundle):
    times=iter([0.,17000.,17001.])
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0,clock=lambda:next(times))==1
    assert called==['h05']


@pytest.fixture
def replacement(tmp_path):
    old=tmp_path/'baseline/build';new=tmp_path/'replacement'
    old.mkdir(parents=True);new.mkdir()
    caller=tmp_path/'upstream/cmfgen_sub.f';caller.parent.mkdir()
    caller.write_text(GUARD_POINT+FIXED_SKIP+FULL_POINT+OBSERVER_CALL+OBSERVER_OUTPUT+FINISH_POINT)
    for root,skip in [(old,False),(new,True)]:
        (root/'cmfgen_sub.f').write_text(B.patched(caller.read_text(),skip_observer_spectrum=skip))
        (root/'cmfgen_fixedj_heat.exe').write_text('synthetic executable '+str(skip))
        manifest=dict(schema='lumina-native-fixedj-integral-re-build-v1',
            observer_spectrum='NOT_EVALUATED_NO_OBSFLUX' if skip else 'UNCHANGED_NATIVE_PATH',
            decay_implementation='UNCHANGED',input_sha256={str(caller):B.capture.sha(caller),
                'libsubs.a':'same-library','prepare_cmfgen_fixedj_heat.py':'new' if skip else 'old'},
            output_sha256={name:B.capture.sha(root/name) for name in
                           ('cmfgen_sub.f','cmfgen_fixedj_heat.exe')})
        (root/'fixedj-heat-build.json').write_text(json.dumps(manifest))
    return old,new


def test_observer_only_build_accepts_driver_change_not_new_physics(replacement):
    old,new=replacement
    report=B.observer_only_build(old,new)
    assert report['observer_spectrum']=='NOT_EVALUATED_NO_OBSFLUX'


@pytest.mark.parametrize('change',['physics','caller','source','executable','decay','flag'])
def test_observer_replacement_rejects_other_changes(replacement,change):
    old,new=replacement;p=new/'fixedj-heat-build.json';m=json.loads(p.read_text())
    if change=='physics':m['input_sha256']['libsubs.a']='different'
    if change=='caller':
        caller=next(Path(p) for p in m['input_sha256'] if Path(p).name=='cmfgen_sub.f')
        caller.write_text(caller.read_text()+'! edited\n')
    if change=='source':
        (new/'cmfgen_sub.f').write_text((new/'cmfgen_sub.f').read_text()+'! extra change\n')
        m['output_sha256']['cmfgen_sub.f']=B.capture.sha(new/'cmfgen_sub.f')
    if change=='executable':(new/'cmfgen_fixedj_heat.exe').write_text('changed')
    if change=='decay':m['decay_implementation']='different'
    if change=='flag':m['observer_spectrum']='UNCHANGED_NATIVE_PATH'
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):B.observer_only_build(old,new)


def bridge_record(bundle):
    expected={}
    for name in B.capture.OUTPUT_NAMES:
        p=bundle/'expected'/name;p.parent.mkdir(exist_ok=True)
        p.write_text('unchanged '+name);expected[name]=B.capture.sha(p)
    record=json.loads((bundle/'bundle-inputs.json').read_text())
    record['observer_build_bridge']=dict(output_sha256=expected)
    (bundle/'bundle-inputs.json').write_text(json.dumps(record))


def test_exact_bridge_then_second_capture(bundle):
    bridge_record(bundle);called=[]
    def run(p):
        called.append(p.name)
        for name in B.capture.OUTPUT_NAMES:
            (p/'native'/name).write_text('unchanged '+name)
        return 0
    assert B.run_bundle(bundle,run)==0
    assert called==['h05','h025']
    r=json.loads((bundle/'bundle-runtime.json').read_text())
    assert r['observer_build_bridge']['status']=='EXACT_PHYSICAL_OUTPUT_REPRODUCTION'
    assert r['final_goal']=='NOT_MET'


@pytest.mark.parametrize('name',B.capture.OUTPUT_NAMES)
def test_bridge_difference_prevents_second_capture(bundle,name):
    bridge_record(bundle);called=[]
    def run(p):
        called.append(p.name)
        for n in B.capture.OUTPUT_NAMES:(p/'native'/n).write_text('unchanged '+n)
        (p/'native'/name).write_text('different')
        return 0
    assert B.run_bundle(bundle,run)==1
    assert called==['h05']
    assert json.loads((bundle/'bundle-runtime.json').read_text())['status']=='STOPPED'


def test_changed_prepared_executable_never_starts_capture(bundle):
    record=json.loads((bundle/'bundle-inputs.json').read_text())
    record['capture_build_sha256']='not-the-actual-sha'
    (bundle/'bundle-inputs.json').write_text(json.dumps(record))
    d=bundle/'h05/build';d.mkdir();(d/'cmfgen_fixedj_heat.exe').write_text('changed')
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0)==1
    assert not called


def test_staging_requires_both_replacement_arguments(tmp_path):
    for kwargs in [dict(replacement_build=tmp_path),dict(bridge_reference=tmp_path)]:
        with pytest.raises(ValueError):B.stage(tmp_path,tmp_path/'output',**kwargs)


@pytest.fixture
def staged_replacement(replacement,tmp_path,monkeypatch):
    old,new=replacement;baseline=old.parent
    native=baseline/'native';(native/'seq_logs').mkdir(parents=True)
    inputs=baseline/'inputs';inputs.mkdir()
    for n in ('heat-state-input.json','fixedj-field-input.json'):(inputs/n).write_text('{}')
    vadat=('T [USE_FIXED_J]\nT [COH_ES]\nT [WRITE_RATES]\nF [FLUX_CAL_ONLY]\n'
           'F [RD_CF_FILE]\n1.0D0 [FRAC_DOP]\n')
    (native/'VADAT').write_text(vadat);(native/'MODEL_SPEC').write_text('800000 [NCF_MAX]\n')
    for n in ('IN_ITS','EDDFACTOR','EDDFACTOR_INFO','NATIVE_HEAT_INPUT.bin'):
        (native/n).write_text('same '+n)
    original=tmp_path/'original';material=tmp_path/'material';original.mkdir();material.mkdir()
    history=dict(material_source=str(material),original_physical_seed=str(original),
                 restart_sha256={},linked_inputs={})
    (native/'seq_logs/feedback-stage.json').write_text(json.dumps(history))
    ref=tmp_path/'reference';(ref/'native/seq_logs').mkdir(parents=True)
    texts=B.refined_grid_inputs(vadat,(native/'MODEL_SPEC').read_text(),frac_dop=.5)
    (ref/'native/VADAT').write_text(texts['vadat'])
    (ref/'native/MODEL_SPEC').write_text(texts['model_spec'])
    for n in ('EDDFACTOR','EDDFACTOR_INFO','NATIVE_HEAT_INPUT.bin'):
        (ref/'native'/n).write_bytes((native/n).read_bytes())
    outputs={}
    for n in B.capture.OUTPUT_NAMES:
        p=ref/'native'/n;p.write_text('captured '+n);outputs[n]=dict(sha256=B.capture.sha(p))
    runtime=dict(status='CAPTURED',srun_exit_code=96,captured_state_exact=True,outputs=outputs,
                 build_sha256=B.capture.sha(old/'cmfgen_fixedj_heat.exe'),fixed_j_sha256='same J',
                 accepted_response_sha256='same response')
    (ref/'native/seq_logs/fixedj-heat-runtime.json').write_text(json.dumps(runtime))
    for n in ('INPUTS','RESTART','RESTART_OVERLAY'):monkeypatch.setattr(B,n,())
    monkeypatch.setattr(B.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
    monkeypatch.setattr(B.capture,'validate_inputs',lambda p:dict(fixed_j_sha256='same J',
        state_manifest=dict(accepted_response_sha256='same response')))
    return baseline,new,ref,tmp_path/'output'


def test_stage_replacement_copies_same_new_binary_and_keeps_seed(staged_replacement):
    baseline,new,ref,out=staged_replacement
    digest=B.capture.sha(baseline/'native/EDDFACTOR')
    r=B.stage(baseline,out,replacement_build=new,bridge_reference=ref)
    assert r['observer_build_bridge']['root']==str(ref)
    assert r['capture_build_sha256']==B.capture.sha(new/'cmfgen_fixedj_heat.exe')
    for n in ('h05','h025'):
        assert B.capture.sha(out/n/'build/cmfgen_fixedj_heat.exe')==r['capture_build_sha256']
        assert B.capture.sha(out/n/'native/EDDFACTOR')==digest
    assert B.capture.sha(baseline/'native/EDDFACTOR')==digest
    assert not (out/'bundle-runtime.json').exists()


@pytest.mark.parametrize('fault',['J','control','output','status','state_flag','binary','response'])
def test_staging_rejects_changed_reference_before_any_capture(staged_replacement,fault):
    baseline,new,ref,out=staged_replacement
    path=ref/'native/seq_logs/fixedj-heat-runtime.json';r=json.loads(path.read_text())
    if fault=='J':(ref/'native/EDDFACTOR').write_text('different J')
    if fault=='control':
        p=ref/'native/VADAT';p.write_text(p.read_text().replace('T [WRITE_RATES]','F [WRITE_RATES]'))
    if fault=='output':(ref/'native/NETRATE').write_text('different')
    if fault=='status':r['status']='FAILED'
    if fault=='state_flag':r['captured_state_exact']=False
    if fault=='binary':r['build_sha256']='different'
    if fault=='response':r['accepted_response_sha256']='different'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):B.stage(baseline,out,replacement_build=new,bridge_reference=ref)
    assert not (out/'bundle-runtime.json').exists()


def heat_bytes(heat,*,radius=3.,counter=70):
    vectors=np.array([[radius,1.],[30.,10.],[0.,0.],[1.,2.],[5.,7.],
                      [2.,3.],[1.,1.],heat,[1e-16,-2e-16]],dtype='<f8')
    pops=np.array([[2.,2.],[3.,3.],[5.,7.],[1.,2.]],dtype='<f8')
    return (b'NATIVEHEATROWV1 '+struct.pack('<8i',1,64,4,2,11,counter,9,7)+
            vectors.tobytes()+pops.tobytes(order='F'))


@pytest.fixture
def numeric_bridge(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    for root in (old,new):
        (root/'native').mkdir(parents=True)
        for name in B.capture.OUTPUT_NAMES:
            (root/'native'/name).write_bytes(heat_bytes([-3.,4.]) if name=='NATIVE_HEAT_ROW.bin'
                                            else b'exact line rates')
    reference=dict(root=str(old),output_sha256={n:B.capture.sha(old/'native'/n)
                                               for n in B.capture.OUTPUT_NAMES})
    return old,new,reference


@pytest.mark.parametrize('steps',[0,1,2])
@pytest.mark.parametrize('point',[0,1])
def test_binary_heat_roundoff_is_not_a_physical_difference(numeric_bridge,steps,point):
    _,new,reference=numeric_bridge;heat=np.array([-3.,4.])
    for _ in range(steps):heat[point]=np.nextafter(heat[point],np.inf)
    (new/'native/NATIVE_HEAT_ROW.bin').write_bytes(heat_bytes(heat))
    r=B.compare_bridge(new,reference)
    assert r['final_goal']=='NOT_MET'
    if steps:
        assert r['status']=='PHYSICAL_OUTPUT_REPRODUCTION_WITH_THERMAL_ROUNDOFF'
        assert r['thermal_roundoff']['max_ulp']==steps
        assert r['thermal_roundoff']['changed_depths_1based']==[point+1]
    else:assert r['status']=='EXACT_PHYSICAL_OUTPUT_REPRODUCTION'


@pytest.mark.parametrize('fault',['three_ulp','sign','state','header','rates','reference'])
def test_heat_roundoff_rule_does_not_mask_other_changes(numeric_bridge,fault):
    old,new,reference=numeric_bridge;heat=np.array([-3.,4.]);kw={}
    for _ in range(2):heat[0]=np.nextafter(heat[0],np.inf)
    if fault=='three_ulp':heat[0]=np.nextafter(heat[0],np.inf)
    if fault=='sign':heat[0]=3.
    if fault=='state':kw['radius']=np.nextafter(3.,np.inf)
    if fault=='header':kw['counter']=71
    if fault=='rates':(new/'native/NETRATE').write_text('changed')
    if fault=='reference':(old/'native/NATIVE_HEAT_ROW.bin').write_bytes(heat_bytes([-3.,5.]))
    (new/'native/NATIVE_HEAT_ROW.bin').write_bytes(heat_bytes(heat,**kw))
    with pytest.raises(ValueError):B.compare_bridge(new,reference)
