"""native 전체 FL 비교에서 원자·전하·준위 주소 정렬을 확인한다."""
from pathlib import Path
import sys
import csv
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_cmfgen_fixedj_full as compare_module
from compare_cmfgen_fixedj_full import align_native, error_metrics, level_layout


def test_cli_treats_all_expected_sha256_values_as_text():
    for name in compare_module.SHA256_ARGUMENTS:
        assert compare_module.argument_type(name) is str
    assert compare_module.argument_type('response') is Path


def test_native_full_rows_align_by_atomic_charge_and_level_with_alias():
    levels=[
        dict(atomic_number='14',ion_number='1',level_number='0',super_level='0'),
        dict(atomic_number='14',ion_number='1',level_number='1',super_level='0'),
    ]
    models={(14,2):dict(label='SiII',n_full=2)}
    lookup,labels=level_layout(levels,models)
    native_index=[
        dict(native_ion='1',Z='14',full_level='2',superlevel='1',eq_sl='5',ion_label='SkII'),
        dict(native_ion='1',Z='14',full_level='1',superlevel='1',eq_sl='5',ion_label='SkII'),
    ]
    for row in native_index:
        row.update(edge_native='1.0',g='2.0')
    native=np.array([[20.,21.],[10.,11.]])
    aligned,groups=align_native(native,native_index,levels,lookup,labels)
    np.testing.assert_array_equal(aligned,[[10.,11.],[20.,21.]])
    assert [item[4] for item in groups]==[2,1]


def test_native_full_alignment_rejects_duplicates_and_wrong_superlevel():
    levels=[dict(atomic_number='14',ion_number='1',level_number=str(i),super_level='0')
            for i in range(2)]
    lookup,labels=level_layout(levels,{(14,2):dict(label='SiII',n_full=2)})
    row=dict(native_ion='1',Z='14',full_level='1',superlevel='1',eq_sl='1',
             ion_label='SiII',edge_native='1.0',g='2.0')
    with pytest.raises(ValueError):
        align_native(np.ones((2,2)),[row,row],levels,lookup,labels)
    bad=dict(row,full_level='2',superlevel='2')
    with pytest.raises(ValueError):
        align_native(np.ones((2,2)),[row,bad],levels,lookup,labels)


def test_scaled_relative_metrics_handle_shared_zero_and_report_worst_location():
    left=np.array([[0.,2.],[4.,8.]])
    right=np.array([[0.,1.],[4.,10.]])
    result=error_metrics(left,right)
    assert result['exact_values']==2
    assert result['max_absolute']==2.
    assert result['max_scaled_relative']==0.5
    assert (result['worst_row'],result['worst_point'])==(0,2)


def test_native_heat_row_converts_cmfgen_internal_units_and_preserves_residual_sign(tmp_path):
    balance=tmp_path/'balance.npz'
    stored=np.array([2.,-1.,0.])
    instantaneous=np.array([0.5,2.,0.])
    scenario=stored+instantaneous
    np.savez(balance,native_stored_net=stored,native_stored_gross=np.array([4.,2.,0.]),
             native_instantaneous_scenario_net=scenario,
             native_instantaneous_scenario_gross=np.array([5.,3.,0.]),
             instantaneous_scenario=instantaneous,stored_decay=np.zeros(3))
    factor=compare_module.CMFGEN_RE_INTERNAL_TO_CGS
    heat=dict(steq_t_native=stored/factor,frequencies=1115020)
    result=compare_module.compare_heat_row(heat,balance,compare_module.sha(balance))
    assert result['stored_policy_candidate']['max_absolute_difference']==pytest.approx(0.,abs=1.e-14)
    assert result['stored_policy_candidate']['max_gross_scaled_absolute_difference']==pytest.approx(0.,abs=1.e-14)
    assert result['instantaneous_scenario']['nonzero_sign_mismatch_points']==1
    assert result['instantaneous_scenario']['max_absolute_difference']==pytest.approx(2.)
    assert result['source_policy']['instantaneous_recomposition_max_absolute']<1.e-14
    assert result['source_policy']['stored_decay_max_absolute']==0.
    assert result['native_steq_t_sign_convention']=='positive=net heating, negative=net cooling'
    assert result['native_steq_t_raw_units']=='CMFGEN radiative-equilibrium internal units'
    assert result['native_steq_t_to_cgs_factor']==factor
    assert result['comparison_units']=='erg cm^-3 s^-1'
    assert result['comparison_sign_convention']=='positive=heating'
    assert result['stored_policy_candidate']['points'][0]['native_steq_t_raw']==pytest.approx(2./factor)
    assert result['stored_policy_candidate']['points'][0]['native_steq_t_cgs']==pytest.approx(2.)
    assert result['stored_policy_candidate']['points'][0]['lumina_net']==2.
    assert result['stored_policy_candidate']['points'][0]['difference']==pytest.approx(0.,abs=1.e-14)
    assert len(result['stored_policy_candidate']['points'])==3
    assert result['interpretation'].startswith('No pass threshold')

    mismatched=compare_module.compare_heat_row(
        dict(steq_t_native=np.array([1.,0.,0.])/factor,frequencies=1115020),
        balance,compare_module.sha(balance))
    assert mismatched['stored_policy_candidate']['max_absolute_difference']==pytest.approx(1.)
    assert mismatched['stored_policy_candidate']['worst_net_relative_point_1based']==2
    assert mismatched['stored_policy_candidate']['max_gross_scaled_absolute_difference']==pytest.approx(0.5)
    with pytest.raises(ValueError,match='봉인 해시'):
        compare_module.compare_heat_row(heat,balance,'0'*64)


def test_run_identity_rejects_field_artifact_not_matching_sealed_manifest(tmp_path):
    native_root=tmp_path/'run';native_root.mkdir()
    (native_root/'VADAT').write_bytes(b'fixed J continuum control fixture')
    field_root=tmp_path/'field';field_root.mkdir()
    (field_root/'EDDFACTOR').write_bytes(b'prepared-field')
    (field_root/'EDDFACTOR_INFO').write_bytes(b'prepared-info')
    manifest=tmp_path/'field.json'
    manifest.write_text(json.dumps({'output_sha256':{
        'EDDFACTOR':compare_module.sha(field_root/'EDDFACTOR'),
        'EDDFACTOR_INFO':compare_module.sha(field_root/'EDDFACTOR_INFO')},
        'input_sha256':{'generation15/J.npy':'1'*64}}))
    build_root=tmp_path/'build';build_root.mkdir()
    exe=build_root/'cmfgen_fixedj_heat.exe';exe.write_bytes(b'fixture-exe')
    build_manifest=build_root/'build.json'
    build_manifest.write_text(json.dumps({'output_sha256':{
        'cmfgen_fixedj_heat.exe':compare_module.sha(exe)}}))
    args=SimpleNamespace(build_manifest=build_manifest,executable=exe,
        field_manifest=manifest,field_root=field_root,native_root=native_root,
        expected_j_sha256='2'*64,
        expected_vadat_sha256=compare_module.sha(native_root/'VADAT'))
    with pytest.raises(ValueError,match='기준 J'):
        compare_module.verify_run_identity(args)
    manifest.write_text(json.dumps({'output_sha256':{
        'EDDFACTOR':compare_module.sha(field_root/'EDDFACTOR'),
        'EDDFACTOR_INFO':compare_module.sha(field_root/'EDDFACTOR_INFO')},
        'input_sha256':{'generation15/J.npy':'2'*64}}))
    args.expected_vadat_sha256='0'*64
    with pytest.raises(ValueError,match='VADAT'):
        compare_module.verify_run_identity(args)
    args.expected_vadat_sha256=compare_module.sha(native_root/'VADAT')
    manifest.write_text(json.dumps({'output_sha256':{
        'EDDFACTOR':'0'*64,'EDDFACTOR_INFO':compare_module.sha(field_root/'EDDFACTOR_INFO')},
        'input_sha256':{'generation15/J.npy':'2'*64}}))
    args.expected_j_sha256='2'*64
    with pytest.raises(ValueError,match='EDDFACTOR'):
        compare_module.verify_run_identity(args)


def write_lmat(path,array):
    values=np.asarray(array,dtype='<f8')
    path.write_bytes(struct.pack('<8sII',b'LMATFL01',*values.shape)+values.tobytes())


def test_complete_same_state_comparison_writes_diagnostic_only_report(tmp_path,monkeypatch):
    root=tmp_path/'native';root.mkdir()
    nd=90;nt=5
    (root/'VADAT').write_bytes(b'fixed J continuum control fixture')
    radius=np.arange(nd,0,-1,dtype=np.float64)
    pops=np.zeros((nt,nd),dtype=np.float64)
    pops[-2]=2.;pops[-1]=1.
    state=b'NATIVEHEATINPV1 '+struct.pack('<4i',1,64,nt,nd)
    state+=radius.astype('<f8').tobytes()+pops.astype('<f8').tobytes(order='F')
    (root/'NATIVE_FULL_INPUT.bin').write_bytes(state)
    (tmp_path/'expected-state.bin').write_bytes(state)

    full=np.vstack((np.full(nd,11.),np.full(nd,22.)))
    fraction=np.vstack((np.full(nd,.1),np.full(nd,.2)))
    occupation=np.vstack((np.full(nd,.8),np.full(nd,.9)))
    write_lmat(root/'NATIVE_FULL_POP.bin',full)
    write_lmat(root/'NATIVE_LTE_FRACTION.bin',fraction)
    write_lmat(root/'NATIVE_OCCUPATION.bin',occupation)
    with (root/'NATIVE_FULL_INDEX.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['native_ion','Z','full_level','superlevel','eq_sl','ion_label','edge_native','g'])
        writer.writerow([1,14,1,1,1,'SkII',1.5,2])
        writer.writerow([1,14,2,1,1,'SkII',1.0,4])
    (root/'NATIVE_FULL_PHASE.txt').write_text(
        'AFTER_CURRENT_STATE_EVAL_LTE_BEFORE_RATE_AND_HEAT_EVALUATION\n7 5 90 2 1\n')
    native_steq_t=np.linspace(-1.,1.,nd)
    vectors=np.vstack((radius,np.zeros(nd),np.zeros(nd),np.ones(nd),np.full(nd,2.),
                       np.ones(nd),np.ones(nd),native_steq_t,np.zeros(nd)))
    heat=b'NATIVEHEATROWV1 '+struct.pack('<8i',1,64,nt,nd,197291,7,9,0)
    heat+=vectors.astype('<f8').tobytes()+pops.astype('<f8').tobytes(order='F')
    (root/'NATIVE_HEAT_ROW.bin').write_bytes(heat)

    levels=[dict(atomic_number=14,ion_number=1,level_number=1,super_level=0),
            dict(atomic_number=14,ion_number=1,level_number=0,super_level=0)]
    level_path=tmp_path/'levels.csv'
    with level_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(levels[0]))
        writer.writeheader();writer.writerows(levels)
    c_full=np.vstack((full[1],full[0]))
    write_lmat(tmp_path/'C-imported-full.bin',c_full)
    c_occupation=tmp_path/'occupation.bin'
    c_occupation.write_bytes(struct.pack('<8sII',b'LMATOCC1',2,nd)+
                             np.vstack((occupation[1],occupation[0])).astype('<f8').tobytes())
    response=tmp_path/'response.npz'
    np.savez(response,fraction=np.vstack((fraction[1],fraction[0])),
             occupation=np.vstack((occupation[1],occupation[0])))
    balance=tmp_path/'balance.npz'
    balance_net=native_steq_t*compare_module.CMFGEN_RE_INTERNAL_TO_CGS
    np.savez(balance,native_stored_net=balance_net,native_stored_gross=np.ones(nd),
             native_instantaneous_scenario_net=balance_net,
             native_instantaneous_scenario_gross=np.ones(nd),
             instantaneous_scenario=np.zeros(nd),stored_decay=np.zeros(nd))
    field_root=tmp_path/'field';field_root.mkdir()
    (field_root/'EDDFACTOR').write_bytes(b'fixed-J field fixture')
    (field_root/'EDDFACTOR_INFO').write_bytes(b'fixed-J field metadata fixture')
    field_manifest=tmp_path/'field-manifest.json'
    field_manifest.write_text(json.dumps({'output_sha256':{
        'EDDFACTOR':compare_module.sha(field_root/'EDDFACTOR'),
        'EDDFACTOR_INFO':compare_module.sha(field_root/'EDDFACTOR_INFO')},
        'input_sha256':{'generation15/J.npy':'7d257602576edf93fe32fa4d7c0e61d162eb26c2d7f8f8683c6c2956d5542841'},
        'layout':{'depths':nd,'frequencies':1115020,'native_quadrature_equal':False},
        'interpolation':'linear-frequency fixture interpolation',
        'quadrature':'native-grid fixture'}))
    executable=tmp_path/'cmfgen_fixedj_heat.exe';executable.write_bytes(b'fixture executable')
    build_manifest=tmp_path/'build-manifest.json'
    build_manifest.write_text(json.dumps({'output_sha256':{
        'cmfgen_fixedj_heat.exe':compare_module.sha(executable)}}))
    model_spec=tmp_path/'MODEL_SPEC';model_spec.write_text('fixture')
    links=tmp_path/'atomic_links.txt';links.write_text('fixture')
    monkeypatch.setattr(compare_module,'load_cmfgen_model_spec',
                        lambda *_:{(14,2):dict(label='SiII',n_full=2)})
    args=SimpleNamespace(native_root=root,out=tmp_path/'comparison.json',
        build_manifest=build_manifest,executable=executable,
        field_manifest=field_manifest,field_root=field_root,
        expected_vadat_sha256=compare_module.sha(root/'VADAT'),
        expected_j_sha256='7d257602576edf93fe32fa4d7c0e61d162eb26c2d7f8f8683c6c2956d5542841',
        expected_state=tmp_path/'expected-state.bin',c_full=tmp_path/'C-imported-full.bin',
        c_occupation=c_occupation,response=response,
        expected_response_sha256=compare_module.sha(response),
        lumina_balance=balance,
        expected_lumina_balance_sha256=compare_module.sha(balance),levels=level_path,
        model_spec=model_spec,atomic_links=links)
    result=compare_module.compare(args)
    assert result['schema']=='lumina-cmfgen-fixedj-full-comparison-v6'
    assert result['exact_injected_state'] and result['heat_row_same_state']
    assert result['fixed_J_source_sha256']=='7d257602576edf93fe32fa4d7c0e61d162eb26c2d7f8f8683c6c2956d5542841'
    assert result['comparisons']['native_full_vs_C_imported_full']['max_absolute']==0.
    assert result['comparisons']['native_LTE_fraction_vs_Lumina_fraction']['max_absolute']==0.
    assert result['native_heat_comparison']['stored_policy_candidate']['max_absolute_difference']==pytest.approx(0.)
    assert result['native_heat_comparison']['stored_policy_candidate']['points'][0]['difference']==pytest.approx(0.)
    assert len(result['native_heat_comparison']['stored_policy_candidate']['points'])==nd
    assert result['native_heat_comparison']['frequency_quadrature']==dict(
        lumina_field_frequency_count=1115020,cmfgen_native_frequency_count=197291,
        native_quadrature_equal=False,field_interpolation='linear-frequency fixture interpolation',
        field_quadrature='native-grid fixture')
    assert result['physical_equivalence']=='DIAGNOSTIC_ONLY_NO_PREDECLARED_THRESHOLD'
    assert result['final_goal']=='NOT_MET'
    saved=json.loads(args.out.read_text())
    assert saved['input_sha256']['native_full_population']['sha256']
    assert saved['input_sha256']['native_VADAT']['sha256']==compare_module.sha(root/'VADAT')
