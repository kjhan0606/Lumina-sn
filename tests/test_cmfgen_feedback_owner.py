"""새 장과 물질의 세대·실제 J 선택·전체 인구 전달을 확인한다."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cmfgen_feedback_owner import choose_step, material_arrays, radiation_arrays, prepare, same_native_inputs


def test_preupdate_comparison_uses_material63_but_postupdate_needs_actual_completion():
    root = Path('/example')
    assert choose_step(root, 63) == root/'field'
    with pytest.raises(ValueError, match='미완료'):
        choose_step(root, 64)
    runtime = dict(field_complete=True, material_complete=True, material_record=64,
                   radiation_produced_from_record=63)
    assert choose_step(root, 64, runtime) == root/'material'
    runtime['radiation_produced_from_record'] = 64
    with pytest.raises(ValueError):
        choose_step(root, 64, runtime)
    with pytest.raises(ValueError):
        choose_step(root, 62)


def test_next_field_phase_uses64B_to65_without_relabelling63():
    root = Path('/example')
    assert choose_step(root, 64, source_record=64) == root/'field'
    runtime = dict(field_complete=True, material_complete=True, material_record=65,
                   radiation_produced_from_record=64)
    assert choose_step(root, 65, runtime, source_record=64) == root/'material'
    with pytest.raises(ValueError):
        choose_step(root, 63, runtime, source_record=64)
    runtime['material_record'] = 64
    with pytest.raises(ValueError, match='미완료'):
        choose_step(root, 65, runtime, source_record=64)


def test_same_new_ne_and_populations_replace_material_not_old_profile_density_or_geometry():
    old = np.array([[1.,2.],[3.,4.],[5.,6.],[10.,20.],[1.,2.]])
    state = old.copy()
    state[:3] *= 1.1
    state[-2] *= 1.2
    profiles = np.array([[1e-14,1e4,10.,1.,2.],[2e-14,2e4,20.,2.,3.]])
    original = profiles.copy()
    columns = np.array([[14,1,0],[16,1,0]])
    copied, pop = material_arrays(state, old, profiles, np.array([0,1]), columns)
    np.testing.assert_array_equal(pop, state[:2].T)
    np.testing.assert_array_equal(copied[:,2], state[-2])
    np.testing.assert_array_equal(copied[:,[0,3,4]], profiles[:,[0,3,4]])
    np.testing.assert_array_equal(profiles, original)
    state[-1,0] *= 2
    with pytest.raises(ValueError, match='고정T'):
        material_arrays(state, old, profiles, np.array([0,1]), columns)


def test_radiation_is_new_actual_angular_J_sorted_in_Hz_not_H_or_source_J():
    array = np.zeros(3, dtype=[('freq','f8'),('field','f8',(8,2))])
    array['freq'] = [3.,2.,1.]
    array['field'][:,0,:] = [[1.,2.],[3.,4.],[5.,6.]]
    array['field'][:,1:,:] = 100.
    nu, field = radiation_arrays(array)
    np.testing.assert_array_equal(nu, [1e15,2e15,3e15])
    np.testing.assert_array_equal(field, [[5.,6.],[3.,4.],[1.,2.]])
    array['freq'][1] = 0
    with pytest.raises(ValueError):
        radiation_arrays(array)


def test_existing_output_is_preserved_before_reading_any_inputs(tmp_path):
    with pytest.raises(ValueError, match='기존'):
        prepare(Path('not-read'), Path('not-read'), 64, tmp_path)


def test_changed_native_source_is_not_compared_to_old_atomic_view(tmp_path):
    from run_cmfgen_lambda_probe import sha
    source = tmp_path/'atomic'
    source.write_bytes(b'original')
    info = dict(target=str(source), sha256=sha(source))
    same_native_inputs({'PHOT':info}, {'PHOT':info})
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='원자료 변경'):
        same_native_inputs({'PHOT':info}, {'PHOT':info})


@pytest.fixture
def same_field_case(tmp_path):
    from prepare_cmfgen_owner_response import write_input
    from run_cmfgen_lambda_probe import sha
    pre=tmp_path/'pre';pre.mkdir()
    root=tmp_path/'run';(root/'field').mkdir(parents=True)
    field=root/'field/FORMAL_PRIMARY_FIELDS.bin';field.write_bytes(b'new actual J65 fixture')
    parent=np.array([[.2,.3],[.8,.7],[10.,20.],[1.,2.]])
    state=parent.copy();state[:2]*=1.1;state[-2]*=1.2
    indexes=np.array([0,1]);columns=np.array([[14,1,0],[16,1,0]])
    profiles=np.array([[1e-14,1e4,10.,1.,2.],[2e-14,2e4,20.,2.,3.]])
    nu=np.array([1e12,1e15,1e18]);radiation=np.array([[.1,.2],[.3,.4],[.5,.6]])
    write_input(pre/'owner_input.bin',nu,profiles,columns,parent[indexes].T,radiation,1e6)
    np.savez(pre/'completed-state.npz',state=parent,profiles=profiles,columns=columns,owner_variable_indexes=indexes)
    manifest=dict(feedback_root=str(root),material_snapshot=dict(record=64),radiation_record=65,
        radiation_material_record=64,input_sha256={str(field):sha(field)},
        output_sha256={n:sha(pre/n) for n in ('owner_input.bin','completed-state.npz')})
    (pre/'fixed-field-owner-inputs.json').write_text(json.dumps(manifest))
    updated_profiles=profiles.copy();updated_profiles[:,2]=state[-2]
    return dict(pre=pre,root=root,state=state,parent=parent,indexes=indexes,columns=columns,
        profiles=updated_profiles,output=tmp_path/'new-input.bin',field_hash=sha(field),epoch=1e6),(nu,radiation,manifest)


def test_same_current_field_blocks_give_exactly_the_full_rebin_input_with_new_ne(same_field_case,tmp_path):
    from prepare_cmfgen_feedback_owner import same_field_material_input
    from prepare_cmfgen_owner_response import write_input
    kwargs,(nu,radiation,_)=same_field_case
    result=same_field_material_input(source_record=64,**{('pre_root' if k=='pre' else k):v for k,v in kwargs.items()})
    write_input(tmp_path/'full-rebin.bin',nu,kwargs['profiles'],kwargs['columns'],
        kwargs['state'][kwargs['indexes']].T,radiation,1e6)
    assert kwargs['output'].read_bytes()==(tmp_path/'full-rebin.bin').read_bytes()
    assert result['same_current_field_blocks_reused'] and not result['rates_and_matrices_reused']
    assert result['radiation_blocks_byte_equal']


@pytest.mark.parametrize('defect',['old_field','wrong_material','different_root','changed_J_hash','epoch'])
def test_only_same_new_field_and_same_parent_can_be_reused(same_field_case,defect):
    from prepare_cmfgen_feedback_owner import same_field_material_input
    kwargs,(_,_,m)=same_field_case
    if defect=='old_field':m['radiation_record']=64
    if defect=='wrong_material':m['material_snapshot']['record']=63
    if defect=='different_root':m['feedback_root']='old root'
    if defect=='changed_J_hash':kwargs['field_hash']='different'
    if defect=='epoch':kwargs['epoch']=2e6
    (kwargs['pre']/'fixed-field-owner-inputs.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        same_field_material_input(source_record=64,**{('pre_root' if k=='pre' else k):v for k,v in kwargs.items()})


@pytest.mark.parametrize('feedback', [False, True, 'generic'])
def test_summary_preserves_actual_field_phase_and_distinguishes_parent_from59(tmp_path, feedback):
    from summarize_cmfgen_fixed_field_owner import summarize
    from run_cmfgen_lambda_probe import sha
    root, response, out = (tmp_path/p for p in ('input','response','summary'))
    root.mkdir();response.mkdir()
    columns = np.array([(Z,c,0) for Z in (14,16) for c in range(1,6)])
    indexes = np.arange(10)
    old = np.ones((12,2))*.2
    old[-2] = 10.;old[-1] = 1.
    parent = old.copy()
    parent[0],parent[1] = .3,.1
    state = old.copy()
    state[0],state[1] = .32,.08
    state[-2] = 12.
    profiles = np.array([[1e-14,1e4,12.,1.,2.],[1e-14,1e4,12.,2.,3.]])
    arrays = dict(state=state,previous59=old,profiles=profiles,owner_variable_indexes=indexes,columns=columns)
    if feedback:
        arrays['previous_parent' if feedback == 'generic' else 'previous63'] = parent
    np.savez(root/'completed-state.npz', **arrays)
    refs=[];rows=[]
    for d in range(2):
        for j,(Z,charge,_) in enumerate(columns):
            fraction=float(state[j,d])
            refs.append(dict(depth_1based=d+1,Z=Z,stage=charge+1,population_cm3=fraction,
                fraction=fraction,Te_K=1e4,ne_cm3=12.))
            rows.append(dict(depth_1based=d+1,Z=Z,stage=charge+1,Te_K=1e4,ne_cm3=12.,n_element=1.,
                pre_fraction=fraction,candidate_fraction=fraction,committed_fraction=fraction,
                scaled_se_residual=1e-15,conservation_residual=1e-16,
                certificate_probability_error_display=1e-16,certificate_relative_error_display=1e-16,
                completed=1,candidate_valid=1,certificate_pass=1,verdict_pass=1,commit_performed=1,
                legacy_topology_pass=1,legacy_numerical_pass=1))
    for path, table in ((root/'reference.csv',refs),(response/'response.csv',rows)):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (response/'owner.log').write_text('CMF_OWNER_RESPONSE_COMPLETE depths=2\n')
    manifest=dict(material_snapshot=dict(record=(65 if feedback == 'generic' else 64) if feedback else 60),
        output_sha256={n:sha(root/n) for n in ('completed-state.npz','reference.csv')},
        max_ne_change_over59=.2,max_native_charge_residual=1e-15,element_total_max_relative_change={})
    if feedback:
        parent_record = 64 if feedback == 'generic' else 63
        manifest.update(material_parent_record=parent_record,radiation_record=parent_record+1,radiation_material_record=parent_record,
                        field_phase_warning='new field produced from63, not converged')
    (root/'fixed-field-owner-inputs.json').write_text(json.dumps(manifest))
    binary=tmp_path/'synthetic-binary-identity';binary.write_bytes(b'not a simulation')
    summarize(root,response,binary,out)
    result=json.loads((out/'fixed-field-owner.json').read_text())
    expected_parent = (64 if feedback == 'generic' else 63) if feedback else 59
    assert result['material_parent_record'] == expected_parent
    assert result['radiation_record'] == (expected_parent+1 if feedback else 60)
    assert result['radiation_material_record'] == expected_parent
    first=next(csv.DictReader((out/'fractions.csv').open()))
    assert float(first['native_parent']) == (.3 if feedback else .2)
    assert float(first['native59']) == .2
    assert result['CMFGEN_converged'] is False
