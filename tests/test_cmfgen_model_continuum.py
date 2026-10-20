"""전 원소 PHOT/Milne 생산자의 물리식·terminal 위상 직접시험."""
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('explicit',[False,True])
def test_int_seq_parser_has_no_audit_import_dependency(tmp_path,monkeypatch,explicit):
    from cmfgen_parser import parse_f_to_s_int_seq
    monkeypatch.setitem(sys.modules,'compare_cmfgen_bf_superlevel_rates',None)
    path=tmp_path/'f_to_s';path.write_text('header\n'+
        'a 2 0 1.0 ignored 1 0'+(' 1' if explicit else '')+'\n'+
        'b 4 12 .9 ignored 2 7'+(' 2' if explicit else '')+'\n')
    mapping,names,sequence,frequency=parse_f_to_s_int_seq(path,2)
    assert names==['a','b']
    np.testing.assert_array_equal(mapping,[1,2]);np.testing.assert_array_equal(sequence,[0,7])
    np.testing.assert_array_equal(frequency,[1.,.9])


@pytest.mark.parametrize('second',['b 4 12 .9 x 2 7 3','b 4 12 .9 x 2 7','invalid'])
def test_int_seq_parser_preserves_prefix_and_layout_rejections(tmp_path,second):
    from cmfgen_parser import parse_f_to_s_int_seq
    path=tmp_path/'f_to_s';path.write_text('a 2 0 1.0 x 1 0 1\n'+second+'\n')
    with pytest.raises(ValueError):parse_f_to_s_int_seq(path,2)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cmfgen_model_continuum as M
from cmfgen_parser import parse_osc


def test_milne_lte_kirchhoff_and_target_probability():
    t=np.array([7000.,12000.,20000.]);ne=np.array([1e7,1e8,1e9])
    upper=np.array([1e5,1e6,1e7]);threshold=1e15
    nu=threshold*np.array([1.,1.3,2.]);sigma=np.array([3e-18,2e-18,1e-18])
    nstar=upper*ne*(M.H*M.H/(2*np.pi*M.ME*M.KB*t))**1.5*(2./(2*6.))
    lower=nstar*np.exp(M.H*threshold/(M.KB*t))
    absorption,stim,eta=M.milne_route(nu,sigma,lower,upper,t,ne,2.,6.,threshold,.3)
    planck=2*M.H*nu[None,:]**3/M.C**2/np.expm1(M.H*nu[None,:]/(M.KB*t[:,None]))
    np.testing.assert_allclose(eta/(absorption-stim),planck,rtol=2e-14)
    unit=M.milne_route(nu,sigma,lower,upper,t,ne,2.,6.,threshold,1.)
    for value,full in zip((absorption,stim,eta),unit,strict=True):
        np.testing.assert_allclose(value,.3*full,rtol=2e-15)


def test_zero_lower_retains_spontaneous_and_negative_net_opacity():
    a,s,e=M.milne_route(np.array([1e15]),np.array([1e-18]),np.array([0.]),
        np.array([1e5]),np.array([10000.]),np.array([1e8]),2.,1.,1e14,1.)
    assert a.item()==0 and s.item()>0 and e.item()>0
    assert (a-s).item()<0


def test_threshold_below_old_grid_is_not_a_route_skip():
    route=M.Route(20,2,0,1,0,0,1.,1.,0.,1e12,
        ((lambda x:np.full_like(x,2e-18),np.array([1e12]),1e12,'tab'),))
    nu=np.array([5.84127859e13,1e14])
    assert np.all(route.sigma(nu)>0)


@pytest.mark.parametrize('bad',['sigma','lower','target','temperature','ne','subthreshold','probability'])
def test_no_silent_physical_repair(bad):
    args=dict(nu=np.array([1e15]),sigma=np.array([1e-18]),n_lower=np.array([1.]),
        n_target=np.array([2.]),temperature=np.array([1e4]),ne=np.array([1e8]),
        g_lower=2.,g_target=1.,threshold=1e14,probability=1.)
    if bad=='sigma':args['sigma'][0]=-1.
    elif bad=='lower':args['n_lower'][0]=-1.
    elif bad=='target':args['n_target'][0]=np.nan
    elif bad=='temperature':args['temperature'][0]=0.
    elif bad=='ne':args['ne'][0]=-1.
    elif bad=='subthreshold':args['threshold']=2e15
    else:args['probability']=0.
    with pytest.raises(ValueError):M.milne_route(**args)


def test_multiple_terminal_phot_files_use_native_gion_eion(tmp_path,monkeypatch):
    paths=[tmp_path/'phot_data_A',tmp_path/'phot_data_B']
    for p in paths:p.write_text('fixed PHOT')
    entry=SimpleNamespace(config='lower',cs_type=20)
    photos={p:SimpleNamespace(n_routes=2,final_state='upper',final_state_excit_eV=float(i),
        final_state_g=float(2+i),split_J=True,entries=[entry]) for i,p in enumerate(paths)}
    ion={(20,5):dict(n_kept=1,levels=dict(config=np.array(['lower']),E_cm=np.array([0.])),
        osc=SimpleNamespace(ionization_eV=10.,z_screen=5.),provenance=dict(phot_path=paths[0]))}
    monkeypatch.setattr(M.atomic,'_route_sigma_model',lambda entry,th,Z,zion:
        (lambda x:np.full_like(x,1e-18),np.array([th]),th,'tab'))
    routes,_,counts=M.compile_routes(ion,{(20,5,1):0},{(20,5)},photos.__getitem__)
    assert counts=={'Z20_stage5':2}
    assert [r.target for r in routes]==[-1,-1]
    assert [r.target_g for r in routes]==[2.,3.]
    assert [r.probability for r in routes]==[1.,1.]
    assert routes[1].threshold_Hz>routes[0].threshold_Hz


def test_all_model_process_policy_is_not_sis_only(tmp_path):
    model=tmp_path/'MODEL';model.write_text('F,F [DIE_Fe2]\nF,F [DIE_CoSIX]\n')
    models={(26,2):dict(label='Fe2'),(27,6):dict(label='CoSIX')}
    sources={key:[dict(path='fixed source',sha256='a'*64)] for key in models}
    policy=M.all_model_policy(model,models,sources)
    assert len(policy['ions'])==2
    assert all(row['external_dr_fit_allowed']==0 for row in policy['ions'])
    model.write_text('T,F [DIE_Fe2]\nF,F [DIE_CoSIX]\n')
    with pytest.raises(ValueError):M.all_model_policy(model,models,sources)


def test_bound_target_uses_actual_temperature_partition_not_phot_sum_g():
    rows=[dict(g='2'),dict(g='2'),dict(g='4')]
    state=dict(temperature=np.array([1e4]),ne=np.array([1e8]),
        full=np.array([[2.],[3.],[7.]]),superlevel=np.array([[10.]]),
        sl_index={(20,4,0):0})
    nu=np.array([1e15]);sigma=np.array([1e-18])
    route=M.Route(20,4,0,1,0,0,1.,6.,0.,1e14,
        ((lambda x:np.full_like(x,1e-18),np.array([1e14]),1e14,'tab'),))
    result=M.continuum_block([route],rows,state,nu)
    expected=M.milne_route(nu,sigma,np.array([2.]),np.array([3.]),
        state['temperature'],state['ne'],2.,2.,1e14,1.)[2]
    wrong=M.milne_route(nu,sigma,np.array([2.]),np.array([10.]),
        state['temperature'],state['ne'],2.,6.,1e14,1.)[2]
    np.testing.assert_array_equal(result['eta_spont'],expected)
    assert abs(result['eta_spont'].item()/wrong.item()-1)>.05


def test_c_population_is_consumed_exactly_and_wrong_material_rejected(tmp_path):
    path=tmp_path/'full.bin';expected=np.array([[1.,0.],[2.,3.]])
    actual=expected.copy();actual[0,0]=np.nextafter(1.,2.)
    path.write_bytes(struct.pack('<8sII',b'LMATFL01',2,2)+actual.astype('<f8').tobytes())
    state=dict(full=expected)
    result=M.consume_c_full_population(path,state)
    np.testing.assert_array_equal(state['full'],actual)
    assert result['values']==4 and result['max_relative_error']>0
    state['full']=expected*2
    with pytest.raises(ValueError):M.consume_c_full_population(path,state)


@pytest.mark.parametrize('block_bins',[1,2,16,1024])
def test_production_blocks_match_direct_target_sum(block_bins):
    rows=[dict(g='2'),dict(g='2'),dict(g='4')]
    state=dict(temperature=np.array([1500.,1e4]),ne=np.array([1e9,1e8]),
        full=np.array([[0.,2.],[3.,2.],[7.,8.]]),
        superlevel=np.array([[10.,20.]]),sl_index={(20,4,-1):0})
    routes=[]
    for target,sl,edge,p in [(1,0,1e12,.3),(-1,-1,1e15,1.)]:
        routes.append(M.Route(20,4,0,target,sl,0,p,6.,0.,edge,
            ((lambda x:np.full_like(x,1e-18),np.array([edge]),edge,'tab'),)))
    nu=np.array([1e12,3e13,1e14,1e15,2e15,3e16,4e16])
    expected=M.continuum_block(routes,rows,state,nu)
    result=M.continuum_grid(routes,rows,state,nu,block_bins)
    for name in expected:np.testing.assert_allclose(result[name],expected[name],rtol=5e-13,atol=0)


def test_material_full_level_reconstruction_and_terminal_consumer(tmp_path):
    path=tmp_path/'material.bin'
    rows=[dict(atomic_number='20',ion_number='4',super_level='0',energy_eV='0',g='2'),
          dict(atomic_number='20',ion_number='4',super_level='0',energy_eV='1',g='4')]
    columns=np.array([[20,4,0],[20,5,-1]],dtype='<i4')
    physical=np.array([[1e14,2e14,1e4,1e8,1e-13,1.]],dtype='<f8')
    pops=np.array([[20.,10.]],dtype='<f8')
    path.write_bytes(struct.pack('<8sIIId',b'LUMCEL01',1,2,2,1e6)+
        columns.tobytes()+physical.tobytes()+pops.tobytes())
    state=M.read_material(path,rows,{(20,5):dict(n_super=1)})
    np.testing.assert_allclose(state['full'].sum(axis=0),20.,rtol=2e-15)
    ratio=2*np.exp(-M.EV/(M.KB*1e4))
    assert state['full'][1,0]/state['full'][0,0]==pytest.approx(ratio)
    route=M.Route(20,5,0,-1,-1,0,1.,1.,0.,1e14,
        ((lambda x:np.full_like(x,1e-18),np.array([1e14]),1e14,'tab'),))
    result=M.continuum_block([route],rows,state,np.array([1e15]))
    assert result['eta_spont'].item()>0
    assert result['chi_abs'].item()==pytest.approx(state['full'][0,0]*1e-18)


def test_osc_level_only_reader_preserves_header_and_levels(tmp_path):
    path=tmp_path/'osc_data'
    path.write_text('2 !Number of energy levels\n1 !Number of transitions\n'
        '100000 !Ionization energy\n2 !Screened nuclear charge\n'
        '! E(cm)\nlower 2.0 0.0 1.0 0.0 1000.0 1 1e8\n'
        'upper 4.0 10000.0 1.0 1.0 1000.0 2 1e8\n'
        'lower -upper 0.1 1e8 5000.0 1-2 1\n')
    full=parse_osc(path);levels=parse_osc(path,include_transitions=False)
    np.testing.assert_array_equal(levels.levels,full.levels)
    assert levels.n_levels==2 and levels.n_transitions==1
    assert levels.declared_n_transitions==full.declared_n_transitions==1
    assert len(levels.transitions)==0 and len(full.transitions)==1
    path.write_text(path.read_text().replace('1 !Number of transitions','2 !Number of transitions'))
    truncated=parse_osc(path)
    assert truncated.n_transitions==1 and truncated.declared_n_transitions==2


def test_occupation_uses_positive_ion_number_not_charge_and_changes_sl_weights():
    rows=[dict(atomic_number='20',ion_number='1',energy_eV='0',g='2'),
          dict(atomic_number='20',ion_number='1',energy_eV='9.9',g='4')]
    models={(20,2):dict(ionization_eV=10.,z_screen=2.)}
    state=dict(temperature=np.array([1e4]),ne=np.array([1e15]),
        columns=np.array([[20,1,0],[20,2,-1]]),superlevel=np.array([[1e14],[1e14]]),
        full=np.zeros((2,1)))
    occupation=M.material_occupation(rows,models,state)
    assert occupation[0,0]==1 and 0<occupation[1,0]<1
    changed={**state,'columns':np.array([[20,4,0],[20,6,-1]])}
    np.testing.assert_array_equal(M.dissolution_vectors(state),M.dissolution_vectors(changed))


def test_dissolved_continuum_is_not_zero_below_primary_edge():
    rows=[dict(atomic_number='20',ion_number='4',super_level='0',energy_eV='0',g='2')]
    model={(20,5):dict(ionization_eV=M.H*1e14/M.EV,z_screen=5.,terminal_g=2.)}
    state=dict(temperature=np.array([1e4]),ne=np.array([1e15]),
        columns=np.array([[20,4,0],[20,5,-1]]),superlevel=np.array([[1e14],[1e14]]),
        sl_index={(20,4,0):0,(20,5,-1):1},full=np.array([[1e14]]),occupation=np.array([[.9]]))
    nu=np.array([.8e14,1.1e14])
    route=M.Route(20,5,0,-1,-1,0,1.,2.,0.,1e14,
        ((lambda x:np.full_like(x,1e-18),np.array([1e14]),1e14,'tab'),),True)
    result={name:np.zeros((1,2)) for name in ('chi_abs','chi_stim','chi_net','eta_spont')}
    summary=M.add_dissolved_continuum(result,[route],rows,model,state,nu)
    assert summary['active_route_sample_terms']>0
    assert result['chi_abs'][0,0]>0 and result['eta_spont'][0,0]>0
    assert result['chi_abs'][0,1]==0 and result['eta_spont'][0,1]==0
