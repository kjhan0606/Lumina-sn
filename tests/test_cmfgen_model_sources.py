"""같은 FL의 실제 FF·국소 선 χ/η 및 원본 MODEL 선택 직접시험."""
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import cmfgen_model_sources as S
import cmfgen_model_continuum as B


def test_full_pair_known_memory_includes_both_states_and_postprocessing():
    result=S.paired_memory_inventory([89,90],1115020,24667)
    assert result['known_allocation_bound_bytes']==21123862912
    assert result['known_allocation_bound_bytes']<32*1024**3
    assert not result['whole_process_peak_RSS_verified']
    assert result['sequential_state_postprocessing']
    assert result['parts']['both_states_BF']==4*179*1115020*8


def test_explicit_continuum_mapping_preserves_requested_order_and_duplicates():
    nu=np.array([1e14,2e14,3e14,4e14])
    mapped=np.array([2.1e14,2.1e14,1.2e14,4.1e14])
    unique,inverse=S.continuum_frequencies(nu,mapped)
    np.testing.assert_array_equal(unique,[1.2e14,2.1e14,4.1e14])
    np.testing.assert_array_equal(unique[inverse],mapped)
    direct,index=S.continuum_frequencies(nu)
    np.testing.assert_array_equal(direct,nu)
    assert index is None


@pytest.mark.parametrize('nu,mapped',[
    ([2.,1.],[2.,1.]),([1.,1.],[1.,1.]),([],[]),([0.],[1.]),
    ([1.],[np.nan]),([1.],[-1.]),([1.,2.],[1.]),([1.],[[1.]])])
def test_invalid_continuum_mapping_rejected(nu,mapped):
    with pytest.raises(ValueError):S.continuum_frequencies(nu,mapped)


def test_continuum_identity_is_exact_even_for_signed_chi():
    nu=np.array([1e14,2e14]);chi=np.array([[-1e-12,1e-15]])
    eta=np.array([[1e-20,1e-26]])
    c,e=S.rescale_continuum(nu,nu,[9000.],chi,eta)
    np.testing.assert_array_equal(c,chi);np.testing.assert_array_equal(e,eta)


def test_continuum_native_thermal_correction_preserves_lte_and_true_absorption():
    nu=np.array([1e14,3e14,9e14]);evaluation=np.array([.99e14,3e14,9.1e14])
    t=np.array([8000.,20000.]);absorption=np.full((2,3),2e-12)
    decay=np.exp(-B.H*evaluation[None,:]/(B.KB*t[:,None]))
    eta=absorption*decay*(2*B.H*evaluation**3/B.C**2)[None,:]
    chi=absorption*(1-decay)
    c,e=S.rescale_continuum(nu,evaluation,t,chi,eta)
    expected_decay=np.exp(-B.H*nu[None,:]/(B.KB*t[:,None]))
    np.testing.assert_allclose(c,absorption*(1-expected_decay),rtol=2e-15)
    np.testing.assert_allclose(e,absorption*expected_decay*(2*B.H*nu**3/B.C**2)[None,:],rtol=3e-15)
    np.testing.assert_array_equal(c[:,1],chi[:,1])
    np.testing.assert_array_equal(e[:,1],eta[:,1])


def test_continuum_native_correction_keeps_inversion_without_floor():
    nu=np.array([1e14,2e14]);evaluation=np.array([.99e14,2.01e14])
    chi=np.full((1,2),-1e-9);eta=np.full((1,2),1e-25)
    c,e=S.rescale_continuum(nu,evaluation,[9000.],chi,eta)
    assert np.all(c<0) and np.all(e>0)


@pytest.mark.parametrize('t,chi,eta',[
    ([0.],[[1.]],[[1.]]),([np.nan],[[1.]],[[1.]]),
    ([8000.],[1.],[[1.]]),([8000.],[[1.]],[[np.nan]]),
    ([8000.],[[1.]],[[ -1.]])])
def test_invalid_continuum_coefficients_rejected(t,chi,eta):
    with pytest.raises(ValueError):S.rescale_continuum([1e14],[1.01e14],t,chi,eta)


def test_mapped_pair_memory_counts_unique_bf_and_thermal_temporaries():
    result=S.paired_memory_inventory([89,90],100,24667,continuum_frequencies=12)
    assert result['parts']['both_states_BF']==4*179*12*8
    assert result['parts']['one_state_postprocessing']==32*90*100*8
    with pytest.raises(ValueError):
        S.paired_memory_inventory([89,90],100,24667,continuum_frequencies=101)


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    supplied=os.environ.get('CMFGEN_SOURCE_TEST_LIB')
    if supplied:return S.SourceKernels(Path(supplied))
    source=Path(os.environ.get('CMFGEN_SOURCE_ROOT','/gpfs/kjhan/cmfgen_src/cur_cmf'))
    if not (source/'subs/gff.f').is_file():pytest.skip('원본 GFF_VEC 소스가 필요함')
    out=tmp_path_factory.mktemp('source-kernels')/'native'
    subprocess.run(['bash',str(ROOT/'scripts/build_cmfgen_source_kernels.sh'),str(source),
        str(ROOT),str(out)],check=True,timeout=60,capture_output=True)
    return S.SourceKernels(out/'model_sources.so')


def test_original_gaunt_grid_matches_scalar_native_function(kernels):
    nu=np.array([1e9,1e13,1e15,1e18]);t=np.array([500.,5000.,50000.,1e7])
    f=kernels.dll.gff_;ptr=ct.POINTER(ct.c_double)
    f.argtypes=[ptr,ptr,ptr];f.restype=ct.c_double
    for charge in (1.,2.,6.):
        actual=kernels.gaunt(nu,t,charge)
        expected=np.empty_like(actual)
        for s,temp in enumerate(t):
            for b,freq in enumerate(nu):
                n4,t4,z=ct.c_double(freq*1e-15),ct.c_double(temp*1e-4),ct.c_double(charge)
                expected[s,b]=f(ct.byref(n4),ct.byref(t4),ct.byref(z))
        np.testing.assert_allclose(actual,expected,rtol=2e-15)
    assert not np.allclose(actual,1.)


def state():
    return dict(columns=np.array([[20,1,0],[20,2,0],[20,3,-1]]),
        superlevel=np.array([[2e6,3e6],[5e6,7e6],[11e6,13e6]]),
        sl_index={(20,1,0):0,(20,2,0):1,(20,3,-1):2},
        temperature=np.array([8000.,18000.]),ne=np.array([1e8,2e8]))


def test_ff_actual_all_stage_charge_squared_and_terminal(kernels):
    material=state();nu=np.array([1e13,1e14,1e15]);models={(20,2):{},(20,3):{}}
    result=S.free_free(nu,material,models,kernels)
    weighted=sum(z*z*material['superlevel'][i,:,None]*kernels.gaunt(nu,material['temperature'],float(z))
                 for i,z in enumerate((1,2,3)))
    x=B.H*nu[None,:]/(B.KB*material['temperature'][:,None])
    expected=3.69e8*material['ne'][:,None]/np.sqrt(material['temperature'][:,None])*weighted*(-np.expm1(-x))/nu**3
    np.testing.assert_allclose(result['chi'],expected,rtol=3e-15)
    assert result['charges']==[1,2,3] and result['native_charges']==[2,3]
    assert np.all(result['chi']>result['native_chi'])
    planck=2*B.H*nu[None,:]**3/B.C**2/np.expm1(x)
    np.testing.assert_allclose(result['eta']/result['chi'],planck,rtol=3e-15)
    doubled=S.free_free(nu,{**material,'ne':material['ne']*2},models,kernels)
    np.testing.assert_array_equal(doubled['chi'],result['chi']*2)


@pytest.mark.parametrize('bad',[0.,-1.,np.nan])
def test_gaunt_rejects_invalid_temperature(kernels,bad):
    with pytest.raises(ValueError):kernels.gaunt(np.array([1e15]),np.array([bad]),1.)


def line(nu0=1e15):
    g=.5;f=.2
    return dict(lower=np.array([0],dtype='i4'),upper=np.array([1],dtype='i4'),
        nu0=np.array([nu0]),f=np.array([f]),g_ratio=np.array([g]),
        A=np.array([B.atomic.A_PREFACTOR*f*g*nu0**2]))


def test_local_line_raw_spontaneous_inversion_and_exact_zero(kernels):
    lines=line();nu=np.array([.9999e15,1e15,1.0001e15])
    profile=dict(width_fraction=1e-4,support_fraction=6e-4)
    pop=np.array([[0.,1.],[4.,2.]])
    chi,eta=kernels.lines(nu,pop,lines,profile)
    phi=np.exp(-((nu-1e15)/1e11)**2)/(1.77245385095516*1e11)
    expected=B.H/(4*np.pi)*1e15*lines['A'][0]*pop[1,:,None]*phi
    np.testing.assert_allclose(eta,expected,rtol=3e-15)
    assert np.all(chi[0]<0) and np.all(chi[1]==0) and np.all(eta>0)
    # native의 정확히0인 χ→lower 흡수1% 치환을 복제하지 않는다.
    assert eta[0,1]/eta[1,1]==2.


def test_line_lte_at_center_and_profile_integral(kernels):
    lines=line();t=1e4;nu0=lines['nu0'][0]
    lower=1e8;upper=lower/.5*np.exp(-B.H*nu0/(B.KB*t))
    width=1e-4;profile=dict(width_fraction=width,support_fraction=6*width)
    nu=np.linspace(nu0*(1-6*width),nu0*(1+6*width),2401)
    chi,eta=kernels.lines(nu,np.array([[lower],[upper]]),lines,profile)
    planck=2*B.H*nu0**3/B.C**2/np.expm1(B.H*nu0/(B.KB*t))
    assert eta[0,1200]/chi[0,1200]==pytest.approx(planck,rel=3e-14)
    trap=getattr(np,'trapezoid',np.trapz)
    # 원본 SET_LINE_OPAC의 Gaussian 분모 상수는 수학적 sqrt(pi)와
    # 2.8e-11 정도 다르다. 시험 오차를 늘리지 않고 실제 식의 적분과 비교한다.
    expected=B.H/(4*np.pi)*nu0*lines['A'][0]*upper*np.sqrt(np.pi)/1.77245385095516*math.erf(6.)
    assert trap(eta[0],nu)==pytest.approx(expected,rel=2e-14)
    outside=np.array([nu0*(1-6.1*width),nu0*(1+6.1*width)])
    for a in kernels.lines(outside,np.array([[lower],[upper]]),lines,profile):
        np.testing.assert_array_equal(a,0)


def test_line_no_arbitrary_wavelength_mask(kernels):
    for nu0 in (1e12,1e17):
        c,e=kernels.lines(np.array([nu0]),np.array([[1e6],[1e4]]),line(nu0),
                         dict(width_fraction=1e-4,support_fraction=6e-4))
        assert c[0,0]>0 and e[0,0]>0


def test_line_invalid_population_rejected(kernels):
    with pytest.raises(ValueError):kernels.lines(np.array([1e15]),np.array([[-1.],[2.]]),
        line(),dict(width_fraction=1e-4,support_fraction=6e-4))


def test_native_gf_selection_keeps_low_levels_top_A_and_ties(tmp_path):
    model=tmp_path/'MODEL';model.write_text('.003 [GF_CUT]\n10 [AT_CUT]\n2 [GF_LEV_CUT]\n3 [MIN_TRANS]\n')
    low=np.arange(7);up=np.full(7,7);g=np.ones(7)
    f=np.array([1e-6,1e-6,.003,1e-6,1e-6,1e-6,1e-6]);A=np.array([1,1,1,10,9,8,8.])
    keep=S.gf_selection(low,up,f,A,g,20,model,8)
    np.testing.assert_array_equal(keep,[True,True,True,True,True,True,True])
    A[-1]=7
    keep=S.gf_selection(low,up,f,A,g,20,model,8)
    np.testing.assert_array_equal(keep,[True,True,True,True,True,True,False])
    # 원자료의 희소 행은0인 A를 포함하는 native DOWN의 순위를 보존해야 한다.
    np.testing.assert_array_equal(S.gf_selection(low[:2],up[:2],f[:2],A[:2],g[:2],20,model,8),True)


def test_fixed_doppler_default_ignores_variable_width_global_profile(tmp_path):
    model=tmp_path/'MODEL';model.write_text('2 [TDOP]\n4 [AMASS_DOP]\n10 [VTURB]\n6 [MAX_DOP]\n1 [FRAC_DOP]\nDOPPLER [GLOBAL_PROF]\n3 [DOP_LIM]\n')
    p=S.fixed_profile(model)
    assert p['kind']=='DOP_FIX' and p['support_native_NUM_DOP']==6
    assert p['width_fraction']==pytest.approx(12.85*np.sqrt(.5+(10/12.85)**2)/299800)


def test_model_line_source_rejects_incomplete_transition_list(tmp_path,monkeypatch):
    path=tmp_path/'osc'
    path.write_text('2 !Number of energy levels\n2 !Number of transitions\n'
        '100000 !Ionization energy\n2 !Screened nuclear charge\n! E(cm)\n'
        'lower 2.0 0.0 1.0 0.0 1000.0 1 1e8\n'
        'upper 4.0 10000.0 1.0 1.0 1000.0 2 1e8\n'
        'lower -upper 0.1 1e8 5000.0 1-2 1\n')
    monkeypatch.setattr(B.atomic,'load_cmfgen_links',lambda p:{(20,2):dict(osc=path)})
    rows=[dict(atomic_number='20',ion_number='1',level_number=str(l)) for l in (0,1)]
    with pytest.raises(ValueError,match='선언/판독'):
        S.load_lines(tmp_path,rows,{(20,2):dict(n_full=2)})
