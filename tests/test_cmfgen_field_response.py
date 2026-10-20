"""전체 native J의 선형 광율·가열 함수와 부분 출력 거부를 직접 확인한다."""
import json
import math
import csv
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_cmfgen_field_response as response


def test_constant_cross_section_linear_field_has_analytic_gamma_and_heating():
    nu=np.linspace(1e15,4e15,25)
    sigma=2e-18;threshold=1e15;p=.7
    J=2e-6+3e-22*nu
    wg,wh=response.linear_sigma_weights(nu,[(p,lambda q:np.full_like(q,sigma),[],threshold,threshold)])
    lo,hi=nu[0],nu[-1];a=2e-6;b=3e-22
    integral=a*math.log(hi/lo)+b*(hi-lo)
    gamma=4*math.pi*p*sigma*integral/response.bf.H_CGS
    heat=4*math.pi*p*sigma*(a*(hi-lo)+.5*b*(hi**2-lo**2)-threshold*integral)
    assert wg@J==pytest.approx(gamma,rel=1e-13)
    assert wh@J==pytest.approx(heat,rel=1e-13)
    assert np.all(wg>=0) and np.all(wh>=0)


def test_nodal_function_matches_existing_all_native_frequency_integrator():
    nu=np.geomspace(1e14,4e17,401);threshold=3.71e14
    sigma=lambda q:2e-18*(threshold/q)**3
    J=1e-8*(1+.4*np.sin(np.log(nu)))
    term=(.4,sigma,np.array([threshold,1e16]),threshold)
    wg,wh=response.linear_sigma_weights(nu,[(*term,threshold)])
    exact=response.bf.raw_field_continuous_sigma_gamma(nu,J,np.array([nu[0],nu[-1]]),np.array([1]),[term])
    assert wg@J==pytest.approx(exact,rel=2e-14)
    assert wh@J>0
    delta=J*np.sin(np.log(nu)*9)
    assert abs(wg@delta)<=wg@abs(delta)


def test_grouped_field_function_preserves_all_nonzero_contributions():
    rng=np.random.default_rng(983)
    old=rng.uniform(.01,1.,(43,7));new=rng.uniform(.01,1.,(43,7));absolute=abs(new-old)
    g=np.zeros(43);h=np.zeros(43)
    g[5:36]=rng.uniform(1e-22,1e-20,31);h[4:37]=rng.uniform(1e-30,1e-28,33)
    g[2]=1e-280;h[40]=1e-290
    expected=np.array([g@old,g@new,g@absolute,h@old,h@new,h@absolute])
    np.testing.assert_allclose(response.apply_field_weights(g,h,old,new,absolute),expected,rtol=2e-14,atol=0)
    np.testing.assert_array_equal(response.apply_field_weights(g*0,h*0,old,new,absolute),np.zeros((6,7)))
    tiny_g=np.zeros(43);tiny_h=np.zeros(43);tiny_g[2]=1e-280;tiny_h[40]=1e-290
    tiny=response.apply_field_weights(tiny_g,tiny_h,old,new,absolute)
    np.testing.assert_allclose(tiny[0],tiny_g[2]*old[2],rtol=2e-14,atol=0)
    np.testing.assert_allclose(tiny[3],tiny_h[40]*old[40],rtol=2e-14,atol=0)


def test_partial_native_field_is_rejected_before_reading_any_EDDFACTOR(tmp_path):
    run=tmp_path/'partial';(run/'seq_logs').mkdir(parents=True)
    (run/'seq_logs/whole-runtime.json').write_text(json.dumps(dict(
        diagnostic_complete=False,returncode=95,coverage=dict(count=5690))))
    with pytest.raises(ValueError,match='부분'):
        response.read_complete_pair(tmp_path/'parent',run)


def test_incomplete_parent_field_is_not_treated_as_a_reference(tmp_path,monkeypatch):
    seed=tmp_path/'seed';seed.mkdir();(seed/'SCRTEMP').write_bytes(b'unchanged')
    run=tmp_path/'run';(run/'seq_logs').mkdir(parents=True);(run/'SCRTEMP').write_bytes(b'unchanged')
    (run/'seq_logs/whole-runtime.json').write_text(json.dumps(dict(diagnostic_complete=True,returncode=97,coverage=dict(count=197291))))
    (run/'seq_logs/whole-inputs.json').write_text(json.dumps(dict(seed=str(seed))))
    monkeypatch.setattr(response.bf,'read_eddfactor',lambda _: (np.ones((3,2)),np.arange(1,4)*1e15,2,0.))
    with pytest.raises(ValueError,match='부모 EDDFACTOR'):
        response.read_complete_pair(seed,run)


def test_time_limit_between_ions_cannot_claim_full_coverage():
    populations={(14,2):dict(n=125),(14,3):dict(n=147),(16,2):dict(n=322),(16,3):dict(n=256)}
    partial={str((14,2)):dict(levels_completed=125)}
    assert not response.target_coverage_complete(populations,partial,125)
    full={str(key):dict(levels_completed=model['n']) for key,model in populations.items()}
    assert response.target_coverage_complete(populations,full,850)
    assert not response.target_coverage_complete(populations,full,849)
    full[str((16,3))]['levels_completed']=255
    assert not response.target_coverage_complete(populations,full,850)


def test_actual_59_population_and_phot_inputs_have_same_identity():
    seed=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702')
    deck=Path('/gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc/atomic_view')
    models=response.bf.atomic.load_cmfgen_model_spec(seed/'MODEL_SPEC',seed/'atomic_links.txt')
    links=response.bf.atomic.load_cmfgen_links(seed/'atomic_links.txt')
    pop,T,ne=response.population_inputs(seed,models,links)
    manifest=json.loads((deck/'cmfgen_bf_channels_manifest.json').read_text())
    provenance=json.loads((deck/'DECK_PROVENANCE.json').read_text())['phot_preprocessing']
    smoother=response.bf.PhotSmoother.from_provenance(provenance)
    assert smoother.controls==response.read_smoothing_controls(seed/'MODEL')
    for key,nf in (((14,2),125),((14,3),147),((16,2),322),((16,3),256)):
        assert pop[key]['population'].shape==(90,nf)
        assert np.isfinite(pop[key]['population']).all()
        assert np.all(pop[key]['population']>=0)
        paths=response.bf.atomic._phot_route_paths(links[key]['phot'])
        assert [response.sha(p) for p in paths]==[r['sha256'] for r in manifest['ions']['Z%d_stage%d'%key]['sources']]


def test_actual_PHOT_channels_match_existing_gamma_without_consuming_new_field():
    seed=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702')
    deck=Path('/gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc/atomic_view')
    links=response.bf.atomic.load_cmfgen_links(seed/'atomic_links.txt')
    manifest=json.loads((deck/'cmfgen_bf_channels_manifest.json').read_text())
    channels=response.bf.parse_cmfgen_bf_channels(deck/'cmfgen_bf_channels.bin',manifest['n_levels'],manifest['n_freq_bins'],manifest)
    with (deck/'levels.csv').open() as f:levels=list(csv.DictReader(f))
    smoother=response.bf.PhotSmoother.from_provenance(json.loads((deck/'DECK_PROVENANCE.json').read_text())['phot_preprocessing'])
    nu=np.geomspace(1e13,4e18,1001);J=np.full_like(nu,1e-8)
    for key in sorted(response.bf.TARGETS):
        i=next(i for i,r in enumerate(levels) if int(r['atomic_number'])==key[0] and int(r['ion_number'])+1==key[1] and int(r['level_number'])==0)
        osc=response.bf.parse_osc(links[key]['osc'])
        terms=response.terms_for_level(links[key]['phot'],channels,i,str(osc.levels['config'][0]),0.,float(osc.ionization_eV),key[0],float(osc.z_screen),smoother)
        assert terms
        wg,wh=response.linear_sigma_weights(nu,terms)
        gamma=response.bf.raw_field_continuous_sigma_gamma(nu,J,np.array([nu[0],nu[-1]]),np.array([1]),[t[:4] for t in terms])
        assert wg@J==pytest.approx(gamma,rel=2e-13)
        assert wh@J>0


@pytest.mark.parametrize('bad',[0.,float('nan'),-1.])
def test_bad_channel_support_is_rejected(bad):
    nu=np.array([1e15,2e15,3e15])
    with pytest.raises(ValueError):
        response.linear_sigma_weights(nu,[(1.,lambda q:np.ones_like(q),[],bad,1e15)])
