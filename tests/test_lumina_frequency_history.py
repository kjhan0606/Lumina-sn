"""양의 모멘트 합성/linear logν/큰 이동/청색 연장 직접 대조."""
import json
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import lumina_frequency_history as H
import lumina_history_spatial as W


@pytest.fixture(scope='module')
def history(tmp_path_factory):
    library = tmp_path_factory.mktemp('frequency-history')/'history.so'
    candidate=['-DCMF_EXPQUAD_ENGINE'] if os.environ.get('LUMINA_EXPQUAD_ENGINE')=='1' else []
    subprocess.run([os.environ.get('CC', 'cc'), '-O2',*candidate, '-std=c11', '-Wall', '-Wextra', '-Werror',
        '-fPIC', '-shared', str(ROOT/'src/cmf_frequency_history.c'), '-lm', '-o', str(library)],
        check=True, capture_output=True, timeout=60)
    return H.Kernel(library)


def test_initialized_cut_prefix_matches_sealed_kernel_exactly(history):
    if os.environ.get('LUMINA_EXPQUAD_ENGINE')=='1':pytest.skip('구적 교체는 수치 비교이며 이전 byte 동일 시험과 다르다')
    old_path=os.environ.get('LUMINA_BASIS_OLD_LIBRARY')
    if old_path is None:pytest.skip('수정 전 봉인 커널이 지정되지 않음')
    old=H.Kernel(old_path)
    pairs=[(0.,0.),(.1,.2),(1.,2.),(-2.,3.),(3.,-2.),(-40.,-10.),
           (1e6,.1),(.1,1e6),(1e-16,2e-16),(1e100,2e100)]
    rng=np.random.default_rng(1021)
    pairs.extend(zip(10.**rng.uniform(-8,5,96),10.**rng.uniform(-8,5,96)))
    for q0,q1 in pairs:
        results=[]
        for kernel in (old,history):
            value,stats=H.Map(),H.Stats()
            status=kernel._map(1.,1.,float(q0),float(q1),.2,.7,.3,1.2,
                ct.byref(value),ct.byref(stats))
            results.append((status,bytes(value),bytes(stats)))
        assert results[0]==results[1]


@pytest.mark.parametrize('q0,q1', [(0.,0.), (1e-12,-1e-12), (.001,.001), (1.,1.), (-2.,-2.),
                                   (-2.,3.), (3.,-2.), (1e5,1e5), (1e5,3e5), (0.,1e5), (1e5,0.)])
def test_positive_moments_against_independent_integrals(history, q0, q1):
    ea0, ea1, eb0, eb1 = .2, .7, .3, 1.2
    mapping = history.map(1., 1., q0, q1, ea0, ea1, eb0, eb1)
    if max(abs(q0),abs(q1)) < 10:
        x, w = np.polynomial.legendre.leggauss(128); u=.5*(x+1); weights=.5*w
        attenuation = np.exp(-u*(q1+.5*(q0-q1)*u))
        a = ea0*u+ea1*(1-u); b=eb0*u+eb1*(1-u)
        exact = [1.,np.exp(-.5*(q0+q1)),np.sum(weights*attenuation*a),
            np.sum(weights*attenuation*u*a),np.sum(weights*attenuation*b),np.sum(weights*attenuation*(1-u)*b)]
        np.testing.assert_allclose(H.values(mapping), exact, rtol=2e-11, atol=1e-15)
    else:
        # η(s)=χ(s)S에서 source의 정확한 두꺼운 극한/균형을 확인한다.
        source = history.map(1., 1., q0, q1, q0, q1, q0, q1)
        np.testing.assert_allclose(source.pa+source.mb, 1-np.exp(-.5*(q0+q1)), rtol=2e-11, atol=0)
    assert np.all(H.values(mapping) >= 0)


def test_time_moment_composition_is_associative_without_subtracting_prefixes(history):
    pieces = [history.map(.2,1.,.3,1.,.2,.4,.5,.8),
              history.map(.3,1.,1.,-.2,.4,.7,.8,1.),
              history.map(.5,1.,-.2,.4,.7,.9,1.,.1)]
    a=history.compose(history.compose(pieces[0],pieces[1]),pieces[2])
    b=history.compose(pieces[0],history.compose(pieces[1],pieces[2]))
    np.testing.assert_allclose(H.values(a), H.values(b), rtol=2e-15, atol=0)


@pytest.mark.parametrize('q0,q1', [(1e-300,2e-300), (1e-12,-2e-12), (.001,.002),
    (.099,-.099), (-.099,.099), (.099,.098), (-.099,-.098), (.1,.03), (.100001,-.07)])
def test_thin_unequal_opacity_moments_keep_independent_integrals_and_fallback(history, q0, q1):
    nodes, weights = np.polynomial.legendre.leggauss(128)
    u=.5*(nodes+1); weights=.5*weights
    attenuation=np.exp(-u*(q1+.5*(q0-q1)*u))
    expected=np.array([np.sum(weights*attenuation*u*u),
        np.sum(weights*attenuation*u*(1-u)), np.sum(weights*attenuation*(1-u)**2)])
    mapping=H.Map();stats=H.Stats()
    assert history._map(1.,1.,q0,q1,1.,0.,0.,1.,ct.byref(mapping),ct.byref(stats))==0
    # 이 endpoint source 선택에서 pa=w0, mb=w2이며 aa+ab로 w1도 분리된다.
    actual=np.array([mapping.pa, mapping.aa-mapping.pa, mapping.mb])
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-15)
    assert np.all(actual>0)
    if max(abs(q0),abs(q1))<.1:
        assert stats.quadrature_nodes==0
    elif os.environ.get('LUMINA_EXPQUAD_ENGINE')!='1':
        assert stats.quadrature_nodes>0


@pytest.mark.parametrize('tau,drift,offset', [(0.,0.,0.), (1.,.13,0.), (-2.,.03,.07), (1e5,.2,0.)])
def test_gray_spectrum_and_linear_time_source_have_the_exact_homogeneous_solution(history,tau,drift,offset):
    nf=33;length=1e12;initial=np.ones(nf)*.4
    value,_=history.apply(.01,length,drift+offset,offset,np.full(nf,tau/length),
        np.full(nf,.2/length),np.full(nf,1.3/length),initial)
    q=tau+3*drift
    if q==0:exact=.4+.5*(.2+1.3)
    else:
        total=-np.expm1(-q)/q;right=(1-total)/q
        exact=np.exp(-q)*.4+(total-right)*.2+right*1.3
    np.testing.assert_allclose(value,exact,rtol=3e-11,atol=1e-15)


def test_spectral_source_and_opacity_match_an_independent_history_integral(history):
    nf=33;h=.02;length=1e12;offset=.375;span=11.25;x=np.arange(nf)
    chi=-.1+.4*np.sin(x*.31)**2;ea=.2+np.sin(x*.5)**2;eb=.1+np.cos(x*.23)**2
    initial=1+np.sin(x*.1)**2;end=x+offset
    prefix=np.r_[0.,np.cumsum(.5*(chi[:-1]+chi[1:]))]
    def integral(q):
        clipped=np.minimum(q,nf-1);j=np.minimum(np.floor(clipped).astype('i8'),nf-2)
        fraction=clipped-j
        return prefix[j]+fraction*chi[j]+.5*fraction*fraction*(chi[j+1]-chi[j])+np.maximum(q-(nf-1),0)*chi[-1]
    def opacity_depth(q,t):return (integral(q)-integral(end))/span+3*span*h*(1-t)
    exact=np.exp(-opacity_depth(end+span,0.))*initial
    cuts=np.unique(np.r_[0.,[(offset+span-k)/span for k in range(1,50) if 0<(offset+span-k)/span<1],1.])
    nodes,weights=np.polynomial.legendre.leggauss(48)
    for a,b in zip(cuts[:-1],cuts[1:]):
        t=.5*(a+b)+.5*(b-a)*nodes
        q=end[None,:]+span*(1-t[:,None])
        source=(1-t[:,None])*np.interp(q,x,ea)+t[:,None]*np.interp(q,x,eb)
        exact+=np.sum(.5*(b-a)*weights[:,None]*np.exp(-opacity_depth(q,t[:,None]))*source,axis=0)
    value,_=history.apply(h,length,(offset+span)*h,offset*h,chi/length,ea/length,eb/length,initial)
    np.testing.assert_allclose(value,exact,rtol=3e-11,atol=1e-15)


@pytest.mark.parametrize('shift',[0.,.125,.5,3.75,24.,512.])
def test_compiled_eight_point_reconstruction_preserves_the_existing_method(history,shift):
    import lumina_semi_lagrangian as S
    x=np.arange(512);initial=2*np.exp(-((x-256)/3.)**2)
    reference,_=S.reconstruct(initial,shift)
    value,report=history.reconstruct(initial,shift)
    np.testing.assert_allclose(value,reference,rtol=2e-12,atol=1e-300)
    assert np.all(value>=0) and report['reconstructions']==1


@pytest.mark.parametrize('offset,span', [(0.,0.), (.25,0.), (0.,.125), (.375,.125), (.75,.5),
    (0.,1.), (.25,1.5), (.75,7.25), (2.375,11.75), (30.,100.), (70.,2.)])
def test_rolling_matches_direct_bin_composition_with_source_and_signed_chi(history, offset, span):
    nf=65; h=.01; length=1e12; x=np.arange(nf)
    chi=(-.2+.5*np.sin(x*.3)**2)/length
    eta_a=(.2+np.sin(x*.4)**2)/length; eta_b=(.1+np.cos(x*.2)**2)/length
    initial=1+np.sin(x*.1)**2
    direct,_=history.apply(h,length,(offset+span)*h,offset*h,chi,eta_a,eta_b,initial,naive=True)
    for width in (1,7,256):
        value,report=history.apply(h,length,(offset+span)*h,offset*h,chi,eta_a,eta_b,initial,slab_width=width)
        np.testing.assert_allclose(value,direct,rtol=3e-12,atol=1e-15)
        assert report['full_bin_maps'] <= nf-1
        assert report['interior_intensity_remappings'] == 0


def test_large_shift_narrow_opacity_is_not_missed(history,tmp_path):
    nf=1025; h=1.1267110173509078e-5; span=256; length=1e12
    x=np.arange(nf); profile=1000*np.exp(-((x-512)/3.)**2)
    prefix=np.r_[0.,np.cumsum(.5*(profile[:-1]+profile[1:]))]
    depth=(prefix[np.minimum(x+span,nf-1)]-prefix[x])/span
    exact=2*np.exp(-depth-3*span*h)
    value,report=history.apply(h,length,span*h,0.,profile/length,np.zeros(nf),np.zeros(nf),np.full(nf,2.))
    error=float(abs(value-exact).max()/exact.max())
    (tmp_path/'frequency-history-large-shift.json').write_text(json.dumps(dict(
        value=value.tolist(),reference=exact.tolist(),max_norm_error=error,report=report,
        actual_model_F_calls=0,final_goal='NOT_JUDGED'),indent=2,allow_nan=False))
    assert error < 1e-10
    assert report['full_bin_maps'] <= nf-1 and report['quadrature_nodes'] == 0


def test_full_frequency_count_uses_one_compiled_scan_and_bounded_window(history,tmp_path):
    nf=1115020;h=1.1267110173509078e-5;shift=1887.1974459832727;length=1e12
    initial=np.full(nf,2.);chi=np.full(nf,.7/length)
    eta_a=np.full(nf,.2/length);eta_b=np.full(nf,1.3/length)
    value,report=history.physical(h,length,shift*h,chi,eta_a,eta_b,initial,slab_width=8192)
    q=.7+3*shift*h;total=-np.expm1(-q)/q;right=(1-total)/q
    exact=2*np.exp(-q)+(total-right)*.2+right*1.3
    error=float(np.max(abs(value-exact))/exact)
    assert error < 1e-10 and report['full_bin_maps'] <= nf-1
    assert report['queue_peak'] <= np.ceil(shift)+1
    (tmp_path/'frequency-history-full-count.json').write_text(json.dumps(dict(
        frequencies=nf,shift_bins=shift,coefficient_origin='synthetic gray; actual grid count/shift scale only',
        max_norm_error=error,first_value=float(value[0]),last_value=float(value[-1]),reference=exact,
        numeric_arrays_bytes=sum(v.nbytes for v in (initial,chi,eta_a,eta_b,value)),
        report=report,actual_model_F_calls=0,physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED'),indent=2,allow_nan=False))


def test_spatial_batch_preserves_moving_grid_and_propagates_signed_gain_response(history):
    count=3;nf=65;h=.01;length=1e12;x=np.arange(nf)
    lengths=length*np.array([[1.,.4,.6],[.7,.2,.5],[.3,.1,.2]])
    deltas=h*np.array([[17.,14.,11.],[11.,8.25,4.],[4.,1.5,0.]])
    chi=np.array([[(-.8+.2*np.sin(x*.3+row+node)**2)/length for node in range(3)] for row in range(count)])
    eta=np.array([[(.1+np.sin(x*.2+row+node)**2)/length for node in range(3)] for row in range(count)])
    initial=1+np.sin(x*.1)**2;preserved=initial.copy();value=initial.copy()
    errors=[];responses=[];zero=np.zeros(nf)
    for row in range(count):
        da,dm,db=deltas[row];whole,left,right=lengths[row]
        coarse,_=history.apply(h,whole,da,db,chi[row,0],eta[row,0],eta[row,2],value,naive=True)
        first,_=history.apply(h,left,da,dm,chi[row,1],eta[row,0],eta[row,1],value,naive=True)
        value,_=history.apply(h,right,dm,db,chi[row,2],eta[row,1],eta[row,2],first,naive=True)
        response,_=history.apply(h,left,da,dm,chi[row,1],zero,zero,np.ones(nf),naive=True)
        response,_=history.apply(h,right,dm,db,chi[row,2],zero,zero,response,naive=True)
        errors.append(abs(value-coarse));responses.append(response)
    result,error,response,report=history.compare(h,lengths,deltas,chi,eta,initial)
    np.testing.assert_allclose(result,value,rtol=3e-12,atol=1e-15)
    np.testing.assert_allclose(error,errors,rtol=3e-10,atol=2e-14)
    np.testing.assert_allclose(response,responses,rtol=3e-12,atol=1e-15)
    np.testing.assert_array_equal(initial,preserved)
    assert np.any(response>1) and report['full_bin_maps']<=3*count*(nf-1)


def test_batch_zero_length_and_gray_manufactured_solution(history):
    nf=8;initial=np.full(nf,.4)
    lengths=np.array([[0.,0.,0.],[1.,.25,.75]])
    deltas=np.array([[.13,.13,.13],[.13,.1175,.08]])
    chi=np.ones((2,3,nf))*.7;eta=np.empty_like(chi)
    eta[0]=0.;eta[1]=np.array([.2,.475,1.3])[:,None]
    value,error,response,_=history.compare(.01,lengths,deltas,chi,eta,initial)
    q=.7+3*.05;total=-np.expm1(-q)/q;right=(1-total)/q
    exact=.4*np.exp(-q)+(total-right)*.2+right*1.3
    np.testing.assert_allclose(value,exact,rtol=3e-12,atol=1e-15)
    np.testing.assert_allclose(error,0.,atol=1e-15)
    np.testing.assert_allclose(response[0],1.,rtol=0.,atol=0.)
    np.testing.assert_allclose(response[1],np.exp(-q),rtol=3e-15,atol=0.)


def test_batch_rejects_nonmonotone_geometry_and_large_leaf_arrays(history):
    with pytest.raises(ValueError):
        history.compare(.01,np.ones((1,3)),np.array([[0.,1.,0.]]),
            np.zeros((1,3,8)),np.zeros((1,3,8)),np.ones(8))
    with pytest.raises(ValueError):
        history.compare(.01,np.ones((4097,3)),np.ones((4097,3)),
            np.zeros((1,3,8)),np.zeros((1,3,8)),np.ones(8))


@pytest.mark.parametrize('panels',[1,8,16])
def test_spatial_refinement_recovers_strong_opacity_gradient_thick_equilibrium(history,tmp_path,panels):
    length=1e12;nf=8;source=.7;tau0=1e5;tau1=3e5
    def coeff(x):
        chi=np.full(nf,(tau0+(tau1-tau0)*x)/length)
        return chi,source*chi
    value,report=W.propagate(history,.01,np.full(nf,.1),lambda a,b:(length*(b-a),0.),
        coeff,lambda x:0.,panels=panels,rtol=1e-6)
    error=float(abs(value-source).max()/source)
    assert error < 1e-5 and report['reconstructions']==1
    assert report['interior_intensity_remappings']==0 and report['spatial_leaves']>=panels
    (tmp_path/'frequency-history-spatial-thick.json').write_text(json.dumps(dict(
        tau_start=tau0,tau_end=tau1,source=source,value=value.tolist(),max_norm_error=error,
        report=report,actual_model_F_calls=0,final_goal='NOT_JUDGED'),indent=2,allow_nan=False))


@pytest.mark.parametrize('tau',[-2.,0.,1e-5,1.,1e5])
def test_spatial_history_preserves_homogeneous_manufactured_solution_without_remapping(history,tau):
    length=1e12;nf=8;drift=.13;initial=np.full(nf,.4)
    value,report=W.propagate(history,.01,initial,lambda a,b:(length*(b-a),drift*(b-a)),
        lambda x:(np.full(nf,tau/length),np.full(nf,(.2+1.1*x)/length)),
        lambda x:drift*(1-x),panels=8)
    q=tau+3*drift;total=-np.expm1(-q)/q;right=(1-total)/q
    exact=.4*np.exp(-q)+(total-right)*.2+right*1.3
    np.testing.assert_allclose(value,exact,rtol=3e-11,atol=1e-15)
    assert report['rounds']==1 and report['formal_vectors']==24+report['branch_checks']
    np.testing.assert_array_equal(initial,np.full(nf,.4))


def test_spatial_resource_failure_is_not_an_accepted_field(history):
    with pytest.raises(W.HistoryLimit) as caught:
        W.propagate(history,.01,np.full(8,.1),lambda a,b:(b-a,0.),
            lambda x:(np.full(8,100.),np.full(8,70.*(1+x*x))),lambda x:0.,
            panels=1,max_leaves=1)
    assert caught.value.reason=='max_leaves'
    assert caught.value.report['accepted_physical_field'] is False
    with pytest.raises(ValueError):
        W.propagate(history,.01,np.ones(1025),lambda a,b:(b-a,0.),
            lambda x:(np.zeros(1025),np.zeros(1025)),lambda x:0.)


@pytest.mark.parametrize('offset,span',[(0.,0.),(.4,.9),(.7,1.25),(31.,.9)])
def test_small_spatial_projection_matches_independent_piece_integrals(history,offset,span):
    nf=33;h=.02;length=1e12;x=np.arange(nf)
    ca=-.1+.3*np.sin(x*.31)**2;cb=.1+.5*np.cos(x*.27)**2
    ea=.2+np.sin(x*.5)**2;eb=.1+np.cos(x*.23)**2;initial=.4+np.sin(x*.1)**2
    nodes,weights=np.polynomial.legendre.leggauss(64);exact=[];transmissions=[]
    for i in range(nf):
        start=i+offset+span;end=i+offset;value=initial[i];transmission=1.
        cuts=sorted(set([0.,1.]+[(start-k)/span for k in range(nf) if span and end<k<start]))
        def at(t):
            q=start-span*t
            return ((1-t)*np.interp(q,x,ca)+t*np.interp(q,x,cb)+3*span*h,
                    (1-t)*np.interp(q,x,ea)+t*np.interp(q,x,eb))
        for a,b in zip(cuts[:-1],cuts[1:]):
            c0,e0=at(a);c1,e1=at(b);width=b-a
            u=.5*(nodes+1);phase=-width*u*(c1+.5*(c0-c1)*u)
            source=width*np.sum(.5*weights*np.exp(phase)*(u*e0+(1-u)*e1))
            t=np.exp(-.5*(c0+c1)*width);value=t*value+source;transmission*=t
        exact.append(value);transmissions.append(transmission)
    value,response,_=history.small(h,length,(offset+span)*h,offset*h,ca/length,cb/length,ea/length,eb/length,initial)
    np.testing.assert_allclose(value,exact,rtol=3e-11,atol=1e-15)
    np.testing.assert_allclose(response,transmissions,rtol=3e-14,atol=1e-15)


@pytest.mark.parametrize('span',[.74,.75,.99,1.,1.01,1.25,1.26,2.])
def test_hybrid_branch_is_counted_and_disagreement_is_not_hidden(history,span):
    nf=8;h=.02;x=np.array([0.,.25,.5,.75,1.])
    chi=np.broadcast_to((1+2*x)[None,:,None],(1,5,nf)).copy()
    eta=np.broadcast_to((.7*(1+2*x[[0,2,4]]))[None,:,None],(1,3,nf)).copy()
    initial=np.full(nf,.1);lengths=np.array([[1.,.5,.5]])
    deltas=np.array([[span*h,.5*span*h,0.]])
    value,error,response,report=history.compare_hybrid(h,lengths,deltas,chi,eta,initial)
    pieces=np.array([span,span*.5,span*.5]);checks=int(np.count_nonzero((pieces>=.75)&(pieces<=1.25)))
    assert report['branch_checks']==checks and report['formal_vectors']==3+checks
    assert np.isfinite(value).all() and np.all(error>=0) and np.isfinite(response).all()
    if checks:assert error.max()>1e-5


@pytest.mark.parametrize('span',[.99,1.01])
def test_adaptive_hybrid_sides_approach_same_manufactured_equation(history,span):
    h=.02;nf=8;drift=span*h
    # η=S(χ+3a)면 I=S가 정확한 이동 특성선 평형이다.
    gamma=3*drift;source=.7;depth=2.+gamma
    exact=source+(.1-source)*np.exp(-depth)
    value,report=W.propagate(history,h,np.full(nf,.1),lambda a,b:(b-a,drift*(b-a)),
        lambda x:(np.full(nf,1+2*x),np.full(nf,source*(1+2*x+gamma))),
        lambda x:drift*(1-x),panels=1,rtol=1e-6)
    np.testing.assert_allclose(value,exact,rtol=1e-5,atol=1e-14)
    assert report['branch_checks']>0 and report['formal_vectors']>3


def test_space_frequency_cross_term_is_refined_against_independent_cubic_exponent(history):
    nf=33;h=.02;span=1.01;x=np.arange(nf)
    ca=-.1+.3*np.sin(x*.31)**2;cb=.1+.5*np.cos(x*.27)**2
    ea=.2+np.sin(x*.5)**2;eb=.1+np.cos(x*.23)**2;initial=np.full(nf,.4)
    nodes,weights=np.polynomial.legendre.leggauss(64);u=.5*(nodes+1);exact=[]
    for i in range(nf):
        start=i+span;end=float(i);value=initial[i]
        cuts=sorted(set([0.,1.]+[(start-k)/span for k in range(nf) if end<k<start]))
        def at(t):
            q=start-span*t
            return ((1-t)*np.interp(q,x,ca)+t*np.interp(q,x,cb)+3*span*h,
                    (1-t)*np.interp(q,x,ea)+t*np.interp(q,x,eb))
        for a,b in zip(cuts[:-1],cuts[1:]):
            width=b-a;c0,_=at(a);cm,_=at(.5*(a+b));c1,_=at(b)
            linear=4*cm-3*c0-c1;quadratic=2*(c1+c0-2*cm)
            total=width*(c0+.5*linear+quadratic/3)
            depth=width*(c0*(1-u)+.5*linear*(1-u*u)+quadratic/3*(1-u*u*u))
            _,source=at(a+width*u)
            value=np.exp(-total)*value+width*np.sum(.5*weights*np.exp(-depth)*source)
        exact.append(value)
    value,report=W.propagate(history,h,initial,lambda a,b:(b-a,span*h*(b-a)),
        lambda t:((1-t)*ca+t*cb,(1-t)*ea+t*eb),lambda t:span*h*(1-t),panels=1,rtol=1e-7)
    np.testing.assert_allclose(value,exact,rtol=1e-6,atol=1e-12)
    assert report['spatial_leaves']>1 and report['branch_checks']>0


@pytest.mark.parametrize('background,amplitude',[(2.,-.8),(2.,2.),(0.,2.)])
def test_same_24_bin_narrow_line_survives_256_noninteger_physical_intervals(history,background,amplitude):
    nf=512;h=1.1267110173509078e-5;x=np.arange(nf);shift=24/256
    value=background+amplitude*np.exp(-((x-256)/3.)**2)
    exact=np.exp(-3*24*h)*(background+amplitude*np.exp(-((x+24-256)/3.)**2))
    zero=np.zeros(nf);limited=0
    for _ in range(256):
        value,report=W.propagate(history,h,value,lambda a,b:(b-a,shift*h*(b-a)),
            lambda x:(zero,zero),lambda x:shift*h*(1-x),panels=1)
        limited+=report['limited_polynomials']
        assert report['reconstructions']==1 and report['interior_intensity_remappings']==0
        assert np.all(value>=0)
    assert float(abs(value-exact).max()/exact.max())<=.01
    if background==0:assert limited>0


def test_adaptive_hybrid_keeps_the_same_256_bin_strong_opacity_history(history):
    nf=512;h=1.1267110173509078e-5;span=256.;x=np.arange(nf)
    chi=1000*np.exp(-((x-256)/3.)**2);zero=np.zeros(nf);initial=np.full(nf,2.)
    prefix=np.r_[0.,np.cumsum(.5*(chi[:-1]+chi[1:]))]
    depth=(prefix[np.minimum(x+256,nf-1)]-prefix[x])/span
    exact=2*np.exp(-depth-3*span*h)
    value,report=W.propagate(history,h,initial,lambda a,b:(b-a,span*h*(b-a)),
        lambda x:(chi,zero),lambda x:span*h*(1-x),panels=1)
    np.testing.assert_allclose(value,exact,rtol=1e-10,atol=1e-14)
    assert report['reconstructions']==1 and report['interior_intensity_remappings']==0


@pytest.mark.parametrize('bad', ['negative_eta','nonfinite_chi','overflow_gain'])
def test_invalid_inputs_do_not_return_an_accepted_field(history,bad):
    chi=np.zeros(8);eta=np.ones(8);initial=np.ones(8)
    if bad=='negative_eta':eta[0]=-1
    elif bad=='nonfinite_chi':chi[0]=np.nan
    else:chi[:]=-1e5
    with pytest.raises(ValueError):history.apply(.1,1.,.1,0.,chi,eta,eta,initial)
