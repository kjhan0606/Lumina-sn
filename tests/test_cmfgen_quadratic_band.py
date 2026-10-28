"""대역 C 연결의 방향·core/접선·청색 이력과 미완료 출력 처리를 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cmfgen_quadratic_band import build,refine,plan,collect
import compare_cmfgen_quadratic_band as band_module
from analyze_cmfgen_spatial_defects import QuadraticBlueKernel


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    out=tmp_path_factory.mktemp('quadratic-band')
    band,_,_=build(out)
    return band,QuadraticBlueKernel(out/'quadratic-band.so')


def inputs():
    a=dict(ptr=np.array([0,4,7],dtype='i4'),obs=np.array([1,1],dtype='i4'),nc=1,
        lm=np.array([.7,.9,1.1,0.,.8,1.,0.]),lp=np.array([1.2,.8,.6,0.,1.1,.7,0.]),
        bm=np.linspace(.01,.06,7),bp=np.linspace(.02,.07,7),
        wp=np.array([.3,.2]),wm=np.array([.25,.25]),
        initial_m=np.linspace(.1,.4,7),initial_p=np.linspace(.2,.5,7))
    chi=np.array([[.2,.4,.6,.8],[.3,.5,.7,.9],[.4,.6,.8,1.]])
    eta=np.array([[.1,.4,.5,.8],[.2,.5,.6,.9],[.3,.6,.7,1.]])
    h=np.array([.02,.03,.04]);inner=np.full((3,2),.8);outer=np.full((3,2),.12)
    return a,chi,eta,h,inner,outer


def test_compiled_band_matches_independent_python_sequence(kernels):
    band,cell=kernels;a,chi,eta,h,inner,outer=inputs();expected=[]
    old_m=a['initial_m'].copy();old_p=a['initial_p'].copy()
    original_m=old_m.copy();original_p=old_p.copy()
    for f in range(3):
        new_m=np.zeros(7);new_p=np.zeros(7);value=0.
        for r in range(2):
            off,end=map(int,a['ptr'][r:r+2]);n=end-off
            new_m[off:end]=cell.leg(a['lm'][off:end-1],chi[f,:n],eta[f,:n],
                a['bm'][off:end],old_m[off:end],1/h[f],outer[f,r])
            boundary=inner[f,r] if r<a['nc'] else new_m[end-1]
            new_p[off:end]=cell.leg(a['lp'][off:end-1],chi[f,:n][::-1],eta[f,:n][::-1],
                a['bp'][off:end],old_p[off:end],1/h[f],boundary)
            value+=a['wm'][r]*new_m[off+a['obs'][r]]+a['wp'][r]*new_p[end-1-a['obs'][r]]
        expected.append(value);old_m,old_p=new_m,new_p
    code,result,stats=band.advance(a,chi,eta,h,inner,outer,seconds=2.)
    assert code==0 and stats['completed']==3 and stats['kernel_status']==0
    np.testing.assert_allclose(result,expected,rtol=2e-14)
    np.testing.assert_array_equal(a['initial_m'],original_m)
    np.testing.assert_array_equal(a['initial_p'],original_p)


def test_timeout_has_no_false_completed_frequency(kernels):
    band,_=kernels;a,chi,eta,h,inner,outer=inputs()
    code,result,stats=band.advance(a,chi,eta,h,inner,outer,seconds=1e-12)
    assert code==4 and stats['completed']==0 and np.isnan(result).all()


def test_unsupported_effective_gain_stays_incomplete(kernels):
    band,_=kernels;a,chi,eta,h,inner,outer=inputs()
    chi[:]=-100.
    code,result,stats=band.advance(a,chi,eta,h,inner,outer,seconds=2.)
    assert code==8 and stats['kernel_status']==4 and stats['completed']==0
    assert np.isnan(result).all()


def test_invalid_array_shape_fails_before_native_call(kernels):
    band,_=kernels;a,chi,eta,h,inner,outer=inputs()
    with pytest.raises(ValueError):band.advance(a,chi,eta[:,:2],h,inner,outer)


def test_refinement_preserves_every_native_value_and_linear_material():
    source=np.array([[1.,4.,2.],[2.,3.,5.]])
    for pieces in (1,2,4):
        fine=refine(source,pieces)
        np.testing.assert_array_equal(fine[:,::pieces],source)
        assert fine.shape==(2,2*pieces+1)
    with pytest.raises(ValueError):refine(source,3)


def test_static_transport_is_invariant_under_source_preserving_refinement(kernels):
    band,_=kernels;g=dict(NP=2,NC=1,R_EXT=np.array([4.,3.,2.,1.]),rays=[])
    initial=[]
    for n in (4,3):
        g['rays'].append(dict(R=g['R_EXT'][:n],LNK=np.array([2]),
            S_M=np.arange(n,dtype=float),S_P=-np.arange(n,dtype=float),
            B_M=np.zeros(n),B_P=np.zeros(n)))
        initial.append(dict(I_M=np.full(n,.2),I_P=np.full(n,.3)))
    g['weights']={'JQW_P':np.array([[.3,.2]]),'JQW_M':np.array([[.25,.25]])}
    chi=np.array([[.2,.4,.6,.8],[.3,.5,.7,.9]])
    eta=np.array([[.1,.2,.5,.8],[.4,.5,.6,.7]])
    outputs=[]
    for pieces in (1,2,4):
        a=plan(g,initial,0,pieces)
        code,result,_=band.advance(a,refine(chi,pieces),refine(eta,pieces),[.02,.03],
            np.full((2,2),.8),np.full((2,2),.12),seconds=2.)
        assert code==0
        outputs.append(result)
    for result in outputs[1:]:np.testing.assert_allclose(result,outputs[0],rtol=3e-14)


def test_intentional_prefix_keeps_last_source_not_first_JIT(monkeypatch,tmp_path):
    import time
    rows=[]
    for i in range(3):
        for jit in (0,1):
            rows.append(dict(index=band_module.FIRST+i,jit=jit,new_freq=jit==0,freq=5.-i*.01,
                dlognu=.01,chi_ext=np.ones(4),eta_ext=np.full(4,float(jit)),
                boundary=[(1.,0.,.8,.4)],native=np.ones(4)))
    monkeypatch.setattr(band_module,'records',lambda *args,**kwargs:iter(rows))
    chosen,calls=collect(tmp_path,{},0,time.monotonic()+1,count=2)
    assert len(chosen)==2 and calls==4
    assert all(np.all(row['eta_ext']==1.) for row in chosen)


def test_first_frequency_keeps_material_integral_but_later_history_resolves_curvature(kernels):
    band,_=kernels;g=dict(NP=2,NC=1,R_EXT=np.array([4.,3.,2.,1.]),rays=[])
    initial=[]
    for n in (4,3):
        g['rays'].append(dict(R=g['R_EXT'][:n],LNK=np.array([2]),
            S_M=np.arange(n,dtype=float),S_P=-np.arange(n,dtype=float),
            B_M=np.array([.07,.1,.12,.18])[:n],B_P=np.array([.05,.09,.14,.2])[:n]))
        initial.append(dict(I_M=np.array([.1,.15,.12,.2])[:n],I_P=np.array([.4,.3,.2,.1])[:n]))
    g['weights']={'JQW_P':np.array([[.3,.2]]),'JQW_M':np.array([[.25,.25]])}
    chi=np.array([[.2,.4,.6,.8]]*3);eta=np.array([[.1,.2,.5,.8]]*3);outputs=[]
    for pieces in (1,2,4):
        code,result,_=band.advance(plan(g,initial,0,pieces),refine(chi,pieces),refine(eta,pieces),
            [.02,.03,.04],np.full((3,2),.8),np.full((3,2),.12),seconds=2.)
        assert code==0;outputs.append(result)
    assert outputs[0][0]==pytest.approx(outputs[2][0],rel=3e-12)
    assert abs(outputs[0][-1]-outputs[2][-1])>1e-5
