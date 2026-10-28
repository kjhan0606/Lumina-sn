"""실제 선 χ/η와 BB율의 이산 에너지 일치. 격자/최종해 수렴 주장은 아니다."""
import ctypes as C
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import cmfgen_model_continuum as B
from lumina_model_linear_lines import LinearModelLines, SourceModelLines
from test_cmfgen_model_sources import kernels, line


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    path=tmp_path_factory.mktemp('sampled-lines')/'lines.so'
    root=Path(__file__).resolve().parents[1]
    subprocess.run([os.environ.get('CC','gcc'),'-O2','-std=c11','-Wall','-Wextra','-Werror',
                    '-fPIC','-shared',str(root/'src/model_line_linear_jbar.c'),'-lm','-o',str(path)],check=True)
    return path


def provider(library,nu,j,*,lines=None,width=.01,support=.06):
    return SourceModelLines(line(100.) if lines is None else lines,
        dict(width_fraction=width,support_fraction=support),library,nu,j,radiation_generation=1)


def weights(nu):
    w=np.zeros(len(nu));w[:-1]+=.5*np.diff(nu);w[1:]+=.5*np.diff(nu)
    return w


def test_same_source_spectrum_energy_for_many_depths_signed_opacity_and_irregular_grid(library,kernels):
    nu=np.array([90.,94.,96.,98.1,99.9,100.,100.7,102.,105.,106.,110.])*1.e13
    lines=line(1.e15);profile=dict(width_fraction=.01,support_fraction=.06)
    pop=np.array([[0.,2.,10.,1.],[4.,4.,1.,20.]])*1.e7
    j=np.column_stack((np.linspace(1.,8.,len(nu)),np.zeros(len(nu)),
                       np.array([.1,9.,.2,8.,.3,7.,.4,6.,.5,5.,.6]),np.ones(len(nu))))*1.e-6
    p=provider(library,nu,j,lines=lines);v=p.contract(0,1,radiation_generation=1);r=p.rates(0,1,v)
    chi,eta=kernels.lines(nu,pop,lines,profile)
    actual=4*np.pi*np.sum((chi*j.T-eta)*weights(nu),axis=1,dtype=np.longdouble)
    expected=B.H*lines['nu0'][0]*(pop[0]*r['absorption'][0]-pop[1]*(r['stimulated'][0]+r['spontaneous'][0]))
    gross=B.H*lines['nu0'][0]*(pop[0]*r['absorption'][0]+pop[1]*(r['stimulated'][0]+r['spontaneous'][0]))
    np.testing.assert_array_less(abs(actual-expected),3.e-15*gross)
    assert np.all(chi[0,1:-1]<0) and np.all(chi[1]==0) and np.all(chi[3,1:-1]<0)
    np.testing.assert_array_equal(chi[:,[0,-1]],0.)
    # 이전 분석적 J 적분은 같은 비균일 선 스펙트럼과 이산 내적이 아니다.
    old=LinearModelLines(lines,profile,library,nu,j,radiation_generation=1)
    old_v=old.contract(0,1,radiation_generation=1);old_r=old.rates(0,1,old_v)
    old_net=B.H*lines['nu0'][0]*(pop[0]*old_r['absorption'][0]-pop[1]*(old_r['stimulated'][0]+old_r['spontaneous'][0]))
    assert np.max(abs(actual-old_net)/gross)>1.e-3
    assert p.signature!=old.signature
    with pytest.raises(ValueError):p.rates(0,1,old_v)


def test_overlapping_lines_close_energy_at_90_depths(library,kernels):
    rng=np.random.default_rng(29);nt=7;ns=90
    nu=np.unique(np.r_[np.linspace(.85e15,1.15e15,71),.999e15,1.e15,1.001e15])
    centers=np.linspace(.98e15,1.02e15,nt);f=rng.uniform(.01,.9,nt);ratio=rng.uniform(.2,2.,nt)
    lines=dict(lower=np.arange(nt,dtype='i4')*2,upper=np.arange(nt,dtype='i4')*2+1,
        nu0=centers,f=f,g_ratio=ratio,A=B.atomic.A_PREFACTOR*f*ratio*centers**2)
    pop=rng.uniform(0.,1.e8,(2*nt,ns));j=rng.uniform(0.,1.e-5,(len(nu),ns))
    p=provider(library,nu,j,lines=lines);v=p.contract(0,nt,radiation_generation=1);r=p.rates(0,nt,v)
    chi,eta=kernels.lines(nu,pop,lines,p.profile)
    actual=4*np.pi*np.sum((chi*j.T-eta)*weights(nu),axis=1,dtype=np.longdouble)
    incoming=pop[lines['lower']]*r['absorption']
    outgoing=pop[lines['upper']]*(r['stimulated']+r['spontaneous'][:,None])
    net=np.sum((incoming-outgoing)*(B.H*centers)[:,None],axis=0,dtype=np.longdouble)
    gross=np.sum((incoming+outgoing)*(B.H*centers)[:,None],axis=0,dtype=np.longdouble)
    np.testing.assert_array_less(abs(actual-net),3.e-14*gross)


def test_profile_mass_scales_spontaneous_without_renormalizing_jbar(library):
    p=provider(library,np.array([90.,99.,100.,101.,110.]),np.full((5,3),3.),support=.01)
    v=p.contract(0,1,radiation_generation=1)
    np.testing.assert_allclose(v['jbar'],3*v['profile_mass'][:,None]*np.ones((1,3)),rtol=2.e-15)
    assert abs(v['profile_mass'][0]-1)>1.e-3
    np.testing.assert_array_equal(p.rates(0,1,v)['spontaneous'],p.lines['A']*v['profile_mass'])


def test_dark_outside_and_unsampled_are_distinct(library):
    lines={key:np.repeat(value,3) for key,value in line(100.).items()}
    lines['nu0']=np.array([100.,120.,105.])
    p=provider(library,np.array([90.,100.,110.]),np.zeros((3,2)),lines=lines,width=.001,support=.006)
    v=p.contract(0,3,radiation_generation=1)
    np.testing.assert_array_equal(v['status'],[2,3,4])
    np.testing.assert_array_equal(v['jbar'][0],[0.,0.])
    assert np.isnan(v['jbar'][1:]).all() and np.isnan(v['profile_mass'][1])
    assert v['profile_mass'][2]==0


def test_support_endpoint_weights_are_global_not_local_or_renormalized(library):
    nu=np.array([90.,94.,100.,106.,110.]);j=np.column_stack((np.arange(5.),np.ones(5)))
    p=provider(library,nu,j);v=p.contract(0,1,radiation_generation=1)
    phi=np.exp(-((nu-100.)/1.)**2)/(1.77245385095516*1.)
    phi[(nu<94.)|(nu>106.)]=0.
    np.testing.assert_allclose(v['jbar'][0],(weights(nu)*phi)@j,rtol=2.e-15)
    np.testing.assert_allclose(v['profile_mass'],[np.sum(weights(nu)*phi)],rtol=2.e-15)


def test_late_invalid_input_preserves_all_outputs(library):
    p=provider(library,[90.,100.,110.],np.ones((3,2)))
    j=p.J.copy();j[-1,-1]=np.nan
    out=np.full((1,2),-7.);mass=np.array([-8.]);status=np.array([99],dtype='u1')
    rc=p.function(3,2,1,*[a.ctypes.data_as(p.ptr) for a in (p.frequency,j,p.lines['nu0'])],
        .01,.06,out.ctypes.data_as(p.ptr),mass.ctypes.data_as(p.ptr),status.ctypes.data_as(C.POINTER(C.c_uint8)))
    assert rc!=0
    np.testing.assert_array_equal(out,[[-7.,-7.]])
    np.testing.assert_array_equal(mass,[-8.]);np.testing.assert_array_equal(status,[99])
