from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_coupled_material import MaterialPacket,jacobian,residual,newton_direction,packet,solve_coupled
import lumina_coupled_material as coupled
from run_lumina_fixed_state_balance import adiabatic
import lumina_frozen_direction as frozen
import lumina_charge_projection as projection


@pytest.fixture(scope='module')
def ad(tmp_path_factory):
    path=tmp_path_factory.mktemp('coupled-ad')/'ad.so'
    subprocess.run(['gcc','-O2','-std=c11','-fPIC','-shared',str(ROOT/'src/cmfgen_adiabatic.c'),
                    '-I'+str(ROOT/'src'),'-lm','-o',str(path)],check=True)
    radius=np.array([4.,3.,2.,1.])*1.e15;epoch=2.e6
    return lambda t,ne,ni,u:adiabatic(path,radius,epoch,t,ne,ni,u)['signed_total']


def local(t,ne):
    nuclei=np.array([1.,2.,4.,8.])*1.e8
    fraction=1/(1+np.exp(-(np.log(t/1.e4)-.2*np.log(ne/1.e8))))
    return MaterialPacket(t,ne,nuclei,1.e-11*fraction+1.e-16*t,
        1.e-6*(t/1.e4)**2*(ne/1.e8)**.1,nuclei*(1+fraction)-ne,np.full(4,1.e-5))


def test_batched_local_derivative_preserves_actual_C_depth_coupling_and_mixed_Jv(ad):
    t=np.array([1.2,1.3,1.5,1.8])*1.e4;ne=np.array([1.6,3.,6.,11.])*1.e8
    x=np.r_[np.log(t),np.log(ne)];h=1.e-4;base=local(t,ne)
    j=jacobian(base,local(t*np.exp(h),ne),local(t*np.exp(-h),ne),
        local(t,ne*np.exp(h)),local(t,ne*np.exp(-h)),h,ad)
    v=np.array([.2,-.3,.5,-.1,-.4,.1,.25,-.3]);d=3.e-5
    def f(x):return residual(local(np.exp(x[:4]),np.exp(x[4:])),ad,base.ne,base.thermal_scale)
    expected=(f(x+d*v)-f(x-d*v))/(2*d)
    np.testing.assert_allclose(j@v,expected,rtol=3.e-7,atol=1.e-9)
    # 순수 국소 2×2 블록으로 바꾸면 사라지는 실제 이웃 결합이 존재해야 한다.
    local_only=j.copy()
    for row in range(8):
        for column in range(8):
            if row%4!=column%4:local_only[row,column]=0.
    # 음성 대조도 위 Jv와 동일한 오차 문턱으로 실패해야 한다(임의 norm 문턱 아님).
    assert not np.allclose(local_only@v,expected,rtol=3.e-7,atol=1.e-9)
    direction=newton_direction(j,f(x))
    np.testing.assert_allclose(j@direction,-f(x),rtol=1.e-12,atol=1.e-14)


def test_wrong_profile_generation_and_invalid_scales_are_rejected(ad):
    t=np.full(4,1.4e4);ne=np.array([1.,2.,4.,8.])*1.e8;h=1.e-4;b=local(t,ne)
    with pytest.raises(ValueError,match='신원'):
        jacobian(b,local(t*1.1,ne),local(t*np.exp(-h),ne),
                 local(t,ne*np.exp(h)),local(t,ne*np.exp(-h)),h,ad)
    with pytest.raises(ValueError):residual(b,ad,np.zeros(4),b.thermal_scale)
    with pytest.raises(ValueError):residual(replace(b,internal_energy=np.full(4,-1.)),ad,b.ne,b.thermal_scale)
    with pytest.raises(np.linalg.LinAlgError):newton_direction(np.zeros((2,2)),np.ones(2))


def test_separate_physical_tolerances_stop_without_expensive_probes_when_base_passes():
    base=MaterialPacket(np.array([1.e4,1.1e4]),np.ones(2),np.ones(2),np.ones(2),
        np.array([2.e-6,-1.e-6]),np.array([1.e-12,-1.e-12]),np.ones(2))
    calls=[]
    def evaluate(t,ne):
        calls.append(1)
        raise AssertionError('이미 전하/RE 기준 안에 든 상태는 더 평가하지 않는다')
    result=solve_coupled(evaluate,base,lambda *args:np.zeros(2),max_evaluations=12,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert result['numerical_converged'] and result['iterations']==0
    assert result['packet'] is base and not result['history'] and not calls
    assert result['charge_scaled_max']==1.e-12
    assert result['thermal_scaled_max']==2.e-6
    assert result['charge_tolerance']==1.e-10 and result['thermal_tolerance']==1.e-3


@pytest.mark.parametrize('charge,heat',[(1.e-9,1.e-6),(1.e-12,2.e-3)])
def test_separate_physical_tolerances_do_not_hide_a_failing_residual(charge,heat):
    base=MaterialPacket(np.array([1.e4,1.1e4]),np.ones(2),np.ones(2),np.ones(2),
        np.full(2,heat),np.full(2,charge),np.ones(2))
    result=solve_coupled(lambda *args:pytest.fail('insufficient budget must not call probes'),base,
        lambda *args:np.zeros(2),max_evaluations=4,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert not result['numerical_converged'] and result['packet'] is base
    assert result['charge_scaled_max']==charge and result['thermal_scaled_max']==heat


@pytest.mark.parametrize('final_scale,expected',[(.01,False),(100.,True)])
def test_convergence_uses_the_candidate_thermal_scale(final_scale,expected):
    # 선탐색 merit는 초기 척도로 감소하지만 최종 수렴은 후보 자신의 척도다.
    def evaluate(t,ne):
        x=np.log(t)
        return MaterialPacket(t,ne,np.ones(2),np.ones(2),
            x-.05+.2*x*x,ne-1.,np.exp(x*np.log(final_scale)/.05))
    zero_ad=lambda *args:np.zeros(2)
    base=evaluate(np.ones(2),np.ones(2))
    solved=solve_coupled(evaluate,base,zero_ad,max_evaluations=5,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-4 if expected else 1.e-3)
    actual=float(np.max(abs(solved['packet'].local_heat/solved['packet'].thermal_scale)))
    assert solved['iterations']==1 and solved['numerical_converged'] is expected
    assert solved['thermal_physical_scaled_max']==pytest.approx(actual)
    assert solved['thermal_scaled_max']==pytest.approx(5.e-4,rel=1.e-6)
    assert (actual <= (1.e-4 if expected else 1.e-3)) is expected


def test_convergence_uses_the_candidate_electron_density():
    def evaluate(t,ne):
        x=np.log(ne)
        return MaterialPacket(t,ne,np.ones(2),np.ones(2),np.log(t),
            x+.05+3.9e-8*x*x,np.ones(2))
    base=evaluate(np.ones(2),np.ones(2))
    solved=solve_coupled(evaluate,base,lambda *args:np.zeros(2),max_evaluations=5,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert solved['iterations']==1 and not solved['numerical_converged']
    assert solved['charge_scaled_max'] < 1.e-10
    assert solved['charge_physical_scaled_max'] > 1.e-10
    assert solved['charge_physical_scaled_max']==pytest.approx(
        np.max(abs(solved['packet'].charge/solved['packet'].ne)))


@pytest.mark.parametrize('method',['frozen','projected'])
def test_replayed_direction_does_not_claim_convergence_on_base_scale(method):
    zero_ad=lambda *args:np.zeros(2)
    base=MaterialPacket(np.ones(2),np.ones(2),np.ones(2),np.ones(2),
        np.full(2,.05),np.zeros(2),np.ones(2))
    f=residual(base,zero_ad,base.ne,base.thermal_scale)
    basis=dict(temperature=base.temperature,ne=base.ne,charge_scale=base.ne,
        thermal_scale=base.thermal_scale,scaled_residual=f,jacobian=np.eye(4),direction=-f)
    def evaluate(t,ne):
        return MaterialPacket(t,ne,np.ones(2),np.ones(2),np.full(2,5.e-11),
            np.zeros(2),np.full(2,.01))
    if method=='frozen':
        result=frozen.trial(evaluate,base,zero_ad,basis,.5)
    else:
        result=projection.trial(evaluate,base,zero_ad,basis,.5,
            np.full(2,.1),np.full(2,2.))
    assert result['iterations']==1 and not result['numerical_converged']
    assert np.max(abs(result['packet'].local_heat/result['packet'].thermal_scale)) > 1.e-10


def test_energy_policy_is_explicit_and_uses_original_v2_row_scale():
    r=dict(temperature=np.full(4,1.e4),ne=np.ones(4),energy=np.ones((4,3)),
           population=np.ones((2,4)),charge_residual=np.zeros(4))
    e=dict(native_adiabatic=np.full(4,2.),native_internal_energy=np.ones(4),
           native_stored_net=np.full(4,3.),native_stored_gross=np.full(4,9.))
    p=packet(r,e,'native_stored')
    np.testing.assert_array_equal(p.local_heat,5.)
    np.testing.assert_array_equal(p.thermal_scale,6.)
    np.testing.assert_array_equal(p.nuclei,2.) # 실제 열장부가 사용한 population 합 소유권
    with pytest.raises(ValueError):packet(r,e,'unspecified')


def test_global_newton_changes_T_and_ne_to_actual_depth_coupled_root(ad):
    target_T=np.array([1.1,1.2,1.4,1.6])*1.e4
    nuclei=np.array([1.,2.,4.,8.])*1.e8;target_ne=2*nuclei
    target_U=1.e-10+1.e-16*target_T
    target_ad=ad(target_T,target_ne,nuclei,target_U)
    def evaluate(t,ne):
        return MaterialPacket(t,ne,nuclei,1.e-10+1.e-16*t,
            target_ad+1.e-7*np.log(t/target_T),target_ne-ne+.1*nuclei*np.log(t/target_T),
            np.full(4,1.e-7))
    base=evaluate(target_T*1.08,target_ne*.98)
    solved=solve_coupled(evaluate,base,ad,max_evaluations=25)
    assert solved['numerical_converged'] and solved['iterations']>0
    np.testing.assert_allclose(solved['packet'].temperature,target_T,rtol=1.e-9,atol=0)
    np.testing.assert_allclose(solved['packet'].ne,target_ne,rtol=1.e-9,atol=0)
    limited=solve_coupled(evaluate,base,ad,max_evaluations=4)
    assert not limited['numerical_converged'] and limited['iterations']==0
    assert limited['packet'] is base


def test_optional_secant_uses_actual_accepted_residual_and_fewer_full_probes():
    target_t=np.array([1.e4,1.2e4]);target_ne=np.array([1.e8,2.e8])
    def zero_ad(t,ne,nuclei,u):return np.zeros_like(t)
    def evaluate(t,ne):
        x=np.log(t/target_t);y=np.log(ne/target_ne)
        return MaterialPacket(t,ne,np.ones(2),np.ones(2),
            1.e-5*(x+.2*y+.2*x*x),
            target_ne*(np.exp(y)-1+.25*x+.05*x*x),np.full(2,1.e-5))
    base=evaluate(target_t*1.8,target_ne*.75)
    default=solve_coupled(evaluate,base,zero_ad,max_evaluations=30,max_log_step=.15)
    secant=solve_coupled(evaluate,base,zero_ad,max_evaluations=30,max_log_step=.15,
                         secant_after_accept=True)
    assert default['numerical_converged'] and secant['numerical_converged']
    assert len(secant['history'])<len(default['history'])-10
    assert len([entry for entry in secant['history'] if entry['phase']=='T+'])==1
    assert len([entry for entry in default['history'] if entry['phase']=='T+'])>1
    assert secant['iterations']>1
    np.testing.assert_allclose(secant['packet'].temperature,target_t,rtol=2.e-10)
    np.testing.assert_allclose(secant['packet'].ne,target_ne,rtol=2.e-10)
    assert np.max(abs(residual(secant['packet'],zero_ad,base.ne,base.thermal_scale)))<=1.e-10
    limited=solve_coupled(evaluate,base,zero_ad,max_evaluations=6,max_log_step=.15,
                          secant_after_accept=True)
    assert not limited['numerical_converged'] and limited['iterations']==2
    assert len(limited['history'])==6
    assert limited['packet'] is not base
    with pytest.raises(ValueError,match='비용/수치'):
        solve_coupled(evaluate,base,zero_ad,secant_after_accept=1)


def test_singular_secant_direction_refreshes_centered_jacobian(monkeypatch):
    target=np.array([2.,3.]);calls=[];directions=[]
    def evaluate(t,ne):
        calls.append(1)
        return MaterialPacket(t,ne,np.ones(2),np.ones(2),
            np.log(t/target),np.log(ne/np.array([5.,7.])),np.ones(2))
    original=coupled.newton_direction
    def fail_second(matrix,values):
        if len(directions)==1:
            directions.append('singular_secant')
            raise np.linalg.LinAlgError('synthetic singular update')
        return original(matrix,values)
    monkeypatch.setattr(coupled,'newton_direction',fail_second)
    base=evaluate(target*1.2,np.array([5.,7.]))
    result=solve_coupled(evaluate,base,lambda *args:np.zeros(2),
        max_evaluations=12,max_log_step=.1,secant_after_accept=True,
        on_direction=lambda *args:directions.append(args[-1]))
    assert directions[:3]==['centered','singular_secant','centered']
    assert sum(entry['phase']=='T+' for entry in result['history'])==2
    assert len(calls)==len(result['history'])+1
    assert result['iterations']>=1


def test_frozen_direction_uses_one_actual_trial_and_no_centered_probes(ad):
    t=np.array([1.2,1.3,1.5,1.8])*1.e4;ne=np.array([1.6,3.,6.,11.])*1.e8;h=1.e-4
    b=local(t,ne);f=residual(b,ad,b.ne,b.thermal_scale)
    j=jacobian(b,local(t*np.exp(h),ne),local(t*np.exp(-h),ne),
        local(t,ne*np.exp(h)),local(t,ne*np.exp(-h)),h,ad)
    direction=newton_direction(j,f)
    basis=dict(temperature=t,ne=ne,charge_scale=b.ne,thermal_scale=b.thermal_scale,
               scaled_residual=f,jacobian=j,direction=direction)
    first=min(1.,.1/float(abs(direction).max()));seen=[]
    def evaluate(t,ne):seen.append((t.copy(),ne.copy()));return local(t,ne)
    result=frozen.trial(evaluate,b,ad,basis,first/2)
    assert len(seen)==1 and len(result['history'])==1
    assert result['iterations']==1 and result['history'][0]['accepted']
    assert result['history'][0]['scaled_max']<result['history'][0]['scaled_start_max']
    before=len(seen)
    with pytest.raises(ValueError):frozen.trial(evaluate,b,ad,basis,first)
    with pytest.raises(ValueError):frozen.trial(evaluate,b,ad,{**basis,'direction':np.zeros(8)},first/2)
    with pytest.raises(ValueError):frozen.trial(evaluate,b,ad,{**basis,'scaled_residual':f+1.e-3},first/2)
    with pytest.raises(ValueError):frozen.trial(evaluate,local(t*1.0001,ne),ad,basis,first/2)
    assert len(seen)==before
