from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_coupled_material import MaterialPacket,jacobian,residual,newton_direction,packet,solve_coupled
from run_lumina_fixed_state_balance import adiabatic


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


def test_energy_policy_is_explicit_and_uses_original_v2_row_scale():
    r=dict(temperature=np.full(4,1.e4),ne=np.ones(4),energy=np.ones((4,3)),charge_residual=np.zeros(4))
    e=dict(native_adiabatic=np.full(4,2.),native_internal_energy=np.ones(4),
           native_stored_net=np.full(4,3.),native_stored_gross=np.full(4,9.))
    p=packet(r,e,'native_stored')
    np.testing.assert_array_equal(p.local_heat,5.)
    np.testing.assert_array_equal(p.thermal_scale,6.)
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
