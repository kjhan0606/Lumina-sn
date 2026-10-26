from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_common_charge import solve


def test_true_charge_root_not_small_step_and_every_trial_evaluated():
    seen=[];initial=np.array([2.,5.,9.]);target=np.array([3.,4.,8.])
    def evaluate(ne):
        seen.append(ne.copy());return dict(charge_residual=target-ne)
    result=solve(evaluate,initial,target-initial,np.ones(3),np.full(3,20.))
    assert result['converged'] and len(seen)==len(result['history'])
    np.testing.assert_allclose(result['ne'],target,rtol=1.e-10,atol=0)


def test_nonmonotone_charge_secant_accepts_only_actual_residual_decrease():
    initial=np.array([1.1,1.2]);fun=lambda ne:(ne-1.)*(ne-3.)
    result=solve(lambda ne:dict(charge_residual=fun(ne)),initial,fun(initial),
                 np.full(2,.5),np.full(2,4.),max_evaluations=20)
    assert result['converged']
    np.testing.assert_allclose(result['ne'],1.,rtol=1.e-10,atol=0)
    norms=[float(np.max(abs(fun(initial)/initial)))]+[
        r['charge_relative'] for r in result['history'] if r['accepted']]
    assert all(b<a for a,b in zip(norms,norms[1:]))


def test_no_root_or_budget_exhaustion_preserves_last_accepted_not_false_convergence():
    initial=np.array([2.]);f=np.array([.1])
    result=solve(lambda ne:dict(charge_residual=ne*.05),initial,f,np.array([1.]),np.array([3.]),
                 max_evaluations=3)
    assert not result['converged']
    np.testing.assert_array_equal(result['ne'],initial)
    assert not any(r['accepted'] for r in result['history'])
    with pytest.raises(ValueError):
        solve(lambda ne:None,initial,f,np.array([3.]),np.array([4.]))
