"""전체 step과 출력 정밀도의 비교는 density floor 없이 수행한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cmfgen_trace_cap_ab import full_step_check,run


def test_tiny_trace_growth_does_not_force_global_rejection_of_full_positive_candidate():
    pre=np.array([[1e-46,1.],[1e6,2e6]])
    sol=np.array([[-22652.,.5],[-.1,0.]])
    after=pre*(1-sol)
    result=full_step_check(pre,after,sol,sol)
    assert result['actual_full_step_within_print_precision']
    assert result['raw_candidate_same_within_print_precision']


def test_partial_step_is_not_accepted_as_full_step():
    pre=np.ones((2,2));sol=np.array([[.5,-2.],[-.1,.2]])
    result=full_step_check(pre,pre*(1-.001*sol),sol,sol)
    assert not result['actual_full_step_within_print_precision']


def test_smaller_than_binary64_density_change_is_not_mistaken_for_clipping():
    pre=np.array([[1e-100,1e10]]);sol=np.array([[1e-30,-1e-25]])
    result=full_step_check(pre,pre.copy(),sol,sol)
    assert result['actual_full_step_within_print_precision']


def test_different_raw_candidate_is_not_hidden_by_tolerance():
    pre=np.ones((2,2));sol=np.ones((2,2))*.2
    result=full_step_check(pre,pre*(1-sol),sol,sol*1.1)
    assert not result['raw_candidate_same_within_print_precision']


def test_existing_output_is_preserved_before_reference_reads(tmp_path,monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','16');monkeypatch.setenv('SLURM_JOB_ID','1')
    with pytest.raises(ValueError,match='기존'):
        run(Path('not-read'),tmp_path,Path('not-read'))
