"""고정 공간경로에서 주파수 세분만 바꾸는 직접 시험."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cmfgen_siii_full_path_kernel import refine_nodes, sweep
from compare_cmfgen_siii_frequency_refinement import advance


def inputs(pieces=4):
    n = 7
    chi = np.linspace(.1, .2, n)
    eta = np.linspace(.2, .4, n)
    sm = refine_nodes(np.arange(n, dtype='f8'), pieces)
    sp = refine_nodes(np.arange(n, 0, -1, dtype='f8'), pieces)
    bm = np.full(len(sm), .03)
    bp = np.full(len(sm), .04)
    prev_m = np.linspace(.1, .3, len(sm))
    prev_p = np.linspace(.3, .4, len(sm))
    return chi, eta, sm, sp, bm, bp, prev_m, prev_p


def test_one_frequency_step_exactly_preserves_existing_sweep():
    chi, eta, *args = inputs()
    for core in (False, True):
        expected = sweep(chi, eta, *args, .02, .7, core, 4)
        observed = advance(chi*2, eta*3, chi, eta, *args,
                           .02, .9, .7, core, 4, 1)
        for actual, truth in zip(observed, expected):
            np.testing.assert_array_equal(actual, truth)


def test_frequency_steps_match_explicit_log_linear_subintervals():
    chi, eta, sm, sp, bm, bp, prev_m, prev_p = inputs()
    old_chi = chi-.15  # 음의 원 χ도 잘라내지 않는다.
    old_eta = eta*.7
    for factor in (2, 4):
        for core in (False, True):
            minus, plus = prev_m, prev_p
            for step in range(1, factor+1):
                w = step/factor
                current_chi = chi if step == factor else (1-w)*old_chi+w*chi
                current_eta = eta if step == factor else (1-w)*old_eta+w*eta
                boundary = .7 if step == factor else (1-w)*.9+w*.7
                plus, minus = sweep(current_chi, current_eta, sm, sp, bm, bp,
                                    minus, plus, .02/factor, boundary, core, 4)
            actual = advance(old_chi, old_eta, chi, eta, sm, sp, bm, bp,
                             prev_m, prev_p, .02, .9, .7, core, 4, factor)
            np.testing.assert_array_equal(actual[0], plus)
            np.testing.assert_array_equal(actual[1], minus)
            assert np.isfinite(actual).all() and np.min(actual) >= 0


def test_spectral_power_manufactured_solution_improves_with_frequency_resolution():
    # I가 공간에서 일정하고 logν에 exp(2u)로 변하면
    # η=(χ+(3-2)b)I가 연속 주파수 방정식을 만족한다.
    n, pieces, b, chi, du = 7, 8, .03, .2, .08
    sm = refine_nodes(np.arange(n, dtype='f8'), pieces)
    sp = refine_nodes(np.arange(n, 0, -1, dtype='f8'), pieces)
    bb = np.full(len(sm), b)
    old_i, target_i = np.exp(2*du), 1.
    previous = np.full(len(sm), old_i)
    opacity = np.full(n, chi)
    eta = np.full(n, chi+b)
    errors = []
    for factor in (1, 2, 4):
        plus, _ = advance(opacity, eta*old_i, opacity, eta,
                          sm, sp, bb, bb, previous, previous, du,
                          old_i, target_i, True, pieces, factor)
        errors.append(float(np.max(abs(plus-target_i))))
    assert errors[1] < errors[0] and errors[2] < errors[1]
