import numpy as np

from scripts.a2_05_l1bf_gate import (
    EXACT_ZERO,
    OUT_OF_GRID,
    VALID,
    conservative_rebin,
)


def test_conservative_rebin_preserves_tiny_positive_tail_after_large_prefix():
    # The last bin's true area is 5e-17, which is lost by subtracting it from
    # cumulative integrals already dominated by O(1e16) low-frequency flux.
    nu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    j_nu = np.array([1.0e16, 1.0e16, 0.0, 0.0, 1.0e-16])
    edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    rebinned, state = conservative_rebin(nu, j_nu, edges)

    assert state.tolist() == [VALID, VALID, EXACT_ZERO, VALID]
    assert rebinned[0] == 1.0e16
    assert rebinned[1] == 5.0e15
    assert rebinned[2] == 0.0
    assert rebinned[3] == 5.0e-17


def test_conservative_rebin_keeps_zero_and_out_of_grid_semantics():
    nu = np.array([1.0, 2.0, 3.0])
    j_nu = np.zeros_like(nu)
    edges = np.array([0.5, 1.0, 2.0, 3.0, 3.5])

    rebinned, state = conservative_rebin(nu, j_nu, edges)

    assert np.array_equal(rebinned, np.zeros(4))
    assert state.tolist() == [OUT_OF_GRID, EXACT_ZERO, EXACT_ZERO, OUT_OF_GRID]
