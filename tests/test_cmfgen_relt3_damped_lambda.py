from __future__ import annotations

import pytest

from run_cmfgen_relt3_damped_lambda import (
    configure,
    control,
    model_physics_signature,
    validate_seed_model_controls,
)


VADAT = """T [FIX_T]
F [FIX_T_AUTO]
MAJOR [SCALE_OPT]
1.05D0 [MAX_LIN]
1.10D0 [MAX_LAM]
T [COMP_BA]
0 [N_FIX_BA]
F [DO_NG]
"""
IN_ITS = """5 [NUM_ITS]
T [DO_LAM_IT]
F [DO_LAM_AUTO]
F [DO_T_AUTO]
"""
MODEL_SETUP = """ Model Started on:               29-Jul-2026 22:36:43
           40     [NUM_ITS]
            F     [DO_LAM_IT]
  1.72510E+04     [RSTAR]
  1.00000E+01     [MAX_LIN]
  1.00000E+01     [MAX_LAM]
            F     [COMP_BA]
            2     [N_FIX_BA]
            T     [DO_NG]
  1.00000E+100    [MAX_CHNG]
"""
MODEL_SEED = """ Model Started on:               01-Aug-2026 19:52:06
            5     [NUM_ITS]
            T     [DO_LAM_IT]
            F     [DO_LAM_AUTO]
            F     [DO_T_AUTO]
  1.72510E+04     [RSTAR]
  1.05000E+00     [MAX_LIN]
  1.10000E+00     [MAX_LAM]
            T     [COMP_BA]
            0     [N_FIX_BA]
            F     [DO_NG]
  1.00000E+100    [MAX_CHNG]
"""


def test_damped_probe_changes_only_lambda_cap():
    configured = configure(VADAT, IN_ITS)
    assert control(configured, "MAX_LAM") == "1.01D0"
    assert control(configured, "MAX_LIN") == "1.05D0"
    assert control(configured, "FIX_T") == "T"
    assert control(configured, "SCALE_OPT") == "MAJOR"
    assert configured.replace("1.01D0 [MAX_LAM]", "1.10D0 [MAX_LAM]") == VADAT


@pytest.mark.parametrize(
    "vadat, in_its",
    [
        (VADAT.replace("1.10D0", "1.20D0"), IN_ITS),
        (VADAT.replace("T [FIX_T]", "F [FIX_T]"), IN_ITS),
        (VADAT + "1.10D0 [MAX_LAM]\n", IN_ITS),
        (VADAT.replace("T [COMP_BA]", "F [COMP_BA]"), IN_ITS),
        (VADAT.replace("0 [N_FIX_BA]", "2 [N_FIX_BA]"), IN_ITS),
        (VADAT.replace("F [DO_NG]", "T [DO_NG]"), IN_ITS),
        (VADAT, IN_ITS.replace("F [DO_T_AUTO]", "T [DO_T_AUTO]")),
        (VADAT, IN_ITS.replace("T [DO_LAM_IT]", "F [DO_LAM_IT]")),
    ],
)
def test_damped_probe_rejects_wrong_seed_contract(vadat, in_its):
    with pytest.raises(ValueError):
        configure(vadat, in_its)


def test_model_identity_ignores_only_run_stamp_and_declared_iteration_controls():
    assert model_physics_signature(MODEL_SEED) == model_physics_signature(MODEL_SETUP)
    assert validate_seed_model_controls(MODEL_SEED) == {
        "NUM_ITS": "5", "DO_LAM_IT": "T", "DO_LAM_AUTO": "F",
        "DO_T_AUTO": "F", "MAX_LIN": "1.05000E+00",
        "MAX_LAM": "1.10000E+00",
        "COMP_BA": "T", "N_FIX_BA": "0", "DO_NG": "F",
    }


def test_model_identity_rejects_physical_parameter_change():
    changed = MODEL_SETUP.replace("1.72510E+04", "1.80000E+04")
    assert model_physics_signature(MODEL_SEED) != model_physics_signature(changed)


@pytest.mark.parametrize(
    "model",
    [
        MODEL_SEED.replace("1.10000E+00", "1.20000E+00"),
        MODEL_SEED.replace("F     [DO_T_AUTO]", "T     [DO_T_AUTO]"),
        MODEL_SEED.replace("F     [DO_NG]", "T     [DO_NG]"),
    ],
)
def test_model_identity_rejects_unplanned_checkpoint_solver_controls(model):
    with pytest.raises(ValueError):
        validate_seed_model_controls(model)
