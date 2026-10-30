from __future__ import annotations

import pytest

from run_cmfgen_relt3_damped_lambda import configure, control


VADAT = """T [FIX_T]
F [FIX_T_AUTO]
MAJOR [SCALE_OPT]
1.05D0 [MAX_LIN]
1.10D0 [MAX_LAM]
"""
IN_ITS = """5 [NUM_ITS]
T [DO_LAM_IT]
F [DO_LAM_AUTO]
F [DO_T_AUTO]
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
        (VADAT, IN_ITS.replace("F [DO_T_AUTO]", "T [DO_T_AUTO]")),
        (VADAT, IN_ITS.replace("T [DO_LAM_IT]", "F [DO_LAM_IT]")),
    ],
)
def test_damped_probe_rejects_wrong_seed_contract(vadat, in_its):
    with pytest.raises(ValueError):
        configure(vadat, in_its)
