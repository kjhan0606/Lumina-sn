import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_cmfgen_lambda_probe import configure, scan_logs


def sealed_vadat():
    return "\n".join(f"{value} [{key}]" for key,value in
        (("FIX_T","T"),("FIX_T_AUTO","F"),("DO_DDT","F"),("TRT_NON_TE","F"),
         ("DO_NG","F"),("USE_FIXED_J","F"),("FLUX_CAL_ONLY","F"),
         ("WRITE_RATES","T"),("WRITE_JH","T")))+"\n"


def test_only_requested_iteration_controls_change():
    original="19 [NUM_ITS]\nF [DO_LAM_IT]\nF [DO_LAM_AUTO]\nF [DO_T_AUTO]\n"
    updated=configure(sealed_vadat(),original)
    assert updated==original.replace("19 [NUM_ITS]","4 [NUM_ITS]").replace("F [DO_LAM_IT]","T [DO_LAM_IT]")


def test_no_temperature_or_nonthermal_reinterpretation():
    with pytest.raises(ValueError,match="FIX_T"):
        configure(sealed_vadat().replace("T [FIX_T]","F [FIX_T]"),"unused")


@pytest.mark.parametrize("text",[
    "Error in MOM_J_REL_V9: excessive iteration count.\n",
    "Error in grey solver\n", "value = NaN\n",
    "Maximum % increase at depth 1 is 2.0 (BA computed) --- iteration 60\n"])
def test_existing_failure_markers_stop_child(tmp_path,text):
    (tmp_path/"OUTGEN").write_text(text)
    assert scan_logs(tmp_path,{}) is not None


def test_incremental_log_tail_and_valid_lambda(tmp_path):
    p=tmp_path/"OUTGEN";cursor={}
    p.write_text("Maximum % increase (LAMBDA) --- iteration 60\n")
    assert scan_logs(tmp_path,cursor) is None
    with p.open("a") as f:f.write("Error in MOM_")
    assert scan_logs(tmp_path,cursor) is None
    with p.open("a") as f:f.write("J_REL_V9: excessive iteration count.\n")
    assert scan_logs(tmp_path,cursor) is not None
