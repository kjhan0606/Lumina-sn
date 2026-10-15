"""비교 계산의 단위·시각·외삽·절대값 시험. SN 물리의 성공 판정 아님."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_cmf_toy_epoch import spectrum_metrics,public_spectrum,absolute_integral,run_nonthermal

marker="TOY_INITIAL_CONVERGED "
for mode in ("ALL_THERMAL_CONTROL", "RESOLVED_NONTHERMAL_SUBSET"):
    assert run_nonthermal([marker+"nonthermal="+mode+" epoch=172800\n"],marker)==mode
for wrong in ([],[marker+"epoch=172800"],[marker+"nonthermal=UNKNOWN"],
              [marker+"nonthermal=ALL_THERMAL_CONTROL nonthermal=RESOLVED_NONTHERMAL_SUBSET"],
              [marker+"nonthermal=ALL_THERMAL_CONTROL"]*2):
    try:
        run_nonthermal(wrong,marker)
    except ValueError:
        pass
    else:
        raise AssertionError("모호한 비열 모형 수용")

reference=[(2500.,1.),(25500.,1.)]
exact=spectrum_metrics(reference,reference)
assert exact["luminosity_ratio"]==1 and exact["absolute_L1_relative"]==0
double=spectrum_metrics([(2500.,2.),(25500.,2.)],reference)
assert double["luminosity_ratio"]==2 and double["absolute_L1_relative"]==1
assert double["unit_area_shape_L1_diagnostic"]==0 and double["amplitude_fitting"] is False
assert absolute_integral([0,1],[-1,1])==.5
for wrong in ([(3000.,1.),(25500.,1.)],[(2500.,-1.),(25500.,1.)],[(2500.,1.),(2500.,2.)]):
    try:
        spectrum_metrics(wrong,reference)
    except ValueError:
        pass
    else:
        raise AssertionError("잘못된 스펙트럼 수용")
path=Path(sys.argv[1])
assert public_spectrum(path,2.0) is None
data=public_spectrum(path,19.48)
assert data is not None and len(data)==23001 and data[0][0]==2500 and data[-1][0]==25500
assert spectrum_metrics(data,data)["absolute_L1_relative"]==0
print("TOY_COMPARISON_SELFTEST_PASS absolute_scale_preserved=1 exact_epoch=1 no_extrapolation=1 public_identity_only=1 SN_agreement=NOT_JUDGED")
