"""비교 계산의 단위·시각·외삽·절대값 시험. SN 물리의 성공 판정 아님."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_cmf_toy_epoch import spectrum_metrics,public_spectrum,absolute_integral,run_nonthermal
from compare_cmf_initial_iterate import validate_iterate

initial_row = dict(r_inner_cm='1', r_outer_cm='2', rho_g_cm3='3', T_K='4',
                   mass_fraction='0.2', n_e_cm3='1', n_ion_cm3='2')
initial = {(0, 14, 2): initial_row}
updated = {(0, 14, 2): dict(initial_row, n_e_cm3='1.1', n_ion_cm3='2.2')}
status = '# UNCONVERGED_INITIAL_ITERATE; completed_iterations=2; physical_history=0'
assert validate_iterate(status, initial, updated) == 2
assert validate_iterate(status.replace('iterations=2', 'iterations=0'), initial, initial) == 0
bad_iterates = [({}, status), (updated, status.replace('history=0', 'history=1')),
                (updated, status.replace('iterations=2', 'iterations=-1'))]
for field in ('r_inner_cm', 'r_outer_cm', 'rho_g_cm3', 'T_K', 'mass_fraction'):
    bad_iterates.append(({(0, 14, 2): dict(initial_row, **{field: '9'})}, status))
for candidate, label in bad_iterates:
    try:
        validate_iterate(label, initial, candidate)
    except ValueError:
        pass
    else:
        raise AssertionError('수치 반복 신원 불일치 수용')

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
