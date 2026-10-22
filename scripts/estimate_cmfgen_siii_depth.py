#!/usr/bin/env python3
"""원 CMFGEN 장 계수와 1450Å ray 기하로 S III EUV의 항 크기를 추정한다.

실제 600–750Å packet/blue intensity를 포획한 것이 아니다. 결론은
중앙광선 native49–50의 조건부 깊이 규모에 한정된다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_cmfgen_mom_capture import sha
from run_cmfgen_center_packet import EXPECTED_ORIGINAL_FIELD
from run_cmfgen_formal_primary import fields


ORIGINAL = Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062/field/FORMAL_PRIMARY_FIELDS.bin')
DRIFT = Path('/gpfs/kjhan/cmfgen_runs/lumina_center_packet70_20261022-Je2p1/frequency-drift-validated.json')
MIDRAY = Path('/gpfs/kjhan/cmfgen_runs/lumina_midray1450_20261022-Kg6cR/midray.json')
TARGETS_A = (625.,675.,715.,1449.9766600510193)


def calculate(out):
    if out.exists():
        raise FileExistsError('기존 EUV 깊이 추정 결과 보존')
    if sha(ORIGINAL) != EXPECTED_ORIGINAL_FIELD:
        raise ValueError('원 CMFGEN 장 신원 변경')
    drift = json.loads(DRIFT.read_text())
    midray = json.loads(MIDRAY.read_text())
    if (drift['native_nodes_zero_based'] != [49,50] or
        midray['native_nodes_zero_based'] != [49,50] or
        drift['frequency_index'] != 60715 or
        [r['direction'] for r in drift['rows']] != ['M','P'] or
        [r['direction'] for r in midray['result']] != ['M','P']):
        raise ValueError('1450Å 원 중앙광선 기하가 아님')
    geometry, formal = fields(ORIGINAL)
    try:
        freq = np.asarray(formal['freq'])
        wavelength = 2997.92458/freq
        radial_length_1e10cm = float(geometry[0,49]-geometry[0,50])
        if radial_length_1e10cm <= 0:
            raise ValueError('native49–50 반경 순서 오류')

        def radial_chi_tau(index):
            chi = formal['field'][index,4]
            return float(.5*(chi[49]+chi[50])*radial_length_1e10cm)

        original_index = 60715-1
        reference_radial_chi = radial_chi_tau(original_index)
        rows=[]
        for target in TARGETS_A:
            index=int(np.argmin(abs(wavelength-target)))
            # 실제 formal 인자의 dLOG_NU=ln(nu_prev/nu_now)를 재현한다.
            alpha=float(1/np.log(freq[index-1]/freq[index]))
            chi_tau=radial_chi_tau(index)
            directions=[]
            for old,new in zip(drift['rows'],midray['result']):
                material=chi_tau*new['material_chi_optical_depth']/reference_radial_chi
                frequency=(alpha+3.)*old['CMFGEN_delta_log_nu_from_b']
                directions.append(dict(direction=old['direction'],
                    material_chi_tau_estimate=material,
                    frequency_coupling_tau_estimate=frequency,
                    frequency_over_material_estimate=frequency/material))
            rows.append(dict(target_wavelength_A=target,
                actual_wavelength_A=float(wavelength[index]),
                formal_frequency_index_one_based=index+1,
                alpha_nu_over_delta_nu=alpha,
                radial_chi_tau=chi_tau,
                directions=directions))
        check=rows[-1]
        if check['formal_frequency_index_one_based'] != 60715 or any(
            abs(row['material_chi_tau_estimate']-actual['material_chi_optical_depth']) > 1e-10 or
            abs(row['frequency_coupling_tau_estimate']-actual['frequency_coupling_optical_depth']) > 1e-10
            for row,actual in zip(check['directions'],midray['result'])):
            raise ValueError('1450Å 실제 packet 깊이를 재현하지 못함')
        result=dict(scope=__doc__,material_record=70,reference_record=71,
            native_nodes_zero_based=[49,50],central_ray_only=True,
            actual_600_750_packet=False,frequency_geometry_extrapolated_from_1450=True,
            source_fraction_not_calculated=True,results=rows,
            input_sha256={str(p):sha(p) for p in (ORIGINAL,DRIFT,MIDRAY)},
            physical_transport_calls=0,SE_RE_updates=0,final_goal='NOT_MET')
        with out.open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(out=str(out),results=rows),ensure_ascii=False))
    finally:
        formal._mmap.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    calculate(args.out.resolve())
