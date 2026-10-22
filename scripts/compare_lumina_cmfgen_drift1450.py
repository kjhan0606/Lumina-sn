#!/usr/bin/env python3
"""1450Å native49–50 중앙 광선의 CMFGEN b·C 특성선 주파수 이동량 대조."""
import argparse
import json
from pathlib import Path

import numpy as np

import lumina_outer_atmosphere as A
from analyze_cmfgen_formal_capture import formal_state
from lumina_outer_native_spatial import NativeSpatialGeometry
from prepare_cmfgen_mom_capture import sha


GEOMETRY = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_reuse_20261021-uTVbnT/boundary/geometry.npz')
PREPARATION = GEOMETRY.parent/'preparation.json'
INVENTORY = Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v2/formal-build.json')


def compare(packet_root, midray, out):
    if out.exists():
        raise FileExistsError('기존 주파수 이동 대조 보존')
    plan_record = json.loads(PREPARATION.read_text())
    if sha(GEOMETRY) != plan_record['output_sha256']['geometry.npz']:
        raise ValueError('기존 C 기하 입력 변경')
    with np.load(GEOMETRY,allow_pickle=False) as saved:
        g = A.OuterGeometry(saved['radius'].copy(),saved['velocity'].copy(),
                            saved['sigma'].copy(),24,
                            tuple(int(v) for v in saved['reference_indices']),
                            saved['native_radius'].copy())
    geometry = NativeSpatialGeometry(g,subdivisions=4,extra_native_indices=(49,50))
    central = [(negative,positive) for impact,_,negative,positive in geometry.paths
               if impact == 0.]
    if len(central) != 1:
        raise ValueError('C 중앙 impact ray가 유일하지 않음')
    packet_file = packet_root/'field/FORMAL_60715_02_out.bin'
    inventory = json.loads(INVENTORY.read_text())['ray_inventory']
    state = formal_state(packet_file,inventory)
    if state['phase'] != (71,60715,2,0,0):
        raise ValueError('CMFGEN 목표 formal packet 불일치')
    alpha = 1/state['args']['dLOG_NU']
    prior = json.loads(midray.read_text())
    if prior['native_nodes_zero_based'] != [49,50] or len(prior['result']) != 2:
        raise ValueError('이전 중앙 광선 진단 범위 불일치')
    rows=[]
    for direction, steps in zip(('M','P'),central[0]):
        at=[i for i,step in enumerate(steps) if not step.exterior and step.index==39]
        selected=[steps[i] for i in at]
        if len(selected)<4 or at != list(range(at[0],at[-1]+1)) or at[0]==0:
            raise ValueError('C 셀39 중앙 광선 공간조각 누락')
        start_z=float(steps[at[0]-1].z)
        end_z=float(steps[at[-1]].z)
        expected_start,expected_end = ((-g.native_radius[49],-g.native_radius[50]) if direction=='M'
                                       else (g.native_radius[50],g.native_radius[49]))
        if not np.isclose(start_z,expected_start,rtol=1e-12,atol=0) or \
           not np.isclose(end_z,expected_end,rtol=1e-12,atol=0):
            raise ValueError('C index39가 목표 native49–50 구간이 아님')
        c_length=float(sum(step.length for step in selected))
        c_drift=float(sum(step.drift for step in selected))
        native=next(row for row in prior['result'] if row['direction']==direction)
        cmf_drift=native['frequency_coupling_optical_depth']/(alpha+3)
        rows.append(dict(direction=direction,C_segments=len(selected),
                         C_start_z_cm=start_z,C_end_z_cm=end_z,
                         C_length_cm=c_length,C_delta_log_nu=c_drift,
                         CMFGEN_delta_log_nu_from_b=cmf_drift,
                         C_over_CMFGEN_delta_log_nu=c_drift/cmf_drift,
                         CMFGEN_effective_tau=native['effective_optical_depth'],
                         CMFGEN_material_chi_tau=native['material_chi_optical_depth']))
    result=dict(scope=__doc__, material_record=70,frequency_index=60715,
                CMFGEN_alpha=alpha,CMFGEN_dLOG_NU=float(state['args']['dLOG_NU']),
                native_nodes_zero_based=[49,50],C_shell_index=39,
                rows=rows,input_sha256={str(p):sha(p) for p in
                    (GEOMETRY,PREPARATION,packet_file,midray,INVENTORY)},
                C_transport_calls=0,CMFGEN_transport_calls=0,final_goal='NOT_MET')
    with out.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(rows=rows,alpha=alpha),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-root',type=Path,required=True)
    parser.add_argument('--midray',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    compare(args.packet_root.resolve(),args.midray.resolve(),args.out.resolve())
