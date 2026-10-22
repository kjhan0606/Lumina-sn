#!/usr/bin/env python3
"""물질70/J71의 1450Å 중앙 광선에서 native49–50 내부 강도만 조건부 재구성한다.

원 광선 endpoint를 재현하되 전체 각도 적분 J나 원래 코드의 실제 midpoint 출력으로 표시하지 않는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import formal_state, geometry, rays
from compare_cmfgen_packet_cell_material import kernel, linear_material_reference
from prepare_cmfgen_mom_capture import sha


INVENTORY = Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v2/formal-build.json')
NATIVE = Path('/gpfs/kjhan/cmfgen_src/cur_cmf')


def diagnose(packet_root, out):
    if out.exists():
        raise FileExistsError('기존 중앙 광선 진단 보존')
    packet = packet_root/'field'
    stem = 'FORMAL_60715_02'
    paths = {suffix:packet/(stem+suffix) for suffix in (
        '_in.bin','_out.bin','_geometry.bin','_rays_in.bin','_rays_out.bin')}
    inventory = json.loads(INVENTORY.read_text())['ray_inventory']
    state = formal_state(paths['_out.bin'], inventory)
    g = geometry(paths['_geometry.bin'])
    blue = rays(paths['_rays_in.bin'])
    actual = rays(paths['_rays_out.bin'])
    if state['phase'] != (71,60715,2,0,0) or g['RAY_POINTS_INSERTED'] or \
       g['rays'][0]['LNK'][49] < 1 or g['rays'][0]['LNK'][50] < 1:
        raise ValueError('목표 native49–50 원 광선·최종 호출이 아님')
    args = state['args']
    geo = g['rays'][0]
    alpha = 1/args['dLOG_NU']
    out.mkdir(parents=False)
    cell, compiler, producers = kernel(out, NATIVE)
    entries = []
    j = 49
    links = geo['LNK']
    for direction in ('M','P'):
        ids = np.array((j,j+1) if direction == 'M' else (j+1,j))
        rid = np.array((links[j]-1,links[j+1]-1) if direction == 'M' else
                       (links[j+1]-1,links[j]-1), dtype=int)
        if abs(rid[1]-rid[0]) != 1:
            raise ValueError('목표 중앙 광선 node 사이에 추가 점이 있음')
        s = geo['S_'+direction]
        length = float(s[rid[1]]-s[rid[0]])
        chi = np.asarray(args['CHI'][ids],dtype=float)
        eta = np.asarray(args['ETA'][ids],dtype=float)
        b = np.asarray(geo['B_'+direction][rid],dtype=float)
        previous = np.asarray(blue['rays'][0]['I_'+direction][rid],dtype=float)
        intensity = actual['rays'][0]['I_'+direction]
        incident = float(intensity[rid[0]])
        endpoint = float(intensity[rid[1]])
        effective = chi+(alpha+3)*b
        tau = float(.5*length*sum(effective))
        tau_chi = float(.5*length*sum(chi))
        tau_frequency = float(.5*length*(alpha+3)*sum(b))
        frequency_emission = alpha*b*previous
        source_frequency_fraction = frequency_emission/(eta+frequency_emission)
        sources = (eta+alpha*b*previous)/effective
        far_depth = j+2 if direction == 'M' else j-1
        far_rid = int(links[far_depth])-1
        far_eff = args['CHI'][far_depth]+(alpha+3)*geo['B_'+direction][far_rid]
        far_source = (args['ETA'][far_depth]+alpha*geo['B_'+direction][far_rid]*
                      blue['rays'][0]['I_'+direction][far_rid])/far_eff
        neighbor = .5*abs(s[far_rid]-s[rid[1]])*(effective[1]+far_eff)
        if not all(np.isfinite([incident,endpoint,length,tau,neighbor,far_source])) or \
           min(effective)<=0 or min(sources)<0 or min(previous)<0 or length<=0 or tau<=0:
            raise ValueError('양의 원 native49–50 material cell 조건 불충족')
        control = cell('control',sources[0],sources[1],far_source,tau,neighbor)
        reconstructed_end = cell('point',incident,sources[0],control,sources[1],tau,tau)
        t_mid = float(length*(.5*effective[0]+.125*(effective[1]-effective[0])))
        sc_mid = cell('point',incident,sources[0],control,sources[1],tau,t_mid)
        linear_end,linear_end_error = linear_material_reference(
            incident,chi,eta,b,previous,length,alpha)
        linear_mid,linear_mid_error = linear_material_reference(
            incident,.5*(chi[0]+chi),.5*(eta[0]+eta),.5*(b[0]+b),
            .5*(previous[0]+previous),length/2,alpha)
        error = abs(reconstructed_end-endpoint)/max(incident,endpoint,reconstructed_end)
        if error>2e-13:
            raise ValueError('원래 SC endpoint가 packet과 불일치')
        entries.append(dict(direction=direction,incident=incident,actual_endpoint=endpoint,
                            reconstructed_endpoint=reconstructed_end,
                            endpoint_relative_error=error,
                            effective_optical_depth=tau,
                            material_chi_optical_depth=tau_chi,
                            frequency_coupling_optical_depth=tau_frequency,
                            frequency_over_material_chi=tau_frequency/tau_chi,
                            frequency_source_fraction=source_frequency_fraction.tolist(),
                            source_endpoints=sources.tolist(),
                            SC_spatial_midpoint_intensity=sc_mid,
                            linear_material_midpoint_intensity=linear_mid,
                            linear_material_endpoint_intensity=linear_end,
                            SC_mid_over_end=sc_mid/endpoint,
                            material_mid_over_SC_mid=linear_mid/sc_mid,
                            linear_mid_error=linear_mid_error,
                            linear_end_error=linear_end_error))
    result = dict(scope=__doc__, wavelength_A=float(2.99792458e18/(args['FREQ']*1e15)),
                  central_ray_only=True, native_nodes_zero_based=[49,50],
                  result=entries, input_sha256={str(p):sha(p) for p in (
                      *paths.values(),INVENTORY,*producers)},
                  compiler=compiler, CMFGEN_actual_midpoint_J=False,final_goal='NOT_MET')
    (out/'midray.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(wavelength_A=result['wavelength_A'],result=entries),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    diagnose(args.packet_root.resolve(),args.out.resolve())
