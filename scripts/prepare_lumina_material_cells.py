#!/usr/bin/env python3
"""원래 반경 경계 안에서 CMFGEN 점값을 Lumina 쉘 평균으로 옮긴다.

원래90개 점 사이의89개 셀을 사용한다. 수밀도와 rho 및 rho*X는 r^3에서
선형 복원해 부피평균하고, 물질 온도는 rho*T의 질량가중 평균이다.
이산화 근사이며 원래 점값 또는 최종3-grid 검증과 같다고 표시하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from analyze_cmfgen_material_inventory import ELEMENTS,sha


def remap(radius,temperature,ne,rho,population,mass_fractions):
    nd=len(radius)
    if (nd<2 or population.shape[0]!=nd or mass_fractions.shape[1]!=nd or
        any(a.shape!=(nd,) for a in (radius,temperature,ne,rho)) or
        not all(np.isfinite(a).all() for a in
                (radius,temperature,ne,rho,population,mass_fractions)) or
        not np.all(np.diff(radius)<0) or np.any(radius<=0) or
        np.any(temperature<=0) or np.any(ne<=0) or np.any(rho<=0) or
        np.any(population<0) or np.any(mass_fractions<0)):
        raise ValueError('원래 점값/반경 순서 오류')
    r=radius[::-1]
    density=.5*(rho[:-1]+rho[1:])[::-1]
    te=(.5*(rho[:-1]*temperature[:-1]+rho[1:]*temperature[1:]))[::-1]/density
    electron=(.5*(ne[:-1]+ne[1:]))[::-1]
    pops=(.5*(population[:-1]+population[1:]))[::-1]
    x=(.5*(mass_fractions[:,:-1]*rho[:-1]+mass_fractions[:,1:]*rho[1:]))[:,::-1]/density
    volume=(4*np.pi/3)*(r[1:]-r[:-1])*(r[1:]**2+r[1:]*r[:-1]+r[:-1]**2)
    return dict(radius_edges=r,Te_K=te,ne_cm3=electron,rho_g_cm3=density,
                population=pops,mass_fractions=x,volume_cm3=volume)


def read_nodes(path):
    with path.open('rb') as stream:
        head=stream.read(struct.calcsize('<8sIIId'))
        magic,nd,nc,nl,epoch=struct.unpack('<8sIIId',head)
        if magic!=b'LUMNOD01' or not 2<=nd<=1000 or not 1<=nc<=10000 or not nl or epoch<=0:
            raise ValueError('원래 점값 입력 header 오류')
        columns=np.frombuffer(stream.read(nc*12),dtype='<i4').reshape(nc,3).copy()
        physical=np.frombuffer(stream.read(nd*32),dtype='<f8').reshape(nd,4).copy()
        pop=np.frombuffer(stream.read(nd*nc*8),dtype='<f8').reshape(nd,nc).copy()
        if stream.read(1):raise ValueError('원래 점값 입력 후행 데이터')
    return columns,physical,pop,nl,epoch


def prepare(seed,out):
    if out.exists():raise ValueError('기존 쉘 입력 보존')
    manifest=seed/'material-seed.json';m=json.loads(manifest.read_text())
    if (m['source_record']!=70 or m['coordinate_role']!=
            'CMFGEN outer-to-inner nodal values; NOT transfer-cell averages'):
        raise ValueError('원래70 점값 입력이 아님')
    for name,expected in m['output_sha256'].items():
        if sha(seed/name)!=expected:raise ValueError('봉인 점값 입력 변경')
    columns,physical,pop,nl,epoch=read_nodes(seed/'native_material_nodes.bin')
    Zs=list(ELEMENTS)
    xrows={int(r['atomic_number']):r for r in csv.DictReader((seed/'atomic/abundances.csv').open())}
    x=np.array([[float(xrows[Z][str(s)]) for s in range(len(physical))] for Z in Zs])
    cells=remap(*physical.T,pop,x)
    ns=len(cells['Te_K']);edges=cells['radius_edges']
    independent={Z:np.full(len(physical),np.nan) for Z in Zs}
    for row in csv.DictReader((seed/'atomic/element_number_densities.csv').open()):
        independent[int(row['atomic_number'])][int(row['shell_id'])]=float(row['N_element_cm3'])
    if not all(np.isfinite(v).all() and np.all(v>=0) for v in independent.values()):
        raise ValueError('원래 고정 총수 프로필 누락/손상')
    maximum=0
    for Z in Zs:
        expected=(.5*(independent[Z][:-1]+independent[Z][1:]))[::-1]
        actual=cells['population'][:,columns[:,0]==Z].sum(axis=1)
        scale=np.maximum(actual,expected)
        error=np.max(np.divide(abs(actual-expected),scale,
                              out=np.zeros_like(actual),where=scale>0))
        maximum=max(maximum,float(error))
    if maximum>1e-12:raise ValueError('점→쉘 고정 원소 총수 보존 실패')
    charge=cells['population']@columns[:,1]
    charge_error=float(np.max(abs(charge/cells['ne_cm3']-1)))
    if charge_error>1e-10:raise ValueError('점→쉘 전하 보존 실패')
    out.mkdir();atomic=out/'atomic';atomic.mkdir()
    for source in (seed/'atomic').iterdir():
        if source.name not in ('abundances.csv','element_number_densities.csv','electron_temperatures.csv'):
            (atomic/source.name).symlink_to(source.resolve())
    with (atomic/'abundances.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['atomic_number',*range(ns)])
        for i,Z in enumerate(Zs):w.writerow([Z,*[format(v,'.17g') for v in cells['mass_fractions'][i]]])
    with (atomic/'element_number_densities.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['atomic_number','shell_id','N_element_cm3'])
        for Z in Zs:
            value=cells['population'][:,columns[:,0]==Z].sum(axis=1)
            for s,n in enumerate(value):w.writerow([Z,s,format(n,'.17g')])
    with (atomic/'electron_temperatures.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['shell_id','T_e'])
        for s,t in enumerate(cells['Te_K']):w.writerow([s,format(t,'.17g')])
    with (out/'cell_map.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['cell','inner_native_depth_1based','outer_native_depth_1based',
            'r_inner_cm','r_outer_cm','volume_cm3'])
        for s in range(ns):w.writerow([s,ns+1-s,ns-s,*[format(v,'.17g') for v in
            (edges[s],edges[s+1],cells['volume_cm3'][s])]])
    cell_physical=np.column_stack((edges[:-1],edges[1:],cells['Te_K'],
        cells['ne_cm3'],cells['rho_g_cm3'],np.ones(ns)))
    with (out/'material_cells.bin').open('xb') as f:
        f.write(struct.pack('<8sIIId',b'LUMCEL01',ns,len(columns),nl,epoch))
        f.write(columns.astype('<i4').tobytes())
        f.write(cell_physical.astype('<f8').tobytes())
        f.write(cells['population'].astype('<f8').tobytes())
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    report=dict(purpose=m['purpose'],source_record=70,source_seed=str(seed),
        source_manifest_sha256=sha(manifest),plan_sha256=m['plan_sha256'],
        record_storage_sha256=m['record_storage_sha256'],cells=ns,native_points=ns+1,
        full_levels=nl,population_variables=len(columns),epoch_seconds=epoch,
        radius_domain_cm=[float(edges[0]),float(edges[-1])],boundary_method=m['boundary_method'],
        inner_gas_temperature_K=m['inner_gas_temperature_K'],
        inner_luminosity_erg_s=m['inner_luminosity_erg_s'],
        coordinate_role='volume averages over every original radial interval, inner-to-outer',
        reconstruction='linear in r^3 for number densities, rho, rho*X and rho*T; no extrapolation',
        native_J_comparison_rule='first identify the actual Lumina J sampling functional; map reference J to that same point/volume functional. A material volume average does not prove a radiation-field volume average.',
        nodal_temperature_policy='original fixed profile mapped by mass; no radiative-equilibrium solve',
        nuclei_relative_error_max=maximum,charge_relative_error_max=charge_error,
        true_mass_fraction_sum_min_max=[float(cells['mass_fractions'].sum(0).min()),
                                       float(cells['mass_fractions'].sum(0).max())],
        physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED',
        grid_equivalence='NOT_JUDGED; final three-grid test still required',
        RT_executions=0,native_executions=0,
        output_sha256={str(p.relative_to(out)):sha(p) for p in (
            out/'material_cells.bin',out/'cell_map.csv',atomic/'abundances.csv',
            atomic/'element_number_densities.csv',atomic/'electron_temperatures.csv')})
    with (out/'material-cells.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('cells','native_points','radius_domain_cm',
        'nuclei_relative_error_max','charge_relative_error_max')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.seed.resolve(),a.out.resolve())
