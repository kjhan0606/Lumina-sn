#!/usr/bin/env python3
"""같은 물질70/J71에서 셀 source, native source/J, 실제 F의 위치 차이를 비교한다.

광학적으로 두꺼운 셀에서 source의 위치·평균 정의가 UV 잔차의 크기를
설명하는지 확인하는 고정장 판독이며 새 formal 해나 중심 J 참조는 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_lumina_outer_fixed_point import INPUT
from diagnose_lumina_shell37_endpoint_sensitivity import PAIR
from prepare_lumina_outer_input import sha
from run_cmfgen_formal_primary import fields


def photons(nu, intensity):
    return float(np.trapezoid(intensity/nu, nu))


def analyze(transport, fine_bands=False):
    transport = Path(transport).resolve()
    pair = json.loads((PAIR/'execution.json').read_text())
    run = json.loads((transport/'execution.json').read_text())
    cells = INPUT/'cells'
    cell_report = json.loads((cells/'cells.json').read_text())
    preparation = json.loads((INPUT/'boundary/preparation.json').read_text())
    references = [Path(name) for name in pair['input_sha256']
                  if Path(name).name == 'FORMAL_PRIMARY_FIELDS.bin']
    grids = [Path(name) for name in preparation['input_sha256']
             if Path(name).name == 'frequency.npy']
    if (len(references) != 1 or len(grids) != 1 or pair['material_record'] != 70 or
            pair['reference_record'] != 71 or run['F_calls_completed'] != 3 or
            run['status'] != 'FIXED_J_NATIVE_SPATIAL_DIAGNOSTIC_COMPLETE'):
        raise ValueError('물질70/J71·완료된 세 수송장 필요')
    reference, grid = references[0], grids[0]
    identity = {str(path):sha(path) for path in (reference,grid,cells/'cells.json')}
    if (identity[str(reference)] != pair['input_sha256'][str(reference)] or
            identity[str(grid)] != preparation['input_sha256'][str(grid)] or
            identity[str(cells/'cells.json')] != pair['input_sha256'][str(cells/'cells.json')]):
        raise ValueError('참조/격자/셀 입력 신원 변경')
    nu = np.load(grid, mmap_mode='r', allow_pickle=False)
    jref = np.load(PAIR/'Jref.npy', mmap_mode='r', allow_pickle=False)
    chi = np.load(cells/'cell_chi.npy', mmap_mode='r', allow_pickle=False)
    eta = np.load(cells/'cell_eta.npy', mmap_mode='r', allow_pickle=False)
    es_view = cell_report['es_view']
    es_path = Path(es_view['path'])
    identity[str(es_path)] = sha(es_path)
    if identity[str(es_path)] != es_view['source_sha256']:
        raise ValueError('원 전자산란 계수 신원 변경')
    es = np.memmap(es_path, dtype='<f8', mode='r',
                   offset=es_view['offset_bytes'], shape=tuple(es_view['shape']))
    outputs = {}
    for label in ('P0-4', 'native-4'):
        path = transport/(label+'.npy')
        identity[str(path)] = sha(path)
        if (run[label]['field']['path'] != str(path) or
                identity[str(path)] != run[label]['field']['sha256']):
            raise ValueError('고정 수송장 신원 변경: '+label)
        outputs[label] = np.load(path, mmap_mode='r', allow_pickle=False)
    if (nu.shape != (1115020,) or jref.shape != (91,len(nu)) or
            any(a.shape != (89,len(nu)) for a in (chi, eta, es)) or
            any(v.shape != jref.shape for v in outputs.values())):
        raise ValueError('전ν 장/계수 차원 불일치')
    geometry, formal = fields(reference)
    try:
        ref_nu = np.asarray(formal['freq'][::-1])*1e15
        lognu = np.log(nu)
        logref = np.log(ref_nu)
        wave = 2.99792458e18/ref_nu
        selected = (wave>=100)&(wave<3000)
        if (len(ref_nu) != 197291 or ref_nu[0] != nu[0] or
                ref_nu[-1] != nu[-1] or not np.all(np.diff(ref_nu)>0)):
            raise ValueError('CMFGEN 주파수 대응 불일치')
        rows = {}
        for shell in (37, 39):
            outer, inner = 88-shell, 89-shell
            if not geometry[0,outer] > geometry[0,inner]:
                raise ValueError('원 radial cell 대응 불일치')
            node = {}
            for index, name in ((outer,'outer'),(inner,'inner')):
                f = formal['field'][::-1]
                node[name] = dict(J=np.asarray(f[:,0,index]),
                    chi=np.asarray(f[:,4,index])/1e10,
                    true_eta=np.asarray(f[:,5,index])/1e10,
                    es=np.asarray(f[:,6,index])/1e10,
                    Jsource=np.asarray(f[:,7,index]))
            sample = {name:np.interp(logref,lognu,values[shell]) for name,values in
                      (('cell_chi',chi),('cell_eta',eta),('cell_es',es),('Jref',jref),
                       ('P0-4',outputs['P0-4']),('native-4',outputs['native-4']))}
            total = {name:value['true_eta']+value['es']*value['Jsource']
                     for name,value in node.items()}
            sources = {name:total[name]/node[name]['chi'] for name in node}
            average_j = .5*(node['outer']['J']+node['inner']['J'])
            average_source = .5*(sources['outer']+sources['inner'])
            midpoint_source = (total['outer']+total['inner'])/(
                node['outer']['chi']+node['inner']['chi'])
            cell_source = (sample['cell_eta']+sample['cell_es']*sample['Jref'])/sample['cell_chi']
            if any(not np.isfinite(value[selected]).all() or np.any(value[selected]<=0) for value in
                   (average_j,average_source,midpoint_source,cell_source,
                    sources['outer'],sources['inner'])):
                raise ValueError('UV source/J 비양수 또는 비유한')
            source_map = dict(native_outer_source=sources['outer'],
                              native_inner_source=sources['inner'],
                              endpoint_source_average=average_source,
                              coefficient_midpoint_source=midpoint_source,
                              Lumina_cell_source=cell_source,
                              CMFGEN_native_J_average=average_j,
                              CMFGEN_native_outer_J=node['outer']['J'],
                              CMFGEN_native_inner_J=node['inner']['J'],
                              Lumina_Jref=sample['Jref'],
                              Lumina_P0_4=sample['P0-4'],
                              Lumina_native_4=sample['native-4'])
            bands = {}
            intervals = ((100,400),(400,800),*((lo,lo+100) for lo in range(800,1600,100)),
                         (1600,3000)) if fine_bands else ((100,400),(400,800),(800,1600),(1600,3000))
            # formal geometry is in 1e10 cm, whereas node chi above is in cm^-1.
            radial_tau = .5*(node['outer']['chi']+node['inner']['chi'])*(
                geometry[0,outer]-geometry[0,inner])*1e10
            if not np.isfinite(radial_tau[selected]).all() or np.any(radial_tau[selected]<=0):
                raise ValueError('UV radial 광학깊이 오류')
            for lo, hi in intervals:
                mask = (wave>=lo)&(wave<hi)
                frequency = ref_nu[mask]
                denominator = photons(frequency,average_j[mask])
                if not denominator>0:
                    raise ValueError('CMFGEN native 평균 대역 광자장 없음')
                bands[f'{lo}-{hi}A'] = dict(points=int(np.count_nonzero(mask)),
                    radial_tau_min_median_max=[float(value) for value in
                        (np.min(radial_tau[mask]),np.median(radial_tau[mask]),
                         np.max(radial_tau[mask]))],
                    **{name+'_over_native_J_average':photons(frequency,value[mask])/denominator
                       for name,value in source_map.items()},
                    native_outer_J_over_source=photons(frequency,node['outer']['J'][mask])/
                        photons(frequency,sources['outer'][mask]),
                    native_inner_J_over_source=photons(frequency,node['inner']['J'][mask])/
                        photons(frequency,sources['inner'][mask]))
            rows[str(shell)] = dict(native_outer_index=outer,native_inner_index=inner,bands=bands)
        return dict(schema=('lumina-fixed-source-location-fine-v2' if fine_bands else
                            'lumina-fixed-source-location-v1'),material_record=70,reference_record=71,
            method='actual CMFGEN native chi/true_eta/es/Jsource, Lumina cell chi/eta/es/Jref; same CMFGEN nu photon bands',
            caveat='source ratios and endpoint-average J are not actual CMFGEN cell-center J or a new formal solution',
            results=rows,physical_RT_executions=0,SE_RE_updates=0,final_goal='NOT_MET',
            input_sha256=dict(identity,**{str(path):sha(path) for path in
                (PAIR/'Jref.npy',cells/'cell_chi.npy',cells/'cell_eta.npy')}))
    finally:
        formal._mmap.close()
        for item in (nu,jref,chi,eta,es,*outputs.values()):
            item._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fine-bands',action='store_true',help='800–1600A를 100A씩 분해하고 셀 광학깊이를 출력')
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError('기존 source 위치 진단 보존')
    record = analyze(a.transport_root,fine_bands=a.fine_bands)
    with a.out.open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(a.out),results=record['results']),ensure_ascii=False))
