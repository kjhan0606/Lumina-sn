#!/usr/bin/env python3
"""CMFGEN native J 끝점 평균과 같은 셀 중심의 4점 보간 대리값을 대조한다.

이는 CMFGEN이 직접 출력한 중심 J가 아니다. 원래 Formal/J71과 완성된
Lumina 고정장만 읽으며 수송·물질·원자자료를 변경하지 않는다.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from diagnose_lumina_outer_fixed_point import INPUT
from diagnose_lumina_shell37_endpoint_sensitivity import PAIR
from prepare_lumina_outer_input import sha
from run_cmfgen_formal_primary import fields


BANDS = ((100,400),(400,800),(800,1600),(1600,3000))
SHELLS = (37,38,39)


def photons(nu,j):
    return float(np.trapezoid(j/nu,nu))


def analyze(transport_root):
    started = time.monotonic()
    transport_root = Path(transport_root)
    pair = json.loads((PAIR/'execution.json').read_text())
    run = json.loads((transport_root/'execution.json').read_text())
    candidate = [Path(name) for name in pair['input_sha256']
                 if Path(name).name == 'FORMAL_PRIMARY_FIELDS.bin']
    if len(candidate) != 1:
        raise ValueError('CMFGEN formal 참조 유일성 오류')
    reference_path = candidate[0]
    if (pair['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or
            run['F_calls_completed'] < 2 or
            run['input_sha256'][str(PAIR/'Jref.npy')] != pair['Jref']['sha256'] or
            run['input_sha256'][str(reference_path)] != sha(reference_path) or
            sha(reference_path) != pair['input_sha256'][str(reference_path)]):
        raise ValueError('물질70/J71/수송 기준 신원 불일치')
    preparation = json.loads((INPUT/'boundary/preparation.json').read_text())
    grid = next(Path(path) for path in preparation['input_sha256']
                if Path(path).name == 'frequency.npy')
    if sha(grid) != preparation['input_sha256'][str(grid)]:
        raise ValueError('Lumina 전주파수 격자 신원 불일치')
    own_nu = np.load(grid,mmap_mode='r',allow_pickle=False)
    fields_by_name = {}
    try:
        for label in ('P0-4','native-2'):
            path = transport_root/(label+'.npy')
            if run[label]['field']['path'] != str(path) or sha(path) != run[label]['field']['sha256']:
                raise ValueError('완성 장 신원 불일치: '+label)
            fields_by_name[label] = np.load(path,mmap_mode='r',allow_pickle=False)
        if own_nu.shape != (1115020,) or any(v.shape != (91,len(own_nu))
                                               for v in fields_by_name.values()):
            raise ValueError('전ν·91표본 형상 불일치')
        geometry,formal = fields(reference_path)
        try:
            radius = geometry[0]*1e10
            ref_nu = np.asarray(formal['freq'][::-1])*1e15
            if (len(radius) != 90 or len(ref_nu) != 197291 or
                    ref_nu[0] != own_nu[0] or ref_nu[-1] != own_nu[-1]):
                raise ValueError('원 CMFGEN 주파수·반경 범위 불일치')
            wave = 2.99792458e18/ref_nu
            selected = (wave>=100)&(wave<3000)
            freq = ref_nu[selected]
            log_own,log_selected = np.log(own_nu),np.log(freq)
            result = {}
            for shell in SHELLS:
                outer,inner = 88-shell,89-shell
                indices = np.arange(outer-1,inner+2)
                order = np.argsort(radius[indices])
                x = radius[indices][order]
                values = np.stack([np.asarray(formal['field'][::-1,0,int(node)])[selected]
                                   for node in indices[order]])
                scale = np.max(values,axis=0)
                safe = np.where(scale>0,scale,1.)
                normalized = values/safe
                center = .5*(radius[outer]+radius[inner])
                proxy_r = PchipInterpolator(x,normalized,axis=0)(center)*safe
                proxy_logr = PchipInterpolator(np.log(x),normalized,axis=0)(np.log(center))*safe
                average = .5*(np.asarray(formal['field'][::-1,0,outer])[selected]+
                              np.asarray(formal['field'][::-1,0,inner])[selected])
                outputs = {name:np.interp(log_selected,log_own,field[shell])
                           for name,field in fields_by_name.items()}
                if (not np.isfinite(proxy_r).all() or not np.isfinite(proxy_logr).all() or
                        np.any(proxy_r < 0) or np.any(proxy_logr < 0)):
                    raise ValueError('4점 셀중심 보간 대리값 오류')
                rows = {}
                for lo,hi in BANDS:
                    mask = (2.99792458e18/freq>=lo)&(2.99792458e18/freq<hi)
                    f = freq[mask]
                    old = photons(f,average[mask])
                    c_r = photons(f,proxy_r[mask])
                    c_logr = photons(f,proxy_logr[mask])
                    if not old>0 or not c_r>0 or not c_logr>0:
                        raise ValueError('참조 대역 광자 적분0/비양수')
                    rows[f'{lo}-{hi}A'] = dict(center_r_over_endpoint_average=c_r/old,
                        center_logr_over_endpoint_average=c_logr/old,
                        **{name+'_over_average':photons(f,value[mask])/old
                           for name,value in outputs.items()},
                        **{name+'_over_center_r_proxy':photons(f,value[mask])/c_r
                           for name,value in outputs.items()})
                result[str(shell)] = dict(outer_native_index=outer,inner_native_index=inner,
                    center_radius_cm=float(center),bands=rows)
            return dict(schema='lumina-reference-center-proxy-v1',
                status='LOCATION_SENSITIVITY_ONLY_NOT_CMFGEN_CENTER_TRUTH',
                method='4 native J nodes, shape-preserving cubic in physical r and log r; same ν integration',
                caveat='CMFGEN did not output J at cell centers; proxy is an inference, not a formal reference',
                result=result,physical_RT_executions=0,SE_RE_updates=0,final_goal='NOT_MET',
                field_sha256={name:run[name]['field']['sha256'] for name in fields_by_name},
                input_sha256={str(path):sha(path) for path in
                    (reference_path,PAIR/'execution.json',
                     grid,transport_root/'P0-4.npy',transport_root/'native-2.npy')},
                elapsed_seconds=time.monotonic()-started,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        finally:
            formal._mmap.close()
    finally:
        own_nu._mmap.close()
        for value in fields_by_name.values():
            value._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('기존 중심점 대리값 결과 보존')
    result = analyze(args.transport_root)
    with args.out.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(args.out),status=result['status'],
                          shell37=result['result']['37']['bands']),ensure_ascii=False),flush=True)
