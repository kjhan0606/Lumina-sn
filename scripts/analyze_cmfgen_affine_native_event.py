#!/usr/bin/env python3
"""실제 affine MOM의 H 제한/반복 상한 사건과 원식의 근을 분리한다.

전주파수 완주나 이온분포의 합격 판정이 아니다. 보존한 같은 입력의
native trace, 독립 비선형 근과 고정 분기의 고정밀 선형해를 함께 남긴다.
"""
import argparse
import ctypes
import csv
import json
from pathlib import Path
import struct
import time

import numpy as np
from scipy.linalg import solve
from scipy.optimize import root

from analyze_cmfgen_mom_capture import capture, first_event, residual, trace
from analyze_cmfgen_formal_capture import rays,geometry,angular_integral
from analyze_cmfgen_initial_closure import angular_support
from prepare_cmfgen_mom_capture import sha
from solve_cmfgen_mom_kernel import (
    first_repair_equation, high_precision_linear, linmon, linmon_matrix,
    relative_distance, tri_matrix,
)


def analyze(run, replay, build, output, blue_response=False):
    started = time.monotonic()
    if output.exists() or output.with_suffix('.csv').exists():
        raise ValueError('기존 반복 상한 분석 보존')
    inv_path = build / 'capture_first_v1/state_inventory.json'
    inv = json.loads(inv_path.read_text())
    state = capture(run / 'MOM_KERNEL.bin', inv)
    event = first_event(run / 'MOM_FAULT.bin')
    if event['kind'] not in (2,3) or (event['kind']==3 and event['iteration']!=100):
        raise ValueError('native H 제한/100회 반복 상한 사건이 아님')
    if state['phase'] != event['phase'] or state['args']['FREQ'] != event['frequency']:
        raise ValueError('사건과 원식의 위상 불일치')
    args, j, q = state['args'], state['modules']['jrel'], state['modules']['ray']
    if args['XM_CHK_OPTION'].strip() != 'NONE':
        raise ValueError('우변 대체 없는 사건만 분석')
    v = lambda k: j[k]['values']
    base, rhs, adv = first_repair_equation(state, event)
    radius = v('R')
    n = len(radius)
    physical_scale = v('GAM_REL') * radius**2
    native = event['raw']
    formal = q['JNU_STORE']['values'] * physical_scale
    if np.any(native <= 0) or np.any(formal <= 0):
        raise ValueError('이 반복 상한 분석의 자연 초기값이 양수가 아님')
    scale = np.maximum(native, formal)  # 변수 단위만 정규화하며 원식을 변경하지 않는다.

    def nonlinear(x):
        return base @ x + adv * linmon(x, radius) - rhs

    def jacobian(x):
        return base + adv[:, None] * linmon_matrix(x, radius)[0]

    def scaled_residual(x):
        den = abs(base) @ abs(x) + abs(adv * linmon(x, radius)) + abs(rhs)
        return float(np.max(np.divide(abs(nonlinear(x)), den, out=np.zeros(n), where=den > 0)))

    # affine 입사 상수와 첫 내부 flux를 포함해 보존된 우변을 독립 재조립한다.
    boundary_bytes = (run / 'AFFINE_INPUT.bin').read_bytes()
    if len(boundary_bytes) != 112:
        raise ValueError('affine 입력 길이 오류')
    hinc, ninc, hprev, nprev, _, _, _ = struct.unpack_from('=7d', boundary_bytes, 36)
    grad = (v('BETA')[0] * physical_scale[0] * hinc * v('Q')[0] / v('DTAU_H')[0]
            if args['INCL_ADVEC_TERMS'] else 0.)
    flux_constant = grad / (v('P_H')[0] + v('W')[0])
    jp, hp = v('GAM_RSQJNU_PREV'), v('GAM_RSQHNU_PREV')
    terms = dict(local_emission=v('DTAUONQ') * v('SOURCE') * radius**2,
                 previous_J=v('PSIPREV') * jp,
                 previous_H=np.zeros(n), previous_N_lower=np.zeros(n),
                 previous_N_upper=np.zeros(n), affine_flux=np.zeros(n))
    terms['previous_H'][1:-1] = -v('HS')[:-2]*hp[:-2] + v('HS')[1:-1]*hp[1:-1]
    terms['previous_N_lower'][1:-1] = -v('EPS_PREV')[:-2]*(jp[:-2]+jp[1:-1])
    terms['previous_N_upper'][1:-1] = v('EPS_PREV')[1:-1]*(jp[1:-1]+jp[2:])
    terms['affine_flux'][1] = -flux_constant
    rebuilt = sum(terms.values())
    rebuilt[0] = (v('PSIPREV')[0]*jp[0] + physical_scale[0]*(v('P_H')[0]*hinc + v('DELTA')[0]*(
        hinc-hprev+(v('GAM_REL_SQ')[0]*(v('SIGMA')[0]+1)-1)*(ninc-nprev))) + grad)
    rebuilt[-1] = physical_scale[-1]*args['DBB']/3/v('CHI')[-1]
    den = sum(abs(x) for x in terms.values()) + abs(rhs) + abs(rebuilt)
    rhs_error = float(np.max(np.divide(abs(rebuilt-rhs), den, out=np.zeros(n), where=den > 0)))
    if rhs_error > 1e-12:
        raise ValueError('affine 원식 우변 재조립 불일치: '+str(rhs_error))

    def flux(x):
        out = (v('HU')[:-1]*x[1:]-v('HL')[:-1]*x[:-1]+v('HS')[:-1]*hp[:-1]
               +v('EPS_PREV')[:-1]*(jp[:-1]+jp[1:])-v('EPS')[:-1]*(x[:-1]+x[1:]))
        out[0] += flux_constant
        return out

    mid_library=build/'capture_first_v1/analysis_v2/mon_mid.so'
    lib=ctypes.CDLL(str(mid_library))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    def support(x):
        middle=np.zeros(n-1)
        lib.native_mid(n,radius,np.ascontiguousarray(x),middle)
        return angular_support(middle,flux(x),q['KMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]),
                               q['NMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]))

    trace_path=replay/'native_trace/MOM_TRACE.bin'
    if not trace_path.exists():trace_path=replay/'MOM_TRACE.bin'
    iterations = trace(trace_path)[-1]['iterations']
    if set(iterations) != set(range(1, event['iteration']+1)):
        raise ValueError('native 첫 사건까지 trace 누락')
    last = iterations[event['iteration']]
    if not np.array_equal(last['TRI0'], event['matrix']) or not np.array_equal(last['RAW0'], native):
        raise ValueError('native trace의 마지막 식/해와 실제 사건이 다름')
    rows = []
    for i, data in iterations.items():
        x=data['RAW0']
        if 'ERR0' in data:
            old, _, slope = data['ERR0'][1];changed=abs(old-x)/x
            change=data['ERR0'][0];where=int(np.argmax(changed))+1
        else:change=None;where=None
        rows.append(dict(iteration=i, max_change=change,
                         worst_depth_1based=where,
                         nonlinear_residual=scaled_residual(x),
                         minimum_J=float(np.min(x/physical_scale))))
    with output.with_suffix('.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    row_scale = abs(base) @ scale + abs(adv * linmon(scale, radius)) + abs(rhs)
    if np.any(row_scale <= 0):
        raise ValueError('원식 행 정규화 오류')
    results = []
    for name, initial in (('native_at_event', native), ('formal', formal)):
        answer = root(lambda y: nonlinear(y*scale)/row_scale, initial/scale,
                      jac=lambda y: jacobian(y*scale)*scale[None, :]/row_scale[:, None],
                      method='hybr', options=dict(xtol=1e-11, maxfev=500))
        x = answer.x*scale
        item = dict(initial=name, reported_success=bool(answer.success), nfev=int(answer.nfev),
                    positive=bool(np.all(x > 0)), residual=scaled_residual(x), solution=x.tolist(),
                    H_MAX_VAL_edges_1based=(np.flatnonzero(abs(flux(x)) > np.maximum(x[:-1], x[1:]))+1).tolist())
        if item['positive']:
            item.update(angular_support=support(x),
                        max_H_over_adjacent_J=float(np.max(abs(flux(x))/np.maximum(x[:-1],x[1:]))))
        if item['positive'] and item['residual'] <= 1e-10:
            D, branches = linmon_matrix(x, radius)
            precise, _ = high_precision_linear(base+adv[:, None]*D, rhs)
            item.update(mp80_relative=relative_distance(x, precise), mp80_positive=bool(np.all(precise > 0)),
                        mp80_residual=scaled_residual(precise),
                        same_mp80_branches=branches == linmon_matrix(precise, radius)[1])
            # 원래 damped log-gradient Picard 갱신의 같은 분기 국소 선형화다.
            alpha = adv/radius
            slope = radius*(D@x)/x
            frozen = base+np.diag(alpha*slope)
            dx_ds = solve(frozen, -np.diag(alpha*x))
            ds_dx = radius[:, None]*D/x[:, None]-np.diag(slope/x)
            derivative = ds_dx@dx_ds
            item['picard_local_spectral_radius'] = {
                str(d): float(np.max(abs(np.linalg.eigvals((1-d)*np.eye(n)+d*derivative))))
                for d in (.1, .8)}
        results.append(item)

    precise, _ = high_precision_linear(tri_matrix(event['matrix']), rhs)
    ray_support = {}
    for side in ('in', 'out'):
        r = rays(run / ('FORMAL_EVENT_rays_'+side+'.bin'))
        ray_support[side] = {k: float(min(np.min(a[k]) for a in r['rays'])) for k in ('I_P', 'I_M')}
    paths = [run/name for name in ('MOM_CAPTURE.bin', 'MOM_KERNEL.bin', 'MOM_FAULT.bin', 'AFFINE_INPUT.bin')]
    paths += [trace_path, mid_library, inv_path, Path(__file__)]
    report = dict(scope='같은 affine 원식의 실제 H 제한/반복 실패·독립 근·국소 반복 안정성; 생산 수리/분포 수렴 아님',
                  phase=event['phase'],event_kind=event['kind'],event_iteration=event['iteration'],bad_1based=event['bad_1based'],frequency_Hz=event['frequency']*1e15,
                  replay_byte_equal={name: (run/name).read_bytes() == (replay/name).read_bytes()
                                     for name in ('MOM_CAPTURE.bin', 'MOM_KERNEL.bin', 'MOM_FAULT.bin')},
                  native_change_last10=rows[-10:], native_nonlinear_residual=scaled_residual(native),
                  native_linear_residual=residual(event['matrix'], native),
                  native_linear_mp80_relative=relative_distance(native, precise),
                  native_angular_support=support(native),formal_nonlinear_residual=scaled_residual(formal),
                  rhs_reconstruction_relative=rhs_error, ray_support=ray_support,
                  J_max=float(np.max(native/physical_scale)),
                  negative_rhs_depths_1based=(np.flatnonzero(rhs < 0)+1).tolist(),
                  root_checks=results, elapsed_seconds=time.monotonic()-started,
                  evidence_sha256={str(p): sha(p) for p in paths}, final_goal='NOT_JUDGED')
    if blue_response:
        # 실제 앞 주파수 광선에서만 얻는다. 임의 모멘트를 주입하거나 다음 이력으로 채택하지 않는다.
        geo=geometry(run/'FORMAL_EVENT_geometry.bin')
        plus,minus=angular_integral(geo,rays(run/'FORMAL_EVENT_rays_in.bin'))
        blue=plus+minus
        jblue=physical_scale*blue[0]
        hblue=np.zeros(n-1)
        lib.native_mid(n,radius,np.ascontiguousarray(physical_scale*blue[1]),hblue)
        hnew=np.zeros(n);hnew[:-1]=hblue
        responses=[]
        for label,Jp,Hp in (('unchanged',jp,hp),('blue_J_only',jblue,hp),('blue_H_only',jp,hnew),('blue_J_H',jblue,hnew)):
            b=rhs.copy()
            b[0]+=v('PSIPREV')[0]*(Jp[0]-jp[0])
            b[1:-1]+=(-v('HS')[:-2]*(Hp[:-2]-hp[:-2])+v('HS')[1:-1]*(Hp[1:-1]-hp[1:-1])
                +v('PSIPREV')[1:-1]*(Jp[1:-1]-jp[1:-1])
                -v('EPS_PREV')[:-2]*(Jp[:-2]+Jp[1:-1]-jp[:-2]-jp[1:-1])
                +v('EPS_PREV')[1:-1]*(Jp[1:-1]+Jp[2:]-jp[1:-1]-jp[2:]))
            answer=root(lambda y:(base@(y*scale)+adv*linmon(y*scale,radius)-b)/row_scale,np.array(results[0]['solution'])/scale,
                        jac=lambda y:jacobian(y*scale)*scale[None,:]/row_scale[:,None],method='hybr',options=dict(xtol=1e-11,maxfev=500))
            x=answer.x*scale
            h=(v('HU')[:-1]*x[1:]-v('HL')[:-1]*x[:-1]+v('HS')[:-1]*Hp[:-1]
                +v('EPS_PREV')[:-1]*(Jp[:-1]+Jp[1:])-v('EPS')[:-1]*(x[:-1]+x[1:]))
            h[0]+=flux_constant
            den=abs(base)@abs(x)+abs(adv*linmon(x,radius))+abs(b)
            rel=float(np.max(np.divide(abs(base@x+adv*linmon(x,radius)-b),den,out=np.zeros(n),where=den>0)))
            item=dict(case=label,reported_success=bool(answer.success),positive=bool(np.all(x>0)),residual=rel,
                      H_MAX_VAL_edges_1based=(np.flatnonzero(abs(h)>np.maximum(x[:-1],x[1:]))+1).tolist(),
                      max_H_over_adjacent_J=float(np.max(abs(h)/np.maximum(x[:-1],x[1:]))),solution=x.tolist())
            if item['positive']:
                mid=np.zeros(n-1);lib.native_mid(n,radius,np.ascontiguousarray(x),mid)
                item['angular_support']=angular_support(mid,h,q['KMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]),q['NMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]))
                D,branches=linmon_matrix(x,radius);mpx,_=high_precision_linear(base+adv[:,None]*D,b)
                item.update(mp80_relative=relative_distance(x,mpx),same_mp80_branches=branches==linmon_matrix(mpx,radius)[1])
            responses.append(item)
        report['blue_history_response']=dict(scope='현재formal/Q/기하/물질/closure 고정의 한 주파수 원인분리; 생산 history 수리 아님',
            source_phase=event['phase'],incoming_rays_sha256=sha(run/'FORMAL_EVENT_rays_in.bin'),
            depth_samples=[dict(depth=d,J_MOM_blue=float(jp[d-1]/physical_scale[d-1]),J_ray_blue=float(blue[0,d-1])) for d in (1,28,35,36,37,61,90)],
            root_checks=responses)
        report['elapsed_seconds']=time.monotonic()-started
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('root_checks', 'evidence_sha256')}, ensure_ascii=False))
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k != 'solution'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'replay', 'build', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--blue-response',action='store_true')
    a = p.parse_args()
    analyze(a.run, a.replay, a.build, a.output,a.blue_response)
