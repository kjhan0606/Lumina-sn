#!/usr/bin/env python3
"""원13003의 세 특성선만 실제 r-nu 계수로 추적한다. 새 전체장/물질 갱신은 없다."""
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq

from lumina_compiled_input import CompiledInput
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import Location, SCALED
from lumina_fullsr_sc_actual import BASE, BASE_SHA, publish
from lumina_linear_native import Positive
import lumina_local_micro as M


def scalar(raw): return Positive(float(raw['mantissa']), int(raw['exponent']))


def relative(a, b):
    if not a.m: return 0.
    if not b.m: raise ValueError('기여율의 영 분모')
    difference = a.e-b.e
    return 0. if difference < -1074 else math.ldexp(a.m/b.m, difference)


def components(provider, interval, x, column):
    """이미 봉인된7평면의2반경·2주파수 값만 읽는다. J=0의 진 source다."""
    source = provider._source; radius, delta, _, _ = interval.point(x)
    location = Location(); prepare = provider._lib.cmf_nodal_input_prepare
    prepare.argtypes = [ct.c_void_p, ct.c_double, ct.c_void_p]; prepare.restype = ct.c_int
    if prepare(provider._ctx, radius, ct.byref(location)) or location.mode not in (0, 1):
        raise ValueError('이번 구성성분 추출은 native 위치만 지원한다')
    coordinate = column+delta/source.dlognu; low = math.floor(coordinate); fraction = coordinate-low
    if not 0 <= low < source.nf-1: raise ValueError('구성성분 추출 주파수 범위 밖')
    result = {}; digest = hashlib.sha256()
    names = ('bf_chi', 'bf_eta', 'ff_chi', 'ff_eta', 'bb_chi', 'bb_eta', 'es_chi')
    for part, name in enumerate(names):
        values = []
        for row in (location.first, location.last):
            offset = 16+8*(source.nf+2*source.ns+(part*source.ns+row)*source.nf+low)
            raw = os.pread(source._fd, 16, offset)
            if len(raw) != 16: raise ValueError('기존 계수 표본 잘림')
            digest.update(offset.to_bytes(8, 'little')); digest.update(raw)
            pair = np.frombuffer(raw, dtype='<f8')
            values.append(float((1-fraction)*pair[0]+fraction*pair[1]))
        result[name] = (1-location.weight)*values[0]+location.weight*values[1]
    chi, eta = provider.at([radius], np.array([coordinate]))
    result.update(radius_cm=radius, frequency_Hz=float(source.frequency[column]*math.exp(delta)),
        chi_total=float(chi[0, 0]), eta_total=float(eta[0, 0]), sampled_bytes_sha256=digest.hexdigest())
    return result


def trace(root, library, radius_pairs=False):
    root = Path(root).resolve(); output = root/'formation.json'
    if output.exists(): raise FileExistsError('기존 형성 위치 추적 보존')
    if M.N.sha(BASE) != BASE_SHA: raise ValueError('원 봉인 변경')
    c = json.loads(BASE.read_text()); started = time.monotonic(); deadline = started+160
    record = dict(status='STARTED', phase='OPEN_INPUTS', final_goal='NOT_MET', accepted_physical_field=False,
        input_plan_sha256=BASE_SHA, kernel_sha256=M.N.sha(library), runner_sha256=M.N.sha(__file__),
        source_input='J0_I0_ZERO', interval=13003, cases=[], columns=c['fine_columns'],
        reference=c['fine_values'], radius_pairs=radius_pairs,
        scope='세 실제 특성선의 위치-주파수 동시 변화와 양의 기여; 전체장/이온분포 아님')
    def save():
        record['elapsed_seconds'] = time.monotonic()-started; publish(output, record)
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('형성 위치 추적 내부160초')
    save()
    try:
        if M.N.sha(c['cpu_library']) != c['sealed_sha256'][c['cpu_library']]: raise ValueError('제공자 바이너리 변경')
        with M.N.open_sealed_native(c['config'], c['preparation'], c['preparation_sha256']) as source:
            if source.identity != c['source_identity']: raise ValueError('원 계수 신원 변경')
            g = M.open_geometry(c['preparation']); interval = M.Interval(g, c['intervals'][0]['step'])
            with CompiledInput(c['cpu_library'], g, source, M.ZeroTrial(source), slab_width=128,
                               cache_rows=128, profiles=1024) as provider:
                provider.bind_zero_trial(); record['dense_preparation'] = provider.prepare_dense()
                h = source.dlognu; span = interval.point(0.)[1]/h
                integers = np.arange(math.floor(span*4), -1, -1, dtype='i8')
                integers = integers[integers/4 < span]
                phases = np.r_[span, integers/4]
                x = np.zeros(len(phases)); radius = np.empty_like(x); delta = np.empty_like(x)
                record.update(phase='CORRELATED_COEFFICIENTS', phase_span=span, maximum_subdivision=4); save()
                chi = np.empty((len(phases), 3)); eta = np.empty_like(chi)
                for i, phase in enumerate(phases):
                    if i%128 == 0: check()
                    if i == 0: point = 0.
                    elif phase == 0: point = 1.
                    else:
                        point = brentq(lambda q: interval.point(float(q))[1]-phase*h,
                                       float(x[i-1]), 1., xtol=4*np.finfo(float).eps, rtol=4*np.finfo(float).eps)
                    x[i] = point; radius[i], delta[i], _, _ = interval.point(float(point))
                    a, b = provider.at([radius[i]], np.array(c['fine_columns'])+delta[i]/h)
                    chi[i], eta[i] = a[0], b[0]
                record['max_phase_roundoff_bins'] = float(np.max(abs(delta/h-phases))); save()
                kernel = FrozenKernel(library, linear_chi=True)
                for subdivision in (1, 2, 4):
                    check(); stride = 4//subdivision
                    selected = np.r_[0, 1+np.flatnonzero(integers%stride == 0)]
                    xx, dd = x[selected], delta[selected]
                    lengths = np.array([interval.geometry(float(a), float(b)) for a, b in zip(xx[:-1], xx[1:])])
                    code, result, maps, report = kernel.chain(lengths, dd[:-1]-dd[1:], chi[selected], eta[selected],
                        np.zeros(3, dtype=SCALED), max_seconds=deadline-time.monotonic())
                    row = dict(phase_subdivisions=subdivision, steps=len(lengths), report=report)
                    if code: raise RuntimeError('실제 경로 chain 미완: '+str(report))
                    actual = np.array([scalar(v).value() for v in result])
                    row.update(values=actual.tolist(), signed_relative=(actual/np.array(c['fine_values'])-1).tolist(), formation=[])
                    for j, column in enumerate(c['fine_columns']):
                        down = Positive.of(1.); terms = [None]*len(lengths)
                        for k in range(len(lengths)-1, -1, -1):
                            terms[k] = scalar(maps[k, j]['source'])*down
                            down = scalar(maps[k, j]['transmission'])*down
                        total = Positive()
                        for value in terms: total = total+value
                        difference = total.difference(scalar(result[j]))
                        if relative(difference, total) > 2e-11: raise ValueError('기여 재합성 불일치')
                        order = sorted(range(len(terms)), key=lambda k: (terms[k].e if terms[k].m else -2**60, terms[k].m), reverse=True)
                        top = []
                        for k in order[:5]:
                            middle = .5*(float(xx[k])+float(xx[k+1]))
                            top.append(dict(step=k, interval_fraction=[float(xx[k]), float(xx[k+1])],
                                final_fraction=relative(terms[k], total),
                                midpoint_components=components(provider, interval, middle, column)))
                        row['formation'].append(dict(column=column, top_emission_intervals=top))
                    record['cases'].append(row); record['phase'] = 'CHAIN_'+str(subdivision); save()
                if radius_pairs:
                    record['fixed_radius_cases'] = []
                    # 실제 D와 proper dl, 같은 phase mesh는 보존하고 r만 cell 중점으로 바꾼다.
                    # cell 경계 양쪽 계수는 의도적으로 다르므로 chain을 cell별로 연결한다.
                    for cells in (4, 16):
                        for subdivision in (2, 4):
                            check(); stride = 4//subdivision
                            selected = np.r_[0, 1+np.flatnonzero(integers%stride == 0)]
                            phase_x = x[selected]; state = np.zeros(3, dtype=SCALED); count = 0
                            for left, right in zip(np.arange(cells)/cells, np.arange(1, cells+1)/cells):
                                check()
                                xx = np.r_[left, phase_x[(phase_x > left)&(phase_x < right)], right]
                                dd = np.array([interval.point(float(q))[1] for q in xx])
                                lengths = np.array([interval.geometry(float(a), float(b)) for a, b in zip(xx[:-1], xx[1:])])
                                radius_mid = interval.point(float(.5*(left+right)))[0]
                                columns = np.array(c['fine_columns'])[None, :]+dd[:, None]/h
                                cc, ee = provider.at([radius_mid], columns.ravel())
                                code, state, _, report = kernel.chain(lengths, dd[:-1]-dd[1:],
                                    cc.reshape((-1, 3)), ee.reshape((-1, 3)), state,
                                    max_seconds=deadline-time.monotonic())
                                if code: raise RuntimeError('고정 반경 대조 미완: '+str(report))
                                count += len(lengths)
                            actual = np.array([scalar(v).value() for v in state])
                            record['fixed_radius_cases'].append(dict(spatial_cells=cells,
                                phase_subdivisions=subdivision, steps=count, values=actual.tolist(),
                                signed_relative=(actual/np.array(c['fine_values'])-1).tolist()))
                            record['phase'] = 'FIXED_RADIUS_'+str(cells)+'_'+str(subdivision); save()
                np.savez_compressed(root/'correlated-profile.npz', x=x, radius=radius, delta=delta, phases=phases, chi=chi, eta=eta)
                record['profile_sha256'] = M.N.sha(root/'correlated-profile.npz')
        record['status'] = 'CORRELATED_NATIVE_THREE_RAYS_COMPLETE_NOT_FIELD'
    except Exception as exc:
        record.update(status='INCOMPLETE', error=type(exc).__name__+': '+str(exc))
    record['phase'] = 'FINISHED'; save()
    return 0 if record['status'].startswith('CORRELATED') else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('library', type=Path)
    p.add_argument('--radius-pairs', action='store_true')
    a = p.parse_args(); raise SystemExit(trace(**vars(a)))
