#!/usr/bin/env python3
"""실제 고정-blue 한 주파수의 모델 내부 전체 광선을 생산 적분 코어로 대조한다.

각도별 외부 입사·core 경계와 blue를 고정한다. native s에서 chi/eta/B/blue를
각각 선형으로 정의하며, 원 CMFGEN SC의 source-ratio 보간과 구별한다.
실제 전주파수 Lumina 특성선·새 산란/물질 고정점·최종 동질성 검증은 아니다.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from analyze_cmfgen_formal_capture import formal_state, geometry, rays, angular_integral
from analyze_cmfgen_mom_capture import capture
from analyze_cmfgen_sweep_context import floop_state
from prepare_cmfgen_mom_capture import sha

ROOT = Path(__file__).resolve().parents[1]


class FixedBlueKernel:
    def __init__(self, library):
        self.library = ct.CDLL(str(library))
        self.function = self.library.lumina_fixed_blue_leg
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        self.function.argtypes = [ct.c_int]+[pointer]*5+[ct.c_double]*2+[ct.c_int, pointer]
        self.function.restype = ct.c_int

    def leg(self, length, chi, eta, b, blue, alpha, incident, pieces):
        arrays = [np.ascontiguousarray(a, dtype='f8') for a in (length, chi, eta, b, blue)]
        n = len(arrays[1])
        if (any(a.ndim != 1 for a in arrays) or arrays[0].shape != (n-1,) or
                any(a.shape != (n,) for a in arrays[1:]) or
                type(pieces) is not int or not 1 <= pieces <= 64):
            raise ValueError('고정-blue 광선 배열/분할 오류')
        result = np.zeros(n)
        code = self.function(n, *arrays, alpha, incident, pieces, result)
        if code: raise ValueError('생산 적분 코어의 고정-blue 진단 실패: '+str(code))
        return result


def build_kernel(out):
    wrapper = ROOT/'scripts/cmfgen_fixed_blue_transport.cpp'
    headers = [ROOT/'src'/name for name in ('cmf_fullsr_linear_map.hpp',
        'cmf_ordered_maps.hpp', 'cmf_positive_moments.h', 'cmf_frequency_history.h')]
    library = out/'fixed-blue.so'
    command = [os.environ.get('CXX', 'c++'), '-O3', '-std=c++17', '-ffp-contract=off',
        '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', '-I'+str(ROOT/'src'),
        str(wrapper), '-o', str(library)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if proc.returncode: raise RuntimeError(proc.stderr)
    return FixedBlueKernel(library), command, [wrapper, *headers, library]


def transport(kernel, args, g, blue, actual, pieces, deadline):
    """실제 모델 바깥 incoming만 고정하고 내부 각 광선을 연속 전달한다."""
    moments = np.zeros((4, g['ND']))
    minimum = float('inf'); cells = 0
    for ip, ray in enumerate(g['rays']):
        if time.monotonic() >= deadline: raise TimeoutError('고정-blue 진단 시간 한도')
        n = min(g['ND'], g['NP']-ip)
        link = ray['LNK'][:n]-1
        if (not np.array_equal(np.diff(link), np.ones(n-1, dtype=int)) or
                not np.array_equal(ray['R'][link], args['R'][:n])):
            raise ValueError('원래점 대응 없는 광선은 이 진단 범위 밖')
        chi, eta = args['CHI'][:n], args['ETA'][:n]
        data, previous = actual['rays'][ip], blue['rays'][ip]
        if n == 1:
            # 외곽에 접하는 영길이 광선은 기존 입사를 양쪽에 그대로 전달한다.
            inward = np.array([data['I_M'][link[0]]]); outward = inward.copy()
        else:
            inward = kernel.leg(np.diff(ray['S_M'][link]), chi, eta,
                ray['B_M'][link], previous['I_M'][link], 1/args['dLOG_NU'],
                float(data['I_M'][link[0]]), pieces)
            inner = float(data['I_P'][link[-1]]) if ip < g['NC'] else float(inward[-1])
            outward = kernel.leg(np.diff(ray['S_P'][link[::-1]]), chi[::-1], eta[::-1],
                ray['B_P'][link[::-1]], previous['I_P'][link[::-1]],
                1/args['dLOG_NU'], inner, pieces)[::-1]
        for k, letter in enumerate('JHKN'):
            moments[k, :n] += (inward*g['weights'][letter+'QW_M'][:n, ip]+
                              outward*g['weights'][letter+'QW_P'][:n, ip])
        minimum = min(minimum, float(inward.min()), float(outward.min()))
        cells += 2*(n-1)
    return moments, dict(rays=len(g['rays']), directional_cells=cells,
                         minimum_intensity=minimum)


def differences(x, y):
    # H/K/N의 부호·작은 값을 임의 floor로 나누지 않는다.
    denominator = np.maximum(abs(x[0]), abs(y[0]))
    delta = abs(x-y)
    if np.any((denominator == 0)[None, :] & (delta != 0)):
        raise ValueError('영 J에서 비영 모멘트 차이')
    relative = np.divide(delta, denominator[None, :], out=np.zeros_like(delta),
                         where=denominator[None, :] > 0)
    return relative


def compare(run, formal_build, inventory, out, pieces=(1, 4, 16), seconds=150):
    if out.exists(): raise FileExistsError('기존 고정-blue 대조 보존')
    if (not 0 < seconds <= 160 or len(pieces) < 2 or
            any(type(n) is not int or not 1 <= n <= 64 for n in pieces) or
            tuple(sorted(set(pieces))) != tuple(pieces)):
        raise ValueError('고정-blue 진단 시간/분할 한도 오류')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 진단 환경 필요')
    started = time.monotonic(); deadline = started+seconds
    inv = json.loads((formal_build/'formal-build.json').read_text())['ray_inventory']
    paths = [run/name for name in ('FORMAL_EVENT_in.bin', 'FORMAL_EVENT_out.bin',
        'FORMAL_EVENT_geometry.bin', 'FORMAL_EVENT_rays_in.bin',
        'FORMAL_EVENT_rays_out.bin', 'FLOOP_STATE.bin', 'MOM_CAPTURE.bin')]
    entry = formal_state(paths[0], inv); final = formal_state(paths[1], inv)
    a = entry['args']; g = geometry(paths[2]); blue = rays(paths[3]); actual = rays(paths[4])
    control = floop_state(paths[5]); mom = capture(paths[6], json.loads(inventory.read_text()))
    if (g['RAY_POINTS_INSERTED'] or not control['coherent_es'] or
            entry['phase'] != final['phase'] or entry['phase'] != mom['phase']):
        raise ValueError('추가점 없는 동일 phase coherent packet 필요')
    source = control['ETA_CLUMP']+control['CHI_SCAT_CLUMP']*mom['args']['JNU_SM']
    if not np.array_equal(source, a['ETA']): raise ValueError('실제 산란 source 신원 불일치')
    reference = sum(angular_integral(g, actual))
    stored = np.array([final['ray'][k+'NU_STORE']['values'] for k in 'JHKN'])
    identity = float(differences(reference, stored).max())
    if identity > 1e-12: raise ValueError('formal 광선/저장 모멘트 재현 실패')
    out.mkdir(); kernel, command, producers = build_kernel(out)
    fields = []; summaries = []
    for count in pieces:
        result, stat = transport(kernel, a, g, blue, actual, count, deadline)
        fields.append(result)
        summaries.append(dict(pieces=count, **stat,
            maximum_difference_over_local_J=differences(result, reference).max(axis=1).tolist()))
    delta = differences(fields[-1], fields[-2])
    samples = []
    for node in (0, 20, 33, 35, 45, 49, 60, 74, 89):
        samples.append(dict(node_zero_based=node, formal_JHKN=reference[:, node].tolist(),
            linear_material_JHKN=fields[-1][:, node].tolist(),
            difference_over_local_J=differences(fields[-1], reference)[:, node].tolist()))
    np.savez(out/'moments.npz', formal=reference, subdivisions=np.array(pieces),
             linear_material=np.array(fields), radius_native=a['R'])
    files = [*paths, formal_build/'formal-build.json', inventory, *producers, Path(__file__)]
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__), *producers[:-1]]:
        shutil.copy2(path, snapshot/path.name)
    report = dict(status='FIXED_BLUE_INTERNAL_TRANSPORT_DIAGNOSTIC',
        scope=__doc__, phase=entry['phase'], frequency_Hz=a['FREQ']*1e15,
        physical_reconstruction='각 native s cell에서 chi/eta/B/blue 각각 선형; 원 SC source 보간과 다름',
        external_incoming='보존된 실제 모델 외곽 inward 강도 고정; 확장층 재계산 없음',
        source_identity_exact=True, formal_angular_identity_max=identity,
        subdivisions=summaries, last_refinement_difference_over_local_J=delta.max(axis=1).tolist(),
        samples=samples, compiler_command=command,
        input_sha256={str(path): sha(path) for path in files},
        moments_sha256=sha(out/'moments.npz'), elapsed_seconds=time.monotonic()-started,
        production_changed=False, new_full_frequency_runs=0, new_material_updates=0,
        physical_equivalence='NOT_JUDGED', final_goal='NOT_MET')
    with (out/'fixed-blue.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], subdivisions=summaries,
        refinement=report['last_refinement_difference_over_local_J'], samples=samples),
        ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'formal-build', 'inventory', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pieces', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--seconds', type=float, default=150)
    args = parser.parse_args()
    compare(args.run.resolve(), args.formal_build.resolve(), args.inventory.resolve(),
            args.out.resolve(), tuple(args.pieces), args.seconds)
