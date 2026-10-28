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
import re
import shutil
import subprocess
import time

import numpy as np

from analyze_cmfgen_formal_capture import formal_state, geometry, rays, angular_integral
from analyze_cmfgen_mom_capture import capture
from analyze_cmfgen_sweep_context import floop_state
from check_cmfgen_local_sc_packet import extended_material
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


def transport(kernel, args, g, blue, actual, pieces, deadline, domain='internal', *, ray_output=None):
    """같은 모델 내부 또는 원 확장층까지 각 광선을 연속 전달한다."""
    if domain not in ('internal', 'extended'): raise ValueError('광선 대조 영역 오류')
    if ray_output is not None and (domain != 'extended' or len(ray_output)):
        raise ValueError('전체 확장 광선만 빈 출력 목록으로 반환한다')
    exterior = extended_material(args, g) if domain == 'extended' else None
    moments = np.zeros((4, g['ND']))
    minimum = float('inf'); cells = 0
    for ip, ray in enumerate(g['rays']):
        if time.monotonic() >= deadline: raise TimeoutError('고정-blue 진단 시간 한도')
        observed_n = min(g['ND'], g['NP']-ip)
        native_link = ray['LNK'][:observed_n]-1
        if (not np.array_equal(np.diff(native_link), np.ones(observed_n-1, dtype=int)) or
                not np.array_equal(ray['R'][native_link], args['R'][:observed_n])):
            raise ValueError('원래점 대응 없는 광선은 이 진단 범위 밖')
        if exterior is None:
            link = native_link; n = observed_n; observed = np.arange(n)
            chi, eta = args['CHI'][:n], args['ETA'][:n]
        else:
            n = len(ray['R']); link = np.arange(n); observed = native_link
            if not np.array_equal(ray['R'], g['R_EXT'][:n]):
                raise ValueError('원 확장층과 다른 광선 물질 좌표')
            chi, eta = (a[:n] for a in exterior)
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
            moments[k, :observed_n] += (inward[observed]*g['weights'][letter+'QW_M'][:observed_n, ip]+
                                      outward[observed]*g['weights'][letter+'QW_P'][:observed_n, ip])
        if ray_output is not None:
            ray_output.append(dict(I_M=inward.copy(), I_P=outward.copy()))
        minimum = min(minimum, float(inward.min()), float(outward.min()))
        cells += 2*(n-1)
    return moments, dict(domain=domain, rays=len(g['rays']), directional_cells=cells,
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


def load_packet(run, formal_build, inventory, prefix):
    """완전한 formal 포획의 실제 총η를 공유한다. 없는 source J를 추정하지 않는다."""
    selected = re.fullmatch(r'FORMAL_(\d{1,6})_(\d{2})', prefix)
    if prefix != 'FORMAL_EVENT' and not selected: raise ValueError('formal packet 이름 오류')
    inv_path = formal_build/'formal-build.json'
    inv = json.loads(inv_path.read_text())['ray_inventory']
    paths = [run/(prefix+suffix) for suffix in (
        '_in.bin', '_out.bin', '_geometry.bin', '_rays_in.bin', '_rays_out.bin')]
    entry = formal_state(paths[0], inv); final = formal_state(paths[1], inv)
    a = entry['args']; g = geometry(paths[2]); blue = rays(paths[3]); actual = rays(paths[4])
    if g['RAY_POINTS_INSERTED'] or entry['phase'] != final['phase']:
        raise ValueError('추가점 없는 동일 phase packet 필요')
    if selected and entry['phase'][1:3] != tuple(map(int, selected.groups())):
        raise ValueError('formal packet 이름과 실제 phase 불일치')
    for name in ('CHI', 'ETA', 'ESEC', 'R', 'V', 'SIGMA'):
        if not np.array_equal(a[name], final['args'][name]):
            raise ValueError('formal 전후 소비 계수/기하 변경: '+name)
    for name in ('FREQ', 'dLOG_NU', 'NC', 'NP', 'ND'):
        if a[name] != final['args'][name]: raise ValueError('formal 전후 인자 변경: '+name)
    if any(g[k] != actual[k] or g[k] != blue[k] for k in ('NC', 'NP', 'ND')):
        raise ValueError('formal 포획의 광선 차원 불일치')
    source_record = dict(kind='SAVED_FORMAL_TOTAL_ETA', decomposition_verified=False)
    if prefix == 'FORMAL_EVENT':
        if inventory is None: raise ValueError('MOM source 재현 inventory 필요')
        loop, snapshot = run/'FLOOP_STATE.bin', run/'MOM_CAPTURE.bin'
        control = floop_state(loop); mom = capture(snapshot, json.loads(inventory.read_text()))
        if not control['coherent_es'] or entry['phase'] != mom['phase']:
            raise ValueError('동일 phase coherent MOM 입력 필요')
        source = control['ETA_CLUMP']+control['CHI_SCAT_CLUMP']*mom['args']['JNU_SM']
        if not np.array_equal(source, a['ETA']): raise ValueError('실제 산란 source 신원 불일치')
        paths.extend([loop, snapshot, inventory])
        source_record = dict(kind='MOM_ENTRY_J_RECONSTRUCTED', decomposition_verified=True)
    paths.append(inv_path)
    return entry, final, g, blue, actual, paths, source_record


def advance_pair(run, formal_build, out, pieces=64, seconds=150):
    """실제715Å 두 주파수에서 고정-blue 공간 차이의 다음 주파수 전달만 분리한다."""
    if out.exists(): raise FileExistsError('기존 두 주파수 전달 대조 보존')
    if type(pieces) is not int or not 1 <= pieces <= 64 or not 0 < seconds <= 160:
        raise ValueError('두 주파수 전달 대조 예산 오류')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 진단 환경 필요')
    started = time.monotonic(); deadline = started+seconds
    first = load_packet(run, formal_build, None, 'FORMAL_44360_02')
    second = load_packet(run, formal_build, None, 'FORMAL_44361_02')
    if first[5][2].read_bytes() != second[5][2].read_bytes():
        raise ValueError('인접 주파수의 광선 기하 변경')
    if first[5][4].read_bytes() != second[5][3].read_bytes():
        raise ValueError('실제 current→blue 광선 이력 연결 불일치')
    if first[0]['phase'][0] != second[0]['phase'][0]:
        raise ValueError('인접 주파수의 물질 반복 세대 변경')
    a0, a1 = first[0]['args'], second[0]['args']
    if (not a0['FREQ'] > a1['FREQ'] > 0 or
            not np.isclose(np.log(a0['FREQ']/a1['FREQ']), a1['dLOG_NU'], rtol=1e-8, atol=1e-12)):
        raise ValueError('실제 인접 주파수 간격과 blue 이력 불일치')
    out.mkdir(); kernel, command, producers = build_kernel(out)
    g = first[2]; first_rays = []; fixed_rays = []; changed_rays = []
    field0, _ = transport(kernel, a0, g, first[3], first[4], pieces, deadline,
                           'extended', ray_output=first_rays)
    fixed, _ = transport(kernel, a1, second[2], second[3], second[4], pieces, deadline,
                          'extended', ray_output=fixed_rays)
    carried = dict(NC=g['NC'], NP=g['NP'], ND=g['ND'], rays=first_rays)
    changed, _ = transport(kernel, a1, second[2], carried, second[4], pieces, deadline,
                            'extended', ray_output=changed_rays)
    truth0 = sum(angular_integral(g, first[4]))
    truth1 = sum(angular_integral(g, second[4]))
    angular = []
    for ip in range(g['NP']-49):
        link = int(g['rays'][ip]['LNK'][49])-1
        parts = [(changed_rays[ip]['I_'+side][link]-fixed_rays[ip]['I_'+side][link])*
                 g['weights']['JQW_'+side][49, ip] for side in ('P', 'M')]
        angular.append(dict(ray_1based=ip+1, core=ip < g['NC'],
            delta_J_plus=float(parts[0]), delta_J_minus=float(parts[1]),
            delta_J=float(sum(parts))))
    samples = []
    for node in (33, 45, 49, 50, 60, 74):
        samples.append(dict(node_zero_based=node, first_formal_J=float(truth0[0, node]),
            first_linear_J=float(field0[0, node]), second_formal_J=float(truth1[0, node]),
            second_fixed_blue_J=float(fixed[0, node]), second_carried_blue_J=float(changed[0, node]),
            propagated_minus_fixed_J=float(changed[0, node]-fixed[0, node])))
    np.savez(out/'pair-moments.npz', first=field0, second_fixed=fixed, second_carried=changed,
             reference_first=truth0, reference_second=truth1)
    files = [*first[5], *second[5], *producers, Path(__file__),
             Path(extended_material.__code__.co_filename)]
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__), Path(extended_material.__code__.co_filename), *producers[:-1]]:
        shutil.copy2(path, snapshot/path.name)
    report = dict(status='FIXED_ETA_TWO_FREQUENCY_HISTORY_DIAGNOSTIC',
        scope=advance_pair.__doc__, actual_blue_chain_byte_equal=True,
        input_total_eta_fixed=True, original_core_and_external_boundaries_fixed=True,
        pieces=pieces, samples=samples,
        maximum_second_fixed_difference_over_local_J=differences(fixed, truth1).max(axis=1).tolist(),
        maximum_second_carried_difference_over_local_J=differences(changed, truth1).max(axis=1).tolist(),
        maximum_blue_effect_over_local_J=differences(changed, fixed).max(axis=1).tolist(),
        native49_directional_blue_effect=dict(plus=float(sum(x['delta_J_plus'] for x in angular)),
            minus=float(sum(x['delta_J_minus'] for x in angular)),
            total=float(sum(x['delta_J'] for x in angular))),
        largest_native49_angular_blue_effect=sorted(angular, key=lambda x: abs(x['delta_J']), reverse=True)[:5],
        compiler_command=command, input_sha256={str(p): sha(p) for p in files},
        output_sha256=sha(out/'pair-moments.npz'), elapsed_seconds=time.monotonic()-started,
        self_consistent_scattering=False, new_full_frequency_runs=0, new_material_updates=0,
        production_changed=False, physical_equivalence='NOT_JUDGED', final_goal='NOT_MET')
    with (out/'pair-history.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('status', 'samples', 'native49_directional_blue_effect',
        'largest_native49_angular_blue_effect', 'maximum_blue_effect_over_local_J')}, ensure_ascii=False))
    return report


def compare(run, formal_build, inventory, out, pieces=(1, 4, 16), seconds=150,
            *, formal_prefix='FORMAL_EVENT', domain='internal'):
    if out.exists(): raise FileExistsError('기존 고정-blue 대조 보존')
    if (not 0 < seconds <= 160 or len(pieces) < 2 or
            any(type(n) is not int or not 1 <= n <= 64 for n in pieces) or
            tuple(sorted(set(pieces))) != tuple(pieces)):
        raise ValueError('고정-blue 진단 시간/분할 한도 오류')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 진단 환경 필요')
    started = time.monotonic(); deadline = started+seconds
    entry, final, g, blue, actual, paths, source_record = load_packet(
        run, formal_build, inventory, formal_prefix)
    a = entry['args']
    reference = sum(angular_integral(g, actual))
    stored = np.array([final['ray'][k+'NU_STORE']['values'] for k in 'JHKN'])
    identity = float(differences(reference, stored).max())
    if identity > 1e-12: raise ValueError('formal 광선/저장 모멘트 재현 실패')
    out.mkdir(); kernel, command, producers = build_kernel(out)
    fields = []; summaries = []
    for count in pieces:
        result, stat = transport(kernel, a, g, blue, actual, count, deadline, domain)
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
    files = [*paths, *producers, Path(__file__), Path(extended_material.__code__.co_filename)]
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__), Path(extended_material.__code__.co_filename), *producers[:-1]]:
        shutil.copy2(path, snapshot/path.name)
    report = dict(status='FIXED_BLUE_TRANSPORT_DIAGNOSTIC', domain=domain,
        scope=__doc__, phase=entry['phase'], frequency_Hz=a['FREQ']*1e15,
        physical_reconstruction='각 native s cell에서 chi/eta/B/blue 각각 선형; 원 SC source 보간과 다름',
        external_incoming=('보존된 실제 모델 외곽 inward 강도 고정; 확장층 재계산 없음'
            if domain == 'internal' else '보존된 원 확장층 최외곽 입사 고정; 원래 물질 외삽 재사용'),
        source_identity_exact=True, source_identity_scope=source_record,
        formal_angular_identity_max=identity,
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
    for name in ('run', 'formal-build', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--formal-prefix', default='FORMAL_EVENT')
    parser.add_argument('--domain', choices=('internal', 'extended'), default='internal')
    parser.add_argument('--pair-history', action='store_true')
    parser.add_argument('--pieces', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--seconds', type=float, default=150)
    args = parser.parse_args()
    if args.pair_history:
        advance_pair(args.run.resolve(), args.formal_build.resolve(), args.out.resolve(),
                     max(args.pieces), args.seconds)
    else:
        compare(args.run.resolve(), args.formal_build.resolve(),
                args.inventory.resolve() if args.inventory else None,
                args.out.resolve(), tuple(args.pieces), args.seconds,
                formal_prefix=args.formal_prefix, domain=args.domain)
