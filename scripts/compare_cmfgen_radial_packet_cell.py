#!/usr/bin/env python3
"""같은 실제 셀의 원 SC source/τ와 생산 Reader 반경 재구성을 분리한다.

생산 Reader를 직접 호출하고 ES와 J를 따로 보간한다. B와 blue의 r-선형
정의는 모든 mesh에서 고정한다. 전주파수 생산 F나 새 물질해를 풀지 않는다.
"""
import argparse
import ctypes as ct
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from analyze_cmfgen_spatial_defects import QuadraticBlueKernel
from check_cmfgen_local_sc_packet import bezier_control, extended_material
from compare_cmfgen_fixed_blue_transport import load_packet
from compare_cmfgen_packet_cell_material import linear_material_reference
import lumina_outer_atmosphere as A
from prepare_cmfgen_mom_capture import sha
from run_cmfgen_formal_primary import fields

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = 'd9073ac830d8a70beffc731dc919140b2b97ccff7e8d960060d42a7f7ca60563'


class ProductionSampler:
    def __init__(self, library, chi, true_eta, es, source_J):
        self.values = np.ascontiguousarray([chi, true_eta, es, source_J], dtype='f8')
        if (self.values.shape != (4, 2) or not np.isfinite(self.values).all() or
                np.any(self.values[1:] < 0) or np.any(self.values[2] <= 0)):
            raise ValueError('두 native 끝점의 유한 χ/진η/ES/J 필요')
        self.library = ct.CDLL(str(library))
        self.function = self.library.lumina_radial_packet_sample
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.function.argtypes = [ct.c_int, pointer, pointer, pointer, pointer]
        self.function.restype = ct.c_int

    def at(self, weight):
        w = np.ascontiguousarray([weight], dtype='f8'); chi = np.empty(1); eta = np.empty(1)
        if self.function(1, w, self.values, chi, eta):
            raise ValueError('생산 Reader 반경 표본화 실패')
        return float(chi[0]), float(eta[0])


def build(out):
    sources = [ROOT/'scripts'/name for name in
               ('cmfgen_radial_packet_cell.cpp', 'cmfgen_fixed_blue_transport.cpp')]
    headers = [ROOT/'src'/name for name in ('cmf_linear_provider.hpp', 'cmf_nodal_input.h',
        'cmf_fullsr_linear_map.hpp', 'cmf_ordered_maps.hpp', 'cmf_positive_moments.h',
        'cmf_frequency_history.h')]
    library = out/'radial-cell.so'
    command = ['c++', '-O3', '-std=c++17', '-ffp-contract=off', '-Wall', '-Wextra',
               '-Werror', '-fPIC', '-shared', '-I'+str(ROOT/'src'), *map(str, sources), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode: raise RuntimeError(result.stderr)
    return library, command, [*sources, *headers, library]


@dataclass
class RadialCell:
    sampler: ProductionSampler
    radius_up: float
    radius_down: float
    impact: float
    B: np.ndarray
    blue: np.ndarray
    alpha: float
    geometry: object = None

    def __post_init__(self):
        self.B, self.blue = [np.asarray(x, dtype='f8') for x in (self.B, self.blue)]
        if (not np.isfinite([self.radius_up, self.radius_down, self.impact, self.alpha]).all() or
                not self.radius_down > self.radius_up > self.impact >= 0 or self.alpha < 0 or
                any(x.shape != (2,) or not np.isfinite(x).all() or np.any(x < 0)
                    for x in (self.B, self.blue))):
            raise ValueError('외향 국소 셀의 반경/고정 B·blue 오류')
        self.z0 = np.sqrt((self.radius_up-self.impact)*(self.radius_up+self.impact))
        self.z1 = np.sqrt((self.radius_down-self.impact)*(self.radius_down+self.impact))

    def point(self, x):
        if not 0 <= x <= 1: raise ValueError('셀 좌표 범위 오류')
        z = self.z0+x*(self.z1-self.z0)
        native = self.radius_up if x == 0 else self.radius_down if x == 1 else None
        radius = float(np.hypot(self.impact, z)) if native is None else native
        weight = (radius-self.radius_up)/(self.radius_down-self.radius_up)
        chi, eta = self.sampler.at(weight)
        b = (1-weight)*self.B[0]+weight*self.B[1]
        blue = (1-weight)*self.blue[0]+weight*self.blue[1]
        ds = self.z1-self.z0
        if self.geometry is not None: ds *= float(A.doppler(self.geometry, self.impact, z, native))
        kappa, emission = chi+(self.alpha+3)*b, eta+self.alpha*b*blue
        if kappa <= 0 or not np.isfinite([kappa, emission, ds]).all() or emission < 0:
            raise ValueError('이번 양의 광학좌표 참조가 지원하지 않는 셀')
        return chi, eta, b, blue, kappa, emission, ds

    def length(self, a, b):
        if self.geometry is None: return (b-a)*(self.z1-self.z0)
        return A.segment_geometry(self.geometry, self.impact,
            self.z0+a*(self.z1-self.z0), self.z0+b*(self.z1-self.z0),
            start_radius=self.radius_up if a == 0 else None,
            end_radius=self.radius_down if b == 1 else None)[0]

    def mesh(self, pieces):
        x = np.linspace(0., 1., pieces+1)
        sample = np.asarray([self.point(float(v))[:4] for v in x]).T
        lengths = np.array([self.length(float(a), float(b)) for a, b in zip(x[:-1], x[1:])])
        return lengths, *sample


def radial_reference(cell, incoming):
    """같은 callback의 독립 formal 적분. 큰 τ의 도착점 경계층도 표본화한다."""
    def optical_integrand(t):
        value = cell.point(t)
        return value[4]*value[6]
    def remaining(x):
        return quad(optical_integrand, x, 1., epsabs=1.e-12, epsrel=3.e-13, limit=120)[0]
    tau = remaining(0.); ymax = -np.expm1(-tau)
    def integrand(y):
        optical = -np.log1p(-ymax*y)
        x = brentq(lambda t: remaining(t)-optical, 0., 1., xtol=2.e-14, rtol=2.e-14)
        value = cell.point(x)
        return value[5]/value[4]
    loose = quad(integrand, 0., 1., epsabs=0., epsrel=1.e-9, limit=160, full_output=1)
    tight = quad(integrand, 0., 1., epsabs=0., epsrel=1.e-11, limit=160, full_output=1)
    if len(loose) != 3 or len(tight) != 3: raise ValueError('반경 재구성 독립 적분 미수렴')
    result = incoming*np.exp(-tau)+ymax*tight[0]
    return float(result), dict(effective_tau=float(tau),
        loose_tight_change=float(ymax*abs(tight[0]-loose[0])), estimated_error=float(ymax*tight[1]))


def sc_reference(incoming, sup, control, target, tau):
    ymax = -np.expm1(-tau)
    def source(y):
        u = 1+np.log1p(-ymax*y)/tau
        return (1-u)**2*sup+2*u*(1-u)*control+u*u*target
    result = quad(source, 0., 1., epsabs=0., epsrel=1.e-12, limit=160, full_output=1)
    if len(result) != 3: raise ValueError('원 SC 연속 source 독립 적분 미수렴')
    return float(incoming*np.exp(-tau)+ymax*result[0])


def compare(run, formal_build, prior, reference, out):
    if out.exists(): raise FileExistsError('기존 공통 재구성 셀 비교 보존')
    started = time.monotonic(); packet = load_packet(run, formal_build, None, 'FORMAL_44360_02')
    entry, _, g, blue, actual, paths, _ = packet
    recorded = json.loads((prior/'pair-history.json').read_text())['input_sha256']
    for path in paths:
        if sha(path) != recorded.get(str(path)): raise ValueError('원 포획 입력 변경')
    if sha(reference) != REFERENCE_SHA: raise ValueError('원 source 분해 장 변경')
    raw_geometry, field = fields(reference)
    try:
        row = field[44359]; a = entry['args']; components = row['field'].copy()
        if a['FREQ'] != row['freq']: raise ValueError('원 source 분해의 주파수 주소 불일치')
        identities = {}
        for name, x, y in (('R', a['R'], raw_geometry[0]), ('V', a['V'], raw_geometry[1]),
            ('SIGMA', a['SIGMA'], raw_geometry[2]), ('CHI', a['CHI'], components[4]),
            ('ES', a['ESEC'], components[6]), ('ETA', a['ETA'], components[5]+components[6]*components[7])):
            scale = np.maximum(abs(x), abs(y))
            error = np.divide(abs(x-y), scale, out=np.zeros_like(scale), where=scale > 0)
            identities[name] = float(error.max())
            if identities[name] > 1.e-12: raise ValueError('원 source 분해 정체성 불일치: '+name)
    finally: field._mmap.close()
    ip, node = 53, 49; ray = g['rays'][ip]; k = int(ray['LNK'][node])-1
    if int(ray['LNK'][node+1])-1 != k+1: raise ValueError('원 내부 셀 링크 불일치')
    native = np.array([node+1, node]); links = np.array([k+1, k])
    incoming = float(actual['rays'][ip]['I_P'][k+1]); saved = float(actual['rays'][ip]['I_P'][k])
    chi, eta = extended_material(a, g); alpha = 1/a['dLOG_NU']
    opacity = chi[:len(ray['R'])]+(alpha+3)*ray['B_P']
    source = (eta[:len(ray['R'])]+alpha*ray['B_P']*blue['rays'][ip]['I_P'])/opacity
    dtau = -.5*np.diff(ray['S_P'])*(opacity[:-1]+opacity[1:])
    control = bezier_control(source[k+1], source[k], source[k-1], dtau[k], dtau[k-1])
    original = sc_reference(incoming, source[k+1], control, source[k], dtau[k])
    if abs(original-saved) > 1.e-10*abs(saved):
        raise ValueError('원 SC 연속 source가 저장 결과를 재현하지 못함')
    simple, simple_accuracy = linear_material_reference(incoming, a['CHI'][native], a['ETA'][native],
        ray['B_P'][links], blue['rays'][ip]['I_P'][links], float(ray['S_P'][k]-ray['S_P'][k+1]), alpha)
    geometry = A.geometry(a['R']*1.e10, a['V']*1.e5, a['SIGMA'])
    out.mkdir(); library, command, producers = build(out); kernel = QuadraticBlueKernel(library)
    common = dict(radius_up=float(a['R'][node+1]*1.e10), radius_down=float(a['R'][node]*1.e10),
        impact=float(a['P'][ip]*1.e10), B=ray['B_P'][links]/1.e10,
        blue=blue['rays'][ip]['I_P'][links], alpha=alpha, geometry=geometry)
    provider = ProductionSampler(library, components[4,native]/1.e10, components[5,native]/1.e10,
                                 components[6,native]/1.e10, components[7,native])
    assembled = ProductionSampler(library, a['CHI'][native]/1.e10, a['ETA'][native]/1.e10,
                                  components[6,native]/1.e10, np.zeros(2))
    cell = RadialCell(provider, **common)
    exact, accuracy = radial_reference(cell, incoming)
    total_control, total_accuracy = radial_reference(RadialCell(assembled, **common), incoming)
    mesh = []
    for pieces in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        lengths, c, e, b, previous = cell.mesh(pieces)
        result = float(kernel.leg(lengths, c, e, b, previous, alpha, incoming)[-1])
        mesh.append(dict(pieces=pieces, intensity=result, relative_error=abs(result-exact)/abs(exact)))
    for path in [Path(__file__), Path(A.__file__), *producers[:-1],
                 ROOT/'scripts/check_cmfgen_local_sc_packet.py']:
        target = out/'source_snapshot'; target.mkdir(exist_ok=True); shutil.copy2(path, target/path.name)
    report = dict(status='SAME_PROVIDER_RADIAL_CELL_COMPARISON', scope=__doc__,
        ray_1based=ip+1, node_zero_based=node, phase=entry['phase'],
        source_component_identity_max_relative=identities, original_SC=saved,
        SC_continuous_source_reference=original, SC_reference_relative_error=abs(original-saved)/abs(saved),
        native_s_linear_reference=simple, native_s_linear_accuracy=simple_accuracy,
        radial_total_eta_control=total_control, radial_total_eta_accuracy=total_accuracy,
        production_reader_radial_reference=exact, radial_reference_accuracy=accuracy, mesh=mesh,
        local_ray54_weighted_J_change=float(g['weights']['JQW_P'][node,ip]*(exact-saved)),
        coefficient_callback='production lumina_native_linear::Reader; no Python replacement of ES*J',
        native_saved_path_cm=float((ray['S_P'][k]-ray['S_P'][k+1])*1.e10),
        production_geometry_path_cm=float(cell.length(0., 1.)),
        B_and_blue='same saved endpoints; fixed linear functions of physical radius at every refinement',
        limitation='고정-blue 국소식이며 생산 전주파수 fullSR 연산자와 같다는 주장이 아님',
        input_sha256={str(p): sha(p) for p in [*paths, prior/'pair-history.json', reference, *producers, Path(__file__)]},
        compiler_command=command, elapsed_seconds=time.monotonic()-started,
        production_default_changed=False, new_full_frequency_runs=0, new_material_updates=0, final_goal='NOT_MET')
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with (out/'radial-cell.json').open('x') as stream: stream.write(encoded)
    print(encoded)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'formal-build', 'prior', 'reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.run.resolve(), args.formal_build.resolve(), args.prior.resolve(), args.reference.resolve(), args.out.resolve())
