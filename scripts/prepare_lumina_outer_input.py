#!/usr/bin/env python3
"""완성 계수에서 외부 기준 노드와 실제 기하만 준비한다. 수송/물질 풀이는 없다."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

import lumina_outer_atmosphere as A
from lumina_outer_transport import CoupledGeometry
from lumina_outer_c_backend import estimate_work,pack_geometry,diffusion_boundary


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_native(path):
    with path.open('rb') as stream:
        tag,n,nc,nl,epoch = struct.unpack('<8sIIId',stream.read(28))
        if (tag != b'LUMNOD01' or not 24 <= n <= 512 or nc < 1 or nl < 1 or
                not np.isfinite(epoch) or epoch <= 0 or
                path.stat().st_size != 28+12*nc+8*n*(4+nc)):
            raise ValueError('native 물질/길이/epoch 오류')
        stream.seek(28+12*nc)
        physical = np.frombuffer(stream.read(n*4*8),dtype='<f8').reshape(n,4).copy()
    if not np.isfinite(physical).all() or np.any(physical <= 0):
        raise ValueError('native 물리 프로필 비유한/비양수')
    return physical,epoch


def read_auxiliary(source,report,nu,indices,physical=None):
    with source.open('rb') as stream:
        tag,ns,nf = struct.unpack('<8sII',stream.read(16))
        if (tag != b'LMPHYS01' or ns != report['cells'] or nf != len(nu) or
                source.stat().st_size != 16+8*(nf+2*ns+7*ns*nf)):
            raise ValueError('완성 계수의 header/격자/길이 오류')
        actual_nu = np.frombuffer(stream.read(8*nf),dtype='<f8')
        if not np.array_equal(actual_nu,nu):
            raise ValueError('봉인 격자와 실제 완성 계수의 ν 불일치')
        t,ne = [np.frombuffer(stream.read(8*ns),dtype='<f8') for _ in range(2)]
        if (not np.isfinite(t).all() or not np.isfinite(ne).all() or
                np.any(t <= 0) or np.any(ne <= 0) or
                (physical is not None and (not np.array_equal(t,physical[:,1]) or
                                         not np.array_equal(ne,physical[:,2])))):
            raise ValueError('완성 계수와 원래 물질의 T/ne 불일치')
        parts = np.empty((7,len(indices),nf))
        for component in range(7):
            for at,index in enumerate(indices):
                if not 0 <= index < ns:
                    raise ValueError('native 기준점 범위 오류')
                stream.seek(16+8*(nf+2*ns+(component*ns+index)*nf))
                raw = stream.read(8*nf)
                if len(raw) != 8*nf:
                    raise ValueError('완성 계수 표본이 잘림')
                parts[component,at] = np.frombuffer(raw,dtype='<f8')
    if not np.isfinite(parts).all():
        raise ValueError('기준 노드 실제 계수 비유한')
    for k in (1,2,3,5,6):
        if np.any(parts[k] < 0):
            raise ValueError('진방출률/FF/전자산란의 음수 입력')
    # cmfgen_sum_raw_coefficients와 같은 괄호/연산 순서다.
    chi = (parts[6]+(parts[0]+parts[2]))+parts[4]
    eta = (parts[1]+parts[3])+parts[5]
    es = parts[6].copy()
    return chi,es,eta


def prepare(config,library,out,include_inner_boundary=False):
    config,library,out = Path(config).resolve(),Path(library).resolve(),Path(out)
    if out.exists():
        raise ValueError('기존 준비 산출물 보존')
    c = json.loads(config.read_text()); plan_path = Path(c['plan'])
    if sha(plan_path) != c['plan_sha256']:
        raise ValueError('봉인 계획 신원 불일치')
    plan = json.loads(plan_path.read_text())
    seed_path = Path(c['seed'])/'material-seed.json'
    seed = json.loads(seed_path.read_text())
    if (seed['source_record'] != plan['initial_native_record'] or
            seed['source_record'] != plan['reference_radiation_material_record'] or
            seed['plan_sha256'] != c['plan_sha256']):
        raise ValueError('기준 물질 세대 불일치')
    material = Path(c['seed'])/'native_material_nodes.bin'
    if sha(material) != seed['output_sha256']['native_material_nodes.bin']:
        raise ValueError('봉인 native 물질 변경')
    physical,epoch = read_native(material)
    grid = Path(c['grid'])
    if sha(grid) != c['grid_sha256'] or grid.stat().st_size > 128*1024**2:
        raise ValueError('전체 격자 신원/준비 메모리 범위 오류')
    nu = np.load(grid,allow_pickle=False)
    if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or
            np.any(nu <= 0) or not np.all(np.diff(nu) > 0)):
        raise ValueError('전체 주파수 값/순서 오류')
    dlog = float(np.log(nu[1]/nu[0]))
    if np.max(abs(np.log(nu[1:]/nu[:-1])-dlog)) > 64*np.finfo(float).eps:
        raise ValueError('전체 주파수가 log-uniform이 아님')
    field = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if sha(field) != plan['reference_field_sha256']:
        raise ValueError('봉인 참조장 변경')
    with field.open('rb') as stream:
        tag = stream.read(16)
        kind,bits,ibits,n,count,nfield = struct.unpack('<6i',stream.read(24))
        if (tag.rstrip() != b'FORMALPRIMARYV1' or (kind,bits,ibits,nfield) != (8,64,32,8) or
                n != len(physical) or count < 1 or
                field.stat().st_size != 40+4*n*8+count*(24+8*n*8)):
            raise ValueError('참조장 기하/header/길이 오류')
        reference_geometry = np.frombuffer(stream.read(4*n*8),dtype='<f8').reshape(4,n)
    if not np.array_equal(physical[:,0],reference_geometry[0]*1e10):
        raise ValueError('native 물질과 실제 참조의 반경 불일치')
    # 원래 승인된 binary radius/epoch 권위값을 보존한다. 직렬화된 V와의
    # 차이는 별도 기록하며 외부 대기를 억지로 homologous로 늘리지 않는다.
    velocity = physical[:,0]/epoch
    g = A.geometry(physical[:,0],velocity,reference_geometry[2])
    geometry = CoupledGeometry(g)
    work = estimate_work(library,geometry,len(nu),dlog)
    source_dir = Path(c['reuse_sources']['boundary_sources']['directory'])
    source_report = source_dir/'sources.json'
    if sha(source_report) != c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('완성 native 계수 보고서 변경')
    report = json.loads(source_report.read_text()); source = source_dir/'sources.bin'
    if (report['cells'] != len(physical) or report['frequencies'] != len(nu) or
            report['input_sha256'][str(material)] != sha(material) or
            report['input_sha256'][str(grid)] != c['grid_sha256'] or
            sha(source) != report['output_sha256']['sources.bin']):
        raise ValueError('완성 native 계수/물질/격자 신원 불일치')
    chi,es,eta = read_auxiliary(source,report,nu,list(g.reference_indices),physical)
    boundary_arrays = {}
    boundary_report = None
    if include_inner_boundary:
        manifest_path = Path(c['cells'])/'material-cells.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['source_record'] != seed['source_record'] or
                manifest['source_manifest_sha256'] != sha(seed_path) or
                manifest['plan_sha256'] != c['plan_sha256'] or
                manifest['native_points'] != len(physical) or
                manifest['cells'] != len(physical)-1 or
                manifest['epoch_seconds'] != epoch or
                manifest['radius_domain_cm'] != [float(physical[-1,0]),float(physical[0,0])] or
                manifest['boundary_method'] != 'DIFFUSION' or
                manifest['boundary_method'] != seed['boundary_method'] or
                manifest['inner_gas_temperature_K'] != physical[-1,1] or
                manifest['inner_gas_temperature_K'] != seed['inner_gas_temperature_K'] or
                manifest['inner_luminosity_erg_s'] != seed['inner_luminosity_erg_s']):
            raise ValueError('실제 확산 경계와 봉인 물질/기하 불일치')
        inner_chi,_,_ = read_auxiliary(source,report,nu,[len(physical)-1],physical)
        inner_chi = inner_chi[0]
        planck,gradient,boundary_report = diffusion_boundary(library,nu,
            float(physical[-1,1]),manifest['inner_luminosity_erg_s'],
            float(physical[-1,0]),inner_chi)
        boundary_arrays = dict(inner_chi=inner_chi,inner_planck=planck,inner_gradient=gradient)
        boundary_report.update(native_index=len(physical)-1,material_manifest_sha256=sha(manifest_path),
            coefficient_origin='original innermost native node; canonical raw sum; not first-cell material')
    packed = pack_geometry(geometry)
    steps,rays,observations,turning,ratio = packed
    out.mkdir()
    np.save(out/'node_chi.npy',chi,allow_pickle=False)
    np.save(out/'node_es.npy',es,allow_pickle=False)
    np.save(out/'node_eta.npy',eta,allow_pickle=False)
    for name,value in boundary_arrays.items():
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    np.savez(out/'geometry.npz',native_radius=g.native_radius,radius=g.radius,velocity=g.velocity,
        sigma=g.sigma,reference_indices=g.reference_indices,
        step_float=np.array([[s.length,s.drift,s.weight] for s in steps]),
        step_int=np.array([[s.exterior,s.index,s.observation_begin,s.observation_count] for s in steps],dtype='i4'),
        ray_int=np.array([[r.minus_begin,r.minus_count,r.plus_begin,r.plus_count,r.core,
                           r.turn_begin,r.turn_count] for r in rays],dtype='i4'),
        ray_mu=np.array([r.inner_mu for r in rays]),
        observation_sample=np.array([o.sample for o in observations],dtype='i4'),
        observation_mu=np.array([o.mu for o in observations]),turning_mu=turning,exterior_ratio=ratio)
    result = dict(schema='lumina-outer-real-input-preparation-v1',status='PREPARED_INPUTS_ONLY',
        ready_for_physical_execution=False,material_record=seed['source_record'],native_nodes=len(physical),
        cells=geometry.n,frequencies=len(nu),rays=len(rays),half_segments=len(steps),
        native_reference_indices=list(g.reference_indices),epoch_seconds=epoch,
        original_outer_radius_cm=float(g.native_radius[0]),extended_outer_radius_cm=float(g.radius[0]),
        inner_velocity_authority='original binary radius/epoch; unchanged approved homologous interior',
        max_relative_velocity_vs_reference=float(np.max(abs(velocity-reference_geometry[1]*1e5)/velocity)),
        c_work_estimate=work,
        node_chi_min=float(chi.min()),nonpositive_reference_chi=int(np.count_nonzero(chi <= 0)),
        node_es_min=float(es.min()),node_eta_zero=int(np.count_nonzero(eta == 0)),
        inner_boundary=boundary_report,
        unresolved=['nonlinear forward-error certification','performance/resource approval for new full transport',
                    'full cell coefficient binding in new driver']+
                    ([] if include_inner_boundary else ['actual inner diffusion boundary not prepared']),
        actual_source_productions=0,physical_RT_executions=0,native_executions=0,
        physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED',
        input_sha256={str(p):sha(p) for p in (config,plan_path,seed_path,material,grid,source_report,library)},
        reused_source_binary_sha256=report['output_sha256']['sources.bin'],
        reference_field_sha256=plan['reference_field_sha256'],
        source_sha256=sha(__file__),
        output_sha256={name:sha(out/name) for name in
                       ['node_chi.npy','node_es.npy','node_eta.npy','geometry.npz']+
                       [k+'.npy' for k in boundary_arrays]})
    (out/'preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','library','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--with-inner-boundary',action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.config,args.library,args.out,args.with_inner_boundary),ensure_ascii=False))
