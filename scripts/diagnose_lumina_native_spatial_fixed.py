#!/usr/bin/env python3
"""같은 물질70/J71·경계에서 P0와 native 공간보간 F(Jref)를 대조한다.

내부 셀만 바꾸며 외부 고리와 확산 경계는 고정한다. 고정장 단회 진단이므로
자기일관 SE/RE·CMFGEN 동질성 판정 또는 스펙트럼을 산출하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time

import numpy as np

from diagnose_lumina_outer_fixed_point import (PROJECT, INPUT, LIBRARY, field_residual)
from diagnose_lumina_shell37_endpoint_sensitivity import PAIR, SOURCE, save_array, save_record
from lumina_outer_c_backend import COuterTransport
from lumina_outer_native_spatial import NativeSpatialGeometry, build_variant, set_native_fixed
from lumina_outer_prepared import PreparedInputs
from prepare_lumina_outer_input import read_auxiliary, sha
from run_cmfgen_formal_primary import fields


def project_native_scattering(field, own_nu):
    """CMFGEN formal J_source를 native 90점×Lumina 전ν에 외삽 없이 투영."""
    reference_nu = np.asarray(field['freq'][::-1])*1e15
    if (len(reference_nu) != 197291 or own_nu[0] != reference_nu[0] or
            own_nu[-1] != reference_nu[-1] or not np.all(np.diff(reference_nu) > 0)):
        raise ValueError('참조 J_source 주파수 범위/순서 불일치')
    log_own,log_ref = np.log(own_nu),np.log(reference_nu)
    projected = np.empty((90,len(own_nu)),dtype='f8')
    difference = 0.
    for node in range(90):
        source = np.asarray(field['field'][::-1,7,node])
        formal_j = np.asarray(field['field'][::-1,0,node])
        if (not np.isfinite(source).all() or np.any(source < 0) or
                not np.isfinite(formal_j).all() or np.any(formal_j < 0)):
            raise ValueError('CMFGEN J_source/J 비유한 또는 음수')
        projected[node] = np.interp(log_own,log_ref,source)
        scale = float(formal_j.max())
        difference = max(difference,float(np.max(np.abs(source-formal_j)))/scale if scale else 0.)
    if not np.isfinite(projected).all() or np.any(projected < 0):
        raise ValueError('native J_source 투영 오류')
    return projected,difference


def _operator(library,plan,loaded):
    return COuterTransport(library,plan,loaded.nu,loaded.cell_chi,loaded.cell_es,
        loaded.cell_eta,*loaded.nodes,*loaded.boundary,work_budget_bytes=16*1024**3,
        weighted_workers=16,moment_count=1,approved_cpu_limit=16)


def run(root):
    root = Path(root).resolve()
    if (socket.gethostname().lower() != 'lageunha' or Path.cwd().resolve() != PROJECT or
            not root.is_dir() or list(root.iterdir())):
        raise ValueError('Lageunha의 확인된 프로젝트·빈 신규 root 필요')
    if shutil.disk_usage(root).free < 8*1024**3:
        raise ValueError('전주파수 장 세 개와 여유 저장소 부족')
    started = time.monotonic()
    pair = json.loads((PAIR/'execution.json').read_text())
    preparation = json.loads((INPUT/'boundary/preparation.json').read_text())
    source_report = json.loads((SOURCE/'sources.json').read_text())
    reference_paths = [Path(name) for name in pair['input_sha256']
                       if Path(name).name == 'FORMAL_PRIMARY_FIELDS.bin']
    if len(reference_paths) != 1:
        raise ValueError('봉인 CMFGEN J71 경로를 유일하게 결정할 수 없음')
    reference_path = reference_paths[0]
    if (pair['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or pair['completed_F_calls'] != 2 or
            pair['material_record'] != 70 or pair['reference_record'] != 71 or
            sha(LIBRARY) != pair['input_sha256'][str(LIBRARY)] or
            sha(PAIR/'Jref.npy') != pair['Jref']['sha256'] or
            sha(PAIR/'Fref.npy') != pair['F_reference']['sha256'] or
            sha(INPUT/'boundary/preparation.json') !=
                pair['input_sha256'][str(INPUT/'boundary/preparation.json')] or
            sha(INPUT/'cells/cells.json') != pair['input_sha256'][str(INPUT/'cells/cells.json')] or
            sha(reference_path) != preparation['reference_field_sha256'] or
            sha(SOURCE/'sources.bin') != preparation['reused_source_binary_sha256'] or
            source_report['output_sha256']['sources.bin'] != preparation['reused_source_binary_sha256']):
        raise ValueError('물질70/J71/원계수/기존 Fref 봉인 신원 불일치')
    record = dict(schema='lumina-native-spatial-fixed-Jref-v1',status='PREPARING',
        physical_equivalence='NOT_JUDGED',final_goal='NOT_MET',F_calls_completed=0,
        resources=dict(host='lageunha',CPUs=16,GPU=0,memory_GiB=64,
                       wall_seconds_max=5400,F_calls_max=3,automatic_retry=0),
        input_sha256={str(path):sha(path) for path in
            (PAIR/'execution.json',PAIR/'Jref.npy',PAIR/'Fref.npy',
             INPUT/'boundary/preparation.json',INPUT/'cells/cells.json',
             SOURCE/'sources.json',SOURCE/'sources.bin',reference_path,LIBRARY)},
        source_sha256={str(Path(__file__)):sha(Path(__file__)),
                       str(Path(__file__).with_name('lumina_outer_native_spatial.py')):
                           sha(Path(__file__).with_name('lumina_outer_native_spatial.py'))})
    save_record(root/'execution.json',record)
    library = build_variant(root/'src')
    record['variant_library_sha256'] = sha(library)
    record['variant_C_sha256'] = sha(root/'src/cmf_outer_transport.c')
    save_record(root/'execution.json',record)
    with PreparedInputs(LIBRARY,INPUT/'boundary',INPUT/'cells') as loaded:
        if loaded.plan.n != 89 or len(loaded.nu) != 1115020:
            raise ValueError('원 native90·전ν 계획이 아님')
        jref = np.load(PAIR/'Jref.npy',mmap_mode='r',allow_pickle=False)
        fref = np.load(PAIR/'Fref.npy',mmap_mode='r',allow_pickle=False)
        if jref.shape != (91,len(loaded.nu)) or fref.shape != jref.shape:
            raise ValueError('기존 장 차원 오류')
        geometry,formal = fields(reference_path)
        try:
            if (formal['field'].shape[1:] != (8,90) or
                    not np.allclose(geometry[0]*1e10,loaded.plan.g.native_radius,
                                    rtol=1e-12,atol=0)):
                raise ValueError('CMFGEN 원반경/장 신원 불일치')
            native_j,j_difference = project_native_scattering(formal,loaded.nu)
        finally:
            formal._mmap.close()
        native_chi,native_es,native_true_eta = read_auxiliary(
            SOURCE/'sources.bin',source_report,loaded.nu,list(range(90)))
        # 원 원자/물질 계수와 같은 native 위치의 CMFGEN 산란 소스를 사용한다.
        native_total_eta = native_true_eta + native_es*native_j
        if not np.isfinite(native_total_eta).all() or np.any(native_total_eta < 0):
            raise ValueError('CMFGEN 산란을 더한 native 총방출률 오류')
        record['source_rule'] = dict(native_index='outer0..inner89',
            native_chi='raw (es+(bf+ff))+bb, signed cm^-1',
            native_eta='raw (bf+ff)+bb + raw_es*CMFGEN formal J_source',
            J_source_field=7,J_formal_field=0,frequency='log(nu)-linear, no extrapolation',
            spatial='linear in physical radius inside original material cells',
            J_source_vs_J_formal_max_row_normalized=j_difference,
            exterior='unchanged P0 auxiliary nodes/power law',
            inner_boundary='unchanged P0 Planck diffusion')
        save_record(root/'execution.json',record)
        results = {}
        for label,parts,use_native in (('P0-4',4,False),('native-2',2,True),('native-4',4,True)):
            plan = NativeSpatialGeometry(loaded.plan.g,subdivisions=parts)
            record['status'] = label+'-PENDING'
            save_record(root/'execution.json',record)
            with _operator(library,plan,loaded) as operator:
                if use_native:
                    set_native_fixed(operator,native_chi,native_total_eta,jref)
                output,info = operator.apply_j(jref)
            path = root/(label+'.npy')
            saved = save_array(path,output)
            results[label] = output
            record[label] = dict(field=saved,half_segments=plan.half_segments,
                actual_ray_threads=info['actual_ray_threads'],
                reference_residual=field_residual(jref,output,jref,loaded.nu),
                difference_from_original_P0=field_residual(fref,output,jref,loaded.nu))
            record['F_calls_completed'] += 1
            record['status'] = label+'-SAVED'
            save_record(root/'execution.json',record)
        record['native_refinement_difference'] = field_residual(
            results['native-2'],results['native-4'],jref,loaded.nu)
        record['P0_segment_refinement_difference'] = record['P0-4']['difference_from_original_P0']
    if (sha(PAIR/'Jref.npy') != pair['Jref']['sha256'] or
            sha(SOURCE/'sources.bin') != source_report['output_sha256']['sources.bin']):
        raise ValueError('원본 참조/원계수 실행 후 신원 변경')
    record['status'] = 'FIXED_J_NATIVE_SPATIAL_DIAGNOSTIC_COMPLETE'
    record['elapsed_seconds'] = time.monotonic()-started
    record['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_record(root/'execution.json',record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.out_root)
    print(json.dumps(dict(root=str(args.out_root),status=result['status'],
        F_calls_completed=result['F_calls_completed'],elapsed_seconds=result['elapsed_seconds'],
        peak_rss_KiB=result['peak_rss_KiB']),ensure_ascii=False,allow_nan=False),flush=True)
