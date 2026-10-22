#!/usr/bin/env python3
"""같은 외부대기 수송 F에서 CMFGEN J71과 저장 J4의 실제 잔차를 짝으로 계산한다."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time

import numpy as np

from lumina_outer_prepared import PreparedInputs
from prepare_lumina_outer_input import sha
from run_cmfgen_formal_primary import fields


PROJECT = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn')
CAMPAIGN = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_campaign_20261021-FAmNPQ')
INPUT = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_reuse_20261021-uTVbnT')
LIBRARY = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_weighted_20261021-B0Fepu/pytest-threadcap-work/outer-c0/outer.so')
CONFIG = Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
DIAGNOSTIC = CAMPAIGN/'comparison-current/diagnostic.json'
STATE = CAMPAIGN/'checkpoint/checkpoint.json'
J4 = CAMPAIGN/'checkpoint/J-000004.npy'
BANDS = ((100., 400.), (400., 800.), (800., 1600.), (1600., 3000.))
H_ERG_S = 6.62607015e-27


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checkpoint(path, value):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def save_array(path, value):
    temporary = path.with_suffix('.npy.partial')
    with temporary.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def field_residual(candidate, applied, reference, frequency):
    ns, nf = candidate.shape
    if applied.shape != candidate.shape or reference.shape != candidate.shape or len(frequency) != nf:
        raise ValueError('잔차 장의 차원/주파수 불일치')
    if not np.isfinite(applied).all() or np.any(applied < 0):
        raise ValueError('반환 F 장 비유한/음수')
    maximum = float(reference.max())
    worst = (0., None, None)
    selected = {}
    band_worst = {f'{lo:g}-{hi:g}A': (0., None) for lo, hi in BANDS}
    masks = [(f'{lo:g}-{hi:g}A',
              (2.99792458e18/frequency >= lo) & (2.99792458e18/frequency < hi))
             for lo, hi in BANDS]
    for shell in range(ns):
        delta = applied[shell]-candidate[shell]
        at = int(np.argmax(np.abs(delta)))
        if abs(delta[at]) > worst[0]:
            worst = (float(abs(delta[at])), shell, at)
        for name, mask in masks:
            scale = 4*np.pi/(H_ERG_S*frequency[mask])
            denominator = float(np.trapezoid(scale*reference[shell, mask], frequency[mask]))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('참조 광자 대역 적분0/비유한')
            signed = float(np.trapezoid(scale*delta[mask], frequency[mask]))/denominator
            unsigned = float(np.trapezoid(scale*abs(delta[mask]), frequency[mask]))/denominator
            if unsigned > band_worst[name][0]:
                band_worst[name] = (unsigned, shell)
            if shell in (0, 37, 38, 45, 75, 88):
                selected.setdefault(shell, {})[name] = dict(signed_over_reference=signed,
                    absolute_over_reference=unsigned)
    return dict(global_max_absolute=worst[0], global_max_over_reference_max=worst[0]/maximum,
        global_max_shell=worst[1], global_max_frequency_Hz=float(frequency[worst[2]]) if worst[2] is not None else None,
        worst_unsigned_band={name:dict(value=value, shell=shell)
                             for name, (value, shell) in band_worst.items()},
        selected_shells=selected)


def run(root):
    root = root.resolve()
    if (socket.gethostname().lower() != 'lageunha' or Path.cwd().resolve() != PROJECT or
            not (PROJECT/'CLAUDE.md').is_file()):
        raise ValueError('Lageunha의 확인된 Lumina-sn 프로젝트에서만 실행')
    if not root.is_dir() or (root/'execution.json').exists() or (root/'Jref.npy').exists():
        raise ValueError('새 증거 root만 사용하고 기존 계산을 덮어쓰지 않음')
    started = time.monotonic()
    config = json.loads(CONFIG.read_text())
    plan_path = Path(config['plan'])
    plan = json.loads(plan_path.read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    state = json.loads(STATE.read_text())
    origin = json.loads((CAMPAIGN/'execution.json').read_text())
    source_names = ('lumina_outer_c_backend.py', 'lumina_outer_prepared.py',
                    'lumina_outer_atmosphere.py', 'lumina_outer_transport.py',
                    'run_cmfgen_formal_primary.py')
    for name in source_names:
        candidates = [value for path, value in origin['source_sha256'].items()
                      if Path(path).name == name]
        if len(candidates) != 1 or sha(Path(__file__).with_name(name)) != candidates[0]:
            raise ValueError('J4를 만든 Python 수송·참조 reader와 소스가 다름: '+name)
    reference_path = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    fixed = {str(path): sha(path) for path in (Path(__file__), CONFIG, plan_path, DIAGNOSTIC,
             STATE, J4, reference_path, LIBRARY, INPUT/'boundary/preparation.json', INPUT/'cells/cells.json')}
    if (config['plan_sha256'] != fixed[str(plan_path)] or
            diagnostic['captured_sha256'][str(CONFIG)] != fixed[str(CONFIG)] or
            diagnostic['captured_sha256'][str(STATE)] != fixed[str(STATE)] or
            diagnostic['comparison_J'] != state['J'] or
            state['J']['sha256'] != fixed[str(J4)] or
            state['contract']['library_sha256'] != fixed[str(LIBRARY)] or
            state['contract']['input_sha256'][str(INPUT/'boundary/preparation.json')] != fixed[str(INPUT/'boundary/preparation.json')] or
            state['contract']['input_sha256'][str(INPUT/'cells/cells.json')] != fixed[str(INPUT/'cells/cells.json')] or
            plan['reference_field_sha256'] != fixed[str(reference_path)] or
            plan['initial_native_record'] != 70 or
            plan['reference_radiation_material_record'] != 70 or
            plan['reference_radiation_record'] != 71 or
            state['completed_iterations'] != 4 or state['accepted_physical_field'] is not False):
        raise ValueError('동일 물질·수송 코드·J4·CMFGEN 기준의 봉인 신원 불일치')
    resources = dict(CPUs=16, GPUs=0, MPI_ranks=0, memory_GiB=64,
                     parent_seconds=3600, F_calls_maximum=2, automatic_retry=0,
                     C_work_budget_GiB=16)
    if shutil.disk_usage(root).free < 4*1024**3:
        raise ValueError('새 장 세 개와 여유 공간 부족')
    record = dict(schema='lumina-outer-reference-fixed-point-pair-v1', status='PREPARING',
                  material_record=70, reference_record=71, input_sha256=fixed,
                  resources=resources, completed_F_calls=0, accepted_physical_field=False,
                  componentwise_verified=False, final_goal='NOT_MET')
    checkpoint(root/'execution.json', record)
    with PreparedInputs(LIBRARY, INPUT/'boundary', INPUT/'cells') as loaded:
        geometry, formal = fields(reference_path)
        try:
            radius = geometry[0]*1e10
            if (len(radius) != 90 or loaded.plan.n != 89 or
                    not np.allclose(radius, loaded.plan.g.native_radius, rtol=1e-12, atol=0) or
                    tuple(loaded.plan.g.reference_indices) != (0, 14)):
                raise ValueError('원래점 반경/외부 두 기준점 대응 오류')
            nu = loaded.nu
            ref_nu = np.asarray(formal['freq'][::-1])*1e15
            if (nu[0] != ref_nu[0] or nu[-1] != ref_nu[-1] or
                    not np.all(np.diff(ref_nu) > 0)):
                raise ValueError('전주파수 범위/순서 불일치; 외삽 금지')
            log_own, log_ref = np.log(nu), np.log(ref_nu)
            projected = np.empty((91, len(nu)), dtype='f8')
            roundtrip = 0.
            for shell in range(91):
                if shell < 89:
                    native = 88-shell
                    original = .5*(formal['field'][::-1, 0, native] +
                                   formal['field'][::-1, 0, native+1])
                else:
                    original = formal['field'][::-1, 0, (0, 14)[shell-89]]
                projected[shell] = np.interp(log_own, log_ref, original)
                if not np.isfinite(projected[shell]).all() or np.any(projected[shell] < 0):
                    raise ValueError('참조 투영 J 비유한/음수')
                back = np.interp(log_ref, log_own, projected[shell])
                roundtrip = max(roundtrip, float(abs(back-original).max()/original.max()))
            record['reference_projection'] = dict(native_nodes=90, material_cells=89,
                auxiliary_native_indices=[0,14], frequency_rule='log(nu)-linear J; no extrapolation',
                roundtrip_max_abs_over_row_max=roundtrip, full_grid=[91,len(nu)],
                J_units='erg s^-1 cm^-2 Hz^-1 sr^-1, comoving')
        finally:
            formal._mmap.close()
        record['Jref'] = save_array(root/'Jref.npy', projected)
        record['status'] = 'REFERENCE_PROJECTED'; checkpoint(root/'execution.json', record)
        j4 = np.load(J4, mmap_mode='r', allow_pickle=False)
        if j4.shape != projected.shape:
            raise ValueError('J4/참조 자체격자 차원 불일치')
        try:
            with loaded.context(16*1024**3, weighted_workers=16, moment_count=1,
                                approved_cpu_limit=16) as operator:
                record['status'] = 'F_REFERENCE_PENDING'; checkpoint(root/'execution.json', record)
                fref, info = operator.apply_j(projected)
                record['F_reference'] = save_array(root/'Fref.npy', fref)
                record['F_reference']['actual_ray_threads'] = info['actual_ray_threads']
                record['completed_F_calls'] = 1
                record['status'] = 'F_REFERENCE_SAVED'; checkpoint(root/'execution.json', record)
                if time.monotonic()-started >= 3000:
                    raise TimeoutError('두번째 F 및 출판을 위한 잔여 시간 부족')
                record['status'] = 'F_J4_PENDING'; checkpoint(root/'execution.json', record)
                f4, info = operator.apply_j(j4)
                record['F_J4'] = save_array(root/'F4.npy', f4)
                record['F_J4']['actual_ray_threads'] = info['actual_ray_threads']
                record['completed_F_calls'] = 2
                record['status'] = 'FIELDS_SAVED'; checkpoint(root/'execution.json', record)
            record['reference_residual'] = field_residual(projected, fref, projected, nu)
            record['J4_residual'] = field_residual(j4, f4, projected, nu)
            record['status'] = 'PAIR_DIAGNOSTIC_COMPLETE'
        finally:
            j4._mmap.close()
    record['elapsed_seconds'] = time.monotonic()-started
    record['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoint(root/'execution.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out_root), ensure_ascii=False, allow_nan=False), flush=True)
