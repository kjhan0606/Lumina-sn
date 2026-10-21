#!/usr/bin/env python3
"""B의 실제 native 구간 세 열만 확인한다. 전ν 비용/장/CMFGEN 일치 판정은 아니다."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

OLD = Path('/gpfs/kjhan/cmfgen_runs/lumina_gpu_batch_local_20261021-xSx3Z2/gpu-local-plan.json')
OLD_SHA = '1865fdfdba8226c267b47a80d902b7a4a7bb2656308cb65b463f09f5f7b81748'
LIMITS = dict(CPUs=1, virtual_GiB=16, wall_seconds=300, internal_seconds=270, interval_seconds=60,
    columns=3, physical_intervals=1, GPUs=0, actual_F_calls=0, SE_RE_updates=0, spatial_relative=1e-5,
    per_column_map_calls=30000, per_column_source_panels=20000000)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def prepare(root):
    root = Path(root).resolve()
    if sha(OLD) != OLD_SHA: raise ValueError('원래 입력·세 점 참조 계약 변경')
    old = json.loads(OLD.read_text()); original_scripts = OLD.parent/'scripts'
    small = {old[k]: old[k+'_sha256'] for k in ('preparation', 'reference', 'origin')}
    small[old['cpu_library']] = old['sealed_sha256'][old['cpu_library']]
    small[old['config']] = old['input_sha256'][old['config']]
    inherited = {p: h for p, h in old['sealed_sha256'].items() if Path(p).parent == original_scripts and p.endswith('.py')}
    sealed = {str(p): sha(p) for p in [*sorted((root/'src').iterdir()), *sorted((root/'scripts').iterdir()), root/'native.so'] if p.is_file()}
    plan = dict(schema='lumina-B-native-selected-once-v1', limits=LIMITS, engine='SPACE_LINEAR_MOMENT_POSITIVE_SOURCE_B',
        original_contract=str(OLD), original_contract_sha256=OLD_SHA, original_scripts=str(original_scripts),
        sealed_sha256=sealed, required_input_sha256=small, inherited_script_sha256=inherited,
        cpu_library=old['cpu_library'], native_library=str(root/'native.so'), config=old['config'],
        preparation=old['preparation'], preparation_sha256=old['preparation_sha256'], source_identity=old['source_identity'],
        interval=next(x for x in old['intervals'] if x['index'] == 13003), columns=old['fine_columns'],
        fine_values=old['fine_values'], fine_relative_criterion=1e-4,
        large_body_identity='봉인된 원래 source identity와 보고서 해시를 상속; 대형 source 본문 SHA 재계산 없음',
        final_goal='NOT_MET', full_frequency_validation=False)
    for group in (small, inherited):
        if any(sha(p) != h for p, h in group.items()): raise ValueError('원래 입력 또는 Python 경로 변경')
    path = root/'selected-plan.json'
    with path.open('x') as stream: json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(sha(path), flush=True)


def run(path, digest):
    path = Path(path).resolve(); root = path.parent
    if sha(path) != digest: raise ValueError('B 세 점 계획 변경')
    plan = json.loads(path.read_text())
    if plan['schema'] != 'lumina-B-native-selected-once-v1' or plan['limits'] != LIMITS: raise ValueError('B 세 점 실행 한도 변경')
    for name in ('sealed_sha256', 'required_input_sha256', 'inherited_script_sha256'):
        if any(sha(p) != h for p, h in plan[name].items()): raise ValueError('B 소스/바이너리/입력 변경')
    with (root/'selected.started').open('x') as stream: stream.write(digest+'\n')
    record = dict(schema=plan['schema'], status='STARTED', plan_sha256=digest, columns=[], GPUh=0,
        actual_F_calls=0, SE_RE_updates=0, full_frequency_validation=False, final_goal='NOT_MET')
    started = time.monotonic(); deadline = started+LIMITS['internal_seconds']
    def publish():
        record.update(elapsed_seconds=time.monotonic()-started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        text = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
        if len(text.encode()) > 1024**2: raise ValueError('세 점 결과1MiB 한도')
        temporary = root/'selected.partial'; temporary.write_text(text); temporary.replace(root/'selected.json')
    code = 1
    try:
        import lumina_local_micro as M
        from lumina_compiled_input import CompiledInput
        from lumina_nodal_driver import CompiledDriverPlan
        from lumina_linear_native import NativeInterval, NativeKernel, Positive, spatial_interval
        publish()
        with M.N.open_sealed_native(plan['config'], plan['preparation'], plan['preparation_sha256']) as source:
            if source.identity != plan['source_identity']: raise ValueError('실제 native source 변경')
            g = M.open_geometry(plan['preparation']); geometry = CompiledDriverPlan(g, source, panels=1, block_width=256)
            spec = plan['interval']; index = spec['index']
            if geometry.arrays['steps'][index].tolist() != spec['step']: raise ValueError('실제 구간 변경')
            field = M.ZeroTrial(source)
            with CompiledInput(plan['cpu_library'], g, source, field, slab_width=128, cache_rows=128, profiles=1024) as provider:
                provider.bind_zero_trial(); record['dense_preparation'] = provider.prepare_dense(); publish()
                kernel = NativeKernel(plan['native_library'])
                for column, reference in zip(plan['columns'], plan['fine_values']):
                    if time.monotonic() >= deadline: raise TimeoutError('B 세 점 전체 시간 한도')
                    interval = NativeInterval(kernel, provider, geometry, index,
                        max_calls=LIMITS['per_column_map_calls'], max_panels=LIMITS['per_column_source_panels'])
                    value, report = spatial_interval(interval.make_maps(column), initial=Positive(),
                        relative=LIMITS['spatial_relative'], seconds=min(LIMITS['interval_seconds'], deadline-time.monotonic()))
                    interval.check(); actual = value.value(); gap = abs(actual/reference-1.)
                    record['columns'].append(dict(column=column, actual=actual, fine_reference=reference, relative_difference=gap,
                        accuracy_met=gap <= plan['fine_relative_criterion'], report=report, map_calls=interval.calls,
                        source_panels=interval.panels, built_bins=interval.bins, tree_combinations=interval.combinations))
                    publish(); print(json.dumps(record['columns'][-1], ensure_ascii=False), flush=True)
                record['status'] = 'B_SELECTED_ACCURACY_MET_NOT_FULL_NU' if all(x['accuracy_met'] for x in record['columns']) else 'B_SELECTED_ACCURACY_NOT_MET'
                code = 0 if record['status'] == 'B_SELECTED_ACCURACY_MET_NOT_FULL_NU' else 1
        if time.monotonic() >= deadline: raise TimeoutError('B 세 점 전체 시간 한도')
    except Exception as error:
        code = 1
        record.update(status='B_SELECTED_NOT_MET', error=type(error).__name__+': '+str(error))
        if hasattr(error, 'last_completed'): record['last_completed_spatial'] = error.last_completed
        if hasattr(error, 'piece'): record['failed_piece'] = error.piece
    finally:
        publish()
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('prepare', 'run')); parser.add_argument('path')
    parser.add_argument('--sha256'); args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.path)
    else: raise SystemExit(run(args.path, args.sha256))
