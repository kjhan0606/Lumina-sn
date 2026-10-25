#!/usr/bin/env python3
"""동일 native90 물질의 전체 자체장을 SC로 계산하고 재개 가능한 상태를 보존한다."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from lumina_compiled_input import CompiledInput
from lumina_fullsr_sc import Kernel, SCPlan, SCALED
import lumina_local_micro as M
from lumina_nodal_driver import CompiledDriverPlan

BASE = Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
BASE_SHA = 'f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab'


def publish(path, record):
    temp = path.with_suffix('.temporary')
    with temp.open('w') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temp.replace(path)


class InputField:
    """완성 자체 J 파일만 연결한다. 참조장 주입은 하지 않는다."""
    def __init__(self, source, path, digest):
        if M.N.sha(path) != digest: raise ValueError('입력 자체장 신원 변경')
        previous = json.loads((Path(path).parent/'sc-result.json').read_text())
        if (previous.get('schema') != 'lumina-fullsr-sc-native90-v1' or
                previous.get('complete_original_grid') is not True or
                previous.get('source_identity') != source.identity or
                previous.get('J') != dict(path=str(Path(path).resolve()), sha256=digest)):
            raise ValueError('동일 물질에서 완성된 자체장 계보 필요')
        self.values = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.values.shape != (source.ns, source.nf) or self.values.dtype != np.dtype('f8'):
            self.values._mmap.close(); raise ValueError('입력 자체장 형상 오류')
        self.ns, self.nf, self.radius, self.frequency = source.ns, source.nf, source.radius, source.frequency
        self.closed = False
        self.identity = dict(role='OWN_NATIVE90_COMPLETED_FIELD', path=str(path), sha256=digest)
    def close(self):
        self.values._mmap.close(); self.closed = True


def verify_stage(root):
    manifest = json.loads((root/'stage.json').read_text())
    for name, digest in manifest['files'].items():
        if M.N.sha(root/'code'/name) != digest: raise ValueError('고정 실행 코드 변경: '+name)
    return M.N.sha(root/'stage.json')


def run(root, *, seconds=2500, threads=16, spatial_subdivisions=4, frequency_subdivisions=1,
        resume=False, input_j=None, input_j_sha256=None):
    if (type(seconds) is not int or not 1 <= seconds <= 2500 or type(threads) is not int or
            not 1 <= threads <= 16 or (input_j is None) != (input_j_sha256 is None)):
        raise ValueError('기존 CPU16·내부2500초 한도 및 자체장 해시 필요')
    root = Path(root).resolve(); output = root/'sc-result.json'; library = root/'fullsr_sc.so'
    if output.exists() and not resume: raise FileExistsError('기존 SC 결과 보존; 재개는 명시적으로 선택')
    if resume and not output.exists(): raise FileNotFoundError('재개할 SC 결과 없음')
    if M.N.sha(BASE) != BASE_SHA: raise ValueError('원 봉인 계획 변경')
    c = json.loads(BASE.read_text()); start = time.monotonic(); deadline = start+seconds
    stage_digest = verify_stage(root)
    identity = dict(input_plan_sha256=BASE_SHA, stage_sha256=stage_digest,
        kernel_sha256=M.N.sha(library), threads=threads, spatial_subdivisions=spatial_subdivisions,
        frequency_subdivisions=frequency_subdivisions,
        input_J='ZERO_OWN_START' if input_j is None else dict(path=str(Path(input_j).resolve()), sha256=input_j_sha256))
    if resume:
        record = json.loads(output.read_text())
        if record['identity'] != identity or record.get('complete_original_grid'):
            raise ValueError('재개 신원 불일치 또는 이미 완성된 계산')
        record['previous_elapsed_seconds'] = record.get('previous_elapsed_seconds', 0.)+record['elapsed_seconds']
        record['invocations'] += 1
        record.update(status='STARTED', phase='OPEN_INPUTS')
    else:
        record = dict(schema='lumina-fullsr-sc-native90-v1', identity=identity, status='STARTED',
            phase='OPEN_INPUTS', final_goal='NOT_MET', accepted_physical_field=False, SE_RE_updates=0,
            complete_original_grid=False, next_frequency=1115019, invocations=1, completed_frequencies=0,
            space_updates=0, observation_updates=0, checkpoint=None,
            limits=dict(cpus=threads, memory_GiB=32, internal_seconds=seconds, parent_seconds=2700,
                        automatic_retries=0),
            scope='원90점·전1115020주파수 자체장; 공간·주파수 수렴과 물리 동질성은 별도 판정')
    moments = field = None

    def save():
        record.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(output, record)

    save()
    try:
        for key in ('cpu_library',):
            if M.N.sha(c[key]) != c['sealed_sha256'][c[key]]: raise ValueError('기존 CPU 바이너리 변경')
        with M.N.open_sealed_native(c['config'], c['preparation'], c['preparation_sha256']) as source:
            if source.identity != c['source_identity'] or (source.ns, source.nf) != (90, 1115020):
                raise ValueError('원 물질/계수/native90 신원 변경')
            geometry = M.open_geometry(c['preparation'])
            original = CompiledDriverPlan(geometry, source, panels=1, block_width=128)
            record.update(source_identity=source.identity, original_layout=original.layout())
            p = json.loads(Path(c['preparation']).read_text()); boundary = []
            for name in ('inner_planck.npy', 'inner_gradient.npy'):
                path = Path(c['preparation']).parent/name
                if M.N.sha(path) != p['output_sha256'][name]: raise ValueError('원 내부 경계 변경')
                boundary.append(np.load(path, mmap_mode='r', allow_pickle=False))
            field = M.ZeroTrial(source) if input_j is None else InputField(source, input_j, input_j_sha256)
            with CompiledInput(c['cpu_library'], geometry, source, field,
                               slab_width=128, cache_rows=128, profiles=1024) as provider:
                if input_j is None: provider.bind_zero_trial()
                else: provider.bind_view()
                record['phase'] = 'PREPARE_DENSE'; save()
                record['dense_preparation'] = provider.prepare_dense(max_seconds=min(300., deadline-time.monotonic()))
                plan = SCPlan(original, provider, c['cpu_library'], spatial_subdivisions=spatial_subdivisions)
                record['layout'] = plan.layout(); record['phase'] = 'FULL_FREQUENCY_SWEEP'; save()
                partial = root/'moments.partial.npy'
                if resume:
                    moments = np.load(partial, mmap_mode='r+', allow_pickle=False)
                    checkpoint = record['checkpoint']
                    if checkpoint:
                        path = root/checkpoint['blue']
                        if M.N.sha(path) != checkpoint['sha256']: raise ValueError('청색 재개 상태 변경')
                        blue = np.load(path, allow_pickle=False)
                    else: blue = np.zeros(plan.nvertex, dtype=SCALED)
                else:
                    if partial.exists(): raise FileExistsError('부분 모멘트 원본 보존')
                    if shutil.disk_usage(root).free < 4*1024**3: raise OSError('출력 여유4GiB 미달')
                    moments = np.lib.format.open_memmap(partial, mode='w+', dtype='f8', shape=(3, source.ns, source.nf))
                    blue = np.zeros(plan.nvertex, dtype=SCALED)
                kernel = Kernel(library)
                while record['next_frequency'] >= 0:
                    remaining = deadline-time.monotonic()
                    if remaining <= 5.: break
                    first = record['next_frequency']; last = max(0, first-16383)
                    code, report = kernel.advance(plan, provider, *boundary, blue, moments,
                        first=first, last=last, tile_width=128, threads=threads,
                        max_seconds=remaining-4., frequency_subdivisions=frequency_subdivisions)
                    record['last_batch'] = report
                    for key in ('completed_frequencies', 'space_updates', 'observation_updates'):
                        record[key] += report[key]
                    if report['completed_frequencies']:
                        moments.flush()
                        name = 'blue-'+str(report['next_frequency'])+'.npy'
                        with (root/name).open('xb') as stream: np.save(stream, blue, allow_pickle=False)
                        record['checkpoint'] = dict(blue=name, sha256=M.N.sha(root/name))
                        record['next_frequency'] = report['next_frequency']
                    save()
                    print(json.dumps(dict(status='SC_PROGRESS', next_frequency=record['next_frequency'],
                        completed=record['completed_frequencies'], elapsed_seconds=record['elapsed_seconds'])), flush=True)
                    if code:
                        record['status'] = 'TIME_LIMIT_PARTIAL' if code == 4 else 'NUMERIC_FAILURE_PARTIAL'
                        break
                if record['next_frequency'] >= 0:
                    if record['status'] == 'STARTED': record['status'] = 'TIME_LIMIT_PARTIAL'
                else:
                    record['phase'] = 'VERIFY_COMPLETE_FIELD'; save()
                    for begin in range(0, source.nf, 16384):
                        values = moments[:, :, begin:begin+16384]
                        if not np.isfinite(values).all() or np.any(values[[0, 2]] < 0):
                            raise ValueError('완성 모멘트의 비유한/음수 값')
                    moments.flush(); moments._mmap.close(); moments = None
                    target = root/'moments.npy'
                    if target.exists(): raise FileExistsError('완성 모멘트 보존')
                    partial.replace(target)
                    values = np.load(target, mmap_mode='r', allow_pickle=False)
                    jpath = root/'J.npy'
                    if jpath.exists(): raise FileExistsError('완성 J 보존')
                    j = np.lib.format.open_memmap(jpath, mode='w+', dtype='f8', shape=(source.ns, source.nf))
                    for begin in range(0, source.nf, 16384): j[:, begin:begin+16384] = values[0, :, begin:begin+16384]
                    j.flush(); j._mmap.close(); values._mmap.close()
                    record.update(status='COMPLETE_TRANSPORT_NOT_PHYSICAL_ACCEPTANCE', complete_original_grid=True,
                        completed_actual_F_calls=1,
                        moments=dict(path=str(target), sha256=M.N.sha(target)),
                        J=dict(path=str(jpath), sha256=M.N.sha(jpath)))
                record['source_report'] = provider.report()
            for values in boundary: values._mmap.close()
    except Exception as exc:
        record.update(status='ERROR_PARTIAL_PRESERVED', error=type(exc).__name__+': '+str(exc))
    finally:
        if moments is not None: moments.flush(); moments._mmap.close()
        if isinstance(field, InputField): field.close()
        record['phase'] = 'FINISHED'; save()
    return 0 if record['complete_original_grid'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--seconds', type=int, default=2500); parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--spatial-subdivisions', type=int, default=4)
    parser.add_argument('--frequency-subdivisions', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--input-j', type=Path); parser.add_argument('--input-j-sha256')
    args = parser.parse_args()
    raise SystemExit(run(**vars(args)))
