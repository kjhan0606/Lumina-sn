#!/usr/bin/env python3
"""봉인 native90의 첫 자체장 최대2회와 동일 원래점 Si/S 율 비교.

준비는 실행하지 않는다. 실제 실행은 새 root에서1회만 하며 실패한 부분장을
완성장으로 출판하지 않는다. J2가 있으면 J1의잔차만 계산한다. SE/RE는 없다.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

import lumina_native_columns as N
from lumina_local_micro import open_geometry
from lumina_compiled_input import CompiledInput
from lumina_nodal_driver import CompiledDriverPlan, DriverKernel, DriverFailure
import lumina_native90_rates as R


LIMITS = dict(cpus=1, memory_GiB=8, wall_seconds=3600, internal_seconds=3300,
              max_F_calls=2, F_seconds=1500, interval_seconds=60,
              max_full_vectors=2000000, max_interval_vectors=200000, max_leaves=4096,
              max_depth=32, cache_MiB=128, output_GiB=10, automatic_retries=0, SE_RE_updates=0)
NATIVE_SHAPE = (90, 1115020)


def write_new(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def prepare(config, preparation, preparation_sha256, library, output):
    config, preparation, library, output = map(lambda p: Path(p).resolve(), (config, preparation, library, output))
    if output.exists(): raise ValueError('기존 native90 계약 보존')
    if DriverKernel(library)._native_driver is None: raise ValueError('봉인 native90 API가 없는 바이너리')
    c = json.loads(config.read_text()); old = json.loads(preparation.read_text())
    with N.open_sealed_native(config, preparation, preparation_sha256) as source:
        if (source.ns, source.nf) != NATIVE_SHAPE: raise ValueError('현재 봉인 native90 전격자 밖')
        geometry = open_geometry(preparation); plan = CompiledDriverPlan(geometry, source, panels=16, block_width=256)
        boundary = {}
        for name in ('inner_planck.npy', 'inner_gradient.npy'):
            path = preparation.parent/name
            if N.sha(path) != old['output_sha256'][name]: raise ValueError('봉인 내부 확산 경계 변경')
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if values.shape != (source.nf,) or not np.isfinite(values).all() or (name == 'inner_planck.npy' and np.any(values < 0)):
                    raise ValueError('실제 내부 경계의 형상/값 오류')
            finally: values._mmap.close()
            boundary[name] = dict(path=str(path), sha256=old['output_sha256'][name])
        source_report = json.loads((Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json').read_text())
        initial_plan = json.loads(Path(c['plan']).read_text())
        # 큰 실제 수송을 시작하기 전에 native90 전체FL 소비 가능성을 확인한다.
        rows, models, routes, state, do_dis, full_check, smoothing = R.material_state(c, initial_plan, source, source_report)
        del rows, models, routes, state
        fixed = [config, preparation, preparation.parent/'geometry.npz', Path(c['plan']),
                 Path(c['seed'])/'material-seed.json', Path(c['seed'])/'native_material_nodes.bin',
                 Path(c['native_full_population']), Path(c['grid'])]
        output_bytes = 2*3*source.ns*source.nf*8+6*4096+10*1024**2
        if output_bytes > LIMITS['output_GiB']*1024**3: raise ValueError('계획된 출력10GiB 초과')
        if shutil.disk_usage(output.parent).free < output_bytes+1024**3: raise ValueError('출력과1GiB 여유 공간 부족')
        record = dict(schema='lumina-native90-first-pair-v1', shape=[source.ns, source.nf],
            config=str(config), config_sha256=N.sha(config), preparation=str(preparation),
            preparation_sha256=preparation_sha256, library=str(library), kernel_sha256=N.sha(library),
            driver_wrapper_sha256=N.sha(Path(__file__).with_name('lumina_nodal_driver.py')),
            source_identity=source.identity, plan_layout=plan.layout(), boundary=boundary,
            input_sha256={str(p): N.sha(p) for p in fixed},
            code_sha256={str(p): N.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            c_source_sha256={str(p): N.sha(p) for suffix in ('*.c','*.h')
                             for p in sorted((library.parents[2]/'src').glob(suffix))},
            limits=LIMITS, initial_J='IDENTICALLY_ZERO_OWN_NATIVE90_NOT_REFERENCE',
            material_preflight=full_check, do_lev_dissolution=do_dis, phot_preprocessing=smoothing,
            expected_new_output_bytes=output_bytes, accepted_physical_field=False, final_goal='NOT_MET')
    write_new(output, record)


class OwnField:
    def __init__(self, source):
        self.ns, self.nf = source.ns, source.nf; self.radius, self.frequency = source.radius, source.frequency
        self.values = np.zeros((source.ns, source.nf)); self.values.flags.writeable = False
        self.identity = dict(role='OWN_NATIVE90_ZERO_START_NOT_REFERENCE', iteration=0)
    @property
    def closed(self): return self.values is None
    def read(self, rows, columns): return self.values[np.ix_(rows, columns)]
    def close(self):
        if isinstance(self.values, np.memmap): self.values._mmap.close()
        self.values = None


def pair_residual(first, second):
    if first.shape != second.shape or first.ndim != 2: raise ValueError('실제 장 쌍 형상 불일치')
    maximum = absolute = 0.
    for begin in range(0, first.shape[1], 65536):
        a, b = first[:, begin:begin+65536], second[:, begin:begin+65536]
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
            raise ValueError('실제 장 쌍의 비유한/음수 값')
        maximum=max(maximum,float(a.max()));absolute=max(absolute,float(abs(b-a).max()))
    return dict(input_iteration=1, output_iteration=2, absolute=absolute,
                relative=absolute/maximum if maximum else (0. if absolute == 0 else None),
                normalization='max(J1); this is J1 residual, NOT J2 residual', J2_residual='NOT_MEASURED')


def run(contract_path, contract_sha256):
    contract_path = Path(contract_path).resolve(); root = contract_path.parent
    if N.sha(contract_path) != contract_sha256: raise ValueError('실제 실행 계약 변경')
    c = json.loads(contract_path.read_text())
    if c['limits'] != LIMITS or c['schema'] != 'lumina-native90-first-pair-v1' or c['shape'] != list(NATIVE_SHAPE):
        raise ValueError('봉인 실제 실행 범위/자원 오류')
    for group in ('input_sha256', 'code_sha256', 'c_source_sha256'):
        if any(N.sha(Path(p)) != value for p, value in c[group].items()): raise ValueError('준비 뒤 입력/소스 변경')
    if N.sha(Path(c['library'])) != c['kernel_sha256']: raise ValueError('준비 뒤 바이너리 변경')
    for entry in c['boundary'].values():
        if N.sha(Path(entry['path'])) != entry['sha256']: raise ValueError('준비 뒤 확산 경계 변경')
    if (root/'campaign.json').exists(): raise ValueError('기존 실제 결과 보존')
    with (root/'campaign.started').open('x') as stream: stream.write(contract_sha256+'\n')
    started=time.monotonic();deadline=started+LIMITS['internal_seconds']
    record=dict(schema='lumina-native90-first-pair-result-v1', contract_sha256=contract_sha256,
                status='STARTED', phase='opening', F_reservations=0, completed=[], SE_RE_updates=0,
                accepted_physical_field=False, final_goal='NOT_MET')
    def checkpoint():
        record.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        path=root/'campaign.partial';path.write_text(json.dumps(record, indent=2, allow_nan=False));path.replace(root/'campaign.json')
    def remaining():
        value=deadline-time.monotonic()
        if value<=0: raise TimeoutError('실제 묶음3300초 한도')
        return value
    checkpoint();code=1
    try:
        with N.open_sealed_native(c['config'],c['preparation'],c['preparation_sha256']) as source:
            if source.identity != c['source_identity'] or (source.ns,source.nf)!=NATIVE_SHAPE:
                raise ValueError('실제 원본 계수 신원/전격자 변경')
            geometry=open_geometry(c['preparation']);plan=CompiledDriverPlan(geometry,source,panels=16,block_width=256)
            field=OwnField(source);kernel=DriverKernel(c['library'])
            planck=np.load(c['boundary']['inner_planck.npy']['path'],mmap_mode='r',allow_pickle=False)
            gradient=np.load(c['boundary']['inner_gradient.npy']['path'],mmap_mode='r',allow_pickle=False)
            try:
                with CompiledInput(c['library'],geometry,source,field,slab_width=128,cache_rows=128,profiles=1024) as provider:
                    provider.bind_view()
                    for iteration in (1,2):
                        seconds=min(1500.,remaining()-300.)
                        if seconds<=0: record['unmet']='time reserved for publication/rates';break
                        record.update(phase='APPLY_PENDING',F_reservations=record['F_reservations']+1);checkpoint()
                        try:
                            value,report=kernel.execute_native90(plan,provider,planck,gradient,c,max_seconds=seconds)
                        except DriverFailure as error:
                            record.update(failed_F=iteration,failed_F_report=error.report,unmet=str(error));checkpoint();break
                        moments=report.pop('moments')
                        if moments.shape!=(3,*NATIVE_SHAPE) or report['physical_intervals']!=c['plan_layout']['physical_intervals']:
                            raise ValueError('완료되지 않은 native 전체장')
                        record['phase']='SAVING_COMPLETE_FIELD';checkpoint()
                        entry=dict(iteration=iteration,input_iteration=iteration-1,report=report,
                            input_J=('IDENTICALLY_ZERO' if iteration==1 else record['completed'][0]['J']))
                        for index,name in enumerate(('J','H','K')):
                            path=root/f'{name}-{iteration:06d}.npy'
                            if path.exists(): raise ValueError('기존 실제 장 덮어쓰기 금지')
                            for begin in range(0,source.nf,65536):
                                block=moments[index,:,begin:begin+65536]
                                if not np.isfinite(block).all() or (name!='H' and np.any(block<0)):
                                    raise ValueError('완료 장의 비유한/음수 모멘트')
                            with path.open('xb') as stream: np.save(stream,moments[index],allow_pickle=False)
                            entry[name]=dict(path=str(path),sha256=N.sha(path),bytes=path.stat().st_size)
                            remaining()
                        record['completed'].append(entry);record['phase']='FIELD_SAVED';checkpoint()
                        del value,moments;gc.collect()
                        if iteration==1:
                            previous=field.values;field.values=np.load(entry['J']['path'],mmap_mode='r',allow_pickle=False)
                            field.identity=dict(role='OWN_NATIVE90_ITERATE_NOT_REFERENCE',iteration=1,J_sha256=entry['J']['sha256'])
                            provider.bind_view();del previous
                field.close()
                if len(record['completed'])==2:
                    a,b=[np.load(e['J']['path'],mmap_mode='r',allow_pickle=False) for e in record['completed']]
                    try:record['J1_residual']=pair_residual(a,b)
                    finally:a._mmap.close();b._mmap.close()
                if record['completed']:
                    record['phase']='RATE_COMPARISON';checkpoint()
                    record['rates']=R.compare(c['config'],source,c,record['completed'],root/'rates',deadline)
                remaining()
                count=len(record['completed'])
                record.update(status='COMPLETED_UNVERIFIED_NATIVE_PAIR' if count==2 else 'INCOMPLETE_NATIVE_PAIR',phase='finished')
                code=0 if count==2 else 1
            finally:
                field.close();planck._mmap.close();gradient._mmap.close()
    except Exception as error:
        record.update(status='INCOMPLETE_OR_FAILED',error_type=type(error).__name__,error=str(error))
    finally:checkpoint()
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'))
    for name in ('config','preparation','preparation-sha256','library','output','contract','contract-sha256'):
        p.add_argument('--'+name)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.config,a.preparation,a.preparation_sha256,a.library,a.output)
    else:raise SystemExit(run(a.contract,a.contract_sha256))
