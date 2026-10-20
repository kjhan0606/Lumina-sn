#!/usr/bin/env python3
"""실제 SC/blue 셀 적분과 node 적분의 차이를 작은 구형 제조해에서 판독한다."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np

from prepare_cmfgen_ray_budget import link
from prepare_cmfgen_mom_capture import sha


def analyze(path):
    with path.open('rb') as f:
        n, nc, np_, nf = struct.unpack('<4i', f.read(16))
        geometry = np.frombuffer(f.read((2*n+np_)*8), dtype='<f8').copy()
    r = geometry[:n]
    dtype = np.dtype([('freq', '<f8'), ('native_flux', '<f8', (n,)),
        ('nodal_source', '<f8', (n,)), ('nodal_work', '<f8', (n,))] +
        [(name + str(order), '<f8', shape) for order in (4, 8) for name, shape in
         (('budget', (n-1, 5)), ('flux', (n,)), ('endpoint', ()))])
    offset = 16 + (2*n+np_)*8
    if path.stat().st_size != offset + nf*dtype.itemsize:
        raise ValueError('부분 ray 셀 자료')
    data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(nf,))
    if any(not np.isfinite(data[name]).all() for name in dtype.names):
        raise ValueError('잘못된 ray 자료')
    freq = data['freq']; weight = np.ones(nf)
    if nf > 1:
        if np.any(np.diff(freq) >= 0):
            raise ValueError('제조해 주파수 coverage')
        weight[0] = .5*(freq[0]-freq[1]); weight[-1] = .5*(freq[-2]-freq[-1])
        weight[1:-1] = .5*(freq[:-2]-freq[2:])
    native = weight @ data['native_flux']
    dr = r[:-1]-r[1:]
    source = weight @ data['nodal_source']; work = weight @ data['nodal_work']
    end = freq[0]*data['nodal_work'][0]-freq[-1]*data['nodal_work'][-1] if nf > 1 else np.zeros(n)
    node_rhs = source-work+end
    nodal = np.sum(.5*dr*(node_rhs[:-1]+node_rhs[1:]))
    report = dict(n=n, nc=nc, nf=nf, native_boundary=float(native[0]-native[-1]),
        nodal_trapezoid_rhs=float(nodal), native_minus_nodal_trapezoid=float(native[0]-native[-1]-nodal), orders={})
    for order in (4, 8):
        b = np.tensordot(weight, data['budget'+str(order)], axes=(0, 0))
        flux = weight @ data['flux'+str(order)]
        delta = flux[:-1]-flux[1:]
        numerical = delta-b[:, 0]-b[:, 2]-b[:, 3]-b[:, 4]
        discrete = delta-b[:, 0]-b[:, 2]
        frequency_end = (freq[0]*data['budget'+str(order)][0, :, 1]-
                         freq[-1]*data['budget'+str(order)][-1, :, 1]) if nf > 1 else np.zeros(n-1)
        continuous = delta-b[:, 0]+b[:, 1]-frequency_end
        scale = max(abs(float(delta.sum())), abs(float(b[:, 0].sum())), abs(float(b[:, 2].sum())))
        frequency_integrated = nf > 1 or np.all(geometry[n:2*n] == 0)
        report['orders'][str(order)] = dict(boundary=float(delta.sum()),
            physical_source=float(b[:, 0].sum()), expansion_work=float(b[:, 1].sum()),
            discrete_blue_exchange=float(b[:, 2].sum()), geometry_interpolation_defect=float(b[:, 3].sum()),
            SC_source_interpolation_defect=float(b[:, 4].sum()), frequency_end=float(frequency_end.sum()),
            numerical_identity_signed_defect=float(numerical.sum()), numerical_identity_L1_defect=float(abs(numerical).sum()),
            physical_discrete_signed_defect=float(discrete.sum()),
            physical_continuous_signed_defect=float(continuous.sum()) if frequency_integrated else None,
            physical_continuous_relative_defect=abs(float(continuous.sum()))/scale if scale and frequency_integrated else None,
            angular_boundary_difference=float((flux[0]-flux[-1])-(native[0]-native[-1])),
            endpoint_reconstruction_max_relative=float(data['endpoint'+str(order)].max()))
    return report


def run(build, out):
    if out.exists():
        raise ValueError('기존 구형 ray 적분 결과 보존')
    out.mkdir(); started = time.monotonic()
    source = Path(__file__).with_name('formal_primary_cell_budget.f90')
    shutil.copy2(source, out / source.name); shutil.copy2(__file__, out / Path(__file__).name)
    exe = out / 'manufactured.exe'; link(source.resolve(), exe, build)
    cases = [('absorb', n, 2*n-2, 64) for n in (17, 33, 65)]
    cases += [('thermal', n, 2*n-2, 64) for n in (17, 65)]
    cases += [('extended', 33, 64, 64)]
    cases += [('sharp', n, 2*n-2, 64) for n in (17, 33, 65)]
    cases += [('sharp', n, 128, 64) for n in (129, 257)]
    cases += [('scatter', n, 128, 64) for n in (65, 129, 257)]
    cases += [('expanding', n, nc, nf) for n, nc, nf in ((17, 32, 64), (33, 64, 128), (65, 128, 256))]
    env = dict(os.environ, LUMINA_FORMAL_PRIMARY='1', LUMINA_FORMAL_CELL_BUDGET='1',
        LUMINA_FORMAL_PRIMARY_WHOLE='', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='')
    results = []
    for mode, n, nc, nf in cases:
        if time.monotonic()-started > 250:
            raise RuntimeError('제조해 총250초 상한; 미완료를 통과로 기록하지 않음')
        case = out / f'{mode}_{n}_{nc}_{nf}'; case.mkdir(); launch = time.monotonic()
        with (case/'stdout').open('xb') as stdout, (case/'stderr').open('xb') as stderr:
            p = subprocess.run([str(exe), mode, str(n), str(nc), str(nf)], cwd=case, env=env,
                stdout=stdout, stderr=stderr, timeout=90)
        row = dict(mode=mode, root=str(case), returncode=p.returncode, seconds=time.monotonic()-launch)
        if p.returncode == 0:
            row.update(analyze(case/'RAY_CELL_BUDGET.bin'))
        results.append(row)
    report = dict(scope='독립 ray 공간/주파수 적분의 제조해; 실제90격자/최종 SiS 수용 판정 아님',
        physical_interpolation='실제 ray node chi/eta/b/blue/D를 native s에서 선형 보간; SC source 차이와 기하 오차는 별도 공개',
        geometry='r^2 dr dmu/2 = p dp ds/(2 D^3); p^2 방향 trapezoid와 native angular 경계량을 별도 비교',
        cases=results, resources='CPU1/4GiB/300s, GPU0', elapsed_seconds=time.monotonic()-started,
        producer_sha256={str(p):sha(p) for p in (Path(__file__),source,build/'ray-budget-build.json',exe)},
        energy_pass='NOT_JUDGED', final_goal='NOT_JUDGED')
    (out/'ray-manufactured.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(out), elapsed_seconds=report['elapsed_seconds'],
        cases=[{k:v for k,v in row.items() if k not in ('root','orders')} | {'order8':row.get('orders',{}).get('8')} for row in results]),ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.build.resolve(),a.out.resolve())
