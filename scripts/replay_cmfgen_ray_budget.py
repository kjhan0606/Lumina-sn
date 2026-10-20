#!/usr/bin/env python3
"""같은 실제 packet에서 관측기 미연결 기준/연결 결과와 계산비용을 비교한다."""
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
from prepare_cmfgen_mom_capture import once, sha


def packet_source():
    source = Path(__file__).with_name('replay_cmfgen_formal_primary.f90').read_text()
    source = once(source, 'use lumina_formal_diag\n', 'use lumina_formal_diag\nuse lumina_ray_cell_budget\n')
    source = once(source, 'call get_command_argument(1,root_dir)', '''real(ldp),allocatable::cell_values(:,:),cell_flux(:)
real(ldp)::cell_error,cell_cpu0,cell_cpu1
integer cell_repeat,cell_order
call get_command_argument(1,root_dir)''')
    marker = "write(6,'(A,I4)')'PRIMARY_PACKET_SOURCE_FIXED_POINT_NOT_ANCHOR_PASS ',fp_calls"
    source = once(source, marker, '''if(cb_enabled())then
 allocate(cell_values(5,n-1),cell_flux(n))
 open(newunit=lu,file='CELL_PACKET.bin',status='new',access='stream',form='unformatted')
 write(lu)n
 do cell_order=4,8,4
  call cpu_time(cell_cpu0)
  do cell_repeat=1,8
   call cb_compute(fa%r,fa%v,fa%p,n,fa%nc,fa%np,fa%freq,cell_order,cell_values,cell_flux,cell_error)
  end do
  call cpu_time(cell_cpu1)
  write(lu)cell_values,cell_flux,cell_error,(cell_cpu1-cell_cpu0)/8
 end do
 close(lu)
 call formal_rays_io('POST_BUDGET_RAYS.bin',fa,.false.,.true.)
 fb=fa;fb%new_freq=.false.
 call formal_rays_io('POST_BUDGET_BLUE.bin',fb,.false.,.false.)
end if
''' + marker)
    return source


def run(build, packet, out):
    if out.exists():
        raise ValueError('기존 실제 packet 적분 결과 보존')
    out.mkdir(); shutil.copy2(__file__, out / Path(__file__).name)
    source = out / 'packet.f90'; source.write_text(packet_source())
    base_exe, observed_exe = out / 'reference.exe', out / 'observed.exe'
    link(source, base_exe, build, reference=True); link(source, observed_exe, build)
    roots = []
    for label, exe, enabled in (('reference', base_exe, ''), ('observed', observed_exe, '1')):
        root = out / label; root.mkdir(); roots.append(root)
        env = dict(os.environ, LUMINA_FORMAL_PRIMARY='1', LUMINA_FORMAL_CELL_BUDGET=enabled,
            LUMINA_FORMAL_PRIMARY_WHOLE='', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='')
        with (root / 'stdout').open('xb') as stdout, (root / 'stderr').open('xb') as stderr:
            p = subprocess.run([str(exe), str(packet)], cwd=root, env=env, stdout=stdout, stderr=stderr, timeout=120)
        if p.returncode != 0:
            raise RuntimeError('실제 packet 관측기 비교 중단: ' + label)
    names = ('PRIMARY_GEOMETRY.bin', 'PRIMARY_FIRST_RAYS.bin', 'PRIMARY_BLUE_FINAL.bin',
        'PRIMARY_RAYS_FINAL.bin', 'PRIMARY_FORMAL_FINAL.bin', 'PRIMARY_H_INTERFACE.bin', 'PRIMARY_REPLAY.csv')
    equal = {name: (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes() for name in names}
    after = dict(rays=(roots[1]/'POST_BUDGET_RAYS.bin').read_bytes() == (roots[1]/'PRIMARY_RAYS_FINAL.bin').read_bytes(),
                 blue=(roots[1]/'POST_BUDGET_BLUE.bin').read_bytes() == (roots[1]/'PRIMARY_BLUE_FINAL.bin').read_bytes())
    if not all(equal.values()) or not all(after.values()):
        raise ValueError('관측기가 실제 packet을 변경함')
    path = roots[1] / 'CELL_PACKET.bin'
    with path.open('rb') as f:
        n, = struct.unpack('<i', f.read(4)); values = np.frombuffer(f.read(), dtype='<f8').copy()
    if len(values) != 2*(5*(n-1)+n+2):
        raise ValueError('부분 packet 셀 적분')
    offset = 0; results = {}; arrays = {}
    for order in (4, 8):
        b = values[offset:offset+5*(n-1)].reshape(n-1, 5); offset += 5*(n-1)
        flux = values[offset:offset+n]; offset += n
        error, seconds = values[offset:offset+2]; offset += 2
        delta = flux[:-1] - flux[1:]
        remainder = delta - b[:, 0] - b[:, 2] - b[:, 3] - b[:, 4]
        scale = max(abs(delta).sum(), *(abs(b[:, i]).sum() for i in (0, 2, 3, 4)))
        results[str(order)] = dict(source=float(b[:, 0].sum()), expansion_work=float(b[:, 1].sum()),
            blue_exchange=float(b[:, 2].sum()), geometry_defect=float(b[:, 3].sum()), source_interpolation_defect=float(b[:, 4].sum()),
            boundary=float(delta.sum()), numerical_identity_signed=float(remainder.sum()), numerical_identity_L1=float(abs(remainder).sum()),
            numerical_identity_L1_over_largest_cell_term=float(abs(remainder).sum()/scale),
            endpoint_reconstruction_max_relative=float(error), seconds_CPU1_per_frequency=float(seconds))
        arrays[str(order)] = b
    difference = float(np.max(abs(arrays['8']-arrays['4'])))
    report = dict(scope='실제66069 packet의 passive 적분/byte 불변/비용; 실제 새 whole 이력이나 앵커 판정 아님',
        packet=str(packet), baseline_and_observer_byte_equal=equal, after_observer_byte_equal=after,
        orders=results, quadrature_4_to_8_max_absolute_cell_term_difference=difference,
        whole_cost_linear_estimate_seconds=1551.566 + results['8']['seconds_CPU1_per_frequency']*197291/16,
        cost_warning='한 실제 주파수의 단순 추정이며 다른 opacity/CPU 병렬 효율의 상한이 아니다.',
        producer_sha256={str(p):sha(p) for p in (Path(__file__),source,build/'ray-budget-build.json',base_exe,observed_exe,path)},
        energy_pass='NOT_JUDGED', final_goal='NOT_JUDGED')
    (out/'packet-cell.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='producer_sha256'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','packet','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.build.resolve(),a.packet.resolve(),a.out.resolve())
