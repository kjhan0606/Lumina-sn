#!/usr/bin/env python3
"""정확한 private 수송 소스에 첫 행렬 전용 포획 fixture만 추가한다."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import ARGUMENTS,FLAGS,command,once,sha


def prepare(parent,out):
    if out.exists():raise ValueError('기존 공동 반응 fixture 보존')
    record=json.loads((parent/'sweep-build.json').read_text())
    for name,h in record['output_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('private 수송 부모 변경: '+name)
    out.mkdir()
    fields=('ND_STORE HN_DEF_ON_NODES HBC NBC HNU_AT_OB HNU_AT_IB NNU_AT_OB NNU_AT_IB '
        'HNU_AT_OB_PREV HNU_AT_IB_PREV NNU_AT_OB_PREV NNU_AT_IB_PREV '
        'R_STORE EXT_RMID_STORE GAM_REL_STORE JNU_STORE HNU_STORE KNU_STORE NNU_STORE').split()
    overlay='\n'.join('full_raw%'+k+'=fresh_raw%'+k for k in fields)
    overlay+='\nfull_raw%RMID_STORE(1:n-1)=fresh_raw%RMID_STORE(1:n-1)'
    packet=Path(__file__).with_name('replay_cmfgen_coupled_packet.f90').read_text().replace('__OVERLAY_RAW__',overlay)
    body=(parent/'mom_jrel_v9.f').read_text()
    body=once(body,'\tINTEGER ICOUNT\n','\tINTEGER ICOUNT,LUM_PACKET_LU\n')
    hook='''\t  IF(ICOUNT/=1.OR..NOT.LUM_SELECTED.OR.ND/=ND_SM)STOP 88
\t  CALL LUMINA_SNAPSHOT_IO('TRIAL_ENTRY.bin',LUM_A,LUM_J,LUM_R,.FALSE.)
\t  CALL LUMINA_SNAPSHOT_IO('TRIAL_KERNEL.bin',LUM_A,LUM_JK,LUM_RK,.FALSE.)
\t  OPEN(NEWUNIT=LUM_PACKET_LU,FILE='TRIAL_MATRIX.bin',STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED')
\t  WRITE(LUM_PACKET_LU) 'LUMINACOUPLEMAT1',ND,FREQ,LUM_TRI,LUM_SLOPE
\t  WRITE(LUM_PACKET_LU) 'LUMINACOUPLEEND1'
\t  CLOSE(LUM_PACKET_LU)
\t  STOP 96
'''
    body=once(body,'\t  CALL THOMAS(TA,TB,TC,XM,ND,1)',hook+'\t  CALL THOMAS(TA,TB,TC,XM,ND,1)')
    (out/'mom_jrel_v9.f').write_text(body)
    (out/'replay_cmfgen_coupled_packet.f90').write_text(packet)
    native=Path(record['native']);formal=Path(record['formal_build']);mom=Path(record['mom_build'])
    includes=['-I'+str(parent),'-I'+str(formal),'-I'+str(mom),'-I'+str(native/'mod')]
    command(['gfortran',*FLAGS,*includes,'-c','mom_jrel_v9.f'],out)
    command(['gfortran','-O','-fopenmp','-ffree-line-length-none',*includes,'-c','replay_cmfgen_coupled_packet.f90'],out)
    shutil.copy2(parent/'libsubs.a',out/'libsubs.a')
    command(['ar','r','libsubs.a','mom_jrel_v9.o'],out)
    common=[parent/(k+'.o') for k in ('cmfgen_mom_acceleration','lumina_affine_boundary','lumina_mom_diag','lumina_formal_diag','lumina_whole_monitor')]
    common += [formal/'ext_rel_grid_v4.o',mom/'mod_jrel_v9.o']
    command(['gfortran','-fopenmp','-m64','replay_cmfgen_coupled_packet.o',*common,'-L.','-L'+str(native/'lib'),
        '-lsubs','-lpgplt','-lmod',native/'lib/libmy_lapack.a',native/'lib/libmy_blas.a','-o','coupled_packet.exe'],out)
    paths=[out/k for k in ('mom_jrel_v9.f','replay_cmfgen_coupled_packet.f90','libsubs.a','coupled_packet.exe')]
    report=dict(scope='같은 주파수의 소스 반응을 첫 MOM 풀이 전 포획; 생산 해/반환 아님',parent=str(parent),
        parent_manifest_sha256=sha(parent/'sweep-build.json'),output_sha256={str(p):sha(p) for p in paths},
        generator_sha256=sha(Path(__file__)),template_sha256=sha(Path(__file__).with_name('replay_cmfgen_coupled_packet.f90')))
    with (out/'coupled-build.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(build=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.out)
