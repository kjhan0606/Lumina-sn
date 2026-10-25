"""완료 반복의 실제 J와 재개 신원을 원자적으로 보존한다. 물리 합격 판정은 아니다."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

PHASES=('ITERATING','ITERATION_COMPLETED','K_NORM_STARTED','K_NORM_COMPLETED',
        'ERROR_ENVELOPE_STARTED','ERROR_ENVELOPE_COMPLETED')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def number(value):
    if np.isnan(value) or value<0:raise ValueError('저장 진단값 손상')
    return float(value) if np.isfinite(value) else 'Infinity'


class FieldCheckpoint:
    def __init__(self,folder,contract,resume_origin=None):
        self.folder=Path(folder)
        if self.folder.exists():raise ValueError('기존 반복 저장 root 보존')
        self.folder.mkdir()
        self.contract=contract;self.origin=resume_origin;self.started=time.monotonic()
        self.events=[];self.kept=[];self.current=None;self.write_seconds=0.;self.error=None

    def record(self,phase,completed,relative,absolute,norm,j):
        entered=time.monotonic()-self.started;save_seconds=0.
        if not 0<=phase<len(PHASES):raise ValueError('알 수 없는 수송 단계')
        if completed and (self.current is None or self.current['iterations']!=completed):
            start=time.monotonic()
            path=self.folder/f'J-{completed:06d}.npy';tmp=path.with_suffix('.npy.tmp')
            with tmp.open('xb') as f:
                np.save(f,j,allow_pickle=False);f.flush();os.fsync(f.fileno())
            value=sha(tmp);os.replace(tmp,path)
            self.current=dict(filename=path.name,sha256=value,iterations=completed)
            self.kept.append(path);save_seconds=time.monotonic()-start
            self.write_seconds+=save_seconds
        self.events.append(dict(phase=PHASES[phase],iterations=completed,
                                entered_seconds=entered,snapshot_seconds=save_seconds))
        state=dict(schema='lumina-field-checkpoint-v1',contract=self.contract,
            phase=PHASES[phase],completed_iterations=completed,
            relative_change=number(relative),absolute_change=number(absolute),K_upper=number(norm),
            J=self.current,resume_origin=self.origin,elapsed_seconds=time.monotonic()-self.started,
            snapshot_seconds_total=self.write_seconds,events=self.events,
            accepted_physical_field=False,componentwise_verified=False)
        state['metadata_sha256']=digest(state)
        tmp=self.folder/'checkpoint.json.tmp'
        with tmp.open('w') as f:
            json.dump(state,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.folder/'checkpoint.json')
        # 새 상태가 가리키는 J를 확정한 뒤, 이 writer가 만든 최근 두 개만 남긴다.
        while len(self.kept)>2:self.kept.pop(0).unlink()


def read_checkpoint(folder,contract):
    folder=Path(folder);path=folder/'checkpoint.json';state=json.loads(path.read_text())
    claimed=state.pop('metadata_sha256')
    if digest(state)!=claimed:raise ValueError('반복 저장 메타데이터 변경')
    if state.get('schema')!='lumina-field-checkpoint-v1' or state.get('contract')!=contract:
        raise ValueError('반복 저장의 입력/바이너리/반복 기준 불일치')
    n=state['completed_iterations'];cap=contract['iteration_cap']
    if type(n) is not int or not 1<=n<=cap:raise ValueError('누적 반복수/상한 불일치')
    relative=float(state['relative_change']);absolute=float(state['absolute_change'])
    if np.isnan(relative) or relative<0 or not np.isfinite(absolute) or absolute<0:
        raise ValueError('반복 저장의 변화량 손상')
    if n==cap and not (n>1 and relative<contract['tolerance']):
        raise ValueError('누적 반복 상한 소진; 재개로 초기화 금지')
    entry=state['J']
    if (not isinstance(entry,dict) or entry['iterations']!=n or
        entry['filename']!=f'J-{n:06d}.npy'):
        raise ValueError('완료 반복의 J 저장 없음/반복수 불일치')
    jpath=folder/entry['filename']
    if sha(jpath)!=entry['sha256']:raise ValueError('저장 J 변경/미완료')
    j=np.load(jpath,mmap_mode='r',allow_pickle=False)
    if j.dtype!=np.dtype('float64') or j.shape!=tuple(contract['shape']):
        raise ValueError('저장 J 차원/형식 불일치')
    flat=j.reshape(-1)
    for begin in range(0,len(flat),131072):
        part=flat[begin:begin+131072]
        if not np.isfinite(part).all() or np.any(part<0):raise ValueError('저장 J 비유한/음수')
    origin=dict(directory=str(folder.resolve()),metadata_sha256=sha(path),J_sha256=entry['sha256'])
    return j,n,relative,absolute,origin
