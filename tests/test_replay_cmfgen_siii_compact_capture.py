"""압축 포획의 광선·청색 상태 재생은 합성 해석해에서 먼저 검사한다."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import replay_cmfgen_siii_compact_capture as compact_replay
from replay_cmfgen_siii_compact_capture import replay_record
from prepare_cmfgen_siii_compact_capture import FIRST


def test_replay_record_constant_source_and_blue_node_check():
    n=6;np_=52;nd=52;depth=.2
    geom=dict(R=np.ones(n),S_P=np.arange(n,0,-1,dtype='f8'),
              S_M=np.arange(n,dtype='f8'),B_P=np.zeros(n),B_M=np.zeros(n),
              MU_P=np.zeros(n),LNK=np.full(nd,3,dtype='i4'))
    g=dict(NC=0,NP=np_,ND=nd,rays=[geom for _ in range(np_)])
    blue=[dict(I_P=np.zeros(n),I_M=np.zeros(n)) for _ in range(np_)]
    minus=lambda k:3.*(-math.expm1(-depth*k))
    plus=lambda k:3.*(-math.expm1(-depth*(2*(n-1)-k)))
    row=dict(np_limit=np_,dlognu=.01,chi_ext=np.full(n,.2),
             eta_ext=np.full(n,.6),
             boundary=[(plus(0),minus(0),plus(n-1),minus(n-1)) for _ in range(np_)],
             points=[[(plus(2),minus(2),0.,0.,0.,0.)
                      for _ in range(np_-id_+1)] for id_ in range(49,53)])
    out,error=replay_record(row,g,blue)
    assert error<2e-13
    np.testing.assert_allclose(out[0]['I_P'][2],plus(2),rtol=2e-13)
    np.testing.assert_allclose(out[0]['I_M'][2],minus(2),rtol=2e-13)
    row['points'][0][0]=(plus(2)*1.01,minus(2),0.,0.,0.,0.)
    with pytest.raises(ValueError,match='재현 실패'):
        replay_record(row,g,blue)


def test_frequency_transition_uses_last_iteration_but_same_frequency_keeps_blue(tmp_path,monkeypatch):
    capture=tmp_path/'capture';field=capture/'field';field.mkdir(parents=True)
    for name in (f'FORMAL_{FIRST}_00_geometry.bin','siii_blue_in.bin','siii_compact.bin'):
        (field/name).write_bytes(b'test')
    monkeypatch.setattr(compact_replay,'geometry',lambda _:dict(NP=1))
    monkeypatch.setattr(compact_replay,'blue_state',
                        lambda path,g,include_rays:(0,['initial']))
    rows=[dict(index=FIRST,jit=0),dict(index=FIRST,jit=1),
          dict(index=FIRST+1,jit=0)]
    monkeypatch.setattr(compact_replay,'records',
                        lambda path,g,include_rays:iter(rows))
    observed=[]
    def replay_one(row,g,blue):
        observed.append((row['index'],row['jit'],blue[:]))
        return [f'output-{len(observed)}'],0.
    monkeypatch.setattr(compact_replay,'replay_record',replay_one)
    out=tmp_path/'report.json'
    compact_replay.replay(capture,out,max_calls=3)
    assert observed==[(FIRST,0,['initial']),
                      (FIRST,1,['initial']),
                      (FIRST+1,0,['output-2'])]
    assert '"formal_calls_replayed": 3' in out.read_text()
    with pytest.raises(FileExistsError):
        compact_replay.replay(capture,out,max_calls=3)
    with pytest.raises(ValueError,match='전대역 주파수 끝'):
        compact_replay.replay(capture,tmp_path/'incomplete.json')
