import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_frozen_direction as frozen
from lumina_frozen_direction import load,sha


def case(tmp_path):
    old=tmp_path/'old-input';old.write_text('unchanged')
    p=tmp_path/'newton-linearization.npz';np.savez(p,direction=np.array([1.,2.]))
    metadata=tmp_path/'inputs.json'
    metadata.write_text(json.dumps(dict(temperature_update=True,thermal_policy='native_instantaneous_scenario',
        input_sha256={str(old):sha(old)},source_commit='1'*40)))
    return p,metadata,old


def test_source_hashes_policy_and_accepted_old_trial_are_checked(tmp_path,monkeypatch):
    monkeypatch.setattr(frozen,'physical_sources',lambda root:{})
    p,m,old=case(tmp_path);before={f:sha(f) for f in (p,m,old)};inputs={}
    basis,identity=load(p,sha(p),sha(m),'native_instantaneous_scenario',inputs)
    assert 'direction' in basis and identity['source_commit']=='1'*40
    assert all(sha(f)==h for f,h in before.items())
    with pytest.raises(ValueError):load(p,'0'*64,sha(m),'native_instantaneous_scenario',{})
    with pytest.raises(ValueError):load(p,sha(p),sha(m),'physical_instantaneous_scenario',{})
    (tmp_path/'result.json').write_text(json.dumps(dict(history=[dict(accepted=True)])))
    with pytest.raises(ValueError,match='이미 수용'):load(p,sha(p),sha(m),'native_instantaneous_scenario',{})


def test_changed_physical_input_is_not_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(frozen,'physical_sources',lambda root:{})
    p,m,old=case(tmp_path);old.write_text('changed')
    with pytest.raises(ValueError,match='원 입력'):load(p,sha(p),sha(m),'native_instantaneous_scenario',{})


def test_changed_physical_implementation_is_rejected_but_wrapper_may_change(tmp_path):
    old=tmp_path/'source';current=tmp_path/'current';run=tmp_path/'run';run.mkdir()
    build=tmp_path/'build';build.mkdir()
    names=['src/producer.c','scripts/run_lumina_common_material.py',
           *['scripts/'+n for n in frozen.EXTRA_PRODUCERS]]
    for root in (old,current):
        for name in names:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('same source')
    (build/'source-before.sha256').write_text('\n'.join(sha(old/n)+'  '+n for n in names[:2])+'\n')
    (current/'scripts/run_lumina_common_material.py').write_text('new numerical wrapper')
    assert frozen.physical_sources(run,current)
    (current/'src/producer.c').write_text('changed physical equations')
    with pytest.raises(ValueError,match='물리 생산자'):frozen.physical_sources(run,current)
