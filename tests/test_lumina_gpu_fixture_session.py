"""모듈별 FixtureDef 두 개도 같은 session의 실제 문맥을 한 번만 만든다."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import test_lumina_gpu_compact_device as D


def setup(monkeypatch,tmp_path):
    calls=[];session=SimpleNamespace();request=SimpleNamespace(session=session)
    monkeypatch.setenv('LUMINA_COMPACT_GPU_LIBRARY',str(tmp_path/'synthetic.so'))
    monkeypatch.setenv('LUMINA_COMPACT_GPU_RECORD_ROOT',str(tmp_path))
    class Runtime:
        def __init__(self,path,free):
            calls.append(('create',free));self.library=Path(path).resolve();self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True;calls.append(('close',))
        def report(self):return dict(synthetic=True)
    monkeypatch.setattr(D,'Runtime',Runtime)
    def probe(*args,**kw):
        calls.append(('probe',));assert sum(c[0]=='probe' for c in calls)==1
        return SimpleNamespace(stdout='32768,0')
    monkeypatch.setattr(D.subprocess,'run',probe)
    return request,calls


def test_imported_fixture_definitions_share_one_context_without_second_utilization_probe(monkeypatch,tmp_path):
    request,calls=setup(monkeypatch,tmp_path)
    first=D.gpu.__wrapped__(request);a=next(first)
    second=D.gpu.__wrapped__(request);assert next(second) is a
    second.close();assert not a.closed and not (tmp_path/'memory.json').exists()
    first.close();assert a.closed and (tmp_path/'memory.json').exists()
    assert sum(c[0]=='create' for c in calls)==sum(c[0]=='close' for c in calls)==1
    assert not hasattr(request.session,'_lumina_compact_gpu_context')


def test_reuse_does_not_allow_binary_change_mid_session(monkeypatch,tmp_path):
    request,calls=setup(monkeypatch,tmp_path)
    first=D.gpu.__wrapped__(request);next(first)
    monkeypatch.setenv('LUMINA_COMPACT_GPU_LIBRARY',str(tmp_path/'other.so'))
    second=D.gpu.__wrapped__(request)
    with pytest.raises(ValueError,match='신원'):next(second)
    first.close();assert sum(c[0]=='create' for c in calls)==1
