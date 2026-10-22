"""오류 페이로드의 열/수치 상태만 검증하며 장치를 열지 않는다."""
import importlib.util
from pathlib import Path
import sys

import pytest
from test_lumina_gpu_measure_local import local


def test_failure_payload_does_not_confuse_original_codes_with_location(local,monkeypatch):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    monkeypatch.delitem(sys.modules,'lumina_gpu_native_actual',raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_failure_location.py'
    spec=importlib.util.spec_from_file_location('isolated_failure_location',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for column in (0,624712,1115019,1999999):
        for status in range(1,8):
            decoded=m.decode(100000100+8*column+status)
            assert decoded['output_column']==column and decoded['characteristic_status']==status
    for code in (205,0,100000100,100000108,116000101,'105',None):assert m.decode(code) is None
    assert local.LIMITS['interval_seconds']==140 and local.LIMITS['slurm_allocation_seconds']==180
    assert local.LIMITS['already_allocated_gpu_seconds']==1064 and local.LIMITS['automatic_retries']==0
