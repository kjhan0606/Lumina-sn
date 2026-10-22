"""Ada 허용이 기존 공유상태 상한이나 다른 아키텍처 허용으로 번지지 않도록 확인한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import pytest
from lumina_gpu_compact import WarpCompactDriver


@pytest.fixture(scope='module')
def device(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]
    target=tmp_path_factory.mktemp('warp-device')/'probe.so'
    subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(root/'src'),str(root/'tests/warp_device_probe.cpp'),'-o',str(target)],
        check=True,capture_output=True,text=True,timeout=60)
    library=ct.CDLL(str(target));library.warp_device_probe.argtypes=[ct.c_int]*6
    library.warp_device_probe.restype=ct.c_int
    library.warp_device_budget_probe.argtypes=[];library.warp_device_budget_probe.restype=ct.c_int
    return library


@pytest.mark.parametrize('properties',[(8,0,108,2048,1024,32),(8,9,100,1536,1024,32),
    (8,9,108,4096,128,32)])
def test_supported_properties_stay_inside_original_bound(device,properties):
    assert device.warp_device_probe(*properties)==1
    assert device.warp_device_budget_probe()==WarpCompactDriver.cooperative_shared_bytes==21233664


@pytest.mark.parametrize('properties',[(9,0,108,2048,1024,32),(8,6,108,1536,1024,32),
    (8,9,109,1536,1024,32),(8,9,0,1536,1024,32),(8,9,-1,1536,1024,32),
    (8,9,100,4097,1024,32),(8,9,100,127,1024,32),(8,9,100,1536,127,32),
    (8,9,100,1536,1024,64)])
def test_unsupported_architecture_or_excess_resources_remain_rejected(device,properties):
    assert device.warp_device_probe(*properties)==0
