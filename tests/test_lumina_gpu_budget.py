"""예약 비중 변경이16GiB 총한도와 할당 경계·기본 경로를 보존하는지 확인한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_driver import creation_preflight

G=1024**3


@pytest.fixture(scope='module')
def budget(tmp_path_factory):
    path=tmp_path_factory.mktemp('gpu-budget')/'budget.so'
    result=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/gpu_budget_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    lib=ct.CDLL(str(path));lib.reservation_probe.argtypes=[ct.c_uint64]*7+[ct.c_int,ct.POINTER(ct.c_uint64)]
    lib.reservation_probe.restype=ct.c_int
    lib.reservation_fits_probe.argtypes=[ct.c_uint64]*4;lib.reservation_fits_probe.restype=ct.c_int
    return lib


def call(lib,*,selected,cap=16*G,context=4*G,available=None,local=9168,stack=16384,threads=221184):
    baseline=79*G;total=80*G
    if available is None:available=baseline-context
    out=ct.c_uint64(17)
    code=lib.reservation_probe(cap,baseline,total,available,local,stack,threads,selected,ct.byref(out))
    return code,out.value


def test_many_resident_threads_fit_total_budget_without_legacy_eight_GiB_limit(budget):
    old=call(budget,selected=0);new=call(budget,selected=1)
    assert old[0]==3 and new[0]==0 and old[1]==new[1] and 8*G<new[1]<16*G
    room=16*G-new[1]
    assert budget.reservation_fits_probe(16*G,new[1],0,room)==1
    assert budget.reservation_fits_probe(16*G,new[1],0,room+1)==0


def test_nvml_baseline_above_cuda_total_is_conservatively_counted_in_selected_mode(budget):
    # 실제 A100 초기화 재현값: NVML과 CUDA의 차이는256KiB다.
    out=ct.c_uint64(17)
    args=(16*G,85095088128,85094825984,81245700096,9168,16384,221184)
    assert budget.reservation_probe(*args,0,ct.byref(out))==1
    assert budget.reservation_probe(*args,1,ct.byref(out))==0
    assert out.value==9769517056
    assert budget.reservation_fits_probe(16*G,out.value,3876449412,4)==1
    assert budget.reservation_fits_probe(16*G,out.value,3876449412,16*G)==0


def test_invalid_huge_baseline_is_not_admitted_by_selected_mode(budget):
    out=ct.c_uint64(17)
    assert budget.reservation_probe(16*G,2**64-1,80*G,76*G,9168,16384,221184,1,ct.byref(out)) in (2,4)


@pytest.mark.parametrize('reserve',[8*G-1,8*G,8*G+1,16*G-1,16*G,16*G+1])
@pytest.mark.parametrize('selected',[0,1])
def test_reservation_boundaries(budget,reserve,selected):
    code,value=call(budget,selected=selected,context=reserve-256*1024**2,local=0,stack=0,threads=1)
    assert value==reserve
    wanted=3 if not selected and reserve>8*G else (4 if reserve>=16*G else 0)
    assert code==wanted


@pytest.mark.parametrize('cap',[G-1,16*G+1])
def test_total_cap_cannot_be_expanded(budget,cap):
    assert call(budget,selected=1,cap=cap)[0]==1


@pytest.mark.parametrize('local,stack,threads',[(2**64-1,1,1),(2**63,0,3),(0,0,0)])
def test_overflow_and_invalid_plans_are_rejected(budget,local,stack,threads):
    assert call(budget,selected=1,local=local,stack=stack,threads=threads)[0] in (1,2)


def test_available_memory_and_live_allocation_remain_checked(budget):
    # baseline보다 가용이 줄어든 비용은 문맥 상계에 포함한다.
    assert call(budget,selected=1,available=G)[0]==4
    reserved=ct.c_uint64()
    assert budget.reservation_probe(16*G,G,80*G,G,0,0,1,1,ct.byref(reserved))==5
    for cap,reserved,live,size,expected in [(100,20,70,10,1),(100,20,70,11,0),
        (100,20,81,1,0),(100,101,0,1,0),(100,100,0,1,0)]:
        assert budget.reservation_fits_probe(cap,reserved,live,size)==expected


def test_failure_report_is_optional_and_preserves_numeric_values(budget):
    assert creation_preflight(SimpleNamespace()) is None
    report=creation_preflight(budget)
    assert report['reason']==3 and report['reserved']==9*G and report['cap']==16*G
    assert report['threads']==221184 and report['kernel_local']==9168
