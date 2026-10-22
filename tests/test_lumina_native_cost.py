"""진단의 국소 트리가 원 전역 인덱스/오차/패널을 바꾸지 않는지 확인한다."""
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_linear import Piece
from lumina_gpu_driver import Location
from lumina_linear_native import DenseView,NativeMap
from test_lumina_linear_native import inputs,endpoint
from test_lumina_characteristic_linear import reference


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    path=tmp_path_factory.mktemp('native-cost')/'probe.so'
    p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/native_cost_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    lib=ct.CDLL(str(path))
    for name in ['native_cost_probe','native_cost_reference']:
        f=getattr(lib,name);f.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p];f.restype=ct.c_int
    return lib


def piece(nf,shift,span):
    host,arrays=inputs(nf=nf);p=Piece()
    p.view=DenseView(host.ns,host.nf,host.ne,host.jmode,host.generation,
        *[getattr(host,k) for k in ('canonical','outer','j','radius','external_radius')])
    p.locations=(Location*2)(Location(1,3,2,.23),Location(1,2,1,.72))
    p.length=1.1;p.h=.001;p.db=shift*p.h;p.da=(shift+span)*p.h
    actual_shift=p.db/p.h;p.span=(p.da-p.db)/p.h;whole=math.floor(actual_shift)
    p.base=min(nf-2,whole);p.nf=nf-p.base;p.offset=actual_shift-whole+whole-p.base;p.block=2;p.count=nf
    p.bins=0 if p.span==0 or p.offset>=p.nf-1 else math.ceil(min(p.nf-1,p.offset+nf-1+p.span))
    p.leaves=1<<max(0,(p.bins+1)//2-1).bit_length()
    p.source_leaves=1<<max(0,(p.bins+31)//32-1).bit_length()
    return p,host,arrays


def run(fun,p,column):
    raw=NativeMap();ct.memset(ct.byref(raw),0xA5,ct.sizeof(raw));before=bytes(raw)
    stats=np.full(4,0,dtype='u8');times=np.full(2,-1.)
    code=fun(ct.byref(p),column,ct.byref(raw),stats.ctypes.data,times.ctypes.data)
    return code,raw,stats,times,before


@pytest.mark.parametrize('nf,column,shift,span',[(129,0,.375,50.7),(129,31,.375,50.7),(129,32,.375,50.7),
    (129,63,.375,50.7),(129,95,3.375,50.7),(129,127,.375,50.7),(129,128,.375,50.7),
    (129,33,.375,0.),(17,15,40.,3.)])
def test_local_cover_matches_global_tree_bits_and_independent_bin_integral(kernel,nf,column,shift,span):
    p,host,arrays=piece(nf,shift,span)
    a=run(kernel.native_cost_probe,p,column);b=run(kernel.native_cost_reference,p,column)
    assert a[0]==b[0]==0
    # 맵의 실제 수치/적응 필드는 같아야 한다. 준비 작업량은 덮개 크기에 따라 다르다.
    for name,_ctype in NativeMap._fields_:
        if name in ('built_bins','tree_combinations'):continue
        x=getattr(a[1],name);y=getattr(b[1],name)
        assert bytes(x)==bytes(y) if isinstance(x,ct.Structure) else x==y
    values=np.array([[endpoint(host,arrays,p.locations[c],i)[part] for i in range(nf)]
                     for part in [0,1] for c in [0,1]])
    expected=reference(values,length=p.length,shift=column+p.db/p.h,span=p.span,delta=p.da-p.db,incident=0.)
    value=a[1].source.positive().value();error=a[1].source_error.positive().value()
    assert abs(value-expected)<=error+2e-13*abs(expected)
    assert a[2][0]>=a[2][1] and np.all(a[3]>=0)


@pytest.mark.parametrize('bad',['column','length','offset','source'])
def test_probe_failure_preserves_physical_output_canary(kernel,bad):
    p,_host,arrays=piece(65,.375,3.4);column=32
    if bad=='column':column=65
    elif bad=='length':p.length=-1.
    elif bad=='offset':p.offset=float('nan')
    else:arrays[0][2,:,35]=-1e100
    result=run(kernel.native_cost_probe,p,column)
    assert result[0]!=0 and bytes(result[1])==result[4]


def test_diagnostic_sample_and_query_budget():
    from lumina_native_cost_diag import samples
    columns=samples(1115020)
    assert len(columns)==35 and columns[0]==0 and columns[-1]==1115019
    assert set((624712,624968,625223))<=set(columns)
    assert len(columns)+32+32+256+512==867<=1024


def test_diagnostic_comparison_has_no_absolute_floor():
    from lumina_native_cost_diag import compare
    from lumina_gpu_compact import CompactMap
    from lumina_linear_native import CPositive
    raw=CompactMap(CPositive(.5,1),CPositive(.5,-2000),CPositive(.5,-2040),0.)
    expected=dict(t=dict(m=.5,e=1),source=dict(m=.5,e=-2000),error=dict(m=.5,e=-2040))
    assert compare(raw,expected)
    expected['source']=dict(m=.5,e=-2001)
    assert not compare(raw,expected)
