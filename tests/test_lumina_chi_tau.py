"""같은 유효 저장 트리의 χ 질의와 전체 질의의 수치·거부 의미를 대조한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_lumina_native_cost import piece,run

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    libs=[]
    for chi_only in (False,True):
        path=tmp_path_factory.mktemp('chi-tau')/'probe.so'
        flags=['-DCMF_GPU_ENVELOPE_UPPER','-DCMF_GPU_ADAPTIVE_LINEAR_EXP','-DCMF_GPU_ENDPOINT_CACHE']
        if chi_only:flags.append('-DCMF_GPU_CHI_ONLY_TAU')
        proc=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
            '-Wall','-Wextra','-Werror','-fPIC','-shared',*flags,'-I'+str(ROOT/'src'),
            str(ROOT/'tests/chi_tau_probe.cpp'),str(ROOT/'tests/native_cost_probe.cpp'),'-o',str(path)],
            capture_output=True,text=True,timeout=60)
        assert proc.returncode==0,proc.stderr
        lib=ct.CDLL(str(path))
        lib.chi_tau_probe.argtypes=[ct.c_int]*2+[ct.c_void_p]+[ct.c_double]*5+[ct.c_void_p]
        lib.chi_tau_probe.restype=ct.c_int
        lib.native_cost_probe.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_void_p]*3
        lib.native_cost_probe.restype=ct.c_int
        assert lib.chi_tau_node_bytes()==144
        libs.append(lib)
    return libs


@pytest.mark.parametrize('block',[1,2,7,32,256])
@pytest.mark.parametrize('shift,span,u',[(.375,50.7,1.),(.375,50.7,.123),
    (3.375,0.,.75),(140.,3.,1.),(0.,128.,.5),(.99999999,.00000002,1.)])
def test_range_all_chi_fields_and_tau_error_are_bit_identical(kernels,block,shift,span,u):
    rng=np.random.default_rng(10222026)
    values=np.ascontiguousarray(np.vstack([rng.uniform(-.2,.5,(2,129)),rng.uniform(0.,1.,(2,129))]))
    result=[]
    for lib in kernels:
        out=np.full(20,-17.)
        assert lib.chi_tau_probe(129,block,values.ctypes.data,shift,span,1.1,.0034,u,out.ctypes.data)==0
        result.append(out)
    assert result[0].tobytes()==result[1].tobytes()


@pytest.mark.parametrize('source',[0.,np.nextafter(0.,1.),1e-80,np.finfo(float).max/4])
def test_tau_remains_independent_of_finite_nonnegative_source_scale(kernels,source):
    values=np.ones((4,65));values[0]*=-.1;values[1]*=.3;values[2:]=source
    result=[]
    for lib in kernels:
        out=np.full(20,-17.)
        assert lib.chi_tau_probe(65,2,values.ctypes.data,.375,32.4,1.1,.0034,1.,out.ctypes.data)==0
        result.append(out)
    assert result[0].tobytes()==result[1].tobytes()


@pytest.mark.parametrize('bad',['negative_eta','nan_eta','infinite_eta','infinite_chi','u'])
def test_failed_input_preserves_output_in_both_paths(kernels,bad):
    values=np.ones((4,65));u=1.
    if bad=='negative_eta':values[2,2]=-1.
    elif bad=='nan_eta':values[2,2]=np.nan
    elif bad=='infinite_eta':values[2,2]=np.inf
    elif bad=='infinite_chi':values[0,2]=np.inf
    else:u=0.
    codes=[]
    for lib in kernels:
        out=np.full(20,-17.)
        codes.append(lib.chi_tau_probe(65,2,values.ctypes.data,.375,32.4,1.1,.0034,u,out.ctypes.data))
        assert np.all(out==-17.)
    assert codes[0]==codes[1]!=0


@pytest.mark.parametrize('column,shift,span',[(0,.375,50.7),(31,.375,50.7),(64,3.375,50.7),
    (127,.375,50.7),(128,.375,50.7),(33,.375,0.),(15,140.,3.)])
def test_full_adaptive_map_errors_and_panels_remain_bit_identical(kernels,column,shift,span):
    p,_host,_arrays=piece(129,shift,span)
    results=[run(lib.native_cost_probe,p,column) for lib in kernels]
    assert results[0][0]==results[1][0]==0
    assert bytes(results[0][1])==bytes(results[1][1])
    np.testing.assert_array_equal(results[0][2],results[1][2])
