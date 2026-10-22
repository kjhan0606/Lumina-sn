"""실패한4097점 급선 기준을 보존한 별도 미세격자 조사. 실제 PHOT이 아니다."""
import json
import ctypes as ct
import os
from pathlib import Path

import numpy as np
from test_lumina_monotone_reconstruction import kernel,run


def test_actual_grid_sample_uses_both_existing_kernels_and_rejects_bad_indices(kernel):
    f=kernel.reconstruction_native_sample_probe
    a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int]*2+[i,a,a,a,i];f.restype=ct.c_int
    values=np.array([144.,36.,4.,0.,0.,4.,36.,144.]);columns=np.array([3],dtype='i4')
    old=np.full(1,-17.);candidate=old.copy();flags=np.full(1,-17,dtype='i4')
    assert f(8,1,columns,values,old,candidate,flags)==0
    assert old[0]==candidate[0]==0. and flags[0]==4
    columns=np.array([3,8],dtype='i4');old=np.full(2,-17.);candidate=old.copy();flags=np.full(2,-17,dtype='i4')
    assert f(8,2,columns,values,old,candidate,flags)==1
    for array in (old,candidate,flags):np.testing.assert_array_equal(array,np.full(2,-17.))


def test_finer_frequency_grid_quantifies_resolution_requirement(kernel,tmp_path):
    rows=[]
    for n in (4097,8193):
        x=np.linspace(0.,1.,n);values=1e-4+np.exp(-((x-.517)/.01)**2)
        worst=0.;functional=0.
        for shift in (.125,.375,.875):
            code,out=run(kernel,values,shift);assert code==0
            target=(np.arange(n-1)+shift)/(n-1);exact=1e-4+np.exp(-((target-.517)/.01)**2)
            worst=max(worst,float((abs(out[:-1]-exact)/exact).max()))
            w=np.exp(-target);functional=max(functional,float(abs(np.sum(w*out[:-1])-np.sum(w*exact))/np.sum(w*exact)))
        rows.append(dict(points=n,maximum_each_output_relative=worst,positive_functional_relative=functional,
            synthetic_1e4_criterion_met=worst<=1e-4 and functional<=1e-4))
    root=Path(os.environ.get('LUMINA_MONOTONE_RESULT_ROOT',str(tmp_path)))
    with (root/'refinement-sharp.json').open('x') as stream:
        json.dump(dict(rows=rows,scope='4097점의실패는보존한다. 더미세한합성격자의통과는native90/실제Si-S PHOT수용이아니다.',
            actual_model_F_calls=0,GPU_seconds=0,final_goal='NOT_MET'),stream,ensure_ascii=False,indent=2)
    assert not rows[0]['synthetic_1e4_criterion_met']
    assert rows[1]['synthetic_1e4_criterion_met']
    assert rows[1]['maximum_each_output_relative']<rows[0]['maximum_each_output_relative']
