"""보존 native ZNET/fine BB 블록의 주소·부호·핵수 보존 직접시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_cmfgen_ir_grid_impact import add_bb_flow, compare_native_bb_flows


def test_bound_bound_flow_conserves_number_and_internal_sl_is_zero():
    target=np.zeros((2,3),dtype=np.longdouble)
    flow=np.array([1.,-2.,3.])
    add_bb_flow(target,0,1,flow)
    np.testing.assert_array_equal(target.sum(axis=0),0)
    before=target.copy()
    add_bb_flow(target,1,1,flow)
    np.testing.assert_array_equal(target,before)


def test_native_znet_comparison_uses_species_and_full_level_keys(tmp_path):
    atoms=[(14,2,'Sk2'),(14,3,'SkIII'),(16,2,'S2'),(16,3,'SIII')]
    rows=[];columns=[];headers=[]
    for i,(z,stage,label) in enumerate(atoms):
        for l in range(2):
            rows.append(dict(atomic_number=str(z),ion_number=str(stage-1),level_number=str(l)))
            columns.append((z,stage-1,l))
        headers.append(f'{i+1} {label}(upper-lower) 1.000000 1 2 1 2\n'+
                       ' '.join(['0.5']*90)+'\n')
    path=tmp_path/'NETRATE';path.write_text('\n'.join(headers))
    pop=np.tile(np.array([10.,2.]*4)[:,None],(1,90))
    lines=dict(lower=np.array([0,2,4,6]),upper=np.array([1,3,5,7]),
               nu0=np.full(4,1.e15),A=np.full(4,3.),g_ratio=np.ones(4))
    report=compare_native_bb_flows(path,lines,np.zeros((4,90)),np.ones(4),pop,
        np.ones((8,90)),np.array(columns),rows,np.arange(8),np.zeros(4,dtype=bool))
    assert report['matched_lines']==4
    for ion in report['by_ion'].values():
        assert ion['matched_lines']==1
        assert ion['max_absolute_row_delta']==pytest.approx(3.)
        assert ion['delta_over_BB_gross_median']==pytest.approx(.5)
        assert ion['number_conservation_relative']==0.
        assert ion['depths'][0]['native_print_L1_bound']==pytest.approx(6.e-7)
    native=compare_native_bb_flows(path,lines,np.zeros((4,90)),np.ones(4),pop,
        np.ones((8,90)),np.array(columns),rows,np.arange(8),np.zeros(4,dtype=bool),2*pop)
    assert native['population']=='native full snapshot'
    for ion in native['by_ion'].values():
        assert ion['max_absolute_row_delta']==pytest.approx(6.)
        assert ion['delta_over_BB_gross_median']==pytest.approx(.5)
