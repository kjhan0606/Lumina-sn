from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_native_line_quadrature import intervals,contract
from test_model_point_lines import kernel


def test_integer_ranges_use_atomic_addresses_and_keep_inactivity_distinct(tmp_path):
    (tmp_path/'MODEL').write_text('BLANK [GLOBAL_LINE]\nF [LAM_SET]\n2 [TDOP]\n'
                                '4 [AMASS_DOP]\n10 [VTURB]\n6 [MAX_DOP]\n1 [FRAC_DOP]\n')
    (tmp_path/'TRANS_INFO').write_text('1 1 2 1.000000 2997.9 10 Sk2(upper-lower)\n'
                                     '2 1 3 0.500000 5995.8 10 Sk2(other-lower)\n')
    (tmp_path/'fort.77').write_text('1 2 4 1.00000E+00\n')
    rows = [dict(atomic_number=14,ion_number=1,level_number=i) for i in range(3)]
    lines = dict(lower=np.array([0,0]),upper=np.array([1,2]),nu0=np.array([1e15,.5e15]))
    first,last,active,report = intervals(tmp_path,rows,{(14,2):dict(label='Sk2')},lines,6)
    assert first.tolist() == [2,0] and last.tolist() == [5,0]
    assert active.tolist() == [True,False]
    assert report['selected_native_inactive_lines'] == 1
    (tmp_path/'fort.77').write_text('1 2 2 1.00000E+00\n')
    with pytest.raises(ValueError,match='공명구간'):
        intervals(tmp_path,rows,{(14,2):dict(label='Sk2')},lines,6)


def test_native_kernel_uses_exact_saved_window_and_global_measure(kernel):
    nu = np.array([.9999e15,1e15,1.0001e15])
    width = np.array([1e10,2e10,3e10])
    j = np.array([2.,3.,4.])
    producer = SimpleNamespace(dll=kernel[0],profile=dict(width_fraction=1e-4),
        lines=dict(nu0=np.array([1e15,1.1e15]),A=np.array([10.,20.]),g_ratio=np.ones(2)))
    result = contract(producer,nu,width,j,[0,0],[2,0],[1,0])
    phi = np.exp(-((nu[:2]-1e15)/1e11)**2)/(1.77245385095516*1e11)
    assert result['jbar'][0] == pytest.approx(np.sum(phi*width[:2]*j[:2]),rel=3e-15)
    assert result['profile_mass'][0] == pytest.approx(np.sum(phi*width[:2]),rel=3e-15)
    assert result['status'].tolist() == [1,5]
    assert result['spontaneous_native_discrete'][1] == 0
    with pytest.raises(ValueError,match='BB 적분'):
        contract(producer,nu,width,j,[0,1],[2,2],[1,0])
