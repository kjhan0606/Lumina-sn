from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_lumina_samej_collision_budget import direct_ci
import cmfgen_model_continuum as B
from build_cmfgen_fullpair_omega import native_edge_frequencies
from types import SimpleNamespace
import pytest


def test_native_ci_scale_is_derived_not_fitted():
    ratio=3.23e23*B.H/(B.KB*1.55e13)
    for temperature in (5000.,19000.,40000.):
        for edge in (1.e14,1.e15,5.e15):
            c=direct_ci(temperature,1.e9,edge,1.e-18,2)
            native=direct_ci(temperature,1.e9,edge,1.e-18,2,True)
            assert np.isclose(native/c,ratio,rtol=1.e-14,atol=0.)
    assert direct_ci(10000.,1.e9,1.e15,0.,2)==0.


def test_native_fedge_keeps_raw_wavenumber_order(tmp_path):
    path=tmp_path/'osc'
    path.write_text('100000.123456789 !Ionization energy\n')
    energies=np.array([90000.123456788,90000.123456789])
    osc=SimpleNamespace(levels={'E_cm':energies})
    result=native_edge_frequencies(path,osc,2)
    expected=(100000.123456789-energies)*B.C*1.e-15
    assert np.array_equal(result,expected)
    assert result[0]>result[1]
    path.write_text('100000. !Other\n')
    with pytest.raises(ValueError,match='파수'):
        native_edge_frequencies(path,osc,2)
