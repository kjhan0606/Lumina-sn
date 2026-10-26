from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from consume_lumina_material_bf90 import addresses,energy_prediction,write_handoff,HEADER


def fixture():
    rows=[dict(atomic_number=16,ion_number=1,super_level=0),
          dict(atomic_number=16,ion_number=2,super_level=0)]
    snapshot=dict(columns=np.array([[16,1,0],[16,2,0],[16,3,-1]]),
        superlevel=np.tile(np.array([[2.],[3.],[5.]]),(1,90)),
        fraction=np.ones((2,90)),temperature=np.full(90,18000.),ne=np.full(90,1.e10),
        saha=np.ones((2,90)))
    roster=dict(lower=np.array([0,1]),target=np.array([1,-1]),Z=np.array([16,16]),
        stage=np.array([2,3]),phot_grid=np.array([0,0]),probability=np.array([.25,.5]),
        threshold=np.array([3.e15,4.e15]),sigma_edge=np.array([1.e-18,2.e-18]))
    rates=np.tile(np.array([[[8.,2.,1.,-1.e-12,2.e-13,3.e-13]],
                            [[12.,3.,2.,1.e-12,3.e-13,4.e-13]]]),(1,90,1))
    return rows,snapshot,roster,rates


def test_exact_addresses_include_terminal_and_reject_wrong_ion():
    rows,snapshot,roster,rates=fixture()
    lo,up=addresses(rows,snapshot['columns'],roster)
    assert np.array_equal(lo,[0,1]) and np.array_equal(up,[1,2])
    roster['target'][0]=0
    with pytest.raises(ValueError,match='상위 이온'):addresses(rows,snapshot['columns'],roster)


def test_full_photon_energy_keeps_negative_pseudo_kinetic_heat():
    rows,snapshot,roster,rates=fixture()
    lo,up=addresses(rows,snapshot['columns'],roster)
    p=energy_prediction(snapshot,roster,rates,lo,up)[16]
    electron=.25*(-2.e-12-3.*5.e-13)+.5*(3.e-12-5.*7.e-13)
    threshold=6.62607015e-27*(.25*3.e15*(2.*8.-3.*3.)+.5*4.e15*(3.*12.-5.*5.))
    np.testing.assert_allclose(p['electron'],electron,rtol=1.e-14)
    np.testing.assert_allclose(p['photon'],electron+threshold,rtol=1.e-14)
    assert np.all(p['electron']<0) and np.all(p['photon']>0)
    assert np.all(p['gross']>=np.abs(p['photon']))


def test_binary_all_nodes_order_and_no_overwrite(tmp_path):
    rows,snapshot,roster,rates=fixture();path=tmp_path/'bf90.bin'
    write_handoff(path,snapshot,roster,rates)
    with path.open('rb') as f:
        assert HEADER.unpack(f.read(HEADER.size))==(b'LMBF90V1',2,2,90,2**32-1,0.,0.)
        np.testing.assert_array_equal(np.fromfile(f,'<f8',90),snapshot['temperature'])
        np.testing.assert_array_equal(np.fromfile(f,'<f8',90),snapshot['ne'])
        for i in range(2):
            assert struct.unpack('<5i',f.read(20))==(i,int(roster['target'][i]),16,i+2,0)
            assert struct.unpack('<3d',f.read(24))==tuple(roster[k][i] for k in ('probability','threshold','sigma_edge'))
            values=np.fromfile(f,'<f8',90*7).reshape(90,7)
            np.testing.assert_array_equal(values[:,0],snapshot['saha'][i])
            np.testing.assert_array_equal(values[:,1:],rates[i])
        assert f.read(1)==b''
    with pytest.raises(FileExistsError):write_handoff(path,snapshot,roster,rates)
