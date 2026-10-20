import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_cmfgen_collision_gaps import (KB_EV, collision_changes, gaussian_mean,
                                         excluded_bb_plane, native_field, read_table, thermal_fractions)


def fixture():
    return [dict(energy_eV=str(i), g=str(2*(i+1)), super_level=str(i)) for i in range(3)]


def test_missing_pairs_exclude_all_radiative_and_tabulated_pairs():
    levels = fixture()
    tables = np.array([1e4, 4e4]), {(0,1):np.array([1.,16.])}
    missing, delta, counts = collision_changes(levels, {(1,2)}, tables, .1,
                                              {0:0,1:1,2:2}, 2e4, 1e8, 3)
    assert counts["missing_cross_SL_pairs"] == 1
    assert missing[2,0] > 0 and missing[0,2] > 0
    assert missing[1,0] == missing[2,1] == 0
    assert np.max(np.abs(missing.sum(axis=0))) < 1e-16
    # Omega(2e4)=4 under native log-log, versus 6 under current linear C.
    assert delta[1,0] < 0
    coeff = 1e8*8.629e-6*(-2)/math.sqrt(2e4)
    assert delta[1,0] == pytest.approx(coeff/2*math.exp(-1/(KB_EV*2e4)))


def test_thermal_stationarity_and_internal_SL_pairs_cancel():
    levels = fixture()
    levels[1]["super_level"] = "0"
    fractions = thermal_fractions(levels, 1e4)
    assert fractions[:2].sum() == pytest.approx(1)
    A, _, counts = collision_changes(levels, set(), (np.array([1.,2.]),{}), .1,
                                     {0:0,2:1}, 1e4, 1e8, 2)
    assert counts["missing_full_pairs"] == 3
    assert counts["missing_cross_SL_pairs"] == 2
    equilibrium = np.array([sum(float(r["g"])*math.exp(-float(r["energy_eV"])/(KB_EV*1e4))
                                for r in levels if r["super_level"] == sl) for sl in ("0","2")])
    assert np.max(np.abs(A@equilibrium)) < 1e-15


def test_binary_identity_duplicate_truncation_rejected(tmp_path):
    path = tmp_path/"collision.bin"
    header = struct.pack("<IIiiiii",0x49474331,1,14,2,1,2,3)
    record = struct.pack("<ii2d",0,1,1.,4.)
    raw = header+struct.pack("<2d",1e4,2e4)+record
    path.write_bytes(raw)
    grid, pairs = read_table(path,14,2,3)
    assert grid.tolist() == [1e4,2e4] and pairs[0,1].tolist() == [1.,4.]
    with pytest.raises(ValueError):
        read_table(path,16,2,3)
    for bad in (raw[:-1], raw+b"x", raw[:16]+struct.pack("<i",2)+raw[20:]+record):
        path.write_bytes(bad)
        with pytest.raises(ValueError):
            read_table(path,14,2,3)


def test_gaussian_integrates_same_constant_linear_field_and_rejects_uncovered():
    nu = np.array([.99e15, .99999e15, 1.00007e15, 1.01e15])
    assert gaussian_mean(nu,np.full(4,3.),1e15) == pytest.approx(3.,rel=1e-12)
    assert gaussian_mean(nu,2.+(nu/1e15-1.),1e15) == pytest.approx(2.,rel=1e-12)
    assert gaussian_mean(nu,np.ones(4),2e15) is None
    with pytest.raises(ValueError):
        gaussian_mean(nu,-np.ones(4),1e15)


def test_native_field_depth_stride_and_excluded_pumping_only(tmp_path):
    path=tmp_path/"input.bin"
    raw=struct.pack("<8sIIIId",b"CMFOWN01",2,2,1,1,1.)
    raw+=struct.pack("<2d3i",.99e13,1.01e13,14,2,0)
    for d in (1.,3.):
        raw+=struct.pack("<7di2d",1.,1e4,1e8,1.,2.,1.,0.,0,d,d)
    path.write_bytes(raw)
    nu,J=native_field(path,2)
    assert J.tolist() == [3.,3.]
    identity=[dict(internal_stage="2",sl_id=str(i)) for i in range(3)]
    rows=[dict(atomic_number="14",ion_number="2",level_number_lower="0",
               level_number_upper="1",nu="1e13",B_lu="2",B_ul="1")]
    plane,counts=excluded_bb_plane(rows,{(14,2):fixture()},identity,14,1e4,nu,J)
    assert plane[1,0] == pytest.approx(6)
    assert plane[0,1] == pytest.approx(3)
    assert np.max(np.abs(plane.sum(axis=0))) == 0
    assert counts["native_field_uncovered"] == 0
    rows[0]["nu"]="2e13"
    plane,counts=excluded_bb_plane(rows,{(14,2):fixture()},identity,14,1e4,nu,J)
    assert np.count_nonzero(plane) == 0 and counts["native_field_uncovered"] == 1
