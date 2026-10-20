"""원자 자료와 물질 프로필을 분리하고 고정 총수를 그대로 내보낸다."""
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_lumina_material_seed import ELEMENTS,atomic_asset,prepare,write_point_profiles


def test_native_element_totals_are_not_reconstructed_from_mass(tmp_path):
    variables=[SimpleNamespace(species=name,index=i+1)
               for i,(name,_) in enumerate(ELEMENTS.values())]
    state=np.arange(1,17,dtype=float).reshape(8,2)
    original=state.copy()
    fractions={Z:np.array([.1,.2]) for Z in ELEMENTS}
    write_point_profiles(tmp_path,variables,state,fractions)
    rows=list(csv.DictReader((tmp_path/'element_number_densities.csv').open()))
    assert len(rows)==12
    assert [float(row['N_element_cm3']) for row in rows]==list(state[:-2].ravel())
    masses=list(csv.DictReader((tmp_path/'abundances.csv').open()))
    assert all(float(row['0'])==.1 and float(row['1'])==.2 for row in masses)
    np.testing.assert_array_equal(state,original)
    assert list(csv.DictReader((tmp_path/'electron_temperatures.csv').open()))==[
        {'shell_id':'0','T_e':'150000'},{'shell_id':'1','T_e':'160000'}]


@pytest.mark.parametrize('name',['config.json','geometry.csv','density.csv',
    'electron_densities.csv','plasma_state.csv','abundances.csv',
    'tau_sobolev.npy','transition_probabilities.npy'])
def test_old_material_and_transport_arrays_are_never_reused(name):
    assert not atomic_asset(name)


@pytest.mark.parametrize('name',['levels.csv','line_list.csv','cmfgen_bf_channels.bin',
    'ige_col_14_1_cmfgen.bin','coldata_cmfgen_manifest.csv'])
def test_explicit_atomic_inputs_remain_available(name):
    assert atomic_asset(name)


def test_existing_output_is_preserved_before_any_input_read(tmp_path):
    with pytest.raises(ValueError,match='보존'):
        prepare(Path('not_read'),'not_read',tmp_path)
