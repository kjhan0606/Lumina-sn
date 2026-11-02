"""공통-J 실제 RE 준비 경로의 산술 보존, 갱신 전 종료, 직렬화 직접시험."""
import os
import json
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_cmfgen_fixedj_heat import (COMP_OPAC, FINISH, FINISH_POINT, FIXED_SKIP,
                                      FULL_CAPTURE, FULL_POINT, GUARD, GUARD_POINT,
                                      patched, read_heat_row, native_state, prepare_state)


def test_private_patch_restores_actual_coefficients_before_re_and_preserves_rest():
    original = ('start\n'+GUARD_POINT+'frequency start\n'+FIXED_SKIP+FULL_POINT+
                'continuum_RE\nadiabatic\nweak_line_RE\n'+FINISH_POINT+'solver\n')
    changed = patched(original)
    assert changed == ('start\n'+GUARD_POINT+GUARD+'frequency start\n'+COMP_OPAC+FULL_POINT+FULL_CAPTURE+
                       'continuum_RE\nadiabatic\nweak_line_RE\n'+FINISH+FINISH_POINT+'solver\n')
    assert changed.count('CALL COMP_OPAC') == 1
    assert changed.count('CALL LUMINA_FIXEDJ_FULL_CAPTURE') == 1
    assert 'EMHNUKT' not in changed  # COMP_OPAC 자체가 actual-nu 지수를 매번 평가한다.
    assert changed.index('CALL EVAL_LTE_V5') < changed.index('CALL LUMINA_FIXEDJ_FULL_CAPTURE')
    assert changed.index('weak_line_RE') < changed.index('CALL LUMINA_FIXEDJ_HEAT_FINISH')
    assert changed.index('CALL LUMINA_FIXEDJ_HEAT_FINISH') < changed.index(FINISH_POINT)
    with pytest.raises(ValueError):
        patched(changed)


@pytest.mark.parametrize('marker', [FIXED_SKIP, GUARD_POINT, FULL_POINT, FINISH_POINT])
def test_patch_rejects_missing_or_duplicate_insertions(marker):
    original = GUARD_POINT+FIXED_SKIP+FULL_POINT+FINISH_POINT
    for source in (original.replace(marker, ''), original+marker):
        with pytest.raises(ValueError):
            patched(source)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('fixedj_actual_heat')
    (root/'kind.f90').write_text('module set_kind_module\ninteger,parameter::ldp=kind(1d0)\nend module\n')
    (root/'driver.f90').write_text('''program direct_heat_test
use set_kind_module
use ieee_arithmetic
implicit none
real(ldp)::pops(4,2),r(2),v(2),sigma(2),t(2),ed(2),atoms(2),clump(2),heat(2),charge(2)
logical::fixedj,ehb,coherent,newmod
integer::ndietot
character(16)::gamma_mode
character(32)::mode
call get_command_argument(1,mode)
fixedj=.true.;ehb=.false.;coherent=.true.;newmod=.false.;ndietot=0;gamma_mode='LOCAL'
if(mode=='not_fixed')fixedj=.false.
if(mode=='ehb')ehb=.true.
if(mode=='noncoherent')coherent=.false.
if(mode=='newmod')newmod=.true.
if(mode=='die_transfer')ndietot=1
if(mode=='gamma_transfer')gamma_mode='RAD_TRANS'
call lumina_fixedj_heat_guard(fixedj,.false.,coherent,ehb,.false.,.false.,.false., &
                             .false.,.false.,'BLANK',newmod,ndietot,gamma_mode)
r=[3d0,1d0];v=[30d0,10d0];sigma=0d0;t=[1d0,2d0];ed=[5d0,7d0]
atoms=[2d0,3d0];clump=1d0;heat=[-3d0,4d0];charge=[1d-16,-2d-16]
pops(1,:)=2d0;pops(2,:)=3d0;pops(3,:)=ed;pops(4,:)=t
if(mode=='load')then
  pops=-1d0
  call lumina_fixedj_heat_load(pops,r,4,2)
  t=pops(4,:);ed=pops(3,:)
end if
if(mode=='nan')heat(1)=ieee_value(0d0,ieee_quiet_nan)
if(mode=='bad_state')pops(4,1)=2d0
if(mode=='bad_radius')r(1)=0.5d0
call lumina_fixedj_heat_finish(pops,r,v,sigma,t,ed,atoms,clump,heat,charge,4,2,11,65,7)
open(unit=71,file='FORBIDDEN_POP_UPDATE',status='new')
write(71,*)'must never reach population update'
end program
''')
    exe = root/'synthetic'
    subprocess.run(['gfortran', '-O', '-ffree-line-length-none', str(root/'kind.f90'),
                    str(SCRIPTS/'cmfgen_fixedj_heat.f90'), str(root/'driver.f90'), '-o', str(exe)],
                   cwd=root, check=True, capture_output=True, timeout=30)

    def run(out, mode='normal', enabled=True, input_bytes=None):
        out.mkdir()
        if input_bytes is not None:
            (out/'NATIVE_HEAT_INPUT.bin').write_bytes(input_bytes)
        env = dict(os.environ)
        env.pop('LUMINA_FIXEDJ_HEAT_ONLY', None)
        if enabled:
            env['LUMINA_FIXEDJ_HEAT_ONLY'] = '1'
        result = subprocess.run([str(exe), mode], cwd=out, env=env,
                                capture_output=True, timeout=10)
        assert not (out/'FORBIDDEN_POP_UPDATE').exists()
        return result
    return run


def test_native_serialization_has_signed_actual_row_exact_state_and_no_update(tmp_path, native):
    output = tmp_path/'capture'
    result = native(output)
    assert result.returncode == 96
    row = read_heat_row(output/'NATIVE_HEAT_ROW.bin')
    np.testing.assert_array_equal(row['steq_t_native'], [-3., 4.])
    np.testing.assert_array_equal(row['temperature_native'], [1., 2.])
    np.testing.assert_array_equal(row['pops'][-2], [5., 7.])
    assert row['frequencies'] == 11 and row['weak_lines'] == 7
    assert row['evaluation_counter'] == 65
    assert row['phase'] == 'BEFORE_POP_UPDATE'
    assert row['kind'] == 'INTEGRAL_RE_NOT_EHB'
    assert row['joint_convergence'] is False


@pytest.mark.parametrize('mode', ['not_fixed', 'ehb', 'noncoherent', 'nan', 'bad_state', 'bad_radius',
                                  'newmod', 'die_transfer', 'gamma_transfer'])
def test_native_rejects_other_operators_and_invalid_state(tmp_path, native, mode):
    output = tmp_path/mode
    assert native(output, mode).returncode == 94
    assert not (output/'NATIVE_HEAT_ROW.bin').exists()


def test_explicit_activation_required(tmp_path, native):
    output = tmp_path/'disabled'
    assert native(output, enabled=False).returncode == 94
    assert not (output/'NATIVE_HEAT_ROW.bin').exists()


def test_reader_rejects_truncation_extra_data_and_bad_header(tmp_path, native):
    output = tmp_path/'original'
    native(output)
    raw = (output/'NATIVE_HEAT_ROW.bin').read_bytes()
    for i, data in enumerate((b'bad', raw[:-1], raw+b'x', b'wrong header!!!!'+raw[16:])):
        path = tmp_path/f'bad-{i}.bin'
        path.write_bytes(data)
        with pytest.raises(ValueError):
            read_heat_row(path)


@pytest.mark.parametrize('index,value', [(0, 2), (1, 32), (2, 2), (3, 1), (4, 0),
                                       (5, 0), (6, 8), (7, -1)])
def test_reader_rejects_changed_schema_dimensions(tmp_path, native, index, value):
    output = tmp_path/'original'
    native(output)
    raw = bytearray((output/'NATIVE_HEAT_ROW.bin').read_bytes())
    struct.pack_into('<i', raw, 16+index*4, value)
    path = tmp_path/'bad.bin'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_heat_row(path)


def input_bytes(state=None):
    if state is None:
        state = np.array([[3., 5.], [7., 11.], [13., 17.], [1.05, 1.95]])
    return (b'NATIVEHEATINPV1 '+struct.pack('<4i', 1, 64, 4, 2)
            +np.asarray([3., 1.], dtype='<f8').tobytes()
            +np.asarray(state, dtype='<f8').tobytes(order='F'))


def test_explicit_input_is_loaded_before_native_state_is_read(tmp_path, native):
    output = tmp_path/'injected'
    raw = input_bytes()
    assert native(output, 'load', input_bytes=raw).returncode == 96
    result = read_heat_row(output/'NATIVE_HEAT_ROW.bin')
    expected = np.frombuffer(raw, dtype='<f8', offset=48).reshape((4, 2), order='F')
    np.testing.assert_array_equal(result['pops'], expected)
    assert (output/'NATIVE_HEAT_INPUT.bin').read_bytes() == raw


@pytest.mark.parametrize('mode', ['missing', 'short', 'extra', 'radius', 'dimension', 'negative', 'nan'])
def test_bad_native_input_does_not_fall_back_to_old_state(tmp_path, native, mode):
    raw = bytearray(input_bytes())
    if mode == 'missing':
        raw = None
    elif mode == 'short':
        raw = raw[:-1]
    elif mode == 'extra':
        raw += b'x'
    elif mode == 'radius':
        struct.pack_into('<d', raw, 32, 4.)
    elif mode == 'dimension':
        struct.pack_into('<i', raw, 24, 5)
    elif mode == 'negative':
        struct.pack_into('<d', raw, 48, -1.)
    elif mode == 'nan':
        struct.pack_into('<d', raw, 48, float('nan'))
    output = tmp_path/mode
    assert native(output, 'load', input_bytes=raw).returncode == 94
    assert not (output/'NATIVE_HEAT_ROW.bin').exists()


def state_fixture():
    columns = np.array([[16, 1, 0], [14, 2, -1], [14, 1, 0]], dtype='i4')
    physical = np.array([[3., 12000., 31., 1.], [1., 15000., 37., 2.]])
    population = np.array([[2., 3., 5.], [7., 11., 13.]])
    variables = [SimpleNamespace(index=1, species='Si', charge=1, superlevel=1, terminal=False),
                 SimpleNamespace(index=2, species='Si', charge=2, superlevel=1, terminal=True),
                 SimpleNamespace(index=3, species='S', charge=1, superlevel=1, terminal=False)]
    return columns, physical, population, variables


def test_native_mapping_is_by_atomic_address_not_current_array_order():
    columns, physical, population, variables = state_fixture()
    result = native_state(columns, physical, population, variables, {'Si': 14, 'S': 16})
    np.testing.assert_array_equal(result[:-2], population[:, [2, 1, 0]].T)
    np.testing.assert_array_equal(result[-2], physical[:, 2])
    np.testing.assert_allclose(result[-1], physical[:, 1]*1.e-4, rtol=0, atol=0)
    np.testing.assert_array_equal(population, [[2., 3., 5.], [7., 11., 13.]])


@pytest.mark.parametrize('mode', ['duplicate', 'address', 'index', 'species', 'radius', 'nonfinite'])
def test_native_mapping_rejects_different_or_ambiguous_state(mode):
    columns, physical, population, variables = state_fixture()
    if mode == 'duplicate': columns[1] = columns[0]
    elif mode == 'address': columns[0, 1] = 4
    elif mode == 'index': variables[1].index = 3
    elif mode == 'species': variables[0].species = 'unknown'
    elif mode == 'radius': physical[1, 0] = 4.
    elif mode == 'nonfinite': population[0, 0] = np.nan
    with pytest.raises(ValueError):
        native_state(columns, physical, population, variables, {'Si': 14, 'S': 16})


def preparation_fixture(tmp_path, monkeypatch):
    import ptf_gated_metrics
    import prepare_lumina_material_cells
    columns, physical, population, variables = state_fixture()
    physical[:, 0] *= 1.e10
    native_run = tmp_path/'native'
    material = tmp_path/'material'
    native_run.mkdir(); material.mkdir()
    names = ('MODEL', 'MODEL_SPEC', 'atomic_links.txt', 'LEVEL_SL_STEQ_LINKS')
    for name in names:
        (native_run/name).write_text(name+'\n')
    (native_run/'SCRTEMP').write_bytes(np.array([3., 1.], dtype='<f8').tobytes()+b'untouched-history')
    nodes = material/'native_material_nodes.bin'
    nodes.write_bytes(b'synthetic-nodal-payload')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = material/'accepted-material.json'
    manifest.write_text(json.dumps(dict(schema='lumina-accepted-nodal-material-v1',
        status='EXPORTED_NOT_TRANSPORT_READY',
        coordinate_role='original outer-to-inner point samples; no averaging, interpolation or remap',
        input_sha256={str(native_run/n): digest(native_run/n) for n in names[:3]},
        output_sha256={nodes.name: digest(nodes)},
        accepted_result_sha256='accepted-result-identity',
        accepted_response_sha256='accepted-response-identity')))
    monkeypatch.setattr(ptf_gated_metrics, 'read_dimensions', lambda p: (2, 5))
    monkeypatch.setattr(ptf_gated_metrics, 'read_variables', lambda p, n: variables)
    monkeypatch.setattr(prepare_lumina_material_cells, 'read_nodes',
                        lambda p: (columns, physical, population, 10, 86400.))
    return native_run, manifest, digest(manifest), digest, physical


def test_preparation_serializes_without_touching_native_or_accepted_input(tmp_path, monkeypatch):
    native_run, manifest, identity, digest, physical = preparation_fixture(tmp_path, monkeypatch)
    protected = [p for d in (native_run, manifest.parent) for p in d.iterdir()]
    before = {p: digest(p) for p in protected}
    out = tmp_path/'out'
    report = prepare_state(manifest, identity, native_run, out)
    assert {p: digest(p) for p in protected} == before
    raw = (out/'NATIVE_HEAT_INPUT.bin').read_bytes()
    assert raw[:16] == b'NATIVEHEATINPV1 '
    assert struct.unpack_from('<4i', raw, 16) == (1, 64, 5, 2)
    payload = np.frombuffer(raw, dtype='<f8', offset=48).reshape((5, 2), order='F')
    np.testing.assert_array_equal(payload[:-2], [[5., 13.], [3., 11.], [2., 7.]])
    np.testing.assert_array_equal(payload[-2], physical[:, 2])
    assert report['actual_native_evaluation'] is False
    assert report['heat_policy'] == 'NOT_YET_MATCHED'
    with pytest.raises(FileExistsError):
        prepare_state(manifest, identity, native_run, out)


@pytest.mark.parametrize('mode', ['manifest', 'nodes', 'model', 'radius', 'native_output', 'material_output'])
def test_preparation_rejects_changed_provenance_and_output_inside_originals(tmp_path, monkeypatch, mode):
    native_run, manifest, identity, digest, physical = preparation_fixture(tmp_path, monkeypatch)
    out = tmp_path/'out'
    if mode == 'manifest': identity = '0'*64
    elif mode == 'nodes': (manifest.parent/'native_material_nodes.bin').write_bytes(b'changed')
    elif mode == 'model': (native_run/'MODEL_SPEC').write_text('changed')
    elif mode == 'radius': physical[0, 0] *= 1.01
    elif mode == 'native_output': out = native_run/'out'
    elif mode == 'material_output': out = manifest.parent/'out'
    with pytest.raises(ValueError):
        prepare_state(manifest, identity, native_run, out)
    assert not out.exists()
