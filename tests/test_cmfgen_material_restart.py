"""초기값 전달 직접시험. 선택적으로 실제 native SCR_READ_V2를 컴파일해 읽는다."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_cmfgen_material_restart as R
from ptf_gated_metrics import ScrtempReader


def fixture_data(nd=3, nt=6):
    radius = np.arange(nd, 0, -1, dtype=float)
    geometry = np.concatenate([radius, radius*100, np.zeros(nd)])
    one = geometry.astype('<f8').tobytes()
    header = (one+b'\0'*(R.REC_BYTES-len(one)))*2
    state = np.arange(1, nt*nd+1, dtype=float).reshape(nt, nd)
    state[0, 0] = 0.
    return header, radius, state


@pytest.mark.parametrize('nd,nt', [(3, 6), (90, 1300)])
def test_exact_state_and_geometry_without_old_history(tmp_path, nd, nt):
    header, radius, state = fixture_data(nd, nt)
    blob, pointer = R.restart_bytes(header, radius, state)
    path = tmp_path/'SCRTEMP'
    path.write_bytes(blob)
    reader = ScrtempReader(path, nd, nt)
    assert reader.nstates == reader.last_it == 1
    np.testing.assert_array_equal(reader.state(1), state)
    np.testing.assert_array_equal(reader.radius, radius)
    assert blob[:len(header)] == header
    assert pointer.splitlines()[1] == '1 1 1 -1000 F'


@pytest.mark.parametrize('change', ['short_header', 'radius', 'nan', 'negative', 'zero_ne', 'zero_T', 'nd'])
def test_bad_state_cannot_be_serialized(change):
    header, radius, state = fixture_data()
    if change == 'short_header': header = header[:-1]
    elif change == 'radius': radius[0] += 1
    elif change == 'nan': state[0, 0] = np.nan
    elif change == 'negative': state[0, 0] = -1
    elif change == 'zero_ne': state[-2, 0] = 0
    elif change == 'zero_T': state[-1, 0] = 0
    elif change == 'nd': state = state[:, :-1]
    with pytest.raises(ValueError): R.restart_bytes(header, radius, state)


def input_fixture(tmp_path):
    native, inputs = tmp_path/'native', tmp_path/'state'
    native.mkdir(); inputs.mkdir()
    header, radius, state = fixture_data()
    blob, pointer = R.restart_bytes(header, radius, state)
    (native/'SCRTEMP').write_bytes(blob)
    for name in ('POINT1', 'POINT2'): (native/name).write_text(pointer)
    model = native/'MODEL'; model.write_text('same physical model\n')
    raw = (b'NATIVEHEATINPV1 '+struct.pack('<4i', 1, 64, *state.shape)
           +radius.astype('<f8').tobytes()+state.astype('<f8').tobytes(order='F'))
    data = inputs/'NATIVE_HEAT_INPUT.bin'; data.write_bytes(raw)
    record = dict(schema='lumina-native-heat-state-input-v1', SL_and_ne_exact=True,
        temperature_conversion='K -> 1e4 K; no equilibration', actual_native_evaluation=False,
        native_run=str(native), input_sha256={str(model): R.sha(model)},
        output_sha256={data.name: R.sha(data)}, points=len(radius), variables=len(state),
        accepted_result_sha256='a'*64, accepted_response_sha256='b'*64)
    manifest = inputs/'heat-state-input.json'
    manifest.write_text(json.dumps(record))
    return native, manifest, record, state


def test_prepare_only_writes_initial_state_not_a_fake_native_solution(tmp_path):
    native, manifest, record, state = input_fixture(tmp_path)
    before = {p: R.sha(p) for d in (native, manifest.parent) for p in d.iterdir()}
    out = tmp_path/'restart'
    report = R.prepare(manifest, R.sha(manifest), out)
    assert {p: R.sha(p) for p in before} == before
    assert {p.name for p in out.iterdir()} == {'SCRTEMP', 'POINT1', 'POINT2', 'restart-seed.json'}
    assert report['status'] == 'INITIAL_STATE_ONLY_NOT_RUN_READY'
    assert report['physical_iterations'] == report['inherited_native_iterations'] == 0
    assert not any(report[k] for k in ('copied_radiation', 'copied_opacities',
        'copied_jacobian', 'fabricated_RVTJ', 'released_temperature', 'computation_started'))
    np.testing.assert_array_equal(ScrtempReader(out/'SCRTEMP', 3, 6).state(1), state)
    with pytest.raises(FileExistsError): R.prepare(manifest, R.sha(manifest), out)


@pytest.mark.parametrize('change', ['manifest', 'payload', 'model', 'points', 'moving', 'inside_native', 'inside_input', 'trailing'])
def test_prepare_rejects_changed_identity_or_unsafe_destination(tmp_path, change):
    native, manifest, record, state = input_fixture(tmp_path)
    identity = R.sha(manifest); out = tmp_path/'restart'
    if change == 'manifest': identity = '0'*64
    elif change == 'payload': (manifest.parent/'NATIVE_HEAT_INPUT.bin').write_bytes(b'wrong')
    elif change == 'model': (native/'MODEL').write_text('changed')
    elif change == 'points':
        path = native/'POINT1'; path.write_text(path.read_text().replace('1 1 1', '2 2 1'))
    elif change == 'moving':
        for name in ('POINT1', 'POINT2'):
            path = native/name; path.write_text(path.read_text().replace('-1000 F', '-1000 T'))
    elif change == 'inside_native': out = native/'restart'
    elif change == 'inside_input': out = manifest.parent/'restart'
    elif change == 'trailing':
        data = manifest.parent/'NATIVE_HEAT_INPUT.bin'
        data.write_bytes(data.read_bytes()+b'x')
        record['output_sha256'][data.name] = R.sha(data)
        manifest.write_text(json.dumps(record)); identity = R.sha(manifest)
    with pytest.raises(ValueError): R.prepare(manifest, identity, out)
    assert not out.exists()


@pytest.fixture(scope='module')
def actual_native_reader(tmp_path_factory):
    source = os.environ.get('CMFGEN_NATIVE_SOURCE')
    if not source: pytest.skip('실제 native I/O 시험은 CMFGEN_NATIVE_SOURCE 지정 시 수행')
    if not shutil.which('gfortran'): pytest.fail('요청한 native I/O 시험에 gfortran이 없음')
    source = Path(source)
    names = ('tools/set_kind_module.f', 'subs/scr_read_v2.f', 'unix/dir_acc_pars_gen.f',
             'unix/gen_asci_open.f', 'unix/term_io.f', 'subs/set_case.f')
    sources = [source/n for n in names]
    hashes = {str(p): R.sha(p) for p in sources}
    build = tmp_path_factory.mktemp('cmfgen_native_restart_reader')
    driver = build/'read_restart.f90'
    driver.write_text('''program read_restart
use set_kind_module
implicit none
integer :: nd,nt,record,nits,num,last
logical :: newmod,rv_written
character(32) :: arg
real(ldp),allocatable :: r(:),v(:),sigma(:),pops(:,:)
call get_command_argument(1,arg)
read(arg,*)nd
call get_command_argument(2,arg)
read(arg,*)nt
allocate(r(nd),v(nd),sigma(nd),pops(nt,nd))
record=0
num=1
call scr_read_v2(r,v,sigma,pops,record,nits,num,last,rv_written,nt,nd,71,newmod)
if(newmod.or.record/=1.or.nits/=1.or.rv_written.or.last/=-1000)stop 1
open(unit=72,file='native-read.bin',form='unformatted',access='stream',status='new')
write(72)r,v,sigma,pops
close(72)
end program
''')
    exe = build/'read_restart'
    subprocess.run(['gfortran', '-O0', '-std=legacy', '-ffixed-line-length-none',
        '-ffree-line-length-none', '-fcheck=all', *map(str, sources), str(driver), '-o', str(exe)],
        cwd=build, check=True, capture_output=True, timeout=30)
    assert {str(p): R.sha(p) for p in sources} == hashes
    (build/'native-source-sha256.json').write_text(json.dumps(hashes, indent=2))
    return exe


@pytest.mark.parametrize('nd,nt', [(3, 6), (90, 1300)])
def test_actual_unmodified_cmfgen_reader_recovers_all_arrays(tmp_path, actual_native_reader, nd, nt):
    header, radius, state = fixture_data(nd, nt)
    blob, pointer = R.restart_bytes(header, radius, state)
    (tmp_path/'SCRTEMP').write_bytes(blob)
    for name in ('POINT1', 'POINT2'): (tmp_path/name).write_text(pointer)
    subprocess.run([str(actual_native_reader), str(nd), str(nt)], cwd=tmp_path,
                   check=True, capture_output=True, timeout=10)
    expected = header[:3*nd*8]+np.asarray(state, dtype='<f8').tobytes(order='F')
    assert (tmp_path/'native-read.bin').read_bytes() == expected
