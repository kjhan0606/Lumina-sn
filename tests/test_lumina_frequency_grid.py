"""전체 참조 범위/단위와 실제 선폭을 보존하는 실행 입력 시험."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_lumina_frequency_grid as G


def fixture(tmp_path):
    field=tmp_path/'field.bin';model=tmp_path/'MODEL';ns=3;frequency=[2.,1.8,1.2,1.]
    with field.open('xb') as stream:
        stream.write(b'FORMALPRIMARYV1 '.ljust(16,b' '))
        stream.write(struct.pack('<6i',8,64,32,ns,len(frequency),8))
        stream.write(np.zeros(4*ns,dtype='<f8').tobytes())
        for index,nu in enumerate(frequency):
            stream.write(struct.pack('<iidd',index+1,1,nu,1.))
            stream.write(np.zeros((8,ns),dtype='<f8').tobytes())
    model.write_text('2 [TDOP]\n4 [AMASS_DOP]\n10 [VTURB]\n6 [MAX_DOP]\n1 [FRAC_DOP]\n')
    return field,model


def test_real_full_range_units_and_doppler_spacing_are_preserved(tmp_path):
    field,model=fixture(tmp_path);out=tmp_path/'grid';before=G.sha(field)
    report=G.prepare(field,before,model,4,out)
    nu=np.load(out/'frequency.npy',allow_pickle=False)
    assert nu[0]==1e15 and nu[-1]==2e15
    step=np.log(nu[1:]/nu[:-1])
    assert np.max(step)<=report['profile']['width_fraction']/4
    np.testing.assert_allclose(step,report['dlognu'],rtol=0,atol=5e-16)
    assert G.sha(field)==before
    assert report['output_sha256']['frequency.npy']==G.sha(out/'frequency.npy')
    assert not report['frequency_resolution_certified'] and report['physical_RT_executions']==0
    assert json.loads((out/'grid.json').read_text())['native_points']==4
    with pytest.raises(ValueError,match='보존'):G.prepare(field,before,model,4,out)


def test_wrong_reference_identity_does_not_create_grid(tmp_path):
    field,model=fixture(tmp_path);out=tmp_path/'grid'
    with pytest.raises(ValueError,match='SHA256'):G.prepare(field,'0'*64,model,4,out)
    assert not out.exists()


@pytest.mark.parametrize('kind',['trailing','reversed'])
def test_incomplete_or_wrong_order_reference_is_not_repaired(tmp_path,kind):
    field,model=fixture(tmp_path)
    if kind=='trailing':field.write_bytes(field.read_bytes()+b'X')
    else:
        data=bytearray(field.read_bytes());struct.pack_into('<d',data,40+32*3+8,.5)
        field.write_bytes(data)
    with pytest.raises(ValueError):G.prepare(field,G.sha(field),model,4,tmp_path/'grid')
