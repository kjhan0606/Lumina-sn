"""원 커널 무변경 압축 포획의 소스 계약과 stream 판독 시험."""
import struct
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_siii_compact_capture import BOUNDARY, HEADER, POINT, records
from prepare_cmfgen_formal_capture import module_source
from prepare_cmfgen_siii_compact_capture import FIRST, LAST, compact_source, replace_subroutine


def test_compact_source_keeps_one_geometry_and_one_stream():
    source=compact_source(module_source())
    assert f'LUMINA_FREQ_INDEX>={FIRST}.AND.LUMINA_FREQ_INDEX<={LAST}' in source
    assert source.count('COMPACT_GEOMETRY_WRITTEN=.TRUE.')==1
    assert source.count("FILE='siii_compact.bin'")==2
    assert source.count('LUMINASIIIRAYV1')==1
    assert source.count('IF(READING)STOP 86')==2


def test_subroutine_replacement_refuses_missing_or_ambiguous_target():
    with pytest.raises(ValueError):
        replace_subroutine('END MODULE','FORMAL_RAYS_IO','SUBROUTINE X\nEND SUBROUTINE')
    duplicate='SUBROUTINE FORMAL_RAYS_IO(X)\nEND SUBROUTINE\n'*2
    with pytest.raises(ValueError):
        replace_subroutine(duplicate,'FORMAL_RAYS_IO','SUBROUTINE X\nEND SUBROUTINE')


def test_compact_record_reconstructs_native_angular_sum(tmp_path):
    nd=np_=52
    g=dict(ND=nd,NP=np_,RAY_POINTS_INSERTED=False,
           rays=[dict(R=np.ones(1),LNK=np.ones(nd,dtype='i4')) for _ in range(np_)],
           weights=dict(JQW_P=np.ones((nd,np_)),JQW_M=np.zeros((nd,np_))))
    header=HEADER.pack(b'LUMINASIIIRAYV1'.ljust(16,b' '),71,FIRST,0,np_,nd,nd,np_-1,1,0,1.,.01)
    coefficients=np.r_[np.array([4.,3.,2.,1.]),np.zeros(2*nd)].astype('<f8').tobytes()
    boundary=b''.join(BOUNDARY.pack(ip,1,1.,0.,1.,0.) for ip in range(1,np_))
    points=b''.join(POINT.pack(ip,id_,1,1.,0.,1.,0.,.01,.01)
                    for id_ in range(49,53)
                    for ip in range(1,np_-id_+2))
    path=tmp_path/'siii_compact.bin'
    path.write_bytes(header+coefficients+boundary+points+struct.pack('<16s',b'LUMINASIIIEND1'.ljust(16,b' ')))
    result=list(records(path,g))
    assert len(result)==1
    np.testing.assert_array_equal(result[0]['native'],[4.,3.,2.,1.])
    assert result[0]['max_roundoff_fraction']==0


def test_compact_record_rejects_wrong_native_j(tmp_path):
    nd=np_=52
    g=dict(ND=nd,NP=np_,RAY_POINTS_INSERTED=False,
           rays=[dict(R=np.ones(1),LNK=np.ones(nd,dtype='i4')) for _ in range(np_)],
           weights=dict(JQW_P=np.ones((nd,np_)),JQW_M=np.zeros((nd,np_))))
    header=HEADER.pack(b'LUMINASIIIRAYV1'.ljust(16,b' '),71,FIRST,0,np_,nd,nd,np_-1,1,0,1.,.01)
    coefficients=np.r_[np.array([5.,3.,2.,1.]),np.zeros(2*nd)].astype('<f8').tobytes()
    boundary=b''.join(BOUNDARY.pack(ip,1,1.,0.,1.,0.) for ip in range(1,np_))
    points=b''.join(POINT.pack(ip,id_,1,1.,0.,1.,0.,.01,.01)
                    for id_ in range(49,53)
                    for ip in range(1,np_-id_+2))
    path=tmp_path/'siii_compact.bin'
    path.write_bytes(header+coefficients+boundary+points+struct.pack('<16s',b'LUMINASIIIEND1'.ljust(16,b' ')))
    with pytest.raises(ValueError,match='광선합'):
        list(records(path,g))
