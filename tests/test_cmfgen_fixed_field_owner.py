"""살아 있는 native run의 완료 물질만 분리하고 J60 bytes는 그대로 유지한다."""
from pathlib import Path
import csv
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_cmfgen_fixed_field_owner as export
from summarize_cmfgen_fixed_field_owner import quantiles,read_response


def test_completed_record_is_read_even_with_incomplete_later_record(tmp_path):
    nd,nt=3,5;state=np.arange(1,nd*nt+1,dtype=float).reshape(nt,nd)
    payload=state.T.astype('<f8').tobytes()
    raw=payload+b'\0'*(export.REC_BYTES-len(payload))
    path=tmp_path/'SCRTEMP'
    path.write_bytes(b'\0'*(export.SCRTEMP_HEADER_RECS*export.REC_BYTES)+raw+b'partial next record')
    _,read,data=export.record_bytes(path,1,nd,nt)
    assert read==raw;np.testing.assert_array_equal(data,state)
    with pytest.raises(ValueError,match='짧음'):export.record_bytes(path,2,nd,nt)


def test_both_committed_pointers_are_required(tmp_path,monkeypatch):
    for name,value in (('POINT1',61),('POINT2',60)):
        (tmp_path/name).write_text(f'header\n{value} {value} 1 -1000 F\n')
    monkeypatch.setattr(export,'read_dimensions',lambda _:(3,5))
    with pytest.raises(ValueError,match='완료되지'):
        export.snapshot(tmp_path,Path('not-read'),61)


def binary_fixture(tmp_path):
    old=np.array([[1.,2.],[3.,4.],[5.,6.],[10.,20.],[1.,2.]])
    new=old.copy();new[:3]*=1.1;new[-2]*=1.2
    columns=np.array([[14,1,0],[16,1,0]],dtype='<i4');nn=2;nb=2
    header=struct.pack('<8sIIIId',b'CMFOWN01',2,nn,2,nb,1e6)
    prefix=header+np.array([1e10,1e20],dtype='<f8').tobytes()+columns.tobytes()
    field=bytes(range(12*nb+8*nn));raw=prefix
    for d in range(2):
        raw+=np.array([1e-14,old[-1,d]*1e4,old[-2,d],1,2],dtype='<f8').tobytes()
        raw+=old[:2,d].astype('<f8').tobytes()+field
    path=tmp_path/'base.bin';path.write_bytes(raw)
    return path,old,new,prefix,field


def test_only_new_populations_and_T_ne_replace_material_not_radiation(tmp_path):
    path,old,new,prefix,field=binary_fixture(tmp_path);out=tmp_path/'new.bin'
    columns,profiles,identity=export.replace_material(path,out,new,old,np.array([0,1]))
    assert identity['radiation_blocks_byte_equal']
    raw=out.read_bytes();assert raw[:len(prefix)]==prefix
    np.testing.assert_array_equal(profiles[:,2],new[-2])
    n=len(prefix)
    for d in range(2):
        p=np.frombuffer(raw[n:n+40],dtype='<f8');n+=40
        np.testing.assert_array_equal(p[1:3],[new[-1,d]*1e4,new[-2,d]])
        np.testing.assert_array_equal(np.frombuffer(raw[n:n+16],dtype='<f8'),new[:2,d]);n+=16
        assert raw[n:n+len(field)]==field;n+=len(field)
    assert n==len(raw)


def test_wrong_original_material_is_not_silently_reused(tmp_path):
    path,old,new,_,_=binary_fixture(tmp_path);old[0,0]*=2
    with pytest.raises(ValueError,match='baseline 불일치'):
        export.replace_material(path,tmp_path/'refused.bin',new,old,np.array([0,1]))


def test_existing_output_is_preserved_before_any_input_reads(tmp_path):
    with pytest.raises(ValueError,match='기존 완료'):
        export.prepare(Path('not-read'),Path('not-read'),60,tmp_path)


def test_distribution_quantiles_use_mass_not_number_of_shells():
    assert quantiles([0.,1.,2.],[1.,1.,100.])==[2.,2.,2.]


@pytest.mark.parametrize('defect',[None,'ne','total','commit','partial'])
def test_only_completed_response_of_same_material_and_actual_commit_is_compared(tmp_path,defect):
    refs={};rows=[]
    for Z in (14,16):
        for stage in range(2,7):
            refs[1,Z,stage]=dict(fraction=.2,Te_K=1e4,ne_cm3=1e7,population_cm3=.2)
            rows.append(dict(depth_1based=1,Z=Z,stage=stage,Te_K=1e4,ne_cm3=1e7,n_element=1.,
                pre_fraction=.2,candidate_fraction=.2,committed_fraction=.2,scaled_se_residual=1e-15,
                conservation_residual=0,certificate_probability_error_display=1e-16,certificate_relative_error_display=1e-16,
                completed=1,candidate_valid=1,certificate_pass=1,verdict_pass=1,commit_performed=1))
    if defect=='ne':rows[0]['ne_cm3']=2e7
    if defect=='total':rows[0]['n_element']=2.
    if defect=='commit':rows[0]['committed_fraction']=.3
    with (tmp_path/'response.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (tmp_path/'owner.log').write_text('working' if defect=='partial' else 'CMF_OWNER_RESPONSE_COMPLETE depths=1\n')
    if defect:
        with pytest.raises(ValueError):read_response(tmp_path,refs,1)
    else:assert len(read_response(tmp_path,refs,1))==10
