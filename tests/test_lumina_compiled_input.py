"""C 원본 캐시·보간·J 세대와 같은 소형 수송 연결. 실제 formal은 없다."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import lumina_outer_atmosphere as A
import lumina_native_columns as N
import lumina_primary_columns as P
import lumina_compiled_input as C
from lumina_compiled_history import SmallCompiledInputTransport
from test_lumina_outer_atmosphere import sample
from test_lumina_frequency_history import history


@pytest.fixture(scope='module')
def input_library(tmp_path_factory):
    target=tmp_path_factory.mktemp('compiled-input')/'input.so'
    subprocess.run([os.environ.get('CC','cc'),'-O2','-std=c11','-Wall','-Wextra','-Werror',
        '-fPIC','-shared',str(ROOT/'src/cmf_nodal_input.c'),'-lm','-o',str(target)],
        check=True,capture_output=True,timeout=60)
    return target


def packet(root,nf=17,physical=False):
    root.mkdir(exist_ok=True);ns=24
    if physical:
        g,chi,es,eta,_=sample();nu=1e14*np.exp(.05*np.arange(8));nf=8
        parts=np.zeros((7,ns,nf));parts[0]=-es;parts[4]=chi;parts[6]=es;parts[1]=eta
    else:
        r=np.geomspace(1e15,2e14,ns);g=A.geometry(r,.06*A.C*r/r[0],np.zeros(ns))
        nu=np.geomspace(1e14,2e14,nf);parts=np.random.default_rng(418).uniform(.1,1.,(7,ns,nf))
        parts[0]*=-1;parts[4]*=-2;parts[[1,3,5],:,-1]=0.
        parts[6,0,0]=1e16;parts[0,0,0]=-1e16;parts[2,0,0]=1.;parts[4,0,0]=.25
        parts[1,0,0]=1e16;parts[3,0,0]=parts[5,0,0]=1.
    temperature=np.full(ns,14000.);ne=np.full(ns,1e8);path=root/'sources.bin'
    path.write_bytes(struct.pack('<8sII',b'LMPHYS01',ns,nf)+nu.astype('<f8').tobytes()+
        temperature.astype('<f8').tobytes()+ne.astype('<f8').tobytes()+parts.astype('<f8').tobytes())
    source=N.NativeSourceColumns(path,nu,g.native_radius,temperature,ne)
    return g,source,parts


class Field:
    def __init__(self,source):
        self.ns,self.nf=source.ns,source.nf;self.radius=source.radius;self.frequency=source.frequency;self.closed=False
        self.values=.3+np.arange(self.ns)[:,None]*.04+np.arange(self.nf)[None,:]**2*.03
        self.values[:,-1]=0.;self.identity=dict(role='SYNTHETIC_INPUT_ONLY')
    def read(self,rows,columns):return self.values[np.ix_(rows,columns)]


def test_canonical_raw_sum_cache_hits_and_last_single_column(input_library,tmp_path):
    g,source,parts=packet(tmp_path,nf=33);field=Field(source)
    before=N.sha(source.path)
    with source,C.CompiledInput(input_library,g,source,field,slab_width=8,cache_rows=4,profiles=8) as provider:
        rows=np.array([23,0,0]);columns=np.array([32,0,1,0])
        values=provider.raw(rows,columns);expected=source.read(rows,columns)
        for a,b in zip(values,expected):np.testing.assert_array_equal(a,b)
        assert values[0][1,1]==.25 and values[2][1,1]==1e16
        first=provider.report();provider.raw(rows,columns);last=provider.report()
        assert first['source_bytes']==last['source_bytes'] and last['cache_hits']>first['cache_hits']
        assert last['cache_peak_rows']<=4 and last['source_bytes']==7*2*(8+1)*8
        assert last['static_cache_bytes']<10000
    assert N.sha(source.path)==before
    with pytest.raises(ValueError):provider.raw([0],[0])


def test_radius_then_es_j_then_frequency_and_legacy_outer_closure(input_library,tmp_path):
    g,source,_=packet(tmp_path);field=Field(source)
    radii=[g.radius[0],g.radius[7],.5*(g.radius[23]+g.radius[24]),g.native_radius[0],
        g.native_radius[7],.5*(g.native_radius[10]+g.native_radius[11]),g.native_radius[-1]]
    columns=np.array([0.,1.,6.375,16.,31.])
    with source,C.CompiledInput(input_library,g,source,field,slab_width=4,cache_rows=8,profiles=32) as provider:
        provider.bind(0,source.nf);actual=provider.at(radii,columns);legacy=N.NodalColumnProvider(g,source,field)
        for row,radius in enumerate(radii):
            left=np.minimum(np.floor(columns).astype('i8'),source.nf-1)
            right=np.minimum(left+1,source.nf-1);fraction=np.where(columns>=source.nf-1,0.,columns-left)
            a=legacy.at(float(radius),left);b=legacy.at(float(radius),right)
            for value,first,last in zip(actual,a,b):
                np.testing.assert_allclose(value[row],first+fraction*(last-first),rtol=2e-12,atol=0)
        # 끝 ν의 native η=0과 기존 외부 폐쇄관계의 작은 값은 구별한다.
        assert actual[1][-1,-1]==0 and actual[1][0,-1]>0
        assert np.any(actual[0]<0)


def test_bind_is_a_snapshot_and_invalidates_only_j_dependent_profiles(input_library,tmp_path):
    g,source,_=packet(tmp_path);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field,slab_width=4,cache_rows=8,profiles=8) as provider:
        provider.bind(4,4);first=provider.at([g.radius[4]],[5.25]);state=provider.report()
        field.values*=2
        unchanged=provider.at([g.radius[4]],[5.25])
        for a,b in zip(first,unchanged):np.testing.assert_array_equal(a,b)
        provider.bind(4,4);changed=provider.at([g.radius[4]],[5.25]);after=provider.report()
        np.testing.assert_array_equal(first[0],changed[0]);assert not np.array_equal(first[1],changed[1])
        assert after['source_bytes']==state['source_bytes'] and after['bindings']==2
        with pytest.raises(ValueError):provider.at([g.native_radius[5]],[7.5])
        with pytest.raises(ValueError):provider.at([g.native_radius[5]],[3.])


def test_dense_arrays_preserve_canonical_values_and_j_generation(input_library,tmp_path):
    g,source,_=packet(tmp_path,nf=33);field=Field(source)
    radii=[g.radius[0],g.radius[7],g.native_radius[0],
           .5*(g.native_radius[10]+g.native_radius[11]),g.native_radius[-1]]
    columns=np.array([0.,1.,6.375,32.,50.])
    with source,C.CompiledInput(input_library,g,source,field,slab_width=4,profiles=8) as provider:
        with pytest.raises(ValueError):provider.prepare_dense()
        provider.bind(0,source.nf);before=provider.at(radii,columns)
        expected=provider.raw([23,0,0],[32,0,1])
        dense=provider.prepare_dense()
        assert dense['ready']==1 and dense['canonical_bytes']==3*24*33*8
        for a,b in zip(before,provider.at(radii,columns)):np.testing.assert_array_equal(a,b)
        for a,b in zip(expected,provider.raw([23,0,0],[32,0,1])):np.testing.assert_array_equal(a,b)
        state=provider.report();provider.at(radii,columns)
        assert provider.report()['source_preads']==state['source_preads']
        field.values*=2;provider.bind(0,source.nf)
        assert provider.report()['dense']['ready']==0 and provider.report()['dense']['profile_bytes']==0
        sparse_profile=provider.at(radii,columns);provider.prepare_dense()
        for a,b in zip(sparse_profile,provider.at(radii,columns)):np.testing.assert_array_equal(a,b)
        assert provider.report()['source_bytes']==state['source_bytes']
        assert not np.array_equal(before[1],sparse_profile[1])
        assert provider.report()['dense']['profile_accesses']>0


def test_dense_prepare_time_failure_does_not_publish_partial_arrays(input_library,tmp_path):
    g,source,_=packet(tmp_path);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field) as provider:
        provider.bind(0,source.nf)
        with pytest.raises(ValueError,match='준비'):provider.prepare_dense(max_seconds=1e-15)
        assert provider.report()['dense']['ready']==0
        assert provider.report()['dense']['canonical_bytes']==0
        assert provider.report()['dense']['profile_bytes']==0
        provider.prepare_dense()
        assert provider.report()['dense']['ready']==1


def test_borrowed_dense_view_has_exact_layout_and_is_invalidated_on_bind(input_library,tmp_path):
    class View(C.ct.Structure):
        _fields_=[(key,C.ct.c_int) for key in ('ns','nf','ne','jmode')]+[
            ('generation',C.ct.c_uint64)]+[(key,C.ct.POINTER(C.ct.c_double))
                for key in ('canonical','outer','j','radius','external_radius')]
    g,source,_=packet(tmp_path,nf=33);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field) as provider:
        get=provider._lib.cmf_nodal_input_dense_view
        get.argtypes=[C.ct.c_void_p,C.ct.POINTER(View)];get.restype=C.ct.c_int
        view=View();view.ns=-17
        assert get(provider._ctx,C.ct.byref(view))!=0 and view.ns==-17
        provider.bind(0,source.nf);provider.prepare_dense()
        assert get(provider._ctx,C.ct.byref(view))==0
        assert (view.ns,view.nf,view.ne,view.jmode,view.generation)==(24,33,g.added+1,1,1)
        canonical=np.ctypeslib.as_array(view.canonical,shape=(3*24*33,)).reshape(3,24,33)
        expected=provider.raw(np.arange(24),np.arange(33))
        for a,b in zip(canonical,expected):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(np.ctypeslib.as_array(view.j,shape=(24*33,)).reshape(24,33),field.values)
        np.testing.assert_array_equal(np.ctypeslib.as_array(view.radius,shape=(24,)),g.native_radius)
        provider.bind(0,source.nf)
        view2=View();view2.ns=-17
        assert get(provider._ctx,C.ct.byref(view2))!=0 and view2.ns==-17
        # 이전 view의 포인터는 여기서 재사용하지 않는다.
        provider.prepare_dense();assert get(provider._ctx,C.ct.byref(view2))==0
        assert view2.generation==2
        field.identity['role']='EXPLICIT_ZERO_J_MICRO_TRIAL_ONLY'
        provider.bind_zero_trial()
        assert get(provider._ctx,C.ct.byref(view2))!=0
        provider.prepare_dense();assert get(provider._ctx,C.ct.byref(view2))==0
        assert view2.generation==3 and view2.jmode==0 and not bool(view2.j)


def test_python_dense_borrow_and_identity_follow_actual_successful_binding(input_library,tmp_path):
    g,source,_=packet(tmp_path,nf=33);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field) as provider:
        with pytest.raises(ValueError):provider.borrow_dense_view()
        field.identity=dict(role='FIRST_SYNTHETIC_FIELD',sequence=1)
        provider.bind(0,source.nf)
        with pytest.raises(ValueError):provider.borrow_dense_view()
        provider.prepare_dense();view=provider.borrow_dense_view()
        assert view.generation==1 and provider.identity['field']==field.identity
        field.identity=dict(role='SECOND_SYNTHETIC_FIELD',sequence=2)
        field.values*=1.7;provider.bind(0,source.nf)
        assert provider.identity['field']==field.identity
        with pytest.raises(ValueError):provider.borrow_dense_view()
        provider.prepare_dense();view=provider.borrow_dense_view();assert view.generation==2
        expected=np.ctypeslib.as_array(view.j,shape=(24*33,)).reshape(24,33)
        np.testing.assert_array_equal(expected,field.values)
        field.identity=dict(role='EXPLICIT_ZERO_J_MICRO_TRIAL_ONLY')
        provider.bind_zero_trial();provider.prepare_dense();view=provider.borrow_dense_view()
        assert view.generation==3 and view.jmode==0 and not view.j
        assert provider.identity['field']==field.identity
    with pytest.raises(ValueError):provider.borrow_dense_view()


def test_explicit_zero_trial_has_no_full_j_array_and_does_not_leak_into_rebinding(input_library,tmp_path):
    g,source,_=packet(tmp_path);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field,slab_width=4,profiles=32) as provider:
        with pytest.raises(ValueError):provider.bind_zero_trial()
        field.identity['role']='EXPLICIT_ZERO_J_MICRO_TRIAL_ONLY'
        provider.bind(0,source.nf);previous=provider.at([g.radius[4]],[5.25])
        field.values[:]=0.;provider.bind_zero_trial()
        radii=[g.radius[4],g.native_radius[7],.5*(g.native_radius[10]+g.native_radius[11])]
        columns=np.array([0.,5.25,16.]);actual=provider.at(radii,columns)
        legacy=N.NodalColumnProvider(g,source,field)
        for row,radius in enumerate(radii):
            left=np.floor(columns).astype('i8');right=np.minimum(left+1,source.nf-1);f=columns-left
            a=legacy.at(float(radius),left);b=legacy.at(float(radius),right)
            for value,x,y in zip(actual,a,b):np.testing.assert_allclose(value[row],x+f*(y-x),rtol=2e-12,atol=0)
        report=provider.report();assert report['field_buffer_bytes']==report['field_view_address_bytes']==0
        assert not np.array_equal(previous[1],provider.at([g.radius[4]],[5.25])[1])
        field.values[:]=.7;provider.bind(0,source.nf)
        assert not np.array_equal(actual[1],provider.at(radii,columns)[1])
        field.values[:]=.8;field.values.flags.writeable=False;provider.bind_view()
        assert not np.array_equal(actual[1],provider.at(radii,columns)[1])


def test_readonly_view_keeps_its_owner_and_rebinding_changes_only_j_profiles(input_library,tmp_path):
    g,source,_=packet(tmp_path);field=Field(source)
    with source,C.CompiledInput(input_library,g,source,field,slab_width=4,cache_rows=8,profiles=32) as provider:
        with pytest.raises(ValueError):provider.bind_view()
        field.values=np.array(field.values,order='C',copy=True);field.values.flags.writeable=False
        provider.bind_view();first=provider.at([g.radius[4]],[5.25]);state=provider.report()
        field.values=2*field.values;field.values.flags.writeable=False
        # C가 보유한 옛 view는 Python field의 속성 교체로 사라지지 않는다.
        for a,b in zip(first,provider.at([g.radius[4]],[5.25])):np.testing.assert_array_equal(a,b)
        provider.bind_view();changed=provider.at([g.radius[4]],[5.25]);after=provider.report()
        assert not np.array_equal(first[1],changed[1])
        assert after['source_bytes']==state['source_bytes'] and after['field_bytes_copied']==0
        assert after['field_view_address_bytes']==source.ns*source.nf*8 and after['field_buffer_bytes']==0


def test_memmap_view_and_nonfinite_selected_j_do_not_bypass_validation(input_library,tmp_path):
    g,source,_=packet(tmp_path);trial=Field(source);path=tmp_path/'native-j.npy';np.save(path,trial.values)
    with source,N.FrozenNativeField(path,source,source.radius,source.frequency) as field:
        with C.CompiledInput(input_library,g,source,field,slab_width=4,profiles=32) as provider:
            provider.bind_view();value=provider.at([g.native_radius[5]],[4.25])
            assert all(np.isfinite(v).all() for v in value)
            field.close()
            with pytest.raises(ValueError):provider.at([g.native_radius[5]],[4.25])
    g,source,_=packet(tmp_path/'invalid');trial=Field(source)
    trial.values[6,3]=np.nan;trial.values.flags.writeable=False
    with source,C.CompiledInput(input_library,g,source,trial) as provider:
        provider.bind_view()
        with pytest.raises(ValueError):provider.at([.5*(source.radius[5]+source.radius[6])],[3.])


@pytest.mark.parametrize('bad',['negative_column','radius','nonfinite','missing_j','bad_cache'])
def test_invalid_inputs_do_not_create_a_transport_result(input_library,tmp_path,bad):
    g,source,_=packet(tmp_path);field=Field(source)
    with source:
        if bad=='bad_cache':
            with pytest.raises(ValueError):C.CompiledInput(input_library,g,source,field,slab_width=8193)
            return
        with C.CompiledInput(input_library,g,source,field,slab_width=4,profiles=8) as provider:
            if bad!='missing_j':provider.bind(0,source.nf)
            with pytest.raises(ValueError):
                provider.at([g.radius[0]*2 if bad=='radius' else g.native_radius[5]],
                    [-1. if bad=='negative_column' else np.nan if bad=='nonfinite' else 4.])


def test_actual_native90_selected_columns_are_read_only_and_not_formal(input_library,tmp_path):
    config=os.environ.get('LUMINA_NATIVE_SOURCE_CONFIG');prep=os.environ.get('LUMINA_NATIVE_SOURCE_PREPARATION')
    digest=os.environ.get('LUMINA_NATIVE_SOURCE_PREPARATION_SHA256')
    if config is None and prep is None and digest is None:pytest.skip('실제 봉인 입력이 지정되지 않음')
    assert config and prep and digest
    with N.open_sealed_native(config,prep,digest) as source,P.open_sealed_reference(config,prep,digest,source) as field:
        preparation=json.loads(Path(prep).read_text());geometry_path=Path(prep).parent/'geometry.npz'
        assert N.sha(geometry_path)==preparation['output_sha256']['geometry.npz']
        with np.load(geometry_path,allow_pickle=False) as data:
            g=A.OuterGeometry(data['radius'].copy(),data['velocity'].copy(),data['sigma'].copy(),
                len(data['radius'])-len(data['native_radius']),tuple(data['reference_indices']),data['native_radius'].copy())
        with C.CompiledInput(input_library,g,source,field,slab_width=4,cache_rows=8,profiles=8) as provider:
            provider.bind(45182,4);radii=[source.radius[0],.5*(source.radius[14]+source.radius[15]),g.radius[4]]
            columns=np.array([45183.,45183.375,45184.]);actual=provider.at(radii,columns)
            legacy=N.NodalColumnProvider(g,source,field);first=np.floor(columns).astype('i8');fraction=columns-first
            for row,radius in enumerate(radii):
                a=legacy.at(float(radius),first);b=legacy.at(float(radius),first+1)
                for value,left,right in zip(actual,a,b):
                    np.testing.assert_allclose(value[row],left+fraction*(right-left),rtol=2e-12,atol=0)
            report=dict(cache=provider.report(),source_identity=source.identity,field_identity=field.identity,
                native_nodes=source.ns,frequencies=source.nf,actual_model_F_calls=0,formal_calls=0,
                full_source_hash_reverified=False,full_field_hash_reverified=False,final_goal='NOT_JUDGED')
    (tmp_path/'compiled-native90-selected-columns.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def test_compiled_provider_same_material_saved_j_return(history,input_library,tmp_path):
    path=Path(os.environ['LUMINA_COMPILED_SAVED_J']);digest=os.environ['LUMINA_COMPILED_SAVED_J_SHA256']
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    record=json.loads(raw);assert record['panels']==16
    g,source,parts=packet(tmp_path,physical=True);chi,es,eta=source.read(np.arange(24),np.arange(8))
    with source:
        candidate=SmallCompiledInputTransport(g,source.frequency,chi,eta,es,14000.,np.zeros(8),
            history=history,panels=16,input_library=input_library,source=source)
        saved=np.array(record['value']);result,report=candidate.apply(saved)
        relative=float(abs(result-saved).max()/saved.max())
        assert relative<=1e-10
        stats=report['compiled_source'];assert stats['source_bytes']==7*24*8*8
        assert stats['source_preads']==7*24 and stats['bindings']==1
        (tmp_path/'compiled-provider-saved-return.json').write_text(json.dumps(dict(
            path=str(path),sha256=digest,actual_relative_residual=relative,reference_residual=record['actual_relative_residual'],
            cache=stats,formal_vectors=report['formal_vectors'],elapsed_seconds=report['elapsed_seconds'],
            production_driver_complete=False,actual_model_F_calls=0,final_goal='NOT_JUDGED'),indent=2,allow_nan=False))
