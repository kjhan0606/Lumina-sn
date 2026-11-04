"""기존 oracle의 읽기 함수만 검사한다. 최상위 검증 실행/출력 작성은 하지 않는다."""
from pathlib import Path
import csv
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from oracle_compare_cmfgen import parse_prrr


@pytest.fixture
def reader():
    return parse_prrr


@pytest.fixture
def prrr(tmp_path):
    path=tmp_path/'SkIIIPRRR'
    path.write_text('''Depth index
1 2
Electron Density
5 7
Ion Density
100 200
SkIII Photoionization Rates
10 20
2 4
SkIII Recombination Rates
3 6
1 2
Radiative Recombination Coefficient for explicitly treated levels.
0.01 0.02
''')
    return path


def test_parent_ground_density_never_becomes_lower_ion_gamma(reader,prrr):
    r=reader(prrr,'SkIII',2,2)
    np.testing.assert_array_equal(r['parent_ground_density'],[100,200])
    np.testing.assert_array_equal(r['photo_flow'],[12,24])
    np.testing.assert_array_equal(r['recombination_flow'],[4,8])
    assert r['gamma'] is None
    assert 'ion_density' not in r
    assert 'alpha' not in r


def test_gamma_uses_explicit_lower_total_not_parent_ground(reader,prrr):
    r=reader(prrr,'SkIII',2,2,lower_ion_density=np.array([2.,4.]))
    np.testing.assert_array_equal(r['gamma'],[6.,6.])
    np.testing.assert_array_equal(r['alpha_explicit_parent_ground'],[0.01,0.02])


@pytest.mark.parametrize('density',[[1.],[-1.,2.],[0.,2.],[np.nan,2.]])
def test_bad_lower_density_cannot_produce_a_gamma(reader,prrr,density):
    with pytest.raises(ValueError):reader(prrr,'SkIII',2,2,lower_ion_density=np.array(density))


@pytest.mark.parametrize('change',['missing_recombination','extra_photo_row','wrong_depth'])
def test_partial_or_misaddressed_rate_matrix_is_rejected(reader,prrr,change):
    text=prrr.read_text()
    if change=='missing_recombination':text=text.replace('SkIII Recombination Rates','other rates')
    if change=='extra_photo_row':text=text.replace('2 4\n','2 4\n5 6\n')
    if change=='wrong_depth':text=text.replace('1 2\nElectron','1 3\nElectron')
    prrr.write_text(text)
    with pytest.raises(ValueError):reader(prrr,'SkIII',2,2)


@pytest.mark.parametrize('depths',['2 1','1 1','1.5 2'])
def test_depths_cannot_be_silently_relabelled(reader,prrr,depths):
    prrr.write_text(prrr.read_text().replace('1 2\nElectron',depths+'\nElectron'))
    with pytest.raises(ValueError):reader(prrr,'SkIII',2,2)


def test_duplicate_chunk_cannot_overwrite_previous_values(reader,prrr):
    prrr.write_text(prrr.read_text()*2)
    with pytest.raises(ValueError):reader(prrr,'SkIII',2,2)


@pytest.fixture
def saved_flows(tmp_path,prrr):
    from summarize_cmfgen_ir_grid_impact import sha
    root=tmp_path/'response'; native=tmp_path/'native'
    evaluation=root/'evaluation-4'
    evaluation.mkdir(parents=True);(native/'seq_logs').mkdir(parents=True)
    rows=[];columns=[];pop=[]
    for label,z,stage in [('Sk2',14,2),('SkIII',14,3),('SkIV',14,4),
                          ('S2',16,2),('SIII',16,3),('SIV',16,4)]:
        for sl in range(2 if stage<4 else 1):
            rows.append([len(rows)+1,z,sl+1,sl+1,len(rows)+1,label])
            columns.append([z,stage-1,sl])
            pop.append([100.,200.] if stage>2 and sl==0 else [1.,2.])
    with (native/'NATIVE_FULL_INDEX.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(
            ['native_ion','Z','full_level','superlevel','eq_sl','ion_label']);writer.writerows(rows)
    labels=['Sk2','SkIII','S2','SIII']
    (native/'MODEL_SPEC').write_text('\n'.join('2, [ '+label+' ] ['+label+'_ISF]' for label in labels))
    for label in labels:
        (native/(label+'PRRR')).write_text(prrr.read_text().replace('SkIII',label))
    columns=np.array(columns);pop=np.array(pop);ne=np.array([5.,7.]);temp=np.array([10000.,11000.])
    pops=np.vstack([pop,ne,temp*1.e-4]);nt=len(pops);radius=np.array([2.,1.])
    vectors=np.stack([radius,np.ones(2),np.ones(2),temp*1.e-4,ne,
                      np.ones(2),np.ones(2),np.zeros(2),np.zeros(2)])
    heat=native/'NATIVE_HEAT_ROW.bin'
    heat.write_bytes(b'NATIVEHEATROWV1 '+struct.pack('<8i',1,64,nt,2,123,1,9,0)+
                     vectors.astype('<f8').tobytes()+pops.astype('<f8').tobytes(order='F'))
    np.savez(evaluation/'response.npz',columns=columns,population=pop,ne=ne,temperature=temp)
    dens=np.array([pop[(columns[:,0]==z)&(columns[:,1]==charge)].sum(axis=0)
                   for z,charge in [(14,1),(14,2),(16,1),(16,2)]])
    flows=np.tile(np.array([[12.,24.],[3.,6.],[1.,2.],[8.,16.]])[None,:,:],(4,1,1))
    np.savez(evaluation/'balance.npz',bf_ion_Z_stage=[[14,2],[14,3],[16,2],[16,3]],
             bf_ion_number_density=dens,bf_ion_number_flows=flows,
             ff_native_upper_ions=[[0.,0.],[1.,2.]])
    selected=dict(root='evaluation-4',response_sha256=sha(evaluation/'response.npz'),
                  balance_sha256=sha(evaluation/'balance.npz'))
    (root/'result.json').write_text(json.dumps(dict(selected_response=selected)))
    runtime=dict(status='CAPTURED',captured_state_exact=True,
        accepted_response_sha256=selected['response_sha256'],material_update=False,new_transport=False,
        outputs={'NATIVE_HEAT_ROW.bin':dict(sha256=sha(heat))})
    (native/'seq_logs/fixedj-heat-runtime.json').write_text(json.dumps(runtime))
    (native/'GENCOOL').write_text('''Depth
1 2
Radius [1.0E+10cm]
2 1

Temperature [1.0E+4K]
1 1.1

Electron Density
5 7

SkIII (ion) Free-Free Cooling
1 2

Net Cooling Rate [ergs/cm**3/sec]
1 2
''')
    return root,native


def test_saved_continuum_flows_use_same_state_and_no_solver(saved_flows):
    from summarize_cmfgen_ir_grid_impact import compare_saved_continuum_flows
    report=compare_saved_continuum_flows(*saved_flows)
    assert report['final_goal']=='NOT_MET' and report['new_physical_evaluations']==0
    assert report['ff']['maximum_absolute_difference']==0
    for ion in report['bf_number_flows'].values():
        assert ion['photo']['maximum_relative_difference']==0
        assert ion['recombination']['maximum_relative_difference']==0


def test_saved_continuum_flows_reject_changed_heat_capture(saved_flows):
    from summarize_cmfgen_ir_grid_impact import compare_saved_continuum_flows
    root,native=saved_flows
    heat=native/'NATIVE_HEAT_ROW.bin';heat.write_bytes(heat.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='봉인 해시'):
        compare_saved_continuum_flows(root,native)


def test_saved_flows_cli_needs_no_sources_or_bb_replay(saved_flows):
    root,native=saved_flows
    script=Path(__file__).resolve().parents[1]/'scripts/summarize_cmfgen_ir_grid_impact.py'
    result=subprocess.run([sys.executable,str(script),'--response-root',str(root),
        '--continuum-native-root',str(native)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['status']=='SAME_STATE_CONTINUUM_NUMBER_FLOWS'
