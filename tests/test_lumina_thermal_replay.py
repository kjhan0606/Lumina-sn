"""열보정 재개 입력의 신원·실제 물리 코드·완료 배열·호출 목록 시험."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_thermal_replay as R
from test_lumina_fixed_field_thermal import case


@pytest.fixture
def preserved(tmp_path):
    root=tmp_path/'run';root.mkdir();(tmp_path/'build').mkdir()
    libraries=[tmp_path/'build'/name for name in ('bridge.so','lines.so')]
    for path in libraries:path.write_bytes(path.name.encode())
    model=tmp_path/'MODEL';model.write_bytes(b'model')
    columns=np.array([[14,1,0],[14,2,0]])
    state=dict(columns=columns,superlevel=np.ones((2,2)))
    entries=[]
    for i in range(3):
        data,heat,_=case(np.array([2.04,3.06])*np.exp(i*1.e-4),[5.,7.])
        data['columns']=columns
        output=root/('evaluation-'+str(i+1));output.mkdir()
        np.savez(output/'response.npz',**data);np.savez(output/'balance.npz',**heat)
        entries.append(dict(root=str(output),response_sha256=R.B.sha(output/'response.npz'),
            balance_sha256=R.B.sha(output/'balance.npz'),fixed_field_thermal_callback=i>0))
    implementation={'lumina_coupled_material.py':'same'}
    inputs={str(p):R.B.sha(p) for p in [*libraries,model]}
    report=dict(state='INCOMPLETE_PRESERVED',field={'J':'F2'},source_identity={'material':'accepted'},
        thermal_implementation_sha256=implementation.copy(),input_sha256=inputs.copy(),
        thermal_initial_response=entries[0],thermal_evaluation_order=entries[1:],charge_history=[{'accepted':True}])
    args=dict(field=report['field'].copy(),source_identity=report['source_identity'].copy(),
        implementation=implementation,state=state,library=libraries[0],line_library=libraries[1],inputs=inputs.copy())
    return root,report,args


def load(preserved):
    root,report,args=preserved;path=root/'result.json';path.write_text(json.dumps(report))
    return R.load(path,R.B.sha(path),**args)


def test_completed_values_are_loaded_without_changing_source_files(preserved):
    root,report,args=preserved
    paths=[Path(e['root'])/name for e in [report['thermal_initial_response'],*report['thermal_evaluation_order']]
           for name in ('response.npz','balance.npz')]
    before={str(p):R.B.sha(p) for p in paths}
    result=load(preserved)
    assert len(result['replay'])==2 and result['first'][2]==report['thermal_initial_response']
    assert result['charge_history']==[{'accepted':True}]
    assert result['provenance']['new_material_evaluations']==0
    for path,digest in before.items():
        assert R.B.sha(Path(path))==digest and args['inputs'][path]==digest


@pytest.mark.parametrize('fault',['live','field','material','implementation','library','input',
    'response','heat','nonthermal','empty','columns','nuclei','pin'])
def test_wrong_or_corrupted_replay_is_never_used(preserved,fault):
    root,report,args=preserved
    if fault=='live':report['state']='STARTED'
    elif fault=='field':args['field']={'J':'old'}
    elif fault=='material':args['source_identity']={'material':'old'}
    elif fault=='implementation':args['implementation']={'lumina_coupled_material.py':'changed'}
    elif fault=='library':args['library'].write_bytes(b'changed')
    elif fault=='input':(root.parent/'MODEL').write_bytes(b'changed')
    elif fault in ('response','heat'):
        name='response.npz' if fault=='response' else 'balance.npz'
        (Path(report['thermal_evaluation_order'][0]['root'])/name).write_bytes(b'changed')
    elif fault=='nonthermal':report['thermal_evaluation_order'][0]['fixed_field_thermal_callback']=False
    elif fault=='empty':report['thermal_evaluation_order']=[]
    elif fault=='columns':args['state']['columns']=args['state']['columns'].copy();args['state']['columns'][0,0]=16
    elif fault=='nuclei':args['state']['superlevel']*=1.01
    else:
        load(preserved)
        with pytest.raises(ValueError,match='보고서 변경'):
            R.load(root/'result.json','0'*64,**args)
        return
    with pytest.raises(ValueError):load(preserved)
