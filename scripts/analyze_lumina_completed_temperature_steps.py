"""같은 J의 기점·두 수용 보폭에서 Si/S BF 장부와 T/ne 영향을 읽기 전용 비교."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_lumina_new_field_response_eval import B, checked_input, original_reference_path


def analyze(ordinary, box):
    paths = [Path(p).resolve() for p in (ordinary, box)]
    reports = [json.loads(p.read_text()) for p in paths]
    inputs = {str(p): B.sha(p) for p in paths}
    a, b = reports
    if (any(r.get('state') != 'COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE' for r in reports)
            or any(a[k] != b[k] for k in ('field', 'source_identity', 'thermal_initial_response'))):
        raise ValueError('같은 장/물질/기점의 완료 비교 필요')
    entries = [a['thermal_initial_response'], a['selected_response'], b['selected_response'],
               *a['thermal_directions'][0]['probes']]
    values = []
    for entry in entries:
        pair = []
        for name, key in (('response.npz', 'response_sha256'), ('balance.npz', 'balance_sha256')):
            p = Path(entry['root'])/name
            if B.sha(p) != entry[key]: raise ValueError('완료 선택/섭동 변경')
            inputs[str(p)] = entry[key]
            with np.load(p, allow_pickle=False) as data: pair.append({k:data[k] for k in data.files})
        values.append(pair)
    prep = json.loads(checked_input(a, 'preparation.json').read_text())
    refpath = original_reference_path(prep, a)
    # read_material은 정확한 SL 주소를 포함하므로 원 실행과 같은 원자모형으로 읽는다.
    import csv
    levels = checked_input(a, 'levels.csv')
    rows = list(csv.DictReader(levels.open()))
    models = B.atomic.load_cmfgen_model_spec(checked_input(a, 'MODEL_SPEC'), checked_input(a, 'atomic_links.txt'))
    reference = B.read_material(refpath, rows, models)
    inputs[str(refpath)] = B.sha(refpath)
    c = values[0][0]['columns']; records = []
    if not np.array_equal(c, reference['columns']): raise ValueError('기준 원자주소 불일치')
    for data, heat in values:
        if not np.array_equal(c, data['columns']): raise ValueError('응답 원자주소 불일치')
    for z in (14, 16):
        for stage in (2, 3):
            ion = (c[:,0]==z)&(c[:,1]==stage-1); parent = (c[:,0]==z)&(c[:,1]==stage)
            species = c[:,0]==z
            fractions = [v[0]['population'][ion].sum(axis=0)/v[0]['population'][species].sum(axis=0) for v in values]
            reference_fraction = reference['superlevel'][ion].sum(axis=0)/reference['superlevel'][species].sum(axis=0)
            dlogT = (np.log(fractions[3])-np.log(fractions[4]))/2.e-4
            dlogne = (np.log(fractions[5])-np.log(fractions[6]))/2.e-4
            for n in range(len(reference_fraction)):
                row = dict(node=n,Z=z,stage=stage,radius_cm=float(reference['physical'][n,0]),
                           CMFGEN_fraction=float(reference_fraction[n]),
                           base_dlog_fraction_dlogT=float(dlogT[n]),base_dlog_fraction_dlogne=float(dlogne[n]))
                for label, (data, heat), frac in zip(('base','ordinary','box'),values[:3],fractions[:3]):
                    idx=np.flatnonzero((heat['bf_ion_Z_stage']==[z,stage]).all(axis=1))
                    if len(idx)!=1:raise ValueError('BF 이온 주소 결손')
                    flows=heat['bf_ion_number_flows'][idx[0],:,n]
                    ni=data['population'][ion,n].sum(); np_=data['population'][parent,n].sum()
                    row.update({label+'_'+k:float(v) for k,v in dict(
                        T=data['temperature'][n],ne=data['ne'][n],fraction=frac[n],
                        gamma=flows[0]/ni,alpha=(flows[1]+flows[2])/(data['ne'][n]*np_),
                        BF_net_over_photo=flows[3]/flows[0]).items()})
                row['box_predicted_dlog_fraction_T']=dlogT[n]*np.log(row['box_T']/row['base_T'])
                row['box_predicted_dlog_fraction_ne']=dlogne[n]*np.log(row['box_ne']/row['base_ne'])
                row['box_actual_dlog_fraction']=np.log(row['box_fraction']/row['base_fraction'])
                records.append(row)
    for p,h in inputs.items():
        if B.sha(Path(p))!=h:raise ValueError('판독 중 입력 변경')
    return dict(scope='SAME_J_COMPLETED_TEMPERATURE_STEPS_NOT_JOINT_CONVERGENCE',
        reference='CMFGEN70 fixed-T nonconverged diagnostic',input_sha256=inputs,
        interpretation='BF gamma/alpha are population-dependent ledger quantities; derivatives are local full-SE probes, not a finite-step proof',
        new_SE=0,new_BF=0,new_transport=0,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--box',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(analyze(args.ordinary,args.box),ensure_ascii=False,allow_nan=False))
