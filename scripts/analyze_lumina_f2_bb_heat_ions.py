#!/usr/bin/env python3
"""저장 F2의 전체 원자선 열흐름을 이온과 전이별로 재합산한다."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cmfgen_model_continuum as B
from run_lumina_full_material_response import BB_RESULT

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('기존 결과 보존')
    root=Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    report_path=root/'result.json'
    if B.sha(report_path)!='b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':raise ValueError('F2 결과 변경')
    r=json.loads(report_path.read_text());response=Path(r['selected_response']['root'])/'response.npz'
    balance=Path(r['selected_response']['root'])/'balance.npz';jbar=root/'jbar.npy';roster=BB_RESULT.parent/'line-roster.npz'
    if B.sha(BB_RESULT)!='b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9':raise ValueError('선 보고서 변경')
    metadata=json.loads(BB_RESULT.read_text());levels=[Path(x) for x in r['input_sha256'] if x.endswith('/atomic/levels.csv')]
    if len(levels)!=1:raise ValueError('원자 준위 주소 오류')
    for path,sha in ((response,r['selected_response']['response_sha256']),(balance,r['selected_response']['balance_sha256']),
      (jbar,r['output_sha256']['jbar.npy']),(roster,metadata['output_sha256']['line-roster.npz']),
      (levels[0],r['input_sha256'][str(levels[0])])):
        if B.sha(path)!=sha:raise ValueError('입력 변경: '+path.name)
    with np.load(response) as f:population=f['population'];fraction=f['fraction'];columns=f['columns']
    with np.load(balance) as f:stored=f['bb'];net_total=f['physical_stored_net']
    with np.load(roster) as f:lines={k:f[k] for k in f.files}
    rows=list(csv.DictReader(levels[0].open()));index={tuple(map(int,c)):i for i,c in enumerate(columns)}
    group=np.array([index[(int(v['atomic_number']),int(v['ion_number']),int(v['super_level']))] for v in rows])
    z=np.array([int(v['atomic_number']) for v in rows]);stage=np.array([int(v['ion_number'])+1 for v in rows])
    lo=lines['lower'];up=lines['upper'];nu=lines['nu0'];A=lines['A'];ratio=lines['g_ratio']
    if not np.array_equal(z[lo],z[up]) or not np.array_equal(stage[lo],stage[up]):raise ValueError('선 이온 경계 오류')
    values=np.load(jbar,mmap_mode='r');bul=A*B.C**2/(2*B.H*nu**3);energy=B.H*nu;output=[]
    for node in (30,33):
        J=np.asarray(values[:,node]);lower=np.asarray(population[group[lo],node]*fraction[lo,node],dtype=np.longdouble)
        upper=np.asarray(population[group[up],node]*fraction[up,node],dtype=np.longdouble)
        forward=lower*(bul/ratio)*J*energy;reverse=upper*(A+bul*J)*energy
        totals=np.array([forward.sum(),reverse.sum()],dtype='f8')
        np.testing.assert_allclose(totals,stored[:,node],rtol=2e-13,atol=0)
        groups=[]
        for Z,ion in sorted(set(zip(z[lo],stage[lo]))):
            choose=(z[lo]==Z)&(stage[lo]==ion)
            heating=forward[choose].sum();cooling=reverse[choose].sum();net=heating-cooling
            groups.append(dict(Z=int(Z),stage=int(ion),heating=float(heating),cooling=float(cooling),net=float(net),
                               absolute_line_net=float(np.abs(forward[choose]-reverse[choose]).sum())))
        net=forward-reverse;top=[]
        for i in np.argsort(np.abs(net))[-12:][::-1]:
            l,u=int(lo[i]),int(up[i]);top.append(dict(Z=int(z[l]),stage=int(stage[l]),
                wavelength_A=float(B.C*1e8/nu[i]),lower=rows[l]['configuration'],upper=rows[u]['configuration'],
                heating=float(forward[i]),cooling=float(reverse[i]),net=float(net[i])))
        output.append(dict(node=node,heating=float(totals[0]),cooling=float(totals[1]),
            total_net=float(forward.sum()-reverse.sum()),overall_thermal_net=float(net_total[node]),
            top_ions=sorted(groups,key=lambda x:abs(x['net']),reverse=True)[:10],groups=groups,top_lines=top))
    values._mmap.close();out.mkdir();result=dict(state='COMPLETE_F2_LINE_HEAT_ION_DECOMPOSITION',nodes=output,
        scope='physical BB photon heating/cooling at fixed F2 state; net line energy is not an independent thermal closure',
        new_transport=0,new_SE=0,new_RE=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'],nodes=[{k:v for k,v in x.items() if k!='groups'} for x in output])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
