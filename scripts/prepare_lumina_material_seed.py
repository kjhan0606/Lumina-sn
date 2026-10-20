#!/usr/bin/env python3
"""봉인 물질70의 전 원소 재시작 입력을 같은 깊이점에서 내보낸다.

이 단계의 공간값은 CMFGEN 점값이다. 수송용 쉘 평균/경계가 아니며,
기존 원자 템플릿의 물질·격자·경계조건을 복사하지 않는다.
"""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from analyze_cmfgen_material_inventory import ELEMENTS, analyze, block, sha
from expand_atomic_data_cmfgen import load_cmfgen_model_spec, load_cmfgen_level_sl_map
from prepare_cmfgen_fixed_field_owner import point_records, record_bytes
from ptf_gated_metrics import read_dimensions, read_variables
from run_cmfgen_lambda_probe import control


def atomic_asset(name):
    return name in {
        'levels.csv','line_list.csv','atom_masses.csv','ionization_energies.csv',
        'zeta_ions.csv','zeta_temps.csv','zeta_data.npy','cmfgen_sigma_bf.bin',
        'cmfgen_bf_channels.bin','cmfgen_bf_channels_manifest.json',
        'ma_radrecomb_target.bin','coldata_cmfgen_manifest.csv',
        'feiii_col_zhang.bin','s2_col_cmfgen.csv','si2_col_cmfgen.csv',
        'siii_col_cmfgen.csv','siii_terms_col_cmfgen.csv','coiii_col_cmfgen.csv',
        'atomic_source_identity.json','atomic_vintage_manifest.csv',
        'cmfgen_phot_smoothing.so'} or (
            name.startswith('ige_col_') and name.endswith('_cmfgen.bin'))


def write_point_profiles(atomic, variables, state, fractions):
    nd=state.shape[1]
    with (atomic/'abundances.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['atomic_number',*range(nd)])
        for Z in ELEMENTS:
            writer.writerow([Z,*[format(x,'.17g') for x in fractions[Z]]])
    with (atomic/'element_number_densities.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['atomic_number','shell_id','N_element_cm3'])
        for Z,(name,_) in ELEMENTS.items():
            total=state[[v.index-1 for v in variables if v.species==name]].sum(axis=0)
            for depth,value in enumerate(total):
                writer.writerow([Z,depth,format(value,'.17g')])
    with (atomic/'electron_temperatures.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['shell_id','T_e'])
        for depth,value in enumerate(state[-1]*1e4):
            writer.writerow([depth,format(value,'.17g')])


def prepare(plan_path,plan_hash,out):
    if out.exists():
        raise ValueError('기존 물질 재시작 입력 보존')
    if sha(plan_path)!=plan_hash:
        raise ValueError('봉인 BRIDGE 계획 신원 불일치')
    plan=json.loads(plan_path.read_text())
    if (plan['purpose_id']!='LUMINA_STATE_SPECTRUM_BRIDGE_2026-10-20' or
            plan['initial_native_record']!=70 or
            plan['reference_radiation_material_record']!=70):
        raise ValueError('원래70/J71의 동일물질 대조만 허용')
    run=Path(plan['initial_material'])
    base=Path(plan['model_atom_template'])
    runtime=Path(plan['initial_runtime'])
    if sha(runtime)!=plan['initial_runtime_sha256']:
        raise ValueError('완료 물질70 실행 신원 변경')
    rt=json.loads(runtime.read_text())
    if (not rt['material_complete'] or rt['material_record']!=70 or
            rt.get('failure') or not rt.get('source_unchanged')):
        raise ValueError('원래70 완료/보존 증거 없음')
    nd,nt=read_dimensions(run)
    if min(point_records(run).values())<70:
        raise ValueError('미완료 인구 포인터')
    header,raw,state=record_bytes(run/'SCRTEMP',70,nd,nt)
    if hashlib.sha256(raw).hexdigest()!=plan['initial_record_storage_sha256']:
        raise ValueError('원래70 저장 인구 변경')
    radius=np.frombuffer(header,dtype='<f8')[:nd].copy()*1e10
    if not np.all(np.diff(radius)<0):
        raise ValueError('원래 깊이의 반경 순서가 아님')
    rho=block(run/'RVTJ','Mass Density (gm/cm^3)',nd)
    clump=block(run/'RVTJ','Clumping Factor',nd)
    if not np.array_equal(clump,np.ones(nd)):
        raise ValueError('현재 재시작의 체적 규약은 smooth model만 지원')
    vadat=(run/'VADAT').read_text()
    epoch=float(control(vadat,'SN_AGE').replace('D','E'))*86400
    luminosity=float(control(vadat,'LSTAR').replace('D','E'))*3.826e33
    if control(vadat,'IB_METH')!='DIFFUSION':
        raise ValueError('등록된 확산 경계가 아님')
    model=load_cmfgen_model_spec(run/'MODEL_SPEC',run/'atomic_links.txt')
    links=load_cmfgen_level_sl_map(run/'LEVEL_SL_STEQ_LINKS')
    levels=list(csv.DictReader((base/'levels.csv').open()))
    keys=set()
    for row in levels:
        key=(int(row['atomic_number']),int(row['ion_number'])+1)
        level=int(row['level_number'])+1
        identity=(*key,level)
        if key not in model or identity in keys:
            raise ValueError('중복/미등록 원자 준위')
        actual=links[model[key]['label'],level]
        if (actual['superlevel']-1!=int(row['super_level']) or
                actual['configuration']!=row['configuration']):
            raise ValueError('원래 FL/SL/준위 이름 불일치')
        keys.add(identity)
    expected={(Z,stage,l) for (Z,stage),m in model.items()
              for l in range(1,int(m['n_full'])+1)}
    if keys!=expected:
        raise ValueError('전 model-atom의 활성 FL 누락')
    variables=read_variables(run,nt-2)
    name_to_z={name:Z for Z,(name,_) in ELEMENTS.items()}
    columns=[(name_to_z[v.species],v.charge,-1 if v.terminal else v.superlevel-1)
             for v in variables]
    if len(set(columns))!=len(columns):
        raise ValueError('중복 SL/terminal 변수')
    fractions={Z:block(run/'SN_HYDRO_FOR_NEXT_MODEL',name+' mass fraction',nd)
               for Z,(_,name) in ELEMENTS.items()}
    if np.max(abs(sum(fractions.values())-1))>1e-7:
        raise ValueError('native 참질량분율 합 불일치; 정규화하지 않음')
    sources=[run/name for name in ('MODEL','MODEL_SPEC','LEVEL_SL_STEQ_LINKS',
              'VADAT','atomic_links.txt','RVTJ','SN_HYDRO_FOR_NEXT_MODEL')]
    sources += [plan_path,runtime,Path(__file__).resolve()]
    before={str(p):sha(p) for p in sources}
    assets=[p for p in base.iterdir() if atomic_asset(p.name) and p.is_file()]
    for path in assets:
        before[str(path)]=sha(path)
    out.mkdir();atomic=out/'atomic';atomic.mkdir()
    for path in assets:
        (atomic/path.name).symlink_to(path.resolve())
    write_point_profiles(atomic,variables,state,fractions)
    profiles=np.column_stack((radius,state[-1]*1e4,state[-2],rho))
    with (out/'native_material_nodes.bin').open('xb') as stream:
        stream.write(struct.pack('<8sIIId',b'LUMNOD01',nd,len(columns),len(levels),epoch))
        stream.write(np.asarray(columns,dtype='<i4').tobytes())
        stream.write(np.asarray(profiles,dtype='<f8').tobytes())
        stream.write(np.asarray(state[:-2].T,dtype='<f8').tobytes())
    with redirect_stdout(io.StringIO()):
        analyze(run,base,70,plan['initial_record_storage_sha256'],out/'inventory.json')
    _,after_raw,_=record_bytes(run/'SCRTEMP',70,nd,nt)
    if after_raw!=raw or any(sha(Path(p))!=h for p,h in before.items()):
        raise ValueError('준비 중 봉인 입력 변경')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    report=dict(purpose=plan['purpose_id'],plan_sha256=plan_hash,
        source_record=70,record_storage_sha256=plan['initial_record_storage_sha256'],
        material_points=nd,coordinate_role='CMFGEN outer-to-inner nodal values; NOT transfer-cell averages',
        epoch_seconds=epoch,full_levels=len(levels),bound_ions=len(model),
        population_variables=len(columns),terminal_variables=sum(v.terminal for v in variables),
        boundary_method='DIFFUSION',inner_radius_cm=float(radius[-1]),
        outer_radius_cm=float(radius[0]),inner_gas_temperature_K=float(state[-1,-1]*1e4),
        inner_luminosity_erg_s=luminosity,solar_luminosity_erg_s=3.826e33,
        number_density_policy='explicit fixed N_element; no rho/X/mass reconstruction',
        true_mass_fraction_policy='unchanged SN_HYDRO_FOR_NEXT_MODEL; no renormalization',
        volume_policy='smooth, clumping factor exactly1; transport remap NOT YET IMPLEMENTED',
        transfer_ready=False,RT_executions=0,native_executions=0,
        input_sha256=before,
        output_sha256={str(p.relative_to(out)):sha(p) for p in (
            atomic/'abundances.csv',atomic/'element_number_densities.csv',
            atomic/'electron_temperatures.csv',out/'native_material_nodes.bin',out/'inventory.json')},
        physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
    with (out/'material-seed.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('material_points','full_levels','bound_ions',
        'population_variables','terminal_variables','inner_radius_cm','inner_luminosity_erg_s',
        'transfer_ready')},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--plan-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();prepare(a.plan.resolve(),a.plan_sha256,a.out.resolve())
