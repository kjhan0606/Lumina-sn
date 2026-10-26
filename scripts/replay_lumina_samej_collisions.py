#!/usr/bin/env python3
"""완료된 동일J 계수로 CI/C3와 준위용해 충돌을 대조한다. 수송·계수 재계산 없음."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np

import cmfgen_model_continuum as B
from run_lumina_reference_response import parse_raw
from run_lumina_pseudo_candidate_overlay import parse_candidate
from run_lumina_pseudo_candidate_overlay import HEADER,RECORD
from analyze_lumina_samej_collision_budget import direct_ci


def predict_ci_heat(bf_path,occupation_path,levels_path,reference):
    """저장 FL 입력으로 직접 CI/C3의 문턱 전자 열량을 독립 계산한다."""
    levels=list(csv.DictReader(levels_path.open()))
    blob=bf_path.read_bytes()
    tag,nl,nr,ns,node,temp,ne=HEADER.unpack_from(blob)
    if tag!=b'LMBFFUL1' or nl!=len(levels):
        raise ValueError('CI 열량 예측 BF 주소 오류')
    records=np.frombuffer(blob,dtype=RECORD,offset=HEADER.size)
    w=np.frombuffer(occupation_path.read_bytes(),dtype='<f8',offset=16).reshape(nl,ns)[:,node]
    groups={}
    for i,r in enumerate(levels):
        if int(r['atomic_number']) in (14,16):
            groups.setdefault((int(r['atomic_number']),int(r['ion_number']),int(r['super_level'])),[]).append(i)
    fraction=np.zeros(nl)
    for ids in groups.values():
        energy=np.array([float(levels[i]['energy_eV']) for i in ids])
        weight=np.array([float(levels[i]['g']) for i in ids])*w[ids]*np.exp(-(energy-energy.min())*B.EV/(B.KB*temp))
        fraction[ids]=weight/weight.sum()
    net={14:np.longdouble(0.),16:np.longdouble(0.)};gross=dict(net)
    for r in records:
        lo,target,zt,qt,grid=map(int,r['ids'])
        row=levels[lo];z=int(row['atomic_number']);q=int(row['ion_number'])
        if z not in net or (target>=0 and grid!=0):
            continue
        lower=(z,q,int(row['super_level']))
        upper=((z,q+1,int(levels[target]['super_level'])) if target>=0 else (zt,qt,-1))
        prob,edge,sigma,saha=map(float,r['values'][:4])
        rate=prob*direct_ci(temp,ne,edge,sigma,q)
        forward=np.longdouble(rate*fraction[lo])*reference[lower]['original']
        reverse=np.longdouble(rate*saha)*reference[upper]['original']
        energy=B.H*edge
        net[z]+=(forward-reverse)*energy;gross[z]+=(forward+reverse)*energy
    return net,gross


def predict_dissolution(bf_path,collision_path,occupation_path,levels_path,reference):
    """C 조립과 독립적으로 저장된 FL 입력에서 용해 순률/문턱 열량을 예측한다."""
    levels = list(csv.DictReader(levels_path.open()))
    z = np.array([int(r['atomic_number']) for r in levels])
    charge = np.array([int(r['ion_number']) for r in levels])
    sl = np.array([int(r['super_level']) for r in levels])
    g = np.array([float(r['g']) for r in levels])
    energy = np.array([float(r['energy_eV']) for r in levels])
    blob = bf_path.read_bytes()
    tag,nl,nr,ns,node,temp,ne = HEADER.unpack_from(blob)
    if tag != b'LMBFFUL1' or nl != len(levels) or node != 49:
        raise ValueError('독립 용해 예측 BF 주소 오류')
    bf = np.frombuffer(blob,dtype=RECORD,offset=HEADER.size)
    oblob = occupation_path.read_bytes()
    if struct.unpack_from('<8sII',oblob) != (b'LMATOCC1',nl,ns):
        raise ValueError('독립 용해 예측 occupation 주소 오류')
    w = np.frombuffer(oblob,dtype='<f8',offset=16).reshape(nl,ns)[:,node]
    groups = {}
    for i in np.flatnonzero(np.isin(z,[14,16])):
        groups.setdefault((int(z[i]),int(charge[i]),int(sl[i])),[]).append(i)
    fraction = np.zeros(nl)
    for indices in groups.values():
        weight = g[indices]*w[indices]*np.exp(-(energy[indices]-energy[indices].min())/(8.617333262145e-5*temp))
        fraction[indices] = weight/weight.sum()
    primary = {}
    for r in bf:
        lo,target,_,_,grid = map(int,r['ids'])
        if z[lo] not in (14,16) or grid != 0:
            continue
        if lo in primary:
            raise ValueError('primary PHOT 중복')
        upper = ((int(z[target]),int(charge[target]),int(sl[target])) if target >= 0
                 else (int(r['ids'][2]),int(r['ids'][3]),-1))
        primary[lo] = upper,float(r['values'][3]),float(r['values'][1])
    cb = collision_path.read_bytes()
    if HEADER.unpack_from(cb) != (b'LMCBPTS1',nl,ns,node,157371,temp,ne):
        raise ValueError('독립 용해 예측 충돌 입력 오류')
    pairs = np.frombuffer(cb,dtype=[('lo','<i4'),('up','<i4'),('omega','<f8')],offset=HEADER.size)
    net = {key:np.longdouble(0.) for key in reference}
    gross = dict(net)
    heat = {14:np.longdouble(0.),16:np.longdouble(0.)}
    counts = {14:[0,0],16:[0,0]}
    scale = 8.63e-6*ne/np.sqrt(temp)
    for r in pairs:
        lo,up = int(r['lo']),int(r['up'])
        lower = int(z[lo]),int(charge[lo]),int(sl[lo])
        upper,saha,edge = primary[lo]
        lost = 1.-w[up]/w[lo]
        if lost < 0:
            raise ValueError('음의 occupation 용해확률')
        rate = scale*r['omega']*np.exp(-(energy[up]-energy[lo])/(8.617333262145e-5*temp))/g[lo]*lost
        forward = np.longdouble(fraction[lo]*rate)*reference[lower]['original']
        reverse = np.longdouble(saha*rate)*reference[upper]['original']
        value = forward-reverse
        net[lower] -= value;net[upper] += value
        gross[lower] += abs(forward)+abs(reverse);gross[upper] += abs(forward)+abs(reverse)
        heat[int(z[lo])] += value*(6.62607015e-27*edge)
        if rate > 0:
            counts[int(z[lo])][0] += 1
            counts[int(z[lo])][1] += int(sl[lo] == sl[up])
    return net,gross,heat,counts


def compare_rows(rows, reference):
    result = []
    for key, value in rows.items():
        native = reference[key]['native_raw']
        delta = value['total'] - native
        flow = value['absolute_flow']
        result.append(dict(Z=key[0], charge=key[1], superlevel=key[2], C=value,
                           native_raw=native, difference=delta,
                           difference_over_C_absolute_flow=abs(delta)/flow if flow else None))
    return result


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '1' or
            os.environ.get('SLURM_MEM_PER_NODE') != '4096'):
        raise ValueError('CPU1·4GiB Slurm 전용')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('기존 재생 결과 보존')
    source = args.source.resolve()
    report_path = source/'run/result.json'
    report = json.loads(report_path.read_text())
    if (report['state'] != 'COMPLETE_SAME_J_COMPARISON' or
            report['input_material'] != 70 or report['native_node'] != 49 or
            not report['source_unchanged']):
        raise ValueError('완료 동일J 입력 물질 신원 오류')
    profile = report['cases']['native_discrete']['profile']
    reference = {(r['Z'],r['charge'],r['superlevel']):r for r in profile}
    if len(reference) != 518 or any(r['C']['bf_col'] != 0. for r in profile):
        raise ValueError('518행의 CI 비활성 대조가 아님')
    original = {key:r['original'] for key,r in reference.items()}
    project = Path(__file__).resolve().parents[1]
    base_path = Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    base = json.loads(base_path.read_text())
    config_path = Path(base['config'])
    config = json.loads(config_path.read_text())
    old = Path('/gpfs/kjhan/cmfgen_runs/lumina_direct_slprofile_20261026-UUiwsU')
    collision = Path('/gpfs/kjhan/cmfgen_runs/lumina_fullpair8_20261026-qXSzbE/fullpair.bin')
    projection = args.projection.resolve() if args.projection else source/'build/model_projection'
    bf = source/'run/native_discrete-bf.bin'
    bb = source/'run/native_discrete-bb.bin'
    occupation = old/'model-occupation-input.bin'
    parameters = old/'model-occupation-parameters.bin'
    sealed = dict(report['cases']['native_discrete']['scalar_files_sha256'])
    for path in (base_path,occupation,parameters,collision):
        sealed[str(path)] = report['input_sha256'][str(path)]
    sealed[str(projection)] = B.sha(projection) if args.projection else report['input_sha256'][str(projection)]
    if args.projection:
        sealed[str(projection.with_name('source-before.sha256'))] = B.sha(projection.with_name('source-before.sha256'))
    levels_path = Path(config['seed'])/'atomic/levels.csv'
    sealed[str(levels_path)] = B.sha(levels_path)
    if args.ci_baseline:
        sealed[str(args.ci_baseline.resolve())] = B.sha(args.ci_baseline)
        ci_baseline = json.loads(args.ci_baseline.read_text())
        if ci_baseline['source_sha256'] != B.sha(report_path) or not ci_baseline['control_exactly_reproduced']:
            raise ValueError('CI1 대조 계보 오류')
    elif args.with_dissolution:
        raise ValueError('용해 비교에는 완료된 CI1 대조 필요')
    sealed.update(base['input_sha256'])
    sealed[str(report_path)] = B.sha(report_path)
    for path in (Path(__file__),project/'scripts/run_lumina_reference_response.py',
                 project/'scripts/analyze_lumina_samej_collision_budget.py',
                 project/'scripts/run_lumina_pseudo_candidate_overlay.py',
                 project/'data/atomic/topion_levels.csv',project/'data/atomic/ionization_reference.csv'):
        sealed[str(path)] = B.sha(path)
    if any(B.sha(Path(p)) != h for p,h in sealed.items()):
        raise ValueError('봉인된 재생 입력 변경')
    out.mkdir()
    (out/'inputs.json').write_text(json.dumps(dict(input_sha256=sealed,
        scope=__doc__,resources='CPU1/4GiB/10min/retry0',
        control='LUMINA_NLTE_BF_COLLISION=0',experiment='LUMINA_NLTE_BF_COLLISION=1',
        unchanged='material70/J70/BF/BB/occupation/omega/solver'),indent=2)+'\n')
    cases = {}
    ci_heat,ci_heat_gross=predict_ci_heat(bf,occupation,levels_path,reference)
    modes = [(0,0),(1,0)]+([(1,1)] if args.with_dissolution else [])
    for enabled,dissolution in modes:
        name = 'CI'+str(enabled)+('_D1' if dissolution else '')
        exported = out/(name+'-imported.bin')
        command = [str(projection),str(Path(config['seed'])/'atomic'),'90',
                   str(Path(config['seed'])/'native_material_nodes.bin'),str(exported),
                   str(occupation),str(parameters),str(bf),str(bb),str(collision)]
        env = dict(os.environ,LUMINA_ARTIS_PARITY='0',LUMINA_NLTE_BF_COLLISION=str(enabled),
            LUMINA_MODEL_COLLISION_DISSOLUTION=str(dissolution),
            LUMINA_TOPION_LEVELS_FILE=str(project/'data/atomic/topion_levels.csv'),
            LUMINA_IONIZATION_REFERENCE_FILE=str(project/'data/atomic/ionization_reference.csv'))
        child = subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
        (out/(name+'.stdout')).write_text(child.stdout)
        (out/(name+'.stderr')).write_text(child.stderr)
        if child.returncode or B.sha(exported) != sealed[config['native_full_population']]:
            raise ValueError(name+' 재생 실행/원 인구 보존 오류')
        candidate,residual = parse_candidate(child.stdout,original,49)
        rows = parse_raw(child.stdout,original)
        heat_matches=re.findall(r'\[MODEL-SE\]\[BF-COLLISION-HEAT\] Z=(\d+) native_node=49 cooling_erg_cm3_s=(\S+)',child.stdout)
        if heat_matches:
            actual_heat={int(z):float(value) for z,value in heat_matches if int(z) in (14,16)}
            if len(actual_heat)!=2:
                raise ValueError('실제 CI/C3 열량 증거 누락')
            for z,value in actual_heat.items():
                if (not np.isfinite(value) or
                    abs(value-float(ci_heat[z])*enabled)>1.e-10*float(ci_heat_gross[z])*enabled):
                    raise ValueError('독립 CI/C3 문턱 열량 불일치')
        elif args.require_ci_heat:
            raise ValueError('CI/C3 열량 소비가 구현되지 않은 바이너리')
        if not enabled and any(rows[k] != reference[k]['C'] for k in rows):
            raise ValueError('CI0 대조가 저장518행을 재현하지 못함')
        if args.ci_baseline and enabled and not dissolution:
            previous = {(r['Z'],r['charge'],r['superlevel']):r['C']
                        for r in ci_baseline['cases']['CI1']['profile']}
            if rows != previous:
                raise ValueError('CI1 대조가 이전518행을 재현하지 못함')
        cases[name] = dict(stationary_residual=residual,profile=compare_rows(rows,reference),
            candidates=[dict(Z=k[0],charge=k[1],superlevel=k[2],population=v)
                        for k,v in candidate.items()],command=command,
            collision_switch=enabled,dissolution_enabled=bool(dissolution),
            CI_threshold_heat_verified=bool(heat_matches),
            CI_threshold_heat_prediction={str(z):float(value)*enabled for z,value in ci_heat.items()},
            nonzero_bf_col_rows=sum(r['bf_col'] != 0. for r in rows.values()))
        if dissolution:
            net,gross,heat,counts = predict_dissolution(bf,collision,occupation,levels_path,reference)
            direct_rows = re.findall(r'\[MODEL-SE\]\[DISSOLUTION-ROW\] Z=(\d+) stage0=(\d+) super0=(-?\d+) total=(\S+) absolute_flow=(\S+)',child.stdout)
            direct = {(int(z),int(q),int(sl)):(float(value),float(flow))
                      for z,q,sl,value,flow in direct_rows}
            if len(direct_rows) != 518 or set(direct) != set(reference):
                raise ValueError('용해 추가항 자체의518행 누락')
            control = {(r['Z'],r['charge'],r['superlevel']):r['C'] for r in cases['CI1']['profile']}
            checks = []
            for key,value in rows.items():
                expected = float(net[key])
                observed = value['total']-control[key]['total']
                # 큰 BB 항과 합산된 double 출력 두 개를 빼는 산술 폭이다.
                rounding = 128*np.finfo(float).eps*(value['absolute_flow']+control[key]['absolute_flow'])
                error = abs(observed-expected)
                if error > rounding+1.e-10*float(gross[key]):
                    raise ValueError('독립 용해 예측과 실제 SE 증가분 불일치')
                isolated,isolated_flow = direct[key]
                if (not np.isfinite([isolated,isolated_flow]).all() or isolated_flow < 0. or
                    abs(isolated-expected) > 1.e-10*float(gross[key]) or
                    abs(isolated_flow-float(gross[key])) > 1.e-10*float(gross[key])):
                    raise ValueError('독립 용해 예측과 추가항 자체 불일치')
                for unchanged in ('bf_rad','bf_col','bb_rad'):
                    if value[unchanged] != control[key][unchanged]:
                        raise ValueError('용해 추가가 기존 다른 항을 바꿈')
                checks.append(dict(Z=key[0],charge=key[1],superlevel=key[2],
                    expected_raw_change=expected,observed_raw_change=observed,
                    isolated_actual_change=isolated,isolated_absolute_flow=isolated_flow,
                    isolated_error=abs(isolated-expected),
                    subtraction_rounding_bound=rounding,error=error))
            logged = re.findall(r'\[MODEL-COLL-BB\]\[DISSOLUTION\] Z=(\d+) pairs=(\d+) intra=(\d+) heat_erg_cm3_s=(\S+)',child.stdout)
            if len(logged) != 2:
                raise ValueError('용해 열량/전쌍 증거 누락')
            for zc,pairs,intra,actual_heat in logged:
                zn = int(zc)
                if ([int(pairs),int(intra)] != counts[zn] or
                    not np.isclose(float(actual_heat),float(heat[zn]),rtol=1.e-9,atol=0.)):
                    raise ValueError('독립 용해 문턱 열량/쌍 수 불일치')
            cases[name]['independent_change_checks'] = checks
            cases[name]['independent_threshold_heat'] = {str(k):float(v) for k,v in heat.items()}
            cases[name]['dissolution_pair_counts'] = counts
    if any(B.sha(Path(p)) != h for p,h in sealed.items()):
        raise ValueError('재생 중 원본 입력 변경')
    result = dict(schema='lumina-sameJ-CI-switch-v1',state='COMPLETE_PAIRED_REPLAY',
        source_result=str(report_path),source_sha256=sealed[str(report_path)],
        cases=cases,input_sha256=sealed,control_exactly_reproduced=True,
        source_unchanged=True,new_BF=0,new_BB=0,new_native=0,new_transport=0,
        public_material_updates=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'],nonzero_bf_col_rows=cases['CI1']['nonzero_bf_col_rows'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--projection',type=Path)
    parser.add_argument('--ci-baseline',type=Path)
    parser.add_argument('--with-dissolution',action='store_true')
    parser.add_argument('--require-ci-heat',action='store_true')
    run(parser.parse_args())
