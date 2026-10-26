#!/usr/bin/env python3
"""원70/J70·node49에서 Si/S C 연산자와 교정 native 입력 SE를 직접 대조한다.

새 수송/native/공개 물질 갱신은 없다. 다른 원소의 보존된 율은 Si/S 해법에
쓰지 않으며 이 혼합 컨테이너를 whole-model 동일장 입력으로 인증하지 않는다.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native90_rates as R
import lumina_native_columns as N
from compare_native_prrr_common_grid import continuum_mapping
from lumina_model_point_rates import ModelPointBoundRates
from lumina_model_point_lines import ModelPointLines
from lumina_model_point_reference import ReferencePointRates,reference_line_rates
import lumina_native_line_quadrature as L
from prepare_cmfgen_fixed_field_owner import record_bytes
from ptf_gated_metrics import read_dimensions,read_variables
from run_cmfgen_formal_primary import fields
from run_cmfgen_native_se_observation import read_witness
from run_lumina_pseudo_candidate_overlay import HEADER,RECORD,parse_candidate

LINE_HEADER = struct.Struct('<8sIIII')
LINE_RECORD = np.dtype([('ids','<i4',(2,)),('values','<f8',(4,))])
RAW = re.compile(r'^\[MODEL-SE\]\[INPUT-RESIDUAL-ROW\] Z=(\d+) stage0=(\d+) '
                 r'super0=(-?\d+) bf_rad=(\S+) bf_col=(\S+) bb_rad=(\S+) bb_col=(\S+) '
                 r'total=(\S+) absolute_flow=(\S+)$',re.M)


def parse_raw(text,original):
    result = {}
    for z,stage,sl,*values in RAW.findall(text):
        key = int(z),int(stage),int(sl)
        numeric = list(map(float,values))
        if key in result or key not in original or not np.isfinite(numeric).all() or numeric[-1] < 0:
            raise ValueError('입력 SE 잔차의 원자 주소/값 오류')
        result[key] = dict(zip(('bf_rad','bf_col','bb_rad','bb_col','total','absolute_flow'),numeric))
    if set(result) != set(original):
        raise ValueError('입력 SE 잔차518행 누락')
    return result


def check_bf_records(blob,point,nl,ns):
    tag,nlevel,nr,npoint,node,t,ne = HEADER.unpack_from(blob)
    if ((tag,nlevel,nr,npoint,node) != (b'LMBFFUL1',nl,len(point.routes),ns,point.node) or
            t != point.temperature or ne != point.ne or
            len(blob) != HEADER.size+nr*RECORD.itemsize):
        raise ValueError('기존 BF 컨테이너 물질/원래점 오류')
    records = np.frombuffer(blob,dtype=RECORD,offset=HEADER.size).copy()
    for i,route in enumerate(point.routes):
        expected_ids = [route.lower,route.target,route.Z if route.target < 0 else 0,
                        route.stage if route.target < 0 else 0,route.phot_grid]
        expected_values = [route.probability,route.threshold_Hz,point.sigma_edge[i],point.saha[i]]
        if (not np.array_equal(records['ids'][i],expected_ids) or
                not np.array_equal(records['values'][i,:4],expected_values)):
            raise ValueError('기존 BF 경로/상위 정규화가 원70과 다름')
    return records


def reuse_bound(manifest_path,out,selected,n_routes):
    manifest_path = Path(manifest_path)
    seal = json.loads(manifest_path.read_text())
    if (seal['schema'] != 'lumina-completed-sameJ-bound-recovery-v1' or
            seal['material_record'] != 70 or seal['native_node'] != 49 or
            seal['completed_selected_routes'] != int(selected.sum()) or
            not re.fullmatch(r'[0-9a-f]{7,40}',seal['producer_commit'])):
        raise ValueError('완료 BF 복구의 상태/코드/범위 오류')
    root = Path(seal['directory'])
    if B.sha(root/'inputs.json') != seal['producer_inputs_sha256']:
        raise ValueError('완료 BF의 원 시작 manifest 변경')
    before = json.loads((root/'inputs.json').read_text())
    project = Path(__file__).resolve().parents[1]
    archived = []
    for name,digest in before['input_sha256'].items():
        path = Path(name)
        if B.sha(path) == digest:
            continue
        relative = path.relative_to(project)
        blob = subprocess.run(['git','show',seal['producer_commit']+':'+str(relative)],
                              cwd=project,capture_output=True,check=True).stdout
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('복구 BF 생산자의 정확한 코드 아카이브 불일치')
        archived.append(str(relative))
    for name,digest in seal['artifact_sha256'].items():
        if B.sha(root/name) != digest:
            raise ValueError('완료 BF 배열 변경')
    bound = np.load(root/'bound-unit-rates.npy',allow_pickle=False)
    total = np.load(root/'total-unit-rates.npy',allow_pickle=False)
    done = np.load(root/'selected-routes-done.npy',allow_pickle=False)
    if (bound.shape != (n_routes,6) or total.shape != bound.shape or
            not np.array_equal(done,selected) or not np.isfinite(bound[selected]).all() or
            not np.isfinite(total[selected]).all() or np.any(total[selected,:3] < 0)):
        raise ValueError('완료 BF 배열의 선택 범위/수치 손상')
    np.save(out/'bound-unit-rates.npy',bound)
    np.save(out/'total-unit-rates.npy',total)
    np.save(out/'selected-routes-done.npy',done)
    return bound,total,dict(manifest=str(manifest_path),manifest_sha256=B.sha(manifest_path),
        producer_commit=seal['producer_commit'],archived_code=archived,numerical_BF_reexecution=0)


def run(args):
    start = time.monotonic()
    if os.environ.get('SLURM_CPUS_PER_TASK') != '1' or os.environ.get('SLURM_MEM_PER_NODE') != '4096':
        raise ValueError('CPU1·4GiB Slurm 전용')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('기존 동일J 결과 보존')
    out.mkdir()
    paths = {name:Path(getattr(args,name)).resolve() for name in
             ('base','baseline','prediction','rate_contract','native_root','collision','point_library','lines_library','projection')}
    base = json.loads(paths['base'].read_text())
    baseline = json.loads(paths['baseline'].read_text())
    prediction = json.loads(paths['prediction'].read_text())
    c = json.loads(Path(base['config']).read_text())
    plan = json.loads(Path(c['plan']).read_text())
    old = paths['baseline'].parent
    native = paths['native_root']/'run/material'
    native_inputs = json.loads((native.parent/'seq_logs/verbose-se-inputs.json').read_text())
    runtime = json.loads((native.parent/'seq_logs/verbose-se-runtime.json').read_text())
    legacy_path = Path(prediction['legacy_native_identity']['inputs'])
    legacy_inputs = json.loads(legacy_path.read_text())
    if (not runtime.get('completed_corrected_step') or baseline['input_identity'] != base['source_identity'] or
            baseline['full_bound_test']['native_node'] != 49 or prediction['node'] != 49 or
            prediction['source_identity'] != base['source_identity'] or
            prediction['field_sha256'] != plan['reference_field_sha256'] or
            B.sha(native/'CONT_FREQ') != prediction['continuum_sha256'] or
            B.sha(legacy_path) != prediction['legacy_native_identity']['inputs_sha256'] or
            native_inputs['fixed_field_sha256'] != legacy_inputs['fixed_field_sha256'] or
            B.sha(paths['rate_contract']) != prediction['prior_rate_sha256']):
        raise ValueError('같은 원70/J70/교정 native 계보 불일치')
    artifacts = dict(bf=old/'bound-full-frequency-contributions.bin',
                     bb=old/'point-lines-contributions.bin',
                     occupation=old/'model-occupation-input.bin',
                     parameters=old/'model-occupation-parameters.bin')
    expected = dict(bf=baseline['full_bound_test']['contribution_sha256'],
                    bb=baseline['point_bb_test']['contributions_sha256'],
                    occupation=baseline['inputs_sha256']['model-occupation-input.bin'],
                    parameters=baseline['inputs_sha256']['model-occupation-parameters.bin'])
    if (any(B.sha(path) != expected[key] for key,path in artifacts.items()) or
            B.sha(paths['collision']) != baseline['point_collision_test']['binary_sha256']):
        raise ValueError('기존 BF/BB/occupation/충돌 입력 변경')
    source_files = [Path(__file__).resolve(),Path(__file__).with_name('lumina_model_point_reference.py'),
                    Path(L.__file__),
                    Path(__file__).with_name('lumina_model_point_rates.py'),
                    Path(__file__).with_name('lumina_model_point_lines.py'),Path(B.__file__),
                    Path(R.__file__),Path(N.__file__),*artifacts.values(),
                    *[paths[k] for k in ('base','baseline','prediction','rate_contract','collision',
                                        'point_library','lines_library','projection')],
                    legacy_path,Path(args.projection).with_name('source-before.sha256'),
                    Path(__file__).resolve().parents[1]/'tests/model_atom_projection_selftest.c',
                    Path(__file__).resolve().parents[1]/'src/model_source_kernels.c',
                    native/'NATIVE_SE_BOUND_000071.bin',native/'LEVEL_SL_STEQ_LINKS']
    if args.reuse_bound:
        source_files.append(Path(args.reuse_bound).resolve())
    identities = {str(path):B.sha(path) for path in source_files}
    (out/'inputs.json').write_text(json.dumps(dict(scope=__doc__,input_sha256=identities,
        fields='original native J70; not the newer direct-coefficient transport',
        valid_elements=[14,16],other_element_rates='PRESERVED_BUT_UNUSED; not a whole-model field',
        resources='CPU1/4GiB/1h/retry0; no RT/native/public material update'),indent=2)+'\n')
    report_path = Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path) != c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('기존 물질 계수 보고서 변경')
    field_path = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    geometry,field = fields(field_path)
    try:
        with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
            rows,models,routes,state,do_dis,full_check,smoothing = R.material_state(
                c,plan,source,json.loads(report_path.read_text()))
            if source.identity != base['source_identity'] or not np.array_equal(geometry[0]*1e10,source.radius):
                raise ValueError('원 물질/장 반경 신원 불일치')
        nu = np.array(field['freq'][::-1]*1e15)
        j = np.array(field['field'][::-1,0,49])
        if len(nu) != 197291 or not np.isfinite(j).all() or np.any(j < 0):
            raise ValueError('원 복사장 주파수/J 오류')
        evaluation = continuum_mapping(native/'CONT_FREQ',nu)
        rate_metadata = json.loads(paths['rate_contract'].with_suffix('.json').read_text())
        for name,values in (('frequency.npy',nu),('continuum-frequency.npy',evaluation)):
            path = paths['rate_contract'].with_name(name)
            if (B.sha(path) != rate_metadata['output_sha256'][name] or
                    not np.array_equal(values,np.load(path,allow_pickle=False))):
                raise ValueError('기존 봉인된 원 주파수/CONT_FREQ 정확한 주소와 불일치')
        width = np.zeros_like(nu)
        width[:-1] += .5*np.diff(nu)
        width[1:] += .5*np.diff(nu)
        point = ModelPointBoundRates(routes,rows,models,state,49,paths['point_library'])
        producer = ReferencePointRates(point)
        original_blob = artifacts['bf'].read_bytes()
        bf = check_bf_records(original_blob,point,len(rows),90)
        selected = np.array([route.Z in (14,16) for route in point.routes])
        recovery = None
        if args.reuse_bound:
            bound,total,recovery = reuse_bound(args.reuse_bound,out,selected,len(routes))
            done = selected.copy()
        else:
            bound = np.lib.format.open_memmap(out/'bound-unit-rates.npy',mode='w+',dtype='f8',shape=(len(routes),6))
            total = np.lib.format.open_memmap(out/'total-unit-rates.npy',mode='w+',dtype='f8',shape=(len(routes),6))
            bound[:] = bf['values'][:,4:]
            total[:] = bound
            bound[selected] = np.nan
            total[selected] = np.nan
            done = np.zeros(len(routes),dtype=bool)
            for count,i in enumerate(np.flatnonzero(selected),1):
                sums = np.zeros((2,6),dtype=np.longdouble)
                for lo in range(0,len(nu),16384):
                    sl = slice(lo,min(len(nu),lo+16384))
                    sums += producer.contract(i,nu[sl],width[sl],j[sl],evaluation[sl],
                                               dissolution_enabled=do_dis)
                bound[i],total[i] = sums.astype('f8')
                done[i] = True
                if count%250 == 0:
                    bound.flush();total.flush()
                    np.save(out/'selected-routes-done.npy',done)
                    print(json.dumps(dict(selected_routes_complete=count,elapsed=time.monotonic()-start)),flush=True)
            bound.flush();total.flush();np.save(out/'selected-routes-done.npy',done)
        if not done[selected].all() or not np.isfinite(total).all():
            raise ValueError('Si/S BF·terminal·pseudo 전체 경로 미완료')
        physical = point.physical_ion_rates(total)
        rate_checks = {}
        for label,ion in prediction['results'].items():
            got = physical[ion['Z'],ion['stage']]
            wanted = np.array([ion['C']['total'][name] for name in ('photo','spontaneous','stimulated')])
            np.testing.assert_allclose(got,wanted,rtol=2e-10,atol=0)
            by_sl = np.zeros((len(ion['per_SL']),3),dtype=np.longdouble)
            for i,route in enumerate(point.routes):
                if (route.Z,route.stage) != (ion['Z'],ion['stage']):
                    continue
                sl = int(rows[route.lower]['super_level'])
                by_sl[sl] += total[i,:3]*route.probability*np.array([
                    state['full'][route.lower,49],point.upper_population[i],point.upper_population[i]])
            expected_sl = np.array([[entry['total'][name] for name in ('photo','spontaneous','stimulated')]
                                    for entry in ion['per_SL']])
            np.testing.assert_allclose(by_sl.astype('f8'),expected_sl,rtol=2e-10,atol=0)
            rate_checks[label] = dict(C=got.tolist(),prior_same_J=wanted.tolist(),
                                      matched_superlevels=len(by_sl))
        line_model = ModelPointLines(Path(plan['initial_material']),rows,models,paths['lines_library'])
        bb_blob = artifacts['bb'].read_bytes()
        tag,nl,nline,ns,node = LINE_HEADER.unpack_from(bb_blob)
        if ((tag,nl,ns,node) != (b'LMBBPTS1',len(rows),90,49) or
                len(bb_blob) != LINE_HEADER.size+nline*LINE_RECORD.itemsize):
            raise ValueError('기존 BB 컨테이너 물질/원래점 오류')
        bb = np.frombuffer(bb_blob,dtype=LINE_RECORD,offset=LINE_HEADER.size).copy()
        if (not np.array_equal(bb['ids'][:,0],line_model.lines['lower']) or
                not np.array_equal(bb['ids'][:,1],line_model.lines['upper']) or
                not np.array_equal(bb['values'][:,0],line_model.lines['nu0']) or
                not np.array_equal(bb['values'][:,3],line_model.lines['A'])):
            raise ValueError('동일 raw OSC 활성 선 명단/A가 아님')
        line_Z = np.array([int(row['atomic_number']) for row in rows])
        line_selected = np.isin(line_Z[bb['ids'][:,0]],[14,16])
        if np.any(line_Z[bb['ids'][:,0]] != line_Z[bb['ids'][:,1]]):
            raise ValueError('원소 사이 BB 전이')
        line_model.lines = {key:value[line_selected] for key,value in line_model.lines.items()}
        physical_lines = line_model.contract(nu,width,j)
        np.savez(out/'physical-selected-lines.npz',**line_model.lines,**physical_lines)
        first,last,active,line_intervals = L.intervals(native,rows,models,line_model.lines,len(nu))
        line_values = L.contract(line_model,nu,width,j,first,last,active)
        np.savez(out/'native-selected-lines.npz',**line_model.lines,**line_values)
        nd,nt = read_dimensions(native)
        wh,witness = read_witness(native/'NATIVE_SE_BOUND_000071.bin')
        variables = read_variables(native,nt-2)
        _,_,source70 = record_bytes(Path(native_inputs['source70'])/'SCRTEMP',70,nd,nt)
        if wh['parent'] != 70 or not np.array_equal(witness['pops'],source70):
            raise ValueError('native raw SE는 원70 입력 물질이어야 함')
        addresses = {({'Si':14,'S':16}[v.species],v.charge,-1 if v.terminal else v.superlevel-1):v.index-1
                     for v in variables if v.species in ('Si','S')}
        original = {key:float(source70[index,49]) for key,index in addresses.items()}
        if len(original) != 518:
            raise ValueError('원 Si/S518개 변수 주소 불일치')
        native_scales = next(iter(prediction['results'].values()))['native_constant_scales']
        physical_complete = bool(np.isin(physical_lines['status'],[1,2]).all())
        cases = {} if physical_complete else dict(physical=dict(state='INCOMPLETE_UNSAMPLED_BB',
            unsampled=int(np.count_nonzero(physical_lines['status']==4)),
            out_of_grid=int(np.count_nonzero(physical_lines['status']==3)),candidate_solved=False))
        for name,matched in (('physical',False),('native_discrete',True)):
            if not matched and not physical_complete:
                continue
            bf_copy = bf.copy();bb_copy = bb.copy()
            bf_copy['values'][selected,4:] = total[selected]
            current_lines = line_values if matched else physical_lines
            bb_copy['values'][line_selected,1] = current_lines['absorption']
            bb_copy['values'][line_selected,2] = current_lines['stimulated']
            bb_copy['values'][line_selected,3] = current_lines['spontaneous_native_discrete' if matched else 'spontaneous']
            if matched:
                scale = np.array([native_scales['photo'],native_scales['reverse'],native_scales['reverse']]*2)
                bf_copy['values'][selected,4:] *= scale
            bf_file = out/(name+'-bf.bin');bb_file = out/(name+'-bb.bin')
            bf_file.write_bytes(original_blob[:HEADER.size]+bf_copy.tobytes())
            bb_file.write_bytes(bb_blob[:LINE_HEADER.size]+bb_copy.tobytes())
            exported = out/(name+'-imported-full.bin')
            command = [str(paths['projection']),str(Path(c['seed'])/'atomic'),'90',
                       str(Path(c['seed'])/'native_material_nodes.bin'),str(exported),
                       str(artifacts['occupation']),str(artifacts['parameters']),str(bf_file),str(bb_file),
                       str(paths['collision'])]
            project = Path(__file__).resolve().parents[1]
            env = dict(os.environ,LUMINA_TOPION_LEVELS_FILE=str(project/'data/atomic/topion_levels.csv'),
                       LUMINA_IONIZATION_REFERENCE_FILE=str(project/'data/atomic/ionization_reference.csv'))
            child = subprocess.run(command,capture_output=True,text=True,timeout=300,env=env)
            (out/(name+'.stdout')).write_text(child.stdout)
            (out/(name+'.stderr')).write_text(child.stderr)
            if child.returncode or B.sha(exported) != B.sha(Path(c['native_full_population'])):
                raise ValueError(name+' C 후보 실행/원 입력 인구 불일치')
            candidate,residual = parse_candidate(child.stdout,original,49)
            raw = parse_raw(child.stdout,original)
            profile = []
            for key,index in addresses.items():
                c_raw = raw[key]
                native_value = float(witness['raw'][index,49])
                delta = c_raw['total']-native_value
                flow = c_raw['absolute_flow']
                profile.append(dict(Z=key[0],charge=key[1],superlevel=key[2],
                    original=original[key],candidate=candidate[key],C=c_raw,native_raw=native_value,
                    difference=delta,difference_over_C_absolute_flow=abs(delta)/flow if flow > 0 else None))
            cases[name] = dict(private_stationary_residual=residual,profile=profile,
                              applied_native_BF_constants=matched,applied_native_BB_profile_mass=matched,
                              scalar_files_sha256={str(path):B.sha(path) for path in (bf_file,bb_file)})
        if (not np.array_equal(j,np.array(field['field'][::-1,0,49])) or
                any(B.sha(Path(path)) != digest for path,digest in identities.items())):
            raise ValueError('실행 중 사용 입력/소스 변경')
        result = dict(schema='lumina-original-J70-SiS-response-v1',state='COMPLETE_SAME_J_COMPARISON',
            input_sha256=identities,input_material=70,native_node=49,
            field_sha256_inherited=plan['reference_field_sha256'],
            consumed_J_column_sha256=hashlib.sha256(j.tobytes()).hexdigest(),
            frequency_points=len(nu),selected_BF_routes=int(selected.sum()),
            selected_BB_lines=int(line_selected.sum()),other_element_rates='PRESERVED_BUT_UNUSED',
            bound_recovery=recovery,native_line_intervals=line_intervals,
            physical_BB_complete=physical_complete,
            same_J_ion_rate_checks=rate_checks,BB_profile_mass_range=[float(line_values['profile_mass'].min()),
                float(line_values['profile_mass'].max())],BB_profile= line_model.profile,
            cases=cases,source_unchanged=True,wall_seconds=time.monotonic()-start,
            native_arithmetic_error_certificate=False,raw_SE_equivalence='NOT_YET_EVALUATED',
            public_material_updates=0,new_transport=0,new_native=0,SE_charge_RE_solved=False,final_goal='NOT_MET')
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(state=result['state'],wall_seconds=result['wall_seconds'])),flush=True)
        return 0
    finally:
        field._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','baseline','prediction','rate-contract','native-root','collision','point-library',
                 'lines-library','projection','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reuse-bound',type=Path)
    raise SystemExit(run(parser.parse_args()))
