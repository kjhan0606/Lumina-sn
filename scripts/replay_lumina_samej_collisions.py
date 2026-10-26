#!/usr/bin/env python3
"""완료된 동일J 계수로 CI/C3 스위치의 영향만 대조한다. 수송·계수 재계산 없음."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cmfgen_model_continuum as B
from run_lumina_reference_response import parse_raw
from run_lumina_pseudo_candidate_overlay import parse_candidate


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
    projection = source/'build/model_projection'
    bf = source/'run/native_discrete-bf.bin'
    bb = source/'run/native_discrete-bb.bin'
    occupation = old/'model-occupation-input.bin'
    parameters = old/'model-occupation-parameters.bin'
    sealed = dict(report['cases']['native_discrete']['scalar_files_sha256'])
    for path in (base_path,projection,occupation,parameters,collision):
        sealed[str(path)] = report['input_sha256'][str(path)]
    sealed.update(base['input_sha256'])
    sealed[str(report_path)] = B.sha(report_path)
    for path in (Path(__file__),project/'scripts/run_lumina_reference_response.py',
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
    for enabled in (0,1):
        name = 'CI'+str(enabled)
        exported = out/(name+'-imported.bin')
        command = [str(projection),str(Path(config['seed'])/'atomic'),'90',
                   str(Path(config['seed'])/'native_material_nodes.bin'),str(exported),
                   str(occupation),str(parameters),str(bf),str(bb),str(collision)]
        env = dict(os.environ,LUMINA_ARTIS_PARITY='0',LUMINA_NLTE_BF_COLLISION=str(enabled),
            LUMINA_TOPION_LEVELS_FILE=str(project/'data/atomic/topion_levels.csv'),
            LUMINA_IONIZATION_REFERENCE_FILE=str(project/'data/atomic/ionization_reference.csv'))
        child = subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
        (out/(name+'.stdout')).write_text(child.stdout)
        (out/(name+'.stderr')).write_text(child.stderr)
        if child.returncode or B.sha(exported) != sealed[config['native_full_population']]:
            raise ValueError(name+' 재생 실행/원 인구 보존 오류')
        candidate,residual = parse_candidate(child.stdout,original,49)
        rows = parse_raw(child.stdout,original)
        if not enabled and any(rows[k] != reference[k]['C'] for k in rows):
            raise ValueError('CI0 대조가 저장518행을 재현하지 못함')
        cases[name] = dict(stationary_residual=residual,profile=compare_rows(rows,reference),
            candidates=[dict(Z=k[0],charge=k[1],superlevel=k[2],population=v)
                        for k,v in candidate.items()],command=command,
            collision_switch=enabled,nonzero_bf_col_rows=sum(r['bf_col'] != 0. for r in rows.values()))
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
    run(parser.parse_args())
