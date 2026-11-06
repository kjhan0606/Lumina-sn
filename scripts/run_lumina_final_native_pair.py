"""완료 물질/J를 두 native 격자에서 독립 평가한다. 수송·SE 갱신·재시도 없음."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import run_cmfgen_fixedj_heat_capture as C
import compare_cmfgen_fixedj_full as F
from run_cmfgen_lambda_probe import INPUTS, RESTART
from run_cmfgen_field_feedback import RESTART_OVERLAY
from prepare_lumina_accepted_material import prepare as export_material
from prepare_cmfgen_fixedj_heat import prepare_state, refined_grid_inputs, read_heat_row
from prepare_cmfgen_fixedj_field import prepare as prepare_field
from extract_cmfgen_line_net_fixture import CMFGEN_RE_INTERNAL_TO_CGS


def final_input(config):
    path=Path(config['result']);r=json.loads(path.read_text())
    if (r.get('state')!='COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE'
            or r.get('source_commit')!=config['producer_commit']
            or r.get('field',{}).get('J',{}).get('sha256')!=config['J_sha256']
            or r.get('thermal_energy_policy')!='native_instantaneous_scenario'
            or r.get('charge_converged') is not True):
        raise ValueError('완료한 동일-J 수용 물질이 아니므로 독립 계산을 시작하지 않음')
    e=r['selected_response']
    for name,key in [('response.npz','response_sha256'),('balance.npz','balance_sha256')]:
        if C.sha(Path(e['root'])/name)!=e[key]:raise ValueError('최종 수용 응답/장부 변경')
    return r, dict(J_sha256=config['J_sha256'],result_sha256=C.sha(path),response_sha256=e['response_sha256'])


def full_comparison(root, response, levels, model):
    """별도 C export를 주장하지 않고 저장된 Lumina FL 복원과 native FL을 비교한다."""
    native = root/'native'
    if C.sha(native/'NATIVE_FULL_INPUT.bin') != C.sha(native/'NATIVE_HEAT_INPUT.bin'):
        raise ValueError('full-level 포획에서 주입 물질이 달라짐')
    if (native/'NATIVE_FULL_PHASE.txt').read_text().splitlines()[0] != F.PHASE:
        raise ValueError('full-level 포획 단계가 다름')
    with levels.open() as stream:
        level_rows = list(csv.DictReader(stream))
    models = F.load_cmfgen_model_spec(model/'MODEL_SPEC', model/'atomic_links.txt')
    lookup, labels = F.level_layout(level_rows, models)
    index = F.read_index(native/'NATIVE_FULL_INDEX.csv', len(level_rows))
    with np.load(response, allow_pickle=False) as candidate:
        columns = {tuple(k): i for i, k in enumerate(candidate['columns'])}
        mapping = [columns[int(r['atomic_number']), int(r['ion_number']), int(r['super_level'])]
                   for r in level_rows]
        expected = {'NATIVE_FULL_POP.bin': candidate['population'][mapping]*candidate['fraction'],
                    'NATIVE_LTE_FRACTION.bin': candidate['fraction'],
                    'NATIVE_OCCUPATION.bin': candidate['occupation']}
    metrics = {}
    for name, values in expected.items():
        aligned, _ = F.align_native(F.read_matrix(native/name), index, level_rows, lookup, labels)
        metrics[name] = F.error_metrics(aligned, values)
    return dict(comparisons=metrics, independent_C_export=False,
                input_sha256={name: C.sha(native/name) for name in (
                    *expected, 'NATIVE_FULL_INPUT.bin', 'NATIVE_FULL_INDEX.csv', 'NATIVE_FULL_PHASE.txt')})


def execute(config_path, expected):
    if C.sha(config_path)!=expected:raise ValueError('독립 묶음 설정 변경')
    config=json.loads(config_path.read_text());out=Path(config['output'])
    if out.exists():raise FileExistsError('기존 독립 출력 보존')
    if shutil.disk_usage(out.parent).free<32*1024**3:raise ValueError('독립 출력 여유32GiB 필요')
    report,identity=final_input(config)
    baseline=Path(config['baseline']);build=Path(config['build'])
    for name,digest in config['pinned_files'].items():
        if C.sha(Path(name))!=digest:raise ValueError('봉인 native 준비 입력 변경')
    history=json.loads((baseline/'native/seq_logs/feedback-stage.json').read_text())
    original=Path(history['original_physical_seed']);material=Path(history['material_source'])
    sources={n:original/n for n in INPUTS+RESTART if n!='setup_links.sh'}
    sources.update({n:material/n for n in RESTART_OVERLAY})
    sources.update({n:baseline/'native'/n for n in ('VADAT','IN_ITS','MODEL_SPEC','EDDFACTOR','EDDFACTOR_INFO')})
    for name,h in history['restart_sha256'].items():
        if C.sha(material/name)!=h:raise ValueError('native 원 checkpoint 변경')
    for item in history['linked_inputs'].values():
        if C.sha(Path(item['target']))!=item['sha256']:raise ValueError('native 원자자료 변경')
    out.mkdir();layout=out/'layout';layout.mkdir()
    for name,path in sources.items():shutil.copy2(path,layout/name)
    for name,item in history['linked_inputs'].items():(layout/name).symlink_to(Path(item['target']).resolve(strict=True))
    inputs=report['input_sha256']
    seeds=[Path(p).parent for p in inputs if Path(p).name=='native_material_nodes.bin' and (Path(p).parent/'material-seed.json').is_file()]
    models=[Path(p).parent for p in inputs if Path(p).name=='MODEL']
    if len(seeds)!=1 or len(models)!=1:raise ValueError('원 material seed/모형 주소 모호')
    export_material(Path(config['result']),identity['result_sha256'],seeds[0],models[0],out/'accepted')
    manifest=out/'accepted/accepted-material.json'
    prepare_state(manifest,C.sha(manifest),models[0],out/'state')
    prepare_field(config['preparation'],config['preparation_sha256'],config['feedback'],config['feedback_sha256'],layout,out/'field')
    grids=refined_grid_inputs
    rows=[];outputs=[]
    with np.load(Path(report['selected_response']['root'])/'balance.npz') as heat:
        lumina=heat['native_instantaneous_scenario_net'].copy()
    if lumina.shape != (90,) or not np.isfinite(lumina).all():
        raise ValueError('최종 열장부 형상/유한성 오류')
    for label,h in [('h05',.5),('h025',.25)]:
        root=out/label
        for name in ('native','inputs','build'):(root/name).mkdir(parents=True)
        for name in sources:
            if name not in ('EDDFACTOR','EDDFACTOR_INFO'):shutil.copy2(layout/name,root/'native'/name)
        for name,item in history['linked_inputs'].items():(root/'native'/name).symlink_to(Path(item['target']).resolve(strict=True))
        changed=grids((layout/'VADAT').read_text(),(layout/'MODEL_SPEC').read_text(),frac_dop=h)
        (root/'native/VADAT').write_text(changed['vadat']);(root/'native/MODEL_SPEC').write_text(changed['model_spec'])
        for folder,names in [('state',('NATIVE_HEAT_INPUT.bin',)),('field',('EDDFACTOR','EDDFACTOR_INFO'))]:
            for name in names:shutil.copy2(out/folder/name,root/'native'/name)
        for folder,name in [('state','heat-state-input.json'),('field','fixedj-field-input.json')]:shutil.copy2(out/folder/name,root/'inputs'/name)
        for name in ('cmfgen_fixedj_heat.exe','fixedj-heat-build.json'):shutil.copy2(build/name,root/'build'/name)
        if C.run(root,identity)!=0:raise RuntimeError('native 포획 실패; 자동 재시도 없음: '+label)
        native=read_heat_row(root/'native/NATIVE_HEAT_ROW.bin')
        values=native['steq_t_native']*CMFGEN_RE_INTERNAL_TO_CGS
        rows.append(values)
        full=full_comparison(root,Path(report['selected_response']['root'])/'response.npz',
                             seeds[0]/'atomic/levels.csv',models[0])
        (root/'full-comparison.json').write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n')
        outputs.append(dict(grid=label,ncf=native['frequencies'],heat_sha256=C.sha(root/'native/NATIVE_HEAT_ROW.bin'),
            full_comparison=full, max_absolute_difference=float(abs(values-lumina).max()),
            rms_difference=float(np.sqrt(np.mean((values-lumina)**2)))))
    with (out/'native-vs-lumina-heat.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['node','lumina_net_cgs','native_h05_net_cgs','native_h025_net_cgs','native_grid_difference'])
        w.writerows((i,float(lumina[i]),float(rows[0][i]),float(rows[1][i]),float(rows[1][i]-rows[0][i])) for i in range(len(lumina)))
    if C.sha(Path(config['result']))!=identity['result_sha256']:raise ValueError('소비 중 최종 보고서 변경')
    summary=dict(state='COMPLETE_INDEPENDENT_FIXED_STATE_HEAT_NOT_JOINT_CONVERGENCE',identity=identity,
        producer_root_converged=report['temperature_block'].get('fixed_field_root_converged',False),
        native_captures=2,new_transport=0,new_material_update=0,comparisons=outputs,
        native_grid_max_change=float(abs(rows[1]-rows[0]).max()),
        caveats=['Si/S 율 판독은 보존된 LINEHEAT/NETRATE/TOTRATE 출력으로 별도 확인',
                 'native 두 격자는 Lumina fine 격자와 다름; 차이를 전부 물리 오류로 단정하지 않음',
                 '고정장 열 평가이며 native SE/스펙트럼/공동수렴 인증 아님'])
    (out/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--sha',required=True)
    a=p.parse_args();print(json.dumps(execute(a.config,a.sha),ensure_ascii=False))
