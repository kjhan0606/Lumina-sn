"""완료 F2의 단일점에서 물질별 BF 율을 정확히 다시 만든다.

저장된 bound 율과 같은 고정 T 캐시를 사용한다. pseudo 및 캐시 사용이
안전하지 않은 경로는 기존 생산 함수가 해당 점에서 다시 적분한다.
물질 인구·복사장 수송·정상해를 갱신하지 않는다.
"""
from pathlib import Path
import json

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_material_rates import ModelMaterialRates, FixedTemperatureBoundRates
from lumina_new_field_response import trapezoid_weights


def reconstruct(node, cache_result=None):
    if type(node) is not int or not 0 <= node < 90:
        raise ValueError('F2 원래점 범위 오류')
    root = Path('/gpfs/kjhan/cmfgen_runs/lumina_F2_response_lageunha_20261028-bBbs8U/run')
    result_path = root/'result.json'
    if B.sha(result_path) != 'b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea':
        raise ValueError('F2 결과 신원 오류')
    result = json.loads(result_path.read_text())
    if result['state'] != 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE':
        raise ValueError('완료 F2 필요')
    preparation = Path('/gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json')
    if B.sha(preparation) != result['input_sha256'][str(preparation)]:
        raise ValueError('수용 입력 신원 오류')
    meta = json.loads(preparation.read_text())
    response = Path(result['selected_response']['root'])/'response.npz'
    field = Path(result['field']['J']['path'])
    grid = Path(meta['grid'])
    bound_path = root/'bound-rates.npy'
    roster_path = root/'route-roster.npz'
    library = root.parent/'build/bridge.so'
    phot = Path('/gpfs/kjhan/cmfgen_runs/lumina_model_spec_428222/cmfgen_phot_smoothing.so')
    for path, expected in ((response,result['selected_response']['response_sha256']),
        (field,result['field']['J']['sha256']),(grid,result['source_identity']['grid_sha256']),
        (bound_path,result['output_sha256']['bound-rates.npy']),
        (roster_path,result['output_sha256']['route-roster.npz']),
        (library,result['input_sha256'][str(library)]),(phot,result['input_sha256'][str(phot)])):
        if B.sha(path) != expected:
            raise ValueError('F2 봉인 입력 변경: '+path.name)
    models = [Path(p) for p in meta['input_sha256'] if p.endswith('/MODEL')]
    if len(models) != 1 or B.sha(models[0]) != meta['input_sha256'][str(models[0])]:
        raise ValueError('수용 모형 변경')
    runroot = models[0].parent
    material = Path(meta['material'])
    if B.sha(material) != meta['input_sha256'][str(material)]:
        raise ValueError('수용 물질 변경')
    atomic = material.parent/'atomic'
    rows, models, routes, policy, _, _ = B.load_model(runroot,atomic,phot)
    state = B.read_material(material,rows,models)
    with np.load(response) as f:
        final_ne = f['ne'].copy()
        expected_columns = f['columns'].copy()
        expected_fraction = f['fraction'][:,node].copy()
        expected_occupation = f['occupation'][:,node].copy()
    if not np.array_equal(state['columns'],expected_columns):
        raise ValueError('F2 원자 주소 불일치')
    freq = np.load(grid,mmap_mode='r');field_values = np.load(field,mmap_mode='r')
    provider = ModelMaterialRates(routes,rows,models,library,freq,trapezoid_weights(freq),
        field_values[node:node+1],radiation_generation=1,dissolution_enabled=B.dissolution_flag(runroot/'MODEL'))
    def point(ne):
        return dict(columns=state['columns'],superlevel=state['superlevel'][:,node:node+1],
                    temperature=state['temperature'][node:node+1],ne=np.array([ne]))
    reference = provider.prepare(point(state['ne'][node]),material_generation=1)
    trial = provider.prepare(point(final_ne[node]),material_generation=2)
    np.testing.assert_allclose(trial.fraction[:,0],expected_fraction,rtol=5e-12,atol=0)
    np.testing.assert_allclose(trial.state['occupation'][:,0],expected_occupation,rtol=5e-12,atol=0)
    reference_bound = np.load(bound_path,mmap_mode='r')[:,node:node+1,:]
    cache = FixedTemperatureBoundRates(provider,reference,reference_bound)
    with np.load(roster_path) as f:
        roster = {k:f[k] for k in f.files}
    if not (np.array_equal(roster['lower'],[r.lower for r in provider.routes]) and
            np.array_equal(roster['target'],[r.target for r in provider.routes]) and
            np.array_equal(roster['phot_grid'],[r.phot_grid for r in provider.routes])):
        raise ValueError('저장 BF 경로와 새 경로 불일치')
    if cache_result is None:
        bound,total = cache.integrate(trial,material_generation=2,radiation_generation=1)
        reused,direct=cache.last_reused_routes,cache.last_direct_routes
    else:
        cache_result=Path(cache_result); saved=json.loads(cache_result.read_text())
        if (saved['state']!='COMPLETE_F2_SINGLE_ELEMENT_LINE_SENSITIVITY' or saved['node']!=node or
            saved['field_sha256']!=result['field']['J']['sha256'] or
            saved['response_sha256']!=result['selected_response']['response_sha256']):
            raise ValueError('보존 단일점 BF 캐시의 세대 불일치')
        cache_path=cache_result.parent/'node-bf.npz'
        if B.sha(cache_path)!=saved['node_bf_sha256']:raise ValueError('보존 단일점 BF 캐시 변경')
        with np.load(cache_path) as f:
            bound=f['bound'];total=f['total']
            np.testing.assert_array_equal(f['saha'],trial.saha)
            np.testing.assert_array_equal(f['fraction'],trial.fraction)
            np.testing.assert_array_equal(f['occupation'],trial.state['occupation'])
        if (bound.shape!=(len(provider.routes),1,6) or total.shape!=bound.shape or
            not np.isfinite(bound).all() or not np.isfinite(total).all() or np.any(total[:,:,:3]<0)):
            raise ValueError('보존 단일점 BF 율 형상/값 오류')
        reused,direct=saved['bound_reused'],saved['bound_direct']
    return dict(state=state,trial=trial,roster=roster,bound=bound,total=total,
                rows=rows,runroot=runroot,library=library,material=material,
                provider=provider,
                reused=reused,direct=direct,
                field_sha256=result['field']['J']['sha256'],response_sha256=result['selected_response']['response_sha256'])
