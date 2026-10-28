"""같은 장·수용 물질의 완료 계수만 읽는다. 열장부/정상해는 재평가한다."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B


COEFFICIENTS = ('bound-rates.npy', 'route-roster.npz', 'jbar.npy',
                'line-status.npy', 'FF-coefficients.npz')
SAMPLED_BB = 'SOURCE_SAMPLED_TRAPEZOID_A_TIMES_MASS_V1'


def coefficient_names(report):
    contract=report.get('BB_contract', 'LEGACY_LINEAR_J_EXACT_A')
    if contract not in (SAMPLED_BB, 'LEGACY_LINEAR_J_EXACT_A'):
        raise ValueError('알 수 없는 완료 BB 적분 계약')
    return COEFFICIENTS+(('line-profile-mass.npy',) if contract==SAMPLED_BB else ())


# 계수 생성·원자자료 해석이 달라졌다면 이전 수치 캐시를 사용하지 않는다.
PRODUCERS = ('cmfgen_model_continuum.py', 'cmfgen_model_sources.py',
             'expand_atomic_data_cmfgen.py', 'cmfgen_parser.py',
             'cmfgen_phot_smoothing.py', 'lumina_model_material_rates.py',
             'lumina_model_point_rates.py', 'lumina_model_linear_lines.py',
             'lumina_model_freefree.py')


def cache_source_signature(path):
    """새 계수 게시 함수만 제외하고 캐시의 수치 helper·전역 바인딩을 고정한다.

    BF/BB/FF 응답 캐시는 S.prepare/_finish_sources의 계수 게시를 사용하지
    않는다. 해당 게시 기능의 추가로 선 명부·프로파일·Gaunt 코드를 검증하지
    않고 넘어가거나 동일 J의 완성 율을 무조건 재생산하지 않는다.
    """
    if path.name!='cmfgen_model_sources.py': return B.sha(path)
    tree=ast.parse(path.read_text())
    main=ast.dump(ast.parse("__name__ == '__main__'",mode='eval').body)
    body=[]
    for index,node in enumerate(tree.body):
        if isinstance(node,ast.FunctionDef) and node.name in ('prepare','_finish_sources','source_output_reserve'):
            defaults=[*node.args.defaults,*[v for v in node.args.kw_defaults if v is not None]]
            if node.decorator_list or any(not isinstance(v,ast.Constant) for v in defaults):
                raise ValueError('계수 게시 함수의 정의 시 실행/전역 부작용 가능성')
            continue
        if isinstance(node,ast.If) and ast.dump(node.test)==main:continue
        if index==0 and isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
        body.append(node)
    return hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[])).encode()).hexdigest()


def load(path, digest, *, field, source_identity, state, inputs, library,
         line_library, routes, lines):
    """봉인 보고서·원 입력·생산자·배열을 검사하고 ne 시작값과 계수를 반환한다.

    저장된 최종 ne는 시작 추정값일 뿐이다. 원 수용 ne의 bound율을 최종
    ne의 total율로 해석하거나 저장된 열장부를 현재 후보에 대입하지 않는다.
    """
    path = Path(path).resolve(); root = path.parent
    if B.sha(path) != digest:
        raise ValueError('완료 응답 보고서 변경')
    report = json.loads(path.read_text())
    if (report.get('state') != 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE' or
            report.get('fixed_accepted_temperature') is not True or
            report.get('charge_converged') is not True or
            report.get('full_BF_integrations') != 1):
        raise ValueError('완료 고정온도·전하 응답만 계수 재사용 가능')
    if report['field'] != field or report['source_identity'] != source_identity:
        raise ValueError('완료 계수의 복사장/수용 물질 불일치')
    old_inputs = report['input_sha256']
    binaries = {str(Path(library).resolve()), str(Path(line_library).resolve())}
    if any(old_inputs.get(p) != h for p, h in inputs.items() if p not in binaries):
        raise ValueError('완료 계수의 원자자료/모형/입력 불일치')
    consumed = {str(path): digest}
    for current, name in ((library, 'bridge.so'), (line_library, 'lines.so')):
        expected = old_inputs[str(root.parent/'build'/name)]
        if B.sha(current) != expected:
            raise ValueError('완료 계수의 생산 바이너리 불일치')
    source_verification={}
    for name in PRODUCERS:
        current = Path(__file__).with_name(name)
        original = root.parent/'source'/'scripts'/name
        expected = B.sha(original)
        current_sha=B.sha(current)
        if cache_source_signature(current) != cache_source_signature(original):
            raise ValueError('완료 계수의 생산 소스 불일치: '+name)
        consumed[str(original)] = expected
        consumed[str(current)] = current_sha
        source_verification[name]=dict(original_sha256=expected,current_sha256=current_sha,
            cache_scope_sha256=cache_source_signature(current))
    names=coefficient_names(report)
    paths = {name: root/name for name in (*names, 'physical-profiles.npz')}
    for name, file in paths.items():
        expected = report['output_sha256'][name]
        if B.sha(file) != expected:
            raise ValueError('완료 계수 파일 변경: '+name)
        consumed[str(file)] = expected
    entry = report['selected_response']
    if (entry not in report['evaluations'] or
            Path(entry['root']).parent != root or
            Path(entry['root']).name not in {'evaluation-'+str(i+1) for i in range(len(report['evaluations']))}):
        raise ValueError('완료 선택 응답 주소 오류')
    response = Path(entry['root'])/'response.npz'
    if B.sha(response) != entry['response_sha256']:
        raise ValueError('완료 선택 응답 변경')
    consumed[str(response)] = entry['response_sha256']
    with np.load(paths['physical-profiles.npz'], allow_pickle=False) as profiles:
        for key in ('temperature', 'ne'):
            if not np.array_equal(profiles['accepted_'+key], state[key]):
                raise ValueError('완료 계수의 기준 T/ne 불일치')
    with np.load(response, allow_pickle=False) as data:
        ne = data['ne'].copy(); pop = data['population']
        charge = data['charge_residual']
        if (not np.array_equal(data['columns'], state['columns']) or
                not np.array_equal(data['temperature'], state['temperature']) or
                ne.shape != state['ne'].shape or not np.isfinite(ne).all() or np.any(ne <= 0) or
                pop.shape != state['superlevel'].shape or not np.isfinite(pop).all() or np.any(pop <= 0) or
                not np.allclose(pop.sum(axis=0), state['superlevel'].sum(axis=0), rtol=1.e-10, atol=0) or
                charge.shape != ne.shape or not np.isfinite(charge).all() or
                np.max(abs(charge/ne)) > 1.e-10):
            raise ValueError('완료 응답의 T/ne/원자주소/핵수/전하 오류')
    ns = len(ne)
    for name, shape in (('bound-rates.npy', (routes, ns, 6)), ('jbar.npy', (lines, ns)),
                        ('line-status.npy', (lines,))):
        values = np.load(paths[name], mmap_mode='r', allow_pickle=False)
        try:
            valid = values.shape == shape and np.isfinite(values).all()
            if name == 'line-status.npy':
                valid = valid and values.dtype == np.dtype('u1') and np.isin(values, [1, 2]).all()
            else:
                valid = valid and values.dtype == np.dtype('f8')
                valid = valid and np.all((values[:, :, :3] if values.ndim == 3 else values) >= 0)
            if not valid:
                raise ValueError('완료 계수의 형상/값 오류: '+name)
        finally:
            values._mmap.close()
    if 'line-profile-mass.npy' in paths:
        values=np.load(paths['line-profile-mass.npy'],mmap_mode='r',allow_pickle=False)
        try:
            if (values.shape!=(lines,) or values.dtype!=np.dtype('f8') or
                    not np.isfinite(values).all() or np.any(values<=0)):
                raise ValueError('완료 BB 자발방출 프로파일 질량 결손')
        finally:values._mmap.close()
    with np.load(paths['FF-coefficients.npz'], allow_pickle=False) as ff:
        shape = (len(np.unique(state['columns'][state['columns'][:, 1] > 0, 1])), ns)
        if (not np.array_equal(ff['temperature'], state['temperature']) or
                any(ff[k].shape != shape or not np.isfinite(ff[k]).all() or np.any(ff[k] < 0)
                    for k in ('heating', 'cooling'))):
            raise ValueError('완료 FF의 온도/형상/값 오류')
    inputs.update(consumed)
    return dict(report=report, paths=paths, ne=ne,
                provenance=dict(result_sha256=digest, result=str(path),
                    producer_verification=source_verification,
                    original_full_BF_seconds=report['full_BF_seconds'],
                    reused_coefficients=list(names),
                    saved_heat_reused=False, saved_ne_role='initial guess; current SE/heat re-evaluated'))
