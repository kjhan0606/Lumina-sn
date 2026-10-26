"""보존한 Newton 방향의 추가 감쇠 후보만 계산한다. 미분 probe는 반복하지 않는다.

방향은 새 물리 증거가 아니라 봉인한 수치 제안이다. 실제 같은 정책의
동적 잔차 감소가 확인된 후보만 수용하며 원 기준점/행렬을 수정하지 않는다.
"""
import json
from pathlib import Path
import re

import numpy as np

from analyze_cmfgen_material_inventory import sha
from lumina_coupled_material import residual
from lumina_material_restart import read_npz


EXTRA_PRODUCERS=('cmfgen_model_continuum.py','cmfgen_model_sources.py','lumina_model_energy.py',
    'lumina_model_freefree.py','lumina_model_balance.py','lumina_native_heat_sources.py',
    'run_lumina_fixed_state_balance.py','compare_lumina_internal_energy.py',
    'expand_atomic_data_cmfgen.py','cmfgen_parser.py','cmfgen_phot_smoothing.py',
    'build_cmfgen_coldata_all.py','build_cmfgen_fullpair_omega.py','l1a_collision.py')


def physical_sources(run,current=None):
    """같은 물리 구현인지 확인한다. 새 운전/감쇠 wrapper만 비교에서 제외한다."""
    run=Path(run);old=run.parent/'source'
    current=Path(__file__).resolve().parents[1] if current is None else Path(current)
    manifest=run.parent/'build/source-before.sha256';expected={}
    for line in manifest.read_text().splitlines():
        digest,name=line.split(maxsplit=1);name=name.strip()
        if (not re.fullmatch('[0-9a-f]{64}',digest) or '..' in Path(name).parts or
            Path(name).is_absolute() or name.split('/')[0] not in ('src','scripts','data')):
            raise ValueError('보존 방향의 source manifest 형식 오류')
        if name!='scripts/run_lumina_common_material.py':expected[name]=digest
    if not any(n.startswith('src/') for n in expected):raise ValueError('보존 방향의 C 소스 신원 누락')
    for name in EXTRA_PRODUCERS:
        relative='scripts/'+name;expected[relative]=sha(old/relative)
    identities={str(manifest):sha(manifest)}
    for name,h in expected.items():
        if sha(old/name)!=h or sha(current/name)!=h:
            raise ValueError('물리 생산자 소스가 바뀐 Newton 방향은 재사용할 수 없음: '+name)
        identities[str(old/name)]=h;identities[str(current/name)]=h
    return identities


def load(path,expected_sha,inputs_sha,policy,inputs):
    path=Path(path).resolve();metadata=path.parent/'inputs.json'
    if (any(not re.fullmatch('[0-9a-f]{64}',h or '') for h in (expected_sha,inputs_sha)) or
        sha(path)!=expected_sha or sha(metadata)!=inputs_sha):
        raise ValueError('보존 Newton 방향/입력 신원 불일치')
    old=json.loads(metadata.read_text())
    if old['temperature_update'] is not True or old['thermal_policy']!=policy:
        raise ValueError('다른 물질/열정책의 보존 Newton 방향')
    code=physical_sources(path.parent)
    previous=old['input_sha256']
    if any(p in inputs and inputs[p]!=h for p,h in previous.items()):
        raise ValueError('보존 방향과 다른 공통 물리 입력')
    if any(sha(Path(p))!=h for p,h in previous.items()):raise ValueError('보존 방향의 원 입력 변경')
    result=path.parent/'result.json'
    if result.exists():
        r=json.loads(result.read_text())
        if any(x.get('accepted') is True for x in r.get('history',[])):
            raise ValueError('이미 수용한 방향을 원 기준점에서 중복 실행할 수 없음')
    inputs.update(previous);inputs.update(code);inputs[str(path)]=expected_sha;inputs[str(metadata)]=inputs_sha
    return read_npz(path),dict(source_commit=old['source_commit'],linearization_sha256=expected_sha,
                               inputs_sha256=inputs_sha,thermal_policy=policy)


def trial(evaluate,base,adiabatic,basis,alpha,*,numerical_tolerance=1.e-10,max_log_step=.1,on_evaluation=None):
    n=base.validate()
    if (not np.isfinite(alpha) or not 0<alpha<1 or not np.isfinite(max_log_step) or max_log_step<=0 or
        not np.isfinite(numerical_tolerance) or numerical_tolerance<=0):
        raise ValueError('보존 방향의 추가 감쇠 비율 오류')
    if not np.array_equal(base.temperature,basis['temperature']) or not np.array_equal(base.ne,basis['ne']):
        raise ValueError('보존 Newton 방향의 기준 T/ne 변경')
    c,h=basis['charge_scale'],basis['thermal_scale'];f=residual(base,adiabatic,c,h)
    if not np.array_equal(f,basis['scaled_residual']):raise ValueError('보존 방향의 기준 잔차/정책 변경')
    if float(abs(f).max())<=numerical_tolerance:raise ValueError('이미 작은 기준 잔차에 추가 후보 불필요')
    j=np.asarray(basis['jacobian']);direction=np.asarray(basis['direction'])
    if j.shape!=(2*n,2*n) or direction.shape!=(2*n,) or not np.isfinite(j).all() or not np.isfinite(direction).all():
        raise ValueError('보존 Newton 선형화 형상/수치 오류')
    error=np.linalg.norm(j@direction+f,np.inf)
    budget=64*(2*n)*np.finfo('f8').eps*(np.linalg.norm(j,np.inf)*np.linalg.norm(direction,np.inf)+np.linalg.norm(f,np.inf))
    if error>budget:raise ValueError('보존 Newton 방향이 같은 선형계를 풀지 않음')
    magnitude=float(abs(direction).max())
    first_alpha=min(1.,max_log_step/magnitude) if magnitude>0 else 1.
    if alpha>=first_alpha:raise ValueError('최초 방향 후보보다 작은 추가 감쇠가 필요')
    x=np.r_[np.log(base.temperature),np.log(base.ne)]
    with np.errstate(over='raise',invalid='raise',under='raise'):values=np.exp(x+alpha*direction)
    candidate=evaluate(values[:n],values[n:]);candidate.validate()
    if not np.array_equal(candidate.temperature,values[:n]) or not np.array_equal(candidate.ne,values[n:]):
        raise ValueError('보존 방향 평가가 요청 T/ne를 소비하지 않음')
    current=float(abs(f).max());actual=float(abs(residual(candidate,adiabatic,c,h)).max())
    accepted=bool(actual<=numerical_tolerance or actual<current*(1.-1.e-4*alpha))
    entry=dict(evaluation=1,phase='frozen_direction',alpha=alpha,accepted=accepted,
        scaled_start_max=current,scaled_max=actual,predicted_scaled_max=float(abs(f+alpha*(j@direction)).max()))
    if on_evaluation is not None:on_evaluation(candidate,entry)
    return dict(numerical_converged=accepted and actual<=numerical_tolerance,
        packet=candidate if accepted else base,history=[entry],iterations=int(accepted))
