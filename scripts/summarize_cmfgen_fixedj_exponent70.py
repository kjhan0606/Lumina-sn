#!/usr/bin/env python3
"""완료된 고정장 지수 교정의 PRRR·SE 예측을 같은 원70 입력에서 재현한다."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_cmfgen_bf_superlevel_rates import parse_prrr_superlevel_rates
from compare_native_prrr_common_grid import printed_half_step
from prepare_cmfgen_fixedj_exponent import patched
from prepare_cmfgen_fixed_field_owner import record_bytes
from ptf_gated_metrics import read_dimensions, read_variables
from run_cmfgen_lambda_probe import sha
from run_lumina_pseudo_candidate_overlay import parse_candidate
from summarize_cmfgen_verbose_se70 import summarize


def compare_printed(actual, prediction):
    actual, prediction = (np.asarray(v, dtype='f8') for v in (actual, prediction))
    if (actual.ndim != 1 or prediction.shape != actual.shape or
            not np.isfinite(actual).all() or not np.isfinite(prediction).all() or
            np.any(actual < 0) or np.any(prediction < 0)):
        raise ValueError('PRRR 예측/출력 차원 또는 값 오류')
    bound = np.array([printed_half_step(v) for v in actual])
    residual = actual-prediction
    allowance = 64*np.finfo(float).eps*np.maximum(actual, prediction)
    return dict(actual_sum=float(actual.sum()), prediction_sum=float(prediction.sum()),
                residual_sum=float(residual.sum()), print_bound_sum=float(bound.sum()),
                maximum_absolute_residual=float(np.max(abs(residual))),
                outside_print_bound=int(np.count_nonzero(abs(residual) > bound+allowance)))


def profile_metrics(reference, candidate):
    reference, candidate = (np.asarray(v,dtype='f8') for v in (reference,candidate))
    if (reference.ndim != 1 or not len(reference) or candidate.shape != reference.shape or
            not np.isfinite(reference).all() or not np.isfinite(candidate).all() or
            np.any(reference < 0) or np.any(candidate < 0)):
        raise ValueError('SL 개체수 대응/수치 오류')
    diff = abs(candidate-reference)
    positive = reference > 0
    total = float(reference.sum())
    major = (reference >= .01*total)&positive
    relative = np.divide(diff,reference,out=np.zeros_like(diff),where=positive)
    return dict(reference_sum=total, candidate_sum=float(candidate.sum()),
                sum_relative=float(candidate.sum()/total-1) if total > 0 else None,
                normalized_L1=float(diff.sum()/total) if total > 0 else None,
                max_absolute_difference=float(diff.max()),
                max_relative_nonzero_reference=float(relative.max()),
                zero_reference_positive_candidate=int(np.count_nonzero((~positive)&(candidate>0))),
                major_definition='at least 1 percent of this reference ion population',
                major_levels=int(np.count_nonzero(major)),
                major_max_relative=float(relative[major].max()) if np.any(major) else None)


def run(root, diagnosis, output, candidate_path=None):
    root, diagnosis, output = map(Path, (root, diagnosis, output))
    if output.exists():
        raise FileExistsError('기존 교정 대조 증거 보존')
    d = json.loads(diagnosis.read_text())
    if d['material_record'] != 70 or d['node'] != 49 or set(d['results']) != {'Sk2','SkIII','S2','SIII'}:
        raise ValueError('원70/원49 네 이온의 사전 예측 필요')
    new_inputs_path = root/'run/seq_logs/verbose-se-inputs.json'
    new_inputs = json.loads(new_inputs_path.read_text())
    runtime_path = root/'run/seq_logs/verbose-se-runtime.json'
    runtime = json.loads(runtime_path.read_text())
    if not runtime.get('completed_corrected_step') or not runtime['corrected_fixed_J_exponent']:
        raise ValueError('정상 완료한 교정 고정장 실행이 아님')
    legacy = d['legacy_native_identity']
    old_inputs_path = Path(legacy['inputs'])
    if sha(old_inputs_path) != legacy['inputs_sha256']:
        raise ValueError('사전 대조의 원 실행 계보 변경')
    old_inputs = json.loads(old_inputs_path.read_text())
    old_material = old_inputs_path.parent.parent/'material'
    material = root/'run/material'
    identity_keys = ('record', 'record_storage_sha256', 'header_sha256', 'depths', 'variables')
    if (any(new_inputs['input_identity70'][k] != old_inputs['input_identity70'][k] for k in identity_keys) or
            new_inputs['fixed_field_sha256'] != old_inputs['fixed_field_sha256'] or
            new_inputs['stage_controls_sha256'] != old_inputs['stage_controls_sha256'] or
            new_inputs['parent_executable'] != old_inputs['executable']):
        raise ValueError('지수 교정 이외의 입력 상태/장/제어/부모 변경')
    old_source = Path(legacy['source'])
    corrected_source = root/'build/cmfgen_sub.f'
    if (sha(old_source) != legacy['source_sha256'] or
            corrected_source.read_text() != patched(old_source.read_text())):
        raise ValueError('정확한 실제ν 지수 이동 이외의 소스 변경')
    if (sha(Path(legacy['executable'])) != legacy['executable_sha256'] or
            sha(Path(new_inputs['executable']['path'])) != new_inputs['executable']['sha256'] or
            sha(material/'CONT_FREQ') != d['continuum_sha256']):
        raise ValueError('바이너리/원 연속 격자 변경')
    for name, digest in new_inputs['fixed_field_sha256'].items():
        if sha(material/name) != digest:
            raise ValueError('실행 뒤 원 고정장 변경')
    results = {}
    evidence_paths = [diagnosis,new_inputs_path,runtime_path,old_inputs_path,old_source,
                      corrected_source,Path(__file__).resolve()]
    for directory in (material,old_material):
        evidence_paths.extend(directory/name for name in ('fort.19','NATIVE_SE_BOUND_000071.bin',
                              'LEVEL_SL_STEQ_LINKS', 'Sk2PRRR','SkIIIPRRR','SkIVPRRR',
                              'S2PRRR','SIIIPRRR','SIVPRRR'))
    if candidate_path is not None:
        candidate_path = Path(candidate_path)
        evidence_paths.append(candidate_path)
        evidence_paths.extend(candidate_path.parent/name for name in ('control.stdout','pseudo.stdout'))
    evidence_hashes = {str(path):sha(path) for path in evidence_paths}
    for label, ion in d['results'].items():
        nsl = len(ion['per_SL'])
        arrays = {}
        for scope, directory in (('old', old_material), ('new', material)):
            for process in ('photoionization', 'recombination'):
                arrays[scope, process] = parse_prrr_superlevel_rates(
                    directory/f'{label}PRRR', label, 90, nsl, process=process)
        actual = arrays['new', 'recombination'][:, 49]
        expected = [r['total']['reverse']*ion['native_constant_scales']['reverse']
                    for r in ion['per_SL']]
        previous = arrays['old', 'recombination'][:, 49]
        result = compare_printed(actual, expected)
        result.update(n_superlevels=nsl,
            original_sum=float(previous.sum()), change=float((actual-previous).sum()),
            C_actual_physical_constants=ion['C']['total']['reverse'],
            C_relative_difference=float(actual.sum()/ion['C']['total']['reverse']-1),
            photo_unchanged_all90=bool(np.array_equal(arrays['new','photoionization'],
                                                     arrays['old','photoionization'])),
            reverse_change_print_bound=float(sum(printed_half_step(v) for v in actual)+
                                             sum(printed_half_step(v) for v in previous)),
            per_SL=[dict(sl=i, actual=float(a), prediction=float(e),
                         residual=float(a-e), print_bound=printed_half_step(a))
                    for i,(a,e) in enumerate(zip(actual, expected))])
        results[label] = result
    old_se = summarize(old_material/'fort.19', old_material/'NATIVE_SE_BOUND_000071.bin',
                       old_material, 49)
    new_se = summarize(material/'fort.19', material/'NATIVE_SE_BOUND_000071.bin', material, 49)
    nd, nt = read_dimensions(material)
    variables = read_variables(material, nt-2)
    sizes = Counter((v.species, v.charge) for v in variables if not v.terminal)
    exchange = {}
    for species, prefix in (('Si','Sk'), ('S','S')):
        for charge, suffix in ((1,'2'),(2,'III'),(3,'IV')):
            label = prefix+suffix
            nsl = sizes[species,charge]
            if nsl <= 0:
                raise ValueError('SE 대응 원 이온 없음')
            old = parse_prrr_superlevel_rates(old_material/f'{label}PRRR',label,nd,nsl,
                                              process='recombination')[:,49]
            new = parse_prrr_superlevel_rates(material/f'{label}PRRR',label,nd,nsl,
                                              process='recombination')[:,49]
            exchange[species,charge] = (float((new-old).sum()),
                                        float(sum(printed_half_step(v) for v in old)+
                                              sum(printed_half_step(v) for v in new)))
    se_effect = {}
    for name, new in new_se['stage_sums'].items():
        old = old_se['stage_sums'][name]
        species, roman = name.split()
        charge = {'II':1,'III':2,'IV':3}[roman]
        own, own_bound = exchange[species,charge]
        lower, lower_bound = exchange.get((species,charge-1),(0.,0.))
        expected_delta = own-lower
        raw_delta = new['final_raw']-old['final_raw']
        raw_bound = own_bound+lower_bound
        delta_cont = new['continuum_increment']-old['continuum_increment']
        cont_bound = new['continuum_increment_print_bound']+old['continuum_increment_print_bound']
        se_effect[name] = dict(original=old, corrected=new, expected_net_BF_delta=expected_delta,
            raw_delta=raw_delta, raw_delta_minus_PRRR=raw_delta-expected_delta,
            PRRR_change_print_bound=raw_bound,
            raw_change_matches_PRRR=bool(abs(raw_delta-expected_delta) <= raw_bound+
                                   128*np.finfo(float).eps*max(abs(raw_delta),abs(expected_delta))),
            continuum_increment_delta=delta_cont,
            continuum_delta_minus_PRRR=delta_cont-expected_delta,
            continuum_difference_print_bound=cont_bound,
            continuum_change_matches_PRRR=bool(abs(delta_cont-expected_delta) <= raw_bound+cont_bound))
    accepted = (all(v['outside_print_bound'] == 0 and v['photo_unchanged_all90']
                    for v in results.values()) and
                all(v['raw_change_matches_PRRR'] and v['continuum_change_matches_PRRR']
                    for v in se_effect.values()))
    header, storage, corrected_state = record_bytes(material/'SCRTEMP',71,nd,nt)
    output_identity = dict(record=71, header_sha256=hashlib.sha256(header).hexdigest(),
                           record_storage_sha256=hashlib.sha256(storage).hexdigest())
    candidate_comparison = None
    if candidate_path is not None:
        candidate = json.loads(candidate_path.read_text())
        if (candidate.get('schema') != 'lumina-private-direct-pseudo-candidate-v1' or
                candidate.get('state') != 'COMPLETE_PRIVATE_OVERLAY' or
                candidate.get('material_record') != 70 or candidate.get('native_node') != 49 or
                candidate.get('source_unchanged') is not True or
                candidate.get('public_material_updates') != 0 or candidate.get('new_transport') != 0):
            raise ValueError('기존 완료 후보의 물질/원래점/보존 규약 불일치')
        _, _, reference_state = record_bytes(Path(new_inputs['reference71'])/'SCRTEMP',71,nd,nt)
        _, _, original_state = record_bytes(Path(new_inputs['source70'])/'SCRTEMP',70,nd,nt)
        addresses = {}
        for v in variables:
            if v.species not in ('Si','S'):
                continue
            key = ({'Si':14,'S':16}[v.species],v.charge,-1 if v.terminal else v.superlevel-1)
            if key in addresses:
                raise ValueError('원 native SL/terminal 주소 중복')
            addresses[key] = v.index-1
        if len(addresses) != 518:
            raise ValueError('Si/S 전체518개 원 변수 주소 필요')
        original = {key:float(original_state[index,49]) for key,index in addresses.items()}
        bound_values, _ = parse_candidate((candidate_path.parent/'control.stdout').read_text(),original,49)
        pseudo_values, _ = parse_candidate((candidate_path.parent/'pseudo.stdout').read_text(),original,49)
        rows = []
        for entry in candidate['stages']:
            species = {14:'Si',16:'S'}[entry['Z']]
            indices = [v.index-1 for v in variables if v.species == species and v.charge == entry['charge']]
            old_total = float(reference_state[indices,49].sum())
            new_total = float(corrected_state[indices,49].sum())
            if not np.isclose(old_total,entry['native71'],rtol=1e-12,atol=0):
                raise ValueError('기존 후보가 비교한 원71의 이온 인구 불일치')
            rows.append(dict(Z=entry['Z'], charge=entry['charge'], original71=old_total,
                corrected71=new_total, bound_candidate=entry['bound_candidate'],
                pseudo_candidate=entry['pseudo_candidate'],
                bound_over_corrected_minus1=entry['bound_candidate']/new_total-1,
                pseudo_over_corrected_minus1=entry['pseudo_candidate']/new_total-1))
        candidate_comparison = dict(stages=rows,
            mapped_variables=len(addresses),
            per_ion_profiles=[dict(Z=z,charge=charge,n_variables=len(keys),
                bound_vs_original71=profile_metrics([reference_state[addresses[k],49] for k in keys],
                                                    [bound_values[k] for k in keys]),
                pseudo_vs_original71=profile_metrics([reference_state[addresses[k],49] for k in keys],
                                                     [pseudo_values[k] for k in keys]),
                bound_vs_corrected71=profile_metrics([corrected_state[addresses[k],49] for k in keys],
                                                     [bound_values[k] for k in keys]),
                pseudo_vs_corrected71=profile_metrics([corrected_state[addresses[k],49] for k in keys],
                                                      [pseudo_values[k] for k in keys]))
                for z,charge in sorted({k[:2] for k in addresses})
                for keys in [[k for k in addresses if k[:2] == (z,charge)]]],
            per_superlevel=[dict(Z=k[0],charge=k[1],superlevel=k[2],
                original70=original[k],original71=float(reference_state[index,49]),
                corrected71=float(corrected_state[index,49]),
                bound_candidate=bound_values[k],pseudo_candidate=pseudo_values[k])
                for k,index in addresses.items()],
            interpretation='existing direct-J partial-channel fixed-ne stationary candidate vs '
            'corrected cached-coefficient-J native one-step; fields and solve conditions still differ; '
            'not a same-operator or self-consistent equivalence acceptance')
    if any(sha(Path(path)) != digest for path,digest in evidence_hashes.items()):
        raise ValueError('검증 도중 근거 파일 변경')
    report = dict(schema='lumina-fixedj-exponent70-comparison-v1',
        status='FIXED_J_OPERATOR_CORRECTION_REPRODUCED' if accepted else 'REWORK',
        same_input_material=70, output_material=71, same_J_material=70, node=49,
        source_change='only actual-frequency EMHNUKT leaves COMPUTE_NEW_CROSS cache',
        input_sha256=evidence_hashes, corrected_material_identity=output_identity,
        existing_C_candidate_comparison=candidate_comparison,
        source_original_preserved=True, results=results, SE_effect=se_effect,
        raw_SE_is_native_input70_not_output71=True,
        scope='same material/field operator correction; not self-consistent populations or spectrum',
        physical_transport_runs=0, corrected_native_material_steps=1,
        final_goal='NOT_MET')
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with output.open('x') as stream:
        stream.write(payload)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','diagnosis','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.diagnosis, args.output, args.candidate)
    print(json.dumps(dict(status=result['status'], final_goal=result['final_goal']), ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'FIXED_J_OPERATOR_CORRECTION_REPRODUCED' else 1)
