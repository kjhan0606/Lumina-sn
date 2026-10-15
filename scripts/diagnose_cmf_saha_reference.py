#!/usr/bin/env python3
"""동일 T/ne에서 초기 Si/S Saha 비를 비교하는 읽기 전용 물리 진단.

공개 T/ne는 이 진단에서만 사용한다. 생산 초기값/원자율을 수정하지 않으며
공개 인구는 출력 비교에만 사용한다. 자기일관 NLTE/수송/최종 수용 시험이 아니다.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from compare_cmf_initial_iterate import material
from read_cmf_common_material import public_block, write_csv

K = 1.380649e-16
H = 6.62607015e-27
ME = 9.1093837015e-28
EV = 1.602176634e-12


def diagnose(run, public, out, raw=None):
    levels = {}
    identities = {}
    with (run/'input/levels.csv').open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row['atomic_number']), int(row['ion_number']))
            if key[0] not in (14, 16) or key[1] not in (1, 2):
                continue
            energy, weight = float(row['energy_eV']), float(row['g'])
            if not math.isfinite(energy) or not weight>0 or not math.isfinite(weight):
                raise ValueError('준위 에너지/통계중량 오류')
            levels.setdefault(key, []).append((energy, weight))
            identities.setdefault(key, []).append((int(row['level_number']), row['configuration']))
    with (run/'input/ionization_energies.csv').open() as handle:
        thresholds = {(int(r['atomic_number']), int(r['ion_number'])):
                      float(r['ionization_energy_eV']) for r in csv.DictReader(handle)}

    def partition(key, temperature, model=levels):
        entries = model[key]
        base = min(e for e, g in entries)
        return math.fsum(g*math.exp(-(e-base)*EV/(K*temperature)) for e,g in entries)

    def ratio(z, temperature, electron):
        if not temperature>0 or not electron>0:
            raise ValueError('양의 온도/전자밀도가 필요함')
        lower, upper = partition((z, 1), temperature), partition((z, 2), temperature)
        chi = thresholds[(z, 1)]
        if not chi>0 or not math.isfinite(chi):
            raise ValueError('전리 문턱 오류')
        value = 2*(2*math.pi*ME*K*temperature/(H*H))**1.5/electron
        value *= upper/lower*math.exp(-chi*EV/(K*temperature))
        if not value>0 or not math.isfinite(value):
            raise ValueError('유한 양의 인접 이온비가 필요함')
        return value, lower, upper

    # 논문의 준위 수/마지막 항에 대응하는 로컬 자료 민감도다. 공개 원자 판본의
    # 동일성이나 비LTE 전이율 재현으로 해석하지 않는다.
    limited, model_sizes = {}, []
    if raw is not None:
        from cmfgen_parser import parse_osc, _CM2EV
        specifications = (
            (14, 1, 'SIL', 'II', 62, '3s2_7g_2Ge[7/2]'),
            (14, 2, 'SIL', 'III', 61, '3s5g3Ge[5]'),
            (16, 1, 'SUL', 'II', 324, '3s_3p3(5So)4p_6Pe'),
            (16, 2, 'SUL', 'III', 98, '3s_3p2(2De)3d_3Pe'))
        for z, q, element, ion, count, last in specifications:
            key = (z, q)
            atom = parse_osc(raw/element/ion/'19apr23/osc_data')
            if atom.n_levels < max(count, len(levels[key])) or str(atom.levels[count-1]['config']) != last:
                raise ValueError(('표6 마지막 항/원자료 크기 불일치', key))
            for i, ((energy, weight), (number, name)) in enumerate(zip(levels[key], identities[key])):
                entry = atom.levels[i]
                if number != i or name != str(entry['config']) or weight != float(entry['g']) or abs(energy-float(entry['E_cm'])*_CM2EV)>1e-7:
                    raise ValueError(('현행 원자/원자료 준위 대응 실패', key, i))
            # 유지 준위의 반올림까지 그대로 둬서 수 차이만 개입한다.
            limited[key] = list(levels[key][:count]) + [
                (float(r['E_cm'])*_CM2EV, float(r['g']))
                for r in atom.levels[len(levels[key]):count]]
            model_sizes.append(dict(Z=z, stage=q, current_levels=len(levels[key]),
                local_raw_levels=atom.n_levels, table6_subset_levels=count, last_configuration=last))

    # 다른 언어로 평가한 식을 이미 저장된 실제 C Saha 결과와 먼저 대조한다.
    _, native = material(run/'toy_initial_material.csv', 'BOOTSTRAP_LTE_SAHA;')
    validation = []
    for (shell,z,q), row in native.items():
        if z not in (14,16) or q!=1 or float(row['n_ion_cm3'])==0:
            continue
        expected = float(native[(shell,z,2)]['n_ion_cm3'])/float(row['n_ion_cm3'])
        predicted, low, high = ratio(z,float(row['T_K']),float(row['n_e_cm3']))
        error = abs(predicted/expected-1)
        if not error<1e-10:
            raise ValueError(('기존 Saha 경로를 재현하지 못함',shell,z,error))
        validation.append(dict(shell=shell,Z=z,relative_error=error))
    if not validation:
        raise ValueError('실제 Saha 검증 행이 없음')

    ph, phys = public_block(public/'phys_toy06_cmfgen.txt',2.)
    records = []
    for z,name in ((14,'si'),(16,'s')):
        header, ions = public_block(public/f'ionfrac_{name}_toy06_cmfgen.txt',2.)
        if len(phys)!=len(ions) or any(a[0]!=b[0] for a,b in zip(phys,ions)):
            raise ValueError('공개 물질/이온분포의 속도점 불일치; 보간하지 않음')
        for p,r in zip(phys,ions):
            temperature,electron = p[ph.index('temp[K]')],p[ph.index('ne[/cm^3]')]
            predicted,lower,upper = ratio(z,temperature,electron)
            f1,f2 = r[header.index(name+'1')],r[header.index(name+'2')]
            if not 0<=f1<=1 or not 0<=f2<=1:
                raise ValueError('공개 분율 범위 오류')
            observed = f2/f1 if f1>0 else None
            entry = dict(Z=z,velocity_km_s=p[0],T_K=temperature,ne_cm3=electron,
                U_II=lower,U_III=upper,chi_eV=thresholds[(z,1)],
                saha_III_II_same_T_ne=predicted,reference_III_II=observed,
                reference_over_saha=observed/predicted if observed is not None else None,
                reference_II_fraction=f1,reference_III_fraction=f2)
            if limited:
                limited_low = partition((z,1), temperature, limited)
                limited_high = partition((z,2), temperature, limited)
                change = (limited_high/limited_low)/(upper/lower)
                entry.update(table6_subset_U_II=limited_low, table6_subset_U_III=limited_high,
                    table6_subset_over_current_saha=change,
                    table6_subset_saha_III_II=predicted*change,
                    reference_over_table6_subset=observed/(predicted*change) if observed is not None else None)
            records.append(entry)
    out.mkdir(exist_ok=False)
    write_csv(out/'same_state.csv',records)
    write_csv(out/'native_saha_validation.csv',validation)
    summary = dict(status='SAME_T_NE_SAHA_DIAGNOSTIC',day=2.,goal_verdict='NOT_JUDGED',
        public_T_ne_used_only_in_this_diagnostic=True,production_input_changed=False,
        NLTE_solved=False,radiation_solved=False,velocity_interpolation=False,
        table6_size_sensitivity=bool(limited), published_atomic_deck_reproduced=False,
        threshold_changed=False, model_sizes=model_sizes,
        native_validation_count=len(validation),
        native_validation_max_relative=max(r['relative_error'] for r in validation),
        compared_points=len(records),
        transition_rows=[r for r in records if 12100<=r['velocity_km_s']<=12500])
    with (out/'summary.json').open('x') as handle:
        json.dump(summary,handle,indent=2,allow_nan=False);handle.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--raw',type=Path)
    args=parser.parse_args()
    print(json.dumps(diagnose(args.run,args.public,args.out,args.raw),indent=2,allow_nan=False))
