#!/usr/bin/env python3
"""물질 재시작에서 원소 수밀도와 동위원소 질량분율의 규약을 분리한다.

완료 SCRTEMP, POP 종별 합과 SN_HYDRO_FOR_NEXT_MODEL을 읽기 전용으로
대조한다. 조성 정규화나 밀도 교체를 수행하지 않는다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from prepare_cmfgen_fixed_field_owner import point_records, record_bytes
from ptf_gated_metrics import read_dimensions, read_variables, read_rvtj_block

ELEMENTS = {14: ('Si', 'SIL'), 16: ('S', 'SUL'), 20: ('Ca', 'CAL'),
            26: ('Fe', 'IRON'), 27: ('Co', 'COB'), 28: ('Ni', 'NICK')}
NATIVE_AMU_G = 1.66e-24
LUMINA_AMU_G = 1.660539066e-24


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def block(path, header, count):
    value = read_rvtj_block(path, header, count)
    if value is None or value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError('물리 벡터 누락/손상: ' + str(path) + ':' + header)
    return value


def isotope_masses(path):
    """NUC_DECAY_DATA의 핵종 목록만 읽고 반응/스펙트럼 행과 섞지 않는다."""
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split('!')[0].split()
        if len(fields) != 3 or fields[2] not in ('s', 'u'):
            continue
        try:
            mass = float(fields[1].replace('D', 'E'))
        except ValueError:
            continue
        key = (fields[0], int(round(mass)))
        if key in result or not np.isfinite(mass) or mass <= 0:
            raise ValueError('중복/잘못된 동위원소 질량')
        result[key] = mass
    if not result:
        raise ValueError('동위원소 질량 목록 없음')
    return result


def summarize(rho, populations, mass_fractions, natural_mass, isotopic_mass):
    arrays = [rho, *populations.values(), *mass_fractions.values()]
    if (not all(a.shape == rho.shape and np.isfinite(a).all() for a in arrays)
            or np.any(rho <= 0) or any(np.any(a < 0) for a in arrays[1:])):
        raise ValueError('비유한/음수 물질 재고')
    natural_sum = np.zeros_like(rho)
    true_sum = np.zeros_like(rho)
    elements = {}
    for Z, n in populations.items():
        x = mass_fractions[Z]
        natural_sum += n*natural_mass[Z]*LUMINA_AMU_G/rho
        true_sum += x
        denom_mass = isotopic_mass.get(Z, natural_mass[Z])
        n_from_mass = x*rho/(denom_mass*NATIVE_AMU_G)
        scale = np.maximum(abs(n), abs(n_from_mass))
        error = np.divide(abs(n-n_from_mass), scale,
                          out=np.zeros_like(n), where=scale>0)
        elements[str(Z)] = dict(
            natural_atomic_mass=natural_mass[Z],
            source_isotope_mass=isotopic_mass.get(Z),
            native_mass_fraction_to_population_max_relative_error=float(error.max()),
            effective_mass_amu_min_max=[
                float(np.min(x*rho/(n*NATIVE_AMU_G))),
                float(np.max(x*rho/(n*NATIVE_AMU_G)))],
            static_natural_mass_population_error_max=float(np.max(abs(
                x*rho/(natural_mass[Z]*LUMINA_AMU_G)/n-1))))
    return dict(elements=elements,
        natural_mass_reconstructed_rho_ratio_min_max=[float(natural_sum.min()),float(natural_sum.max())],
        native_true_mass_fraction_sum_min_max=[float(true_sum.min()),float(true_sum.max())],
        worst_natural_mass_depth_1based=int(np.argmax(abs(natural_sum-1)))+1,
        original_density_unchanged=True,populations_renormalized=False,
        physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')


def analyze(run, atomic, record, expected_record_sha, out):
    if out.exists():
        raise ValueError('기존 물질 재고 증거 보존')
    nd, nt = read_dimensions(run)
    if min(point_records(run).values()) < record:
        raise ValueError('아직 완료되지 않은 인구')
    header, raw, state = record_bytes(run/'SCRTEMP', record, nd, nt)
    if hashlib.sha256(raw).hexdigest() != expected_record_sha:
        raise ValueError('완료 물질 record 신원 불일치')
    variables = read_variables(run, nt-2)
    rho = block(run/'RVTJ', 'Mass Density (gm/cm^3)', nd)
    masses = {int(r['atomic_number']):float(r['mass_amu'])
              for r in csv.DictReader((atomic/'atom_masses.csv').open())}
    if set(masses) != set(ELEMENTS):
        raise ValueError('현재 물질의 전 원소 집합과 원자 질량표 불일치')
    im = isotope_masses(run/'NUC_DECAY_DATA')
    iso = {Z:im[name,56] for Z,(_,name) in ELEMENTS.items() if Z in (26,27,28)}
    pop, frac, pop_errors = {}, {}, {}
    inputs = [run/'MODEL',run/'MODEL_SPEC',run/'LEVEL_SL_STEQ_LINKS',
              run/'RVTJ',run/'NUC_DECAY_DATA',run/'SN_HYDRO_FOR_NEXT_MODEL',
              atomic/'atom_masses.csv',Path(__file__).resolve()]
    for Z, (species, name) in ELEMENTS.items():
        pop[Z] = state[[v.index-1 for v in variables if v.species==species]].sum(axis=0)
        frac[Z] = block(run/'SN_HYDRO_FOR_NEXT_MODEL', name+' mass fraction',nd)
        p = run/('POP'+name)
        lines = p.read_text().splitlines()
        headers = [s.strip() for s in lines if re.match(r'^\s*'+name+r'/He abundance:',s)]
        if len(headers)!=1:
            raise ValueError('원소별 독립 POP 합 누락')
        independent = block(p, headers[0],nd)
        pop_errors[str(Z)] = float(np.max(abs(pop[Z]/independent-1)))
        if pop_errors[str(Z)] > 1e-7:
            raise ValueError('종별 독립 합과 SCRTEMP 인구 불일치')
        if Z in iso:
            isotope_fraction = block(run/'SN_HYDRO_FOR_NEXT_MODEL',name+' 56 mass fraction',nd)
            if not np.array_equal(frac[Z],isotope_fraction):
                raise ValueError('단일56 핵종으로 설명되지 않는 원소; 합성 질량 가정 금지')
        inputs.append(p)
    before = {str(p):sha(p) for p in inputs}
    report = summarize(rho,pop,frac,masses,iso)
    report.update(record=record,depths=nd,source=str(run),atomic_view=str(atomic),
        record_storage_sha256=expected_record_sha,
        header_sha256=hashlib.sha256(header).hexdigest(),
        independent_POP_species_max_relative_error=pop_errors,
        native_amu_g=NATIVE_AMU_G,lumina_amu_g=LUMINA_AMU_G)
    report['isotope_diagnosis'] = '원래 출력에서 Fe/Co/Ni의 원소 질량분율과56 핵종 분율이 byte 동일한 경우에만 핵종 질량으로 환산했다.'
    _, raw_after, _ = record_bytes(run/'SCRTEMP',record,nd,nt)
    if raw_after!=raw or any(sha(Path(p))!=h for p,h in before.items()):
        raise ValueError('판독 중 원래 물질/생산자 변경')
    report['input_sha256']=before
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','atomic','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--record',type=int,required=True)
    ap.add_argument('--record-sha256',required=True)
    a=ap.parse_args()
    analyze(a.run.resolve(),a.atomic.resolve(),a.record,a.record_sha256,a.out.resolve())
