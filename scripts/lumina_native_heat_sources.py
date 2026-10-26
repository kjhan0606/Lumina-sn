"""원 native의 외부열원 벡터와 PREVENT_LOW_T 규약. 장/물질 갱신은 없다.

Fortran E12.4의 세 자리 지수에서 E가 생략되는 형식을 보존해 읽는다.
출력에 없는 열원을 임의의0으로 대체하지 않으며, 알려진 writer의 SUM=0
생략 계약을 caller가 명시한 경우에만 전체0 벡터를 허용한다.
"""
import re

import numpy as np


def fortran_number(token):
    text=token.strip().replace('D','E').replace('d','e')
    if 'e' not in text.lower():
        text=re.sub(r'(?<=[0-9.])([+-]\d+)$',r'E\1',text)
    result=float(text)
    if not np.isfinite(result):raise ValueError('native 열원 비유한 수치')
    return result


def vector(text,label,count):
    lines=text.splitlines();indices=[i for i,line in enumerate(lines) if line.strip()==label]
    if len(indices)!=1:raise ValueError('native 벡터 제목 누락/중복: '+label)
    values=[]
    for line in lines[indices[0]+1:]:
        if not line.strip():
            if values:break
            continue
        try:row=[fortran_number(token) for token in line.split()]
        except ValueError:break
        values.extend(row)
    if len(values)!=count:raise ValueError('부분 native 벡터: '+label)
    return np.asarray(values)


def cooling_vector(text,label,count,*,omitted_sum_zero=False):
    """GENCOOL의 실제 Depth 주소를 사용한다. 깊이 블록 순서를 가정하지 않는다."""
    lines=text.splitlines();starts=[i for i,line in enumerate(lines) if line.strip()=='Depth']
    if not starts:raise ValueError('GENCOOL Depth 블록 누락')
    output=np.full(count,np.nan);covered=np.zeros(count,dtype=bool);present=[]
    for first,last in zip(starts,starts[1:]+[len(lines)]):
        chunk=lines[first+1:last];nonempty=[i for i,line in enumerate(chunk) if line.strip()]
        if not nonempty:raise ValueError('부분 GENCOOL Depth')
        try:indices=np.asarray([int(x)-1 for x in chunk[nonempty[0]].split()])
        except ValueError as error:raise ValueError('GENCOOL Depth 정수 주소 오류') from error
        if (not len(indices) or np.any(indices<0) or np.any(indices>=count) or
            len(np.unique(indices))!=len(indices) or covered[indices].any()):
            raise ValueError('GENCOOL Depth 누락/중복 주소')
        covered[indices]=True
        headings=[i for i,line in enumerate(chunk) if line.strip()==label]
        present.append(bool(headings))
        if len(headings)>1:raise ValueError('GENCOOL 열원 중복')
        if headings:output[indices]=vector('\n'.join(chunk),label,len(indices))
    if not covered.all():raise ValueError('부분 GENCOOL 전90점')
    if not any(present):
        if not omitted_sum_zero:raise ValueError('확인되지 않은 열원0 대체 거부')
        return np.zeros(count)
    if not all(present) or not np.isfinite(output).all():raise ValueError('부분 GENCOOL 열원 벡터')
    return output


def artificial_heat(temperature,electron_density,positive_ion_density,minimum_temperature):
    """PREVENT_LOW_T의 EHB cgs 출력. 임의 floor/clamp나 exp cutoff가 없다."""
    t,ne,ni=[np.asarray(a,dtype='f8') for a in (temperature,electron_density,positive_ion_density)]
    if (t.ndim!=1 or ne.shape!=t.shape or ni.shape!=t.shape or not len(t) or
        not np.isfinite([t,ne,ni]).all() or np.any(t<=0) or np.any(ne<=0) or np.any(ni<=0) or
        not np.isfinite(minimum_temperature) or minimum_temperature<0):
        raise ValueError('native 인공 가열 물질 오류')
    if minimum_temperature==0:return np.zeros_like(t)
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        value=1.25663706e-19*ne*ni*np.exp(1000.*(minimum_temperature*1.e-4-t*1.e-4))
    if not np.isfinite(value).all():raise ValueError('native 인공 가열 비유한')
    return value


def instantaneous_profile(run,columns,population,radius,epoch):
    """원90점에 이미 있는 순수56 핵종과 C 원소 총수로 만드는 비교 시나리오.

    새700→90점 보간·isotope 진화·질량 재환산을 하지 않는다. 원소의 전
    인구가 해당56 핵종임을 원 파일에서 확인한 경우에만 n_Z를 사용한다.
    원 native 가열/물질/reference에 이 값을 설치하지 않는다.
    """
    from pathlib import Path
    import cmfgen_model_continuum as B
    from lumina_model_energy import instantaneous_local_decay_heating
    run=Path(run);hydro=run/'SN_HYDRO_FOR_NEXT_MODEL';nuclear=run/'NUC_DECAY_DATA'
    identities={str(p):B.sha(p) for p in (hydro,nuclear)}
    text=hydro.read_text();columns=np.asarray(columns);pop=np.asarray(population)
    r=np.asarray(radius);ns=len(r)
    if (columns.ndim!=2 or columns.shape[1]!=3 or pop.shape!=(len(columns),ns) or
        not np.isfinite(pop).all() or np.any(pop<0) or not np.isfinite(r).all() or np.any(r<=0)):
        raise ValueError('순간열의 현재 원소/반경 인구 오류')
    if not np.allclose(vector(text,'Radius grid (10^10 cm)',ns)*1.e10,r,rtol=8.e-11,atol=0):
        raise ValueError('순간열 native90 반경/주소 불일치')
    ages=re.findall(r'^Time\(days\) since explosion:\s*(\S+)\s*$',text,re.M)
    if len(ages)!=1 or not np.isclose(fortran_number(ages[0])*86400,epoch,rtol=1.e-12,atol=0):
        raise ValueError('순간열 현재 epoch 불일치')
    densities={};channels=[];reaction_data=[]
    for line in nuclear.read_text().splitlines():
        fields=line.split('!')[0].split()
        if len(fields)!=8 or fields[6] not in ('F','S'):continue
        name=fields[0];z={'NICK':28,'COB':27}.get(name)
        if z is None or round(fortran_number(fields[1]))!=56 or round(fortran_number(fields[4]))!=56:
            raise ValueError('현재 전모형 밖의 붕괴 채널')
        mass_fraction=vector(text,name+' mass fraction',ns)
        isotope_fraction=vector(text,name+' 56 mass fraction',ns)
        if (not np.array_equal(mass_fraction,isotope_fraction) or np.any(mass_fraction<0) or
            np.any(mass_fraction>1) or not np.any(columns[:,0]==z)):
            raise ValueError('원소 총수를 순수56 핵종 수로 사용할 수 없음')
        key=name+'56'
        if key in densities:raise ValueError('중복 붕괴 핵종')
        half_life_days=fortran_number(fields[2]);energy_MeV=fortran_number(fields[5])
        if half_life_days<=0 or energy_MeV<0:raise ValueError('반감기/반응 에너지 오류')
        densities[key]=pop[columns[:,0]==z].sum(axis=0)
        decay_constant=np.log(2.)/(half_life_days*86400.)
        channels.append((key,decay_constant,energy_MeV*1.602176634e-6))
        reaction_data.append(dict(parent=key,Z=z,half_life_days=half_life_days,
                                  decay_constant_s=decay_constant,energy_MeV=energy_MeV))
    if set(densities)!={'NICK56','COB56'}:raise ValueError('Ni56/Co56 전 붕괴 채널 누락')
    value=instantaneous_local_decay_heating(densities,channels)
    if any(B.sha(Path(p))!=h for p,h in identities.items()):raise ValueError('순간열 원자료 변경')
    return value,dict(input_sha256=identities,channels=reaction_data,
        parent_number_owner='same C candidate total per element; pure56 mass-fraction identity verified',
        density_mass_conversion=False,new_interpolation=False,isotope_evolution=False,
        installed_in_native_or_material=False)
