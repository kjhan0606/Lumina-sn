#!/usr/bin/env python3
"""검증된 private formal build에 읽기 전용 ray material 보존만 연결한다."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_cmfgen_mom_capture import FLAGS, command, once, sha


def link(source, exe, build, reference=False, extra_objects=()):
    record = json.loads((build / 'ray-budget-build.json').read_text())
    base, parent, formal, mom, native = [Path(record[k]) for k in ('base', 'parent', 'formal', 'mom', 'native')]
    objects = [build / (name + '.o') for name in ('cmfgen_sc_cell_integral', 'cmfgen_ray_cell_budget')]
    objects += [base / (name + '.o') for name in ('cmfgen_formal_primary', 'cmfgen_positive_sc')]
    objects += [(build if (build/'cmfgen_formal_primary_whole.o').exists() else base) / 'cmfgen_formal_primary_whole.o']
    objects += [parent / (name + '.o') for name in ('cmfgen_mom_acceleration', 'lumina_affine_boundary', 'lumina_mom_diag', 'lumina_formal_diag', 'lumina_whole_monitor')]
    objects += [formal / 'ext_rel_grid_v4.o', mom / 'mod_jrel_v9.o']
    command(['gfortran', '-O', '-fopenmp', '-ffree-line-length-none',
        *['-I' + str(p) for p in (build, base, parent, formal, mom, native / 'mod')],
        str(source), *map(str, extra_objects), *map(str, objects), '-L' + str(base if reference else build), '-L' + str(native / 'lib'),
        '-lsubs', '-lpgplt', '-lmod', str(native / 'lib/libmy_lapack.a'),
        str(native / 'lib/libmy_blas.a'), '-o', str(exe)], build)


def prepare(base, out, whole=False, manufactured=None, packet=None):
    if out.exists():
        raise ValueError('기존 ray 적분 build 보존')
    baseline = json.loads((base / 'formal-primary-build.json').read_text())
    if baseline.get('source_interpolation') != 'positive_continuous_quadratic_Bezier':
        raise ValueError('검증된 연속 SC baseline이 필요함')
    for name, expected in baseline['output_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('기준 build 변경: ' + name)
    parent = Path(baseline['parent'])
    transport = json.loads((parent / 'sweep-build.json').read_text())
    formal, mom, native = [Path(transport[k]) for k in ('formal_build', 'mom_build', 'native')]
    out.mkdir()
    shutil.copy2(__file__,out/Path(__file__).name)
    original = (base / 'cmf_formal_rel_v4.f').read_text()
    changed = once(original, '\tUSE LUMINA_FORMAL_PRIMARY_WHOLE,ONLY:PW_REMEMBER\n',
        '\tUSE LUMINA_FORMAL_PRIMARY_WHOLE,ONLY:PW_REMEMBER\n\tUSE LUMINA_RAY_CELL_BUDGET,ONLY:CB_CAPTURE_RAY\n')
    marker = '            CALL SOLVE_CMF_FORMAL_V2(CHI_RAY,ETA_RAY,IP,FREQ,NU_ON_dNU,INNER_BND_METH,b_planck,dBdTAU,NRAY,NP,NC)'
    if changed.count(marker) != 2:
        raise ValueError('실제 두 ray 호출이 아님')
    changed = changed.replace(marker, marker + '\n\t    CALL CB_CAPTURE_RAY(IP,NRAY,CHI_RAY(1:NRAY),ETA_RAY(1:NRAY),FREQ,NU_ON_dNU)')
    (out / 'cmf_formal_rel_v4.f').write_text(changed)
    sources = [Path(__file__).with_name(name) for name in ('cmfgen_sc_cell_integral.f90', 'cmfgen_ray_cell_budget.f90')]
    for source in sources:
        shutil.copy2(source, out / source.name)
    names = ['cmfgen_sc_cell_integral.f90', 'cmfgen_ray_cell_budget.f90']
    if whole:
        if manufactured is None or packet is None:
            raise ValueError('전체 diagnostic 전에 제조해/실제 packet 증거가 필요함')
        m, p = [json.loads(path.read_text()) for path in (manufactured, packet)]
        if (any(c['returncode'] for c in m['cases']) or len(m['cases']) < 12 or
                not all(p['baseline_and_observer_byte_equal'].values()) or
                not all(p['after_observer_byte_equal'].values())):
            raise ValueError('제조해 완료/실제 장 불변 증거 미달')
        body = (base/'cmfgen_formal_primary_whole.f90').read_text()
        body = once(body, 'use lumina_formal_diag\n',
            'use lumina_formal_diag\nuse lumina_ray_cell_budget,only:cb_enabled,cb_frequency_end,cb_finish\n')
        marker = 'pw_count=pw_count+1;pw_previous_freq=freq;pw_calls=pw_calls+fp_calls\n'
        body = once(body, marker,
            'if(cb_enabled())call cb_frequency_end(index,expected,freq,weight,r,v,pw_input%p,n,pw_input%nc,pw_input%np,j,h)\n'+marker)
        body = once(body, 'close(pw_csv);close(pw_field);pw_open=.false.\n',
            'call cb_finish()\nclose(pw_csv);close(pw_field);pw_open=.false.\n')
        (out/'cmfgen_formal_primary_whole.f90').write_text(body)
        names.append('cmfgen_formal_primary_whole.f90')
    includes = ['-I' + str(p) for p in (out, base, parent, formal, mom, native / 'mod')]
    names.append('cmf_formal_rel_v4.f')
    for name in names:
        flags = FLAGS if name.endswith('.f') else ['-O', '-fopenmp', '-ffree-line-length-none']
        command(['gfortran', *flags, *includes, '-c', name], out)
    shutil.copy2(base / 'libsubs.a', out / 'libsubs.a')
    command(['ar', 'r', 'libsubs.a', 'cmf_formal_rel_v4.o'], out)
    paths = [out / name for name in ('cmfgen_sc_cell_integral.f90', 'cmfgen_ray_cell_budget.f90', 'cmf_formal_rel_v4.f',
        'cmfgen_sc_cell_integral.o', 'cmfgen_ray_cell_budget.o', 'cmf_formal_rel_v4.o', 'libsubs.a')]
    report = dict(scope='실제 ray chi/eta를 별도 복사하는 private 관측 build; 수송 산술 불변',
        base=str(base), parent=str(parent), formal=str(formal), mom=str(mom), native=str(native),
        baseline_sha256=sha(base / 'formal-primary-build.json'),
        output_sha256={str(p): sha(p) for p in paths},
        whole_executable_released=False, final_goal='NOT_JUDGED', generator_sha256=sha(Path(__file__)))
    (out / 'ray-budget-build.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if whole:
        exe = out/'cmfgen_formal_primary.exe'
        link(native/'new_main/cmfgen.o', exe, out, extra_objects=[base/'cmfgen_sub.o'])
        extra = [out/'cmfgen_formal_primary_whole.f90',out/'cmfgen_formal_primary_whole.o',exe,
                 native/'new_main/cmfgen.o',base/'cmfgen_sub.o',base/'cmfgen_formal_primary.o',base/'cmfgen_positive_sc.o']
        release = dict(baseline)
        release.update(scope='v11 장/물리 불변인 전주파수 cell 에너지 측정 diagnostic',
            whole_executable_released=True, cell_energy_integral='ray_characteristic_physical_coefficients_v1',
            physical_cell_interpolation='native s에서 실제 ray chi/eta/b/blue/D 선형 보간; SC source/기하 결함 별도',
            output_sha256={str(p):sha(p) for p in [*paths,*extra]},
            baseline_build=str(base), baseline_manifest_sha256=sha(base/'formal-primary-build.json'),
            manufactured_evidence_sha256=sha(manufactured), packet_evidence_sha256=sha(packet),
            release_scope='pre-pop 전체 에너지 진단만; 실제 보존/인구/T/앵커 합격 아님')
        (out/'formal-primary-build.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
        report.update(whole_executable_released=True,
            whole_manifest_sha256=sha(out/'formal-primary-build.json'),
            manufactured_evidence=str(manufactured),packet_evidence=str(packet))
        (out/'ray-budget-build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(build=str(out), whole_executable_released=whole)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--whole',action='store_true');p.add_argument('--manufactured',type=Path);p.add_argument('--packet',type=Path)
    a = p.parse_args(); prepare(a.base.resolve(), a.out.resolve(),a.whole,a.manufactured,a.packet)
