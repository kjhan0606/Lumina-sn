#!/usr/bin/env python3
"""원90점 저장 물질에서 실제 C 단열식에 세 에너지 장부를 전달한다.

물리/원 native/영점 정렬 native를 분리한다. native 수렴해·Jv·전자 열평형
또는 최종 이온분포·스펙트럼 동등성을 인증하지 않는다. 파일을 게시하지 않는다.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_energy import ModelThermalEnergy


DP=C.POINTER(C.c_double)


class AdiabaticInput(C.Structure):
    _fields_=[('n_shells',C.c_size_t),('epoch_s',C.c_double)]+[
        (name,DP) for name in ('radius','velocity','T','n_atom','ne','u')]


class AdiabaticCell(C.Structure):
    _fields_=[(name,C.c_double) for name in ('temperature_gradient','velocity_divergence',
        'electron_fraction_gradient','internal_energy_gradient','signed_total','cooling','heating')]


def compare(root):
    root=root.resolve();report=json.loads((root/'result.json').read_text())
    if report['state']!='COMPLETE_FROZEN_MATERIAL_BF_90':raise ValueError('완료 물질90 필요')
    snapshot_path=root/'material-snapshot.npz'
    if B.sha(snapshot_path)!=report['output_sha256'][snapshot_path.name]:raise ValueError('후보 물질 변경')
    with np.load(snapshot_path) as f:snapshot={k:f[k] for k in f.files}
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    config_path=Path(json.loads(base_path.read_text())['config']);c=json.loads(config_path.read_text())
    plan_path=Path(c['plan']);plan=json.loads(plan_path.read_text());run=Path(plan['initial_material'])
    levels=Path(c['cells'])/'atomic/levels.csv';material=Path(c['seed'])/'native_material_nodes.bin'
    import csv
    rows=list(csv.DictReader(levels.open()))
    models=B.atomic.load_cmfgen_model_spec(run/'MODEL_SPEC',run/'atomic_links.txt')
    project=Path(__file__).resolve().parents[1]
    paths=(base_path,config_path,plan_path,levels,material,run/'MODEL',run/'MODEL_SPEC',run/'atomic_links.txt')
    identities={str(p):B.sha(p) for p in paths}
    for p in (levels,material,run/'MODEL',run/'MODEL_SPEC',run/'atomic_links.txt'):
        if identities[str(p)]!=report['data_sha256'][str(p)]:raise ValueError('원 물질/모형 변경')
    owner=ModelThermalEnergy.from_model(run,rows,models,snapshot['columns'],
                                      project/'data/atomic/ionization_reference.csv')
    identities.update(owner.input_sha256)
    with material.open('rb') as f:
        tag,ns,nc,nl,epoch=struct.unpack('<8sIIId',f.read(28))
        columns=np.fromfile(f,dtype='<i4',count=3*nc).reshape(nc,3)
        physical=np.fromfile(f,dtype='<f8',count=4*ns).reshape(ns,4)
    if (tag!=b'LUMNOD01' or ns!=90 or nl!=len(rows) or not np.array_equal(columns,snapshot['columns']) or
        not np.array_equal(physical[:,1],snapshot['temperature']) or
        not np.array_equal(physical[:,2],snapshot['ne'])):raise ValueError('반경/후보 물질90 정체성 불일치')
    # C API는 inner→outer다. 원 자료의 outer→inner 주소는 결과에서 복원한다.
    r=np.ascontiguousarray(physical[::-1,0]);v=np.ascontiguousarray(r/epoch)
    temperature=np.ascontiguousarray(snapshot['temperature'][::-1]);ne=np.ascontiguousarray(snapshot['ne'][::-1])
    nuclei=np.ascontiguousarray(snapshot['superlevel'].sum(axis=0)[::-1])
    energies={mode:np.ascontiguousarray(owner.internal_energy(snapshot['superlevel'],snapshot['fraction'],mode),
                                      dtype='f8') for mode in owner.MODES}
    codes=[project/p for p in ('scripts/lumina_model_energy.py','scripts/compare_lumina_internal_energy.py',
        'src/cmfgen_adiabatic.c','src/cmfgen_adiabatic.h')]
    code_sha={str(p):B.sha(p) for p in codes};actual={};independent_error={}
    with tempfile.TemporaryDirectory(prefix='lumina-energy-') as temporary:
        library=Path(temporary)/'adiabatic.so'
        command=['gcc','-O2','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',
                 str(project/'src/cmfgen_adiabatic.c'),'-lm','-o',str(library)]
        subprocess.run(command,check=True,capture_output=True)
        dll=C.CDLL(str(library));fun=dll.cmfgen_adiabatic_v3_homologous_evaluate
        fun.argtypes=[C.POINTER(AdiabaticInput),C.POINTER(AdiabaticCell)];fun.restype=C.c_int
        for mode,u in energies.items():
            rev=np.ascontiguousarray(u[::-1]);arrays=(r,v,temperature,nuclei,ne,rev)
            value=AdiabaticInput(ns,epoch,*[a.ctypes.data_as(DP) for a in arrays]);cells=(AdiabaticCell*ns)()
            if fun(C.byref(value),cells):raise ValueError('실제 C 단열 에너지 소비 실패')
            actual[mode]={name:np.array([getattr(cell,name) for cell in cells])[::-1]
                          for name,_ in AdiabaticCell._fields_}
            # 영점 차이의 전파는 Q차이=n_atom*v*D_r(u차이)다. 다른 세 항은 불변이다.
        for mode in ('native_static_raw','native_static_aligned'):
            difference=np.asarray(energies[mode]-energies['physical_neutral'],dtype=np.longdouble)
            radius=np.asarray(physical[:,0],dtype=np.longdouble)
            neighbor=np.r_[np.arange(1,ns),ns-2]
            expected=snapshot['superlevel'].sum(axis=0)*(radius/epoch)*(
                difference-difference[neighbor])/(radius-radius[neighbor])
            measured=actual[mode]['signed_total']-actual['physical_neutral']['signed_total']
            scale=np.maximum(abs(actual[mode]['signed_total']),abs(actual['physical_neutral']['signed_total']))
            error=abs(measured-expected)
            if np.any(error>1.e-11*scale):raise ValueError('단열 장부차 독립 차분과 불일치')
            independent_error[mode]=float(np.max(np.divide(error,scale,out=np.zeros_like(error),where=scale>0)))
        binary_sha=B.sha(library)
    summaries={}
    for mode in energies:
        delta=actual[mode]['signed_total']-actual['physical_neutral']['signed_total']
        summaries[mode]=dict(energy_min_erg_per_atom=float(energies[mode].min()),
            energy_max_erg_per_atom=float(energies[mode].max()),
            maximum_adiabatic_delta_erg_cm3_s=float(abs(delta).max()),
            worst_native_node=int(np.argmax(abs(delta))))
    if any(B.sha(Path(p))!=h for p,h in {**identities,**code_sha}.items()):raise ValueError('후처리 중 입력/코드 변경')
    return dict(scope=__doc__,source_BF_result_sha256=B.sha(root/'result.json'),
        input_sha256=identities,code_sha256=code_sha,compiled_binary_sha256=binary_sha,
        summaries=summaries,independent_delta_error_over_adiabatic_scale=independent_error,
        points=[dict(native_node=i,**{mode:dict(internal_energy=float(energies[mode][i]),
            **{name:float(a[i]) for name,a in actual[mode].items()}) for mode in owner.MODES})
            for i in (0,22,49,65,89)],
        energy_reference='physical: neutral ground; native_raw: original first-present SL; aligned: one constant per element',
        physical_neutral_offsets_in_native_raw=False,new_transport=0,new_native=0,
        new_BF_BB_integrals=0,material_updates=0,native_residual_Jv_equivalence=False,final_goal='NOT_MET')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bf',type=Path,required=True)
    print(json.dumps(compare(p.parse_args().bf),indent=2,allow_nan=False))
