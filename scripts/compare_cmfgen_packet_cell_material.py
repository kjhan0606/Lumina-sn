#!/usr/bin/env python3
"""기존 실제 packet의 같은 s-선형 물질에서 SC와 독립 양의 formal 해를 비교한다.

이는 native의 유일한 연속 물질 복원이라는 주장이 아니다. 주어진 ray node 사이에
원래 관측기와 같은 chi/eta/b/blue 선형 모델을 사전 고정한 조건부 비교다.
원래 SC source의 암묵적 연속 eta와 이를 혼동하지 않는다. 생산 필드는 바꾸지 않는다.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.integrate import quad

from analyze_cmfgen_formal_capture import formal_state,geometry,rays
from prepare_cmfgen_mom_capture import sha


def linear_material_reference(iup,chi,eta,b,blue,length,alpha):
    """해석적 선형 effective opacity의 양의 formal 적분; 경계 flux 역산 없음."""
    chi,eta,b,blue=[np.asarray(v,dtype=float) for v in (chi,eta,b,blue)]
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in (chi,eta,b,blue)):
        raise ValueError('두 끝점의 유한 물질 자료 필요')
    if min(chi)<=0 or min(eta)<0 or min(b)<0 or min(blue)<0 or not np.isfinite([iup,length,alpha]).all() or min(iup,alpha)<0 or length<=0:
        raise ValueError('양의 물질/길이와 비음수 입사강도 필요')
    effective=chi+(alpha+3)*b;tau=.5*length*effective.sum()
    if min(effective)<=0 or tau<=0 or not np.isfinite(tau):raise ValueError('양의 유한 cell 필요')
    a=effective[1]/effective.sum();up_fraction=effective[0]/effective.sum()
    ymax=-np.expm1(-tau)
    def integrand(z):
        # y=1-exp(-x), x는 downstream에서 거슬러 간 optical depth다.
        # 작은 tau에서 양끝 subtraction을 피하고 큰 tau의 경계층 전체를 포함한다.
        x=-np.log1p(-ymax*z);q=x/tau
        distance=q/(a+np.sqrt(a*a*(1-q)+up_fraction*up_fraction*q))
        cc=distance*chi[0]+(1-distance)*chi[1];ee=distance*eta[0]+(1-distance)*eta[1]
        bb=distance*b[0]+(1-distance)*b[1];previous=distance*blue[0]+(1-distance)*blue[1]
        return (ee+alpha*bb*previous)/(cc+(alpha+3)*bb)
    factor=ymax;coordinate='exponential_optical_depth'
    if tau<=50:
        # 얇거나 중간인 optical cell의 극단적 chi 대비는 y좌표의 endpoint 급변을 만든다.
        # 같은 정확한 formal 적분을 원래 공간좌표에서 계산한다(지지 절단 없음).
        def integrand(u):
            cc=(1-u)*eta[0]+u*eta[1]
            bb=(1-u)*b[0]+u*b[1];previous=(1-u)*blue[0]+u*blue[1]
            remaining=length*(1-u)*(effective[1]+.5*(effective[0]-effective[1])*(1-u))
            return length*(cc+alpha*bb*previous)*np.exp(-remaining)
        factor=1.;coordinate='native_space_fraction'
    def checked(relative):
        result=quad(integrand,0,1,epsabs=0,epsrel=relative,limit=250,full_output=1)
        if len(result)!=3:raise ValueError('독립 formal 적분이 수렴하지 않음: '+str(result[3]))
        return result[0],result[1]
    loose,error1=checked(1e-10);tight,error2=checked(1e-12)
    incident=np.exp(np.log(iup)-tau) if iup>0 else 0.
    return incident+factor*tight,dict(quadrature_change=abs(factor*(loose-tight)),
        estimated_error=factor*error2,optical_depth=tau,coordinate=coordinate)


def kernel(out,native):
    source=Path(__file__).resolve();scripts=source.parent
    wrapper=out/'cell_wrapper.f90'
    wrapper.write_text('''subroutine point(i,s,c,e,t,x,out)bind(c)
use iso_c_binding
use lumina_sc_cell_integral
real(c_double),value::i,s,c,e,t,x
real(c_double)::out
out=sc_cell_intensity_at(i,s,c,e,t,x)
end subroutine
subroutine control(s,e,f,t,n,out)bind(c)
use iso_c_binding
use lumina_positive_sc
real(c_double),value::s,e,f,t,n
real(c_double)::out
out=sc_control(s,e,f,t,n)
end subroutine
subroutine material(i,chi,eta,b,blue,length,alpha,tol,out,error,evaluations,status)bind(c)
use iso_c_binding
use lumina_material_cell
real(c_double),value::i,length,alpha,tol
real(c_double)::chi(2),eta(2),b(2),blue(2),out,error
integer(c_int)::evaluations,status
call material_cell_transfer(i,chi,eta,b,blue,length,alpha,tol,out,error,evaluations,status)
end subroutine
''')
    producers=[scripts/'cmfgen_positive_sc.f90',scripts/'cmfgen_sc_cell_integral.f90',scripts/'cmfgen_material_cell.f90',source]
    for path in producers:shutil.copy2(path,out/path.name)
    library=out/'cell.so'
    command=['gfortran','-shared','-fPIC','-O','-ffree-line-length-none','-I'+str(native/'mod'),
        str(out/'cmfgen_positive_sc.f90'),str(out/'cmfgen_sc_cell_integral.f90'),str(out/'cmfgen_material_cell.f90'),str(wrapper),'-o',str(library)]
    subprocess.run(command,cwd=out,check=True,capture_output=True,timeout=30)
    lib=C.CDLL(str(library));lib.point.argtypes=[C.c_double]*6+[C.POINTER(C.c_double)]
    lib.control.argtypes=[C.c_double]*5+[C.POINTER(C.c_double)]
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.material.argtypes=[C.c_double]+[ptr]*4+[C.c_double]*3+[
        C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_int),C.POINTER(C.c_int)]
    def call(name,*args):
        if name=='material':
            i,chi,eta,b,blue,length,alpha,tol=args
            value=C.c_double();error=C.c_double();evaluations=C.c_int();status=C.c_int()
            lib.material(i,*[np.ascontiguousarray(v,dtype=float) for v in (chi,eta,b,blue)],length,alpha,tol,
                C.byref(value),C.byref(error),C.byref(evaluations),C.byref(status))
            return value.value,error.value,evaluations.value,status.value
        value=C.c_double();getattr(lib,name)(*map(float,args),C.byref(value));return value.value
    return call,command,[*producers,wrapper,library]


def compare(packet,build,native,out):
    if out.exists():raise ValueError('기존 동일 물질 packet 비교 보존')
    inventory=json.loads((build/'formal-build.json').read_text())['ray_inventory']
    inputs=[packet/name for name in ('PRIMARY_GEOMETRY.bin','PRIMARY_FORMAL_FINAL.bin',
                                     'PRIMARY_RAYS_FINAL.bin','PRIMARY_BLUE_FINAL.bin')]
    g=geometry(inputs[0]);a=formal_state(inputs[1],inventory)['args']
    accepted=rays(inputs[2]);previous=rays(inputs[3])
    if g['RAY_POINTS_INSERTED'] or a['P'][0]!=0:
        raise ValueError('이번 범위는 원래 radial node만 있는 중앙 ray')
    out.mkdir();cell,compiler,producers=kernel(out,native)
    geo=g['rays'][0];data=accepted['rays'][0];blue_data=previous['rays'][0]
    alpha=1/a['dLOG_NU'];rows=[];x,w=np.polynomial.legendre.leggauss(8)
    for depth in (33,58,59,60,61):
        j=depth-1;k=int(geo['LNK'][j])-1
        if int(geo['LNK'][j+1])-1!=k+1:raise ValueError('중간 node가 있는 cell은 별도 범위')
        for direction in ('M','P'):
            ids=np.array([j,j+1]) if direction=='M' else np.array([j+1,j])
            rayids=np.array([k,k+1]) if direction=='M' else np.array([k+1,k])
            chi=a['CHI'][ids];eta=a['ETA'][ids];b=geo['B_'+direction][rayids]
            blue=blue_data['I_'+direction][rayids]
            length=float(geo['S_'+direction][rayids[1]]-geo['S_'+direction][rayids[0]])
            up=float(data['I_'+direction][rayids[0]]);actual=float(data['I_'+direction][rayids[1]])
            far=int(rayids[1]+(1 if direction=='M' else -1))
            jf=j+2 if direction=='M' else j-1
            far_b=float(geo['B_'+direction][far]);far_blue=float(blue_data['I_'+direction][far])
            far_effective=a['CHI'][jf]+(alpha+3)*far_b
            far_source=(a['ETA'][jf]+alpha*far_b*far_blue)/far_effective
            effective=chi+(alpha+3)*b
            neighbor=.5*abs(geo['S_'+direction][far]-geo['S_'+direction][rayids[1]])*(effective[1]+far_effective)
            reference,accuracy=linear_material_reference(up,chi,eta,b,blue,length,alpha)
            started=time.process_time()
            for repeat in range(50):
                repaired,estimate,evaluations,status=cell('material',up,chi,eta,b,blue,length,alpha,1e-10)
            seconds=(time.process_time()-started)/50
            if status or abs(repaired-reference)>5e-9*max(up,reference):
                raise ValueError('같은 물질의 새 cell 해가 독립 reference와 불일치: '+str((depth,direction,status)))
            refinements=[]
            for n in (1,2,4,8,16,32,64):
                grid=np.linspace(0,1,n+1)
                def linear(values,u):return values[0]+u*(values[1]-values[0])
                eff=linear(effective,grid)
                sources=(linear(eta,grid)+alpha*linear(b,grid)*linear(blue,grid))/eff
                taus=.5*length/n*(eff[:-1]+eff[1:]);incoming=up;source_defect=0.
                for index in range(n):
                    f=sources[index+2] if index+1<n else far_source
                    next_tau=taus[index+1] if index+1<n else neighbor
                    control=cell('control',sources[index],sources[index+1],f,taus[index],next_tau)
                    u=(x+1)/(2*n)+grid[index];local=(x+1)/2
                    t=length/n*local*(eff[index]+.5*local*(eff[index+1]-eff[index]));q=t/taus[index]
                    sc=(1-q)**2*sources[index]+2*q*(1-q)*control+q*q*sources[index+1]
                    emission=linear(eta,u)+alpha*linear(b,u)*linear(blue,u)
                    source_defect+=length/n*np.dot(w/2,linear(effective,u)*sc-emission)
                    incoming=cell('point',incoming,sources[index],control,sources[index+1],taus[index],taus[index])
                if n==1 and abs(incoming-actual)>2e-13*max(up,actual,sources.max()):
                    raise ValueError('재구성한 원래 SC 끝점이 actual과 불일치')
                refinements.append(dict(cells=n,SC_endpoint=incoming,
                    endpoint_relative_error=abs(incoming-reference)/max(up,reference),
                    integrated_scalar_source_reconstruction_defect=source_defect))
            rows.append(dict(outer_depth_1based=depth,direction=direction,reference_endpoint=reference,
                actual_endpoint=actual,actual_source_endpoint_reproduced=True,reference_accuracy=accuracy,
                material_endpoint=repaired,material_relative_error=abs(repaired-reference)/max(up,reference),
                material_error_estimate=estimate,material_source_evaluations=evaluations,
                material_seconds_CPU1=seconds,
                refinement=refinements))
    report=dict(scope=__doc__,central_ray_only=True,frequency_Hz=a['FREQ']*1e15,
        physical_reconstruction='원래 관측기와 같은 s-선형 chi/eta/b/blue; 양쪽 SC/reference에 고정',
        limitation='고정 blue·입사강도의 국소 비교. 원래 연속 물질의 유일한 복원/전체 ray/전주파수 58–61 대표 아님',
        results=rows,compiler_command=compiler,
        input_sha256={str(p):sha(p) for p in [*inputs,build/'formal-build.json',*producers]},
        production_changed=False,energy_pass='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'same-material-packet.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(results=rows,production_changed=False),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','build','native','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();compare(*(getattr(args,k).resolve() for k in ('packet','build','native','out')))
