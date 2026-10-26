"""동일 private 물질의 BF/BB/전쌍 충돌을 기존 C 일반 원소 해법으로 전달한다.

원소별 고정장 후보를 만드는 연결부이며 공통 전하/RE 수렴이나 물질 게시를
수행하지 않는다. 원소·점의 누락은 상위 전모형 driver가 검증한다.
"""
import ctypes as C
from pathlib import Path

import numpy as np

DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int32);UP=C.POINTER(C.c_uint8)


class BFView(C.Structure):
    _fields_=[(n,C.c_size_t) for n in ('n_levels','n_routes','n_shells')]+[
        (n,C.c_uint64) for n in ('radiation_generation','material_generation')]+[
        ('material_model_sha256',C.c_char*65)]+[
        (n,IP) for n in ('offset','target','terminal_ion')]+[
        (n,DP) for n in ('probability','threshold_Hz','sigma_edge')]+[
        ('phot_grid',IP)]+[(n,DP) for n in ('saha','temperature','electron_density')]+[
        ('shell_ready',UP),('rates',DP)]


class BBView(C.Structure):
    _fields_=[(n,C.c_size_t) for n in ('n_levels','n_lines','n_shells','native_shell')]+[
        (n,C.c_uint64) for n in ('radiation_generation','material_generation')]+[
        ('material_model_sha256',C.c_char*65),('lower',IP),('upper',IP)]+[
        (n,DP) for n in ('frequency_Hz','absorption','stimulated','spontaneous')]+[('status',UP)]


def array(values,dtype='f8'):
    a=np.array(values,dtype=dtype,order='C',copy=True);a.flags.writeable=False
    return a


def ptr(a):
    return a.ctypes.data_as({'f':DP,'i':IP,'u':UP}[a.dtype.kind])


class MaterialBridge:
    def __init__(self,library,atomic,nodes,occupation,parameters):
        self.library=C.CDLL(str(library));prefix='nlte_model_material_bridge_'
        self.open=getattr(self.library,prefix+'open');self.open.argtypes=[C.c_char_p]*4;self.open.restype=C.c_void_p
        self.free=getattr(self.library,prefix+'free');self.free.argtypes=[C.c_void_p];self.free.restype=None
        self.ion_index=getattr(self.library,prefix+'ion_index');self.ion_index.argtypes=[C.c_void_p,C.c_int,C.c_int]
        self.ion_index.restype=C.c_int
        self.prepare_c=getattr(self.library,prefix+'prepare')
        self.prepare_c.argtypes=[C.c_void_p,C.c_uint64,DP,DP,IP,C.c_size_t,DP,DP,DP,C.POINTER(BFView)]
        self.prepare_c.restype=C.c_int
        self.begin=getattr(self.library,prefix+'element_begin')
        self.begin.argtypes=[C.c_void_p,C.c_uint64,C.c_int,C.c_int,C.POINTER(BBView)];self.begin.restype=C.c_void_p
        self.push_c=getattr(self.library,prefix+'element_push')
        self.push_c.argtypes=[C.c_void_p,C.c_uint64,IP,IP,DP,C.c_size_t];self.push_c.restype=C.c_int
        self.push_points_c=getattr(self.library,prefix+'push_points')
        self.push_points_c.argtypes=[C.POINTER(C.c_void_p),C.c_uint64,IP,IP,DP,C.c_size_t,C.c_size_t]
        self.push_points_c.restype=C.c_int
        self.finish_c=getattr(self.library,prefix+'element_finish')
        self.finish_c.argtypes=[C.c_void_p,C.c_uint64,DP,C.c_size_t,DP];self.finish_c.restype=C.c_int
        self.finish_points_c=getattr(self.library,prefix+'finish_points')
        self.finish_points_c.argtypes=[C.POINTER(C.c_void_p),C.c_uint64,C.c_size_t,DP,C.c_size_t,DP]
        self.finish_points_c.restype=C.c_int
        self.free_element=getattr(self.library,prefix+'element_free');self.free_element.argtypes=[C.c_void_p]
        self.free_element.restype=None
        self.finalize_c=getattr(self.library,prefix+'finalize')
        self.finalize_c.argtypes=[C.c_void_p,C.c_uint64,DP,C.c_size_t,DP,DP];self.finalize_c.restype=C.c_int
        self.handle=self.open(*[str(Path(p)).encode() for p in (atomic,nodes,occupation,parameters)])
        self.elements={};self.id=0;self.buffers=None;self.ready=False
        if not self.handle:raise ValueError('실제 C 원자모형/원 물질 소유자 초기화 실패')

    def close(self):
        for handle in self.elements.values():self.free_element(handle)
        self.elements.clear()
        if self.handle:self.free(self.handle);self.handle=None
        self.buffers=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def prepare(self,state,fraction,saha,roster,rates,*,trial_id,radiation_generation):
        if self.elements or type(trial_id) is not int or trial_id<=self.id or trial_id<=1:
            raise ValueError('사용 중/재사용된 private trial 신원')
        columns=array(state['columns'],'i4');pop=array(state['superlevel'])
        te=array(state['temperature']);ne=array(state['ne']);w=array(state['occupation']);f=array(fraction)
        sa=array(saha);r=array(rates)
        lo=array(roster['lower'],'i4');target=array(roster['target'],'i4');nl=w.shape[0];ns=len(te);nr=len(lo)
        if (columns.ndim!=2 or columns.shape[1]!=3 or pop.shape!=(len(columns),ns) or
            ne.shape!=(ns,) or w.shape!=(nl,ns) or f.shape!=w.shape or sa.shape!=(nr,ns) or
            r.shape!=(nr,ns,6) or target.shape!=(nr,) or np.any(lo<0) or np.any(lo>=nl) or
            np.any(np.diff(lo)<0) or type(radiation_generation) is not int or radiation_generation<=0):
            raise ValueError('공통 trial BF/물질 형상·주소 오류')
        terminal=np.full(nr,-1,dtype='i4')
        for i in np.flatnonzero(target<0):
            ip=self.ion_index(self.handle,int(roster['Z'][i]),int(roster['stage'][i]))
            if target[i]!=-1 or ip<0:raise ValueError('C 원자모형에 없는 terminal 주소')
            terminal[i]=ip
        offset=np.asarray(np.r_[0,np.cumsum(np.bincount(lo,minlength=nl))],dtype='i4')
        probability=array(roster['probability']);edge=array(roster['threshold']);sigma=array(roster['sigma_edge'])
        grid=array(roster['phot_grid'],'i4');ready=np.ones(ns,dtype='u1')
        if any(a.shape!=(nr,) for a in (probability,edge,sigma,grid)):
            raise ValueError('공통 trial PHOT 메타데이터 형상 오류')
        v=BFView(nl,nr,ns,radiation_generation,trial_id,b'',ptr(offset),ptr(target),ptr(terminal),
                 ptr(probability),ptr(edge),ptr(sigma),ptr(grid),ptr(sa),ptr(te),ptr(ne),ptr(ready),ptr(r))
        self.id=trial_id;self.ready=False
        # C는 immutable borrowed view를 원소 stream이 모두 닫힐 때까지 사용한다.
        buffers=(columns,pop,te,ne,w,f,sa,r,lo,target,terminal,offset,probability,edge,sigma,grid,ready,v)
        self.buffers=(buffers,self.buffers)
        if self.prepare_c(self.handle,trial_id,ptr(te),ptr(ne),ptr(columns),len(columns),
                          ptr(pop),ptr(w),ptr(f),C.byref(v)):
            raise ValueError('C private trial의 파생 상태/원소 총수/세대 불일치')
        self.shape=pop.shape;self.nl=nl;self.ns=ns;self.radiation_generation=radiation_generation
        self.buffers=buffers;self.ready=True

    def begin_element(self,Z,node,lines,absorption,stimulated,spontaneous,status):
        key=int(Z),int(node)
        if not self.ready:raise ValueError('준비되지 않았거나 실패한 private trial')
        if key in self.elements:raise ValueError('중복 원소/점 stream')
        lo=array(lines['lower'],'i4');up=array(lines['upper'],'i4');nu=array(lines['nu0'])
        a,s,e=[array(x) for x in (absorption,stimulated,spontaneous)];st=array(status,'u1');n=len(lo)
        if not n or any(v.shape!=(n,) for v in (lo,up,nu,a,s,e,st)):
            raise ValueError('원소 BB roster/율 형상 오류')
        view=BBView(self.nl,n,self.ns,node,self.radiation_generation,self.id,b'',
                    ptr(lo),ptr(up),ptr(nu),ptr(a),ptr(s),ptr(e),ptr(st))
        handle=self.begin(self.handle,self.id,Z,node,C.byref(view))
        if not handle:raise ValueError('C 원소 BF/BB/용해/충돌 stream 조립 실패 '+str(key))
        self.elements[key]=handle

    def push(self,Z,node,lower,upper,omega):
        lo=array(lower,'i4');up=array(upper,'i4');w=array(omega)
        if lo.ndim!=1 or not len(lo) or up.shape!=lo.shape or w.shape!=lo.shape:
            raise ValueError('동일 trial 충돌 블록 형상 오류')
        if self.push_c(self.elements[Z,node],self.id,ptr(lo),ptr(up),ptr(w),len(lo)):
            raise ValueError('C 전 FL쌍 순서/물질 신원/율 불일치')

    def push_points(self,Z,lower,upper,omega):
        lo=array(lower,'i4');up=array(upper,'i4');w=array(omega)
        if lo.ndim!=1 or not len(lo) or up.shape!=lo.shape or w.shape!=(self.ns,len(lo)):
            raise ValueError('전90점 충돌 블록 형상 오류')
        handles=(C.c_void_p*self.ns)(*[self.elements[Z,s] for s in range(self.ns)])
        if self.push_points_c(handles,self.id,ptr(lo),ptr(up),ptr(w),self.ns,len(lo)):
            self.ready=False
            raise ValueError('C 전점 충돌 블록 소비 실패')

    def finish(self,Z,node,output):
        if (output.shape!=self.shape or output.dtype!=np.dtype('f8') or
            not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError('private 원소 후보 출력 형상/소유권 오류')
        key=Z,node;ledger=np.empty(7)
        if self.finish_c(self.elements[key],self.id,ptr(output),output.size,ptr(ledger)):
            raise ValueError('C 전원소 고정장 정상 SE 후보 실패 '+str(key))
        self.free_element(self.elements.pop(key));return ledger

    def finalize(self,output):
        if self.elements or output.shape!=self.shape or output.dtype!=np.dtype('f8') or not output.flags.c_contiguous:
            raise ValueError('전원소·전점 미완결 또는 후보 출력 형상 오류')
        energy=np.empty((self.ns,3));charge=np.empty(self.ns)
        if self.finalize_c(self.handle,self.id,ptr(output),output.size,ptr(energy),ptr(charge)):
            self.ready=False
            raise ValueError('C 전원소 후보의 원자 내부에너지/전하 잔차 소유권 실패')
        self.ready=False
        return dict(energy=energy,charge_residual=charge)

    def finish_points(self,Z,output):
        if (output.shape!=self.shape or output.dtype!=np.dtype('f8') or
            not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError('전점 private 후보 출력 형상/소유권 오류')
        handles=(C.c_void_p*self.ns)(*[self.elements[Z,s] for s in range(self.ns)])
        ledger=np.empty((self.ns,7))
        if self.finish_points_c(handles,self.id,self.ns,ptr(output),output.size,ptr(ledger)):
            self.ready=False
            raise ValueError('C 전점 원소 정상 SE 후보 실패 Z='+str(Z))
        for s in range(self.ns):self.free_element(self.elements.pop((Z,s)))
        return ledger
