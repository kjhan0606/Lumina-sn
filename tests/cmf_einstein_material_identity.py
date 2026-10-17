"""실제 생산 C 함수와 저장된 강선8개의 소광/율계 항등식. 새 수송은 없다."""
import csv
import ctypes as C
from decimal import Decimal as D, getcontext
import json
import math
from pathlib import Path
import sys

getcontext().prec = 60
base = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
lib = C.CDLL(str(Path(sys.argv[1]).resolve()))
class Value(C.Structure):
    _fields_ = [('value', C.c_double), ('validity', C.c_int), ('generation', C.c_uint64)]
class Material(C.Structure):
    _fields_ = [(key, C.c_double) for key in (
        'raw_chi', 'emission', 'effective_chi', 'effective_tau')]+[
        ('srce_chk', C.c_int), ('zero', C.c_int)]
tau = lib.a208_signed_einstein_counted
tau.argtypes = [C.c_double]*5+[C.c_uint64, C.c_void_p]
tau.restype = Value
material = lib.line_net_sobolev_material
material.argtypes = [C.c_double]*5+[C.c_int, C.c_uint, C.POINTER(Material)]
material.restype = C.c_int
pi = D('3.14159265358979323846264338327950288419716939937510')
h = D('6.62607015e-27')
lines = {}
for z in (14,16):
    with (base/f'line-response-20261016.IaZLH0/lines_Z{z}.csv').open() as f:
        lines.update({int(r['line']):r for r in csv.DictReader(f)})
count = 0
capture = Path(sys.argv[2]) if len(sys.argv)>2 else None
with (capture or base/'fine-material-20261016.z5VQYD/lines.csv').open() as f:
    for row in csv.DictReader(f):
        lid = int(row['line']); r = lines[lid]
        lo,up,nu = (D(row[k]) for k in ('n_lower_NLTE','n_upper_NLTE','nu'))
        Bl,Bu,A,J = (D(r[k]) for k in ('B_lu','B_ul','A_ul','Jbar'))
        # 비교 기준도 실제 C에 전달된 binary64 입력을 정확히 표현한다.
        lo,up,nu,Bl,Bu,A,J = map(lambda v:D.from_float(float(v)),(lo,up,nu,Bl,Bu,A,J))
        t = tau(float(Bl),float(Bu),172800,float(lo),float(up),1,None)
        assert t.validity == 1
        out = Material()
        assert material(float(up),float(A),float(nu),t.value,172800,0,1,C.byref(out)) == 0
        unit = h*nu/(4*pi)
        expected_chi = unit*(lo*Bl-up*Bu)
        expected_net = up*(A+Bu*J)-lo*Bl*J
        actual_chi = D.from_float(out.raw_chi)
        actual_net = (D.from_float(out.emission)-actual_chi*J)/unit
        chi_error = abs(actual_chi/expected_chi-1)
        net_error = abs(actual_net-expected_net)
        roundoff_scale = abs(up*A)+abs((lo*Bl-up*Bu)*J)
        assert chi_error < D('2e-15')
        assert net_error < D('2e-15')*roundoff_scale
        assert (actual_net>0) == (expected_net>0)
        old_chi = D(row['integrated_chi'])
        if capture:
            assert abs(old_chi/expected_chi-1)<D('2e-15')
            assert abs(D(row['integrated_eta'])/(unit*up*A)-1)<D('2e-15')
        else:
            assert abs(old_chi/expected_chi-1)>D('1e-6')
        print(json.dumps(dict(line=lid,chi_relative_error=float(chi_error),
            new_net_cm3_s=float(actual_net),SE_net_cm3_s=float(expected_net),
            net_absolute_difference=float(net_error),old_chi_relative_error=float(old_chi/expected_chi-1))))
        count += 1
assert count == 8
# 생산 τ→소광/방출 경로의 열평형과 인구역전. J 소비만 검사하는 우회가 아니다.
nu=2e15;T=9500;Bl=3e9;Bu=2e9;hb=float(h);cb=2.99792458e10;kb=1.380649e-16
A=2*hb*nu**3/cb**2*Bu
for up in (1e8*(Bl/Bu)*math.exp(-hb*nu/(kb*T)), 2e8):
    lo=1e8;t=tau(Bl,Bu,172800,lo,up,1,None);out=Material()
    assert material(up,A,nu,t.value,172800,0,1,C.byref(out))==0
    assert (out.raw_chi<0)==(lo*Bl-up*Bu<0)
    if out.raw_chi>0:
        B=2*hb*nu**3/cb**2/math.expm1(hb*nu/(kb*T))
        assert abs(out.emission/out.raw_chi/B-1)<2e-15
print(f'EINSTEIN_MATERIAL_IDENTITY PASS actual_lines=8 LTE_source=1 signed_inversion=1 production_capture={int(capture is not None)} GPU=0')
