"""SC 셀 내부 적분을 관측 끝점 차감이 아닌 독립 이중 적분과 비교한다."""
import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

NATIVE = Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0,str(SCRIPTS))
from compare_cmfgen_packet_cell_material import linear_material_reference


@pytest.fixture(scope='module')
def cell(tmp_path_factory):
    root = tmp_path_factory.mktemp('sc_cell_integral')
    wrapper = root / 'wrapper.f90'
    wrapper.write_text('''subroutine cell_area(i,s,c,e,t,out)bind(c)
use iso_c_binding
use lumina_sc_cell_integral
real(c_double),value::i,s,c,e,t
real(c_double)::out
out=sc_cell_integrated_intensity(i,s,c,e,t)
end subroutine
subroutine cell_control(s,e,f,t,n,out)bind(c)
use iso_c_binding
use lumina_positive_sc,only:sc_control
real(c_double),value::s,e,f,t,n
real(c_double)::out
out=sc_control(s,e,f,t,n)
end subroutine
subroutine cell_point(i,s,c,e,t,x,out)bind(c)
use iso_c_binding
use lumina_sc_cell_integral
real(c_double),value::i,s,c,e,t,x
real(c_double)::out
out=sc_cell_intensity_at(i,s,c,e,t,x)
end subroutine
subroutine cell_budget(i,s,c,e,chi,eta,b,blue,doppler,length,alpha,n,out)bind(c)
use iso_c_binding
use lumina_sc_cell_integral
real(c_double),value::i,s,c,e,length,alpha
integer(c_int),value::n
real(c_double)::chi(2),eta(2),b(2),blue(2),doppler(2),out(5)
call sc_physical_cell_budget(i,s,c,e,chi,eta,b,blue,doppler,length,alpha,n,out)
end subroutine
''')
    so = root / 'cell.so'
    subprocess.run(['gfortran', '-shared', '-fPIC', '-O', '-ffree-line-length-none',
        '-I' + str(NATIVE / 'mod'), str(SCRIPTS / 'cmfgen_positive_sc.f90'),
        str(SCRIPTS / 'cmfgen_sc_cell_integral.f90'), str(wrapper), '-o', str(so)],
        cwd=root, check=True, capture_output=True, timeout=30)
    lib = ctypes.CDLL(str(so))
    lib.cell_area.argtypes = [ctypes.c_double] * 5 + [ctypes.POINTER(ctypes.c_double)]
    lib.cell_control.argtypes = [ctypes.c_double] * 5 + [ctypes.POINTER(ctypes.c_double)]
    lib.cell_point.argtypes = [ctypes.c_double] * 6 + [ctypes.POINTER(ctypes.c_double)]
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    lib.cell_budget.argtypes = [ctypes.c_double] * 4 + [ptr] * 5 + [ctypes.c_double] * 2 + [ctypes.c_int, ptr]
    def call(name, *args):
        if name == 'cell_budget':
            i, s, c, e, chi, eta, b, blue, doppler, length, alpha, n = args
            out = np.zeros(5)
            lib.cell_budget(i, s, c, e, *[np.array(v, dtype=float) for v in (chi, eta, b, blue, doppler)], length, alpha, n, out)
            return out
        out = ctypes.c_double(); getattr(lib, name)(*args, ctypes.byref(out)); return out.value
    return call


def independent_points(iup, source, tau, positions):
    x, w = np.polynomial.legendre.leggauss(160); x = (x + 1) / 2; w = w / 2
    u = positions[:, None] / tau * x
    s = source[0] * (1 - u)**2 + 2 * source[1] * u * (1 - u) + source[2] * u**2
    return np.exp(-positions) * iup + positions * np.sum(w * np.exp(-positions[:, None] * (1 - x)) * s, axis=1)


@pytest.mark.parametrize('source', [(1., 2., 4.), (4., 0., 2.), (0., 0., 0.)])
def test_cell_integral_against_independent_continuous_double_integral(cell, source):
    x, w = np.polynomial.legendre.leggauss(220)
    for tau in np.geomspace(1e-12, 100., 25):
        positions = (x + 1) * tau / 2
        values = independent_points(.7, source, tau, positions)
        exact = np.dot(w * tau / 2, values)
        actual = cell('cell_area', .7, *source, tau)
        assert actual == pytest.approx(exact, rel=2e-11, abs=0)
        for fraction in (0., .1, .7, 1.):
            t = tau * fraction
            point = cell('cell_point', .7, *source, tau, t)
            reference = independent_points(.7, source, tau, np.array([t]))[0]
            assert point == pytest.approx(reference, rel=2e-11, abs=0)


def test_optically_thin_source_area_does_not_lose_leading_term(cell):
    for tau in (1e-12, 1e-30, 1e-140):
        area = cell('cell_area', 0., 1., 1., 1., tau)
        assert area == pytest.approx(.5 * tau**2, rel=1e-12, abs=0)


def test_constant_source_deep_cell_and_exact_zero(cell):
    for tau in (0., 1e-280, 1., 1e6, 1e280):
        assert cell('cell_area', 2., 2., 2., 2., tau) == pytest.approx(2 * tau, rel=1e-14, abs=0)
        assert cell('cell_point', 2., 2., 2., 2., tau, tau) == pytest.approx(2., rel=1e-14)
        assert cell('cell_area', 0., 0., 0., 0., tau) == 0


def test_thick_boundary_layer_is_not_smeared_over_cell_width(cell):
    for tau in (.1, 1., 10., 1e6):
        actual = cell('cell_budget', 2., 1., 1., 1., [tau, tau], [tau, tau], [0, 0], [0, 0], [1, 1], 1., 0., 8)
        assert actual[0] == pytest.approx(np.expm1(-tau), rel=3e-14, abs=0)
        np.testing.assert_array_equal(actual[1:], 0.)


def test_physical_source_and_blue_are_independent_of_SC_interpolation(cell):
    chi = np.array([1., 3.]); eta = np.array([.4, 5.]); b = np.array([.1, .2])
    blue = np.array([.8, .2]); doppler = np.array([1.1, 1.02]); length = 2.; alpha = 4.
    source = (eta + alpha * b * blue) / (chi + (alpha + 3) * b)
    sup, target = source; control = .5 * (sup + target); iup = .3
    tau = .5 * length * np.sum(chi + (alpha + 3) * b)
    end = cell('cell_point', iup, sup, control, target, tau, tau)
    boundary = end / doppler[1]**3 - iup / doppler[0]**3
    budgets = [cell('cell_budget', iup, sup, control, target, chi, eta, b, blue, doppler, length, alpha, n) for n in (4, 8)]
    defect = [abs(boundary - (v[0] + v[2] + v[3] + v[4])) for v in budgets]
    assert defect[1] < defect[0]
    assert defect[1] < 1e-5 * abs(boundary)
    assert abs(budgets[1][3]) > 1e-4 and abs(budgets[1][4]) > 1e-4
    assert abs(boundary - budgets[1][0] - budgets[1][2]) > 1e-2 * abs(boundary)


def test_same_accurate_endpoints_do_not_define_a_unique_material_reconstruction(cell):
    # u in[0,1], chi=K(1+u), eta=K, tau=K(u+u^2/2).
    # SC uses S=1-tau/(3K): identical endpoint source values, but
    # int(chi*S-eta)du=K/8 exactly. A huge integral mismatch can coexist
    # with an optically thick endpoint that approaches the correct 1/2.
    for scale in (10.,1e3,1e6):
        tau=1.5*scale
        end=cell('cell_point',1.,1.,.75,.5,tau,tau)
        budget=cell('cell_budget',1.,1.,.75,.5,[scale,2*scale],[scale,scale],
                    [0,0],[0,0],[1,1],1.,0.,8)
        assert budget[4]==pytest.approx(scale/8,rel=2e-14)
        assert end==pytest.approx(.5+1/(3*scale),rel=0,abs=2e-8 if scale==10 else 2e-15)
        assert budget[0]+budget[4]==pytest.approx(end-1,rel=2e-9)
        # With constant chi and independently prescribed linear eta, the
        # same SC endpoint source interpolation IS consistent: no defect.
        matched=cell('cell_budget',1.,1.,.75,.5,[scale,scale],[scale,.5*scale],
                    [0,0],[0,0],[1,1],1.,0.,8)
        assert abs(matched[4])<1e-14*scale


@pytest.mark.parametrize('moving',[False,True])
def test_same_analytic_material_against_independent_RTE_and_local_refinement(cell,moving):
    import json
    from scipy.integrate import solve_ivp,quad
    chi=np.array([2.,6.]) if moving else np.array([20.,60.])
    eta=np.array([1.,4.]) if moving else np.array([25.,10.])
    b=np.array([.2,.4]) if moving else np.zeros(2)
    blue=np.array([3.,.5]) if moving else np.zeros(2)
    alpha=10. if moving else 0.;iup=.3
    def linear(values,u):return values[0]+u*(values[1]-values[0])
    effective=chi+(alpha+3)*b
    def optical(u):return effective[0]*u+.5*(effective[1]-effective[0])*u*u
    def source(u):return (linear(eta,u)+alpha*linear(b,u)*linear(blue,u))/linear(effective,u)
    def derivative(u,y):
        return [linear(eta,u)+alpha*linear(b,u)*linear(blue,u)-linear(effective,u)*y[0]]
    reference=solve_ivp(derivative,(0,1),[iup],method='DOP853',rtol=2e-13,atol=2e-15,dense_output=True)
    assert reference.success
    # 독립적인 양의 formal integral로 reference 끝점을 다시 확인한다.
    endpoint=iup*np.exp(-optical(1))+quad(lambda u:
        (linear(eta,u)+alpha*linear(b,u)*linear(blue,u))*np.exp(-(optical(1)-optical(u))),
        0,1,epsabs=1e-13,epsrel=1e-13)[0]
    assert endpoint==pytest.approx(reference.y[0,-1],rel=2e-12)
    independent,accuracy=linear_material_reference(iup,chi,eta,b,blue,1.,alpha)
    assert independent==pytest.approx(endpoint,rel=2e-12)
    assert accuracy['quadrature_change']<1e-11
    def weighted_material(u,intensity):
        doppler=np.exp(-b[0]*u-.5*(b[1]-b[0])*u*u)
        return (linear(eta,u)-linear(chi,u)*intensity+
                alpha*linear(b,u)*(linear(blue,u)-intensity))/doppler**3
    doppler_end=np.exp(-.5*b.sum())
    reference_net=endpoint/doppler_end**3-iup
    gx,gw=np.polynomial.legendre.leggauss(256);u=(gx+1)/2
    reference_material=np.dot(gw/2,weighted_material(u,reference.sol(u)[0]))
    assert reference_material==pytest.approx(reference_net,rel=2e-11,abs=2e-12)
    measurements=[]
    for n in (2,4,8,16,32,64):
        grid=np.linspace(0,1,n+1);tau=np.diff(optical(grid));s=source(grid)
        incoming=iup;integrals=[]
        for order in (48,96):
            x,w=np.polynomial.legendre.leggauss(order);integral=0.;incoming=iup
            for i in range(n):
                control=.5*(s[i]+s[i+1]) if i==n-1 else cell('cell_control',s[i],s[i+1],s[i+2],tau[i],tau[i+1])
                positions=grid[i]+(x+1)/(2*n)
                intensity=np.array([cell('cell_point',incoming,s[i],control,s[i+1],tau[i],
                    optical(pos)-optical(grid[i])) for pos in positions])
                integral+=np.dot(w/(2*n),weighted_material(positions,intensity))
                incoming=cell('cell_point',incoming,s[i],control,s[i+1],tau[i],tau[i])
            integrals.append(integral)
        assert integrals[0]==pytest.approx(integrals[1],rel=1e-10,abs=1e-11)
        net=incoming/doppler_end**3-iup
        measurements.append(dict(cells=n,endpoint_error=abs(incoming-endpoint),
            physical_balance_error=abs(net-integrals[1]),quadrature_change=abs(integrals[0]-integrals[1])))
    assert measurements[0]['physical_balance_error']>1e-4
    assert measurements[-1]['endpoint_error']<measurements[0]['endpoint_error']/10
    assert measurements[-1]['physical_balance_error']<measurements[0]['physical_balance_error']/10
    print('SAME_MATERIAL_RTE '+json.dumps(dict(moving=moving,reference_endpoint=endpoint,
        reference_balance_error=abs(reference_net-reference_material),refinement=measurements)))
