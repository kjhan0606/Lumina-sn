"""선형 nodal 장의 적분 시험. 수송 격자 수렴이나 CMFGEN 해 동등성 시험은 아니다."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import mpmath as mp
import numpy as np

from lumina_model_linear_lines import LinearModelLines


class LinearLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='lumina-linear-bb-')
        cls.library=Path(cls.directory.name)/'lines.so'
        root=Path(__file__).resolve().parents[1]
        subprocess.run([os.environ.get('CC','gcc'),'-O2','-std=c11','-Wall','-Wextra','-Werror',
                        '-shared','-fPIC',str(root/'src/model_line_linear_jbar.c'),'-lm',
                        '-o',str(cls.library)],check=True)
        mp.mp.dps=65

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def provider(self,nu,j,centers=(100.,),width=.01,support=.06,generation=70):
        n=len(centers)
        lines=dict(lower=np.arange(n,dtype='i4')*2,upper=np.arange(n,dtype='i4')*2+1,
                   nu0=np.array(centers),f=np.ones(n),A=np.full(n,7.),g_ratio=np.full(n,.5))
        return LinearModelLines(lines,dict(width_fraction=width,support_fraction=support),
                                self.library,nu,j,radiation_generation=generation)

    def reference(self,nu,j,center,width,support):
        # 이진 입력과 C의 double 지지 끝점을 그대로 사용하되 mp 적분은 독립 계산한다.
        lo=mp.mpf(float(center*(1-support)));hi=mp.mpf(float(center*(1+support)))
        c=mp.mpf(float(center));w=c*mp.mpf(float(width));den=mp.mpf('1.77245385095516')
        result=[mp.mpf(0) for _ in range(j.shape[1])];mass=mp.mpf(0)
        for k in range(len(nu)-1):
            a=max(lo,mp.mpf(float(nu[k])));b=min(hi,mp.mpf(float(nu[k+1])))
            if a>=b:continue
            left=mp.mpf(float(nu[k]));right=mp.mpf(float(nu[k+1]))
            # Gaussian 중심과 꼬리를 포함해 고정밀 수치 적분한다. C의 모멘트 공식을 복제하지 않는다.
            x0=(a-c)/w;x1=(b-c)/w
            breaks=sorted(set([x0,x1]+[mp.mpf(x) for x in (-4,-2,0,2,4) if x0<x<x1]))
            mass+=mp.quad(lambda x:mp.exp(-x*x)/den,breaks)
            for p in range(j.shape[1]):
                jl=mp.mpf(float(j[k,p]));jr=mp.mpf(float(j[k+1,p]))
                result[p]+=mp.quad(lambda x:mp.exp(-x*x)/den*
                    (jl+(jr-jl)*(c+w*x-left)/(right-left)),breaks)
        return np.array([float(x) for x in result]),float(mass)

    def check_reference(self,nu,j,**kw):
        provider=self.provider(nu,j,**kw);v=provider.contract(0,len(provider.lines['nu0']),radiation_generation=70)
        for k,center in enumerate(provider.lines['nu0']):
            expected,mass=self.reference(nu,j,center,provider.profile['width_fraction'],
                                          provider.profile['support_fraction'])
            np.testing.assert_allclose(v['jbar'][k],expected,rtol=3.e-14,atol=0)
            np.testing.assert_allclose(v['profile_mass'][k],mass,rtol=3.e-14,atol=0)
        return provider,v

    def test_constant_and_linear_90_points(self):
        nu=np.array([90.,94.,97.,100.,102.,106.,110.])
        slopes=np.linspace(0.,.01,90)
        j=2.+(nu[:,None]-90)*slopes
        self.check_reference(nu,j)

    def test_no_internal_node_is_not_missing_support(self):
        p,v=self.check_reference(np.array([90.,110.]),np.array([[1.,3.],[5.,7.]]))
        self.assertTrue(np.all(v['status']==1))
        self.assertTrue(np.all(v['jbar']>0))
        rates=p.rates(0,1,v)
        np.testing.assert_array_equal(rates['spontaneous'],[7.])

    def test_node_insertion_same_piecewise_linear_field(self):
        nu=np.array([90.,99.,101.,110.]);j=np.array([[1.,4.],[8.,2.],[3.,9.],[7.,5.]])
        p=self.provider(nu,j);a=p.contract(0,1,radiation_generation=70)
        fine=np.sort(np.r_[nu,94.,99.3,100.,100.8,105.])
        q=self.provider(fine,np.column_stack([np.interp(fine,nu,j[:,k]) for k in range(2)]))
        b=q.contract(0,1,radiation_generation=70)
        np.testing.assert_allclose(a['jbar'],b['jbar'],rtol=3.e-14,atol=0)
        with self.assertRaises(ValueError):q.rates(0,1,a)

    def test_narrow_segments_and_tails(self):
        nu=np.array([90.,94.,94.000000001,95.999999999,96.,99.999999999,
                     100.,100.000000001,103.999999999,104.,105.999999999,106.,110.])
        j=np.column_stack((np.linspace(.1,7.,len(nu)),np.linspace(9.,.2,len(nu))))
        self.check_reference(nu,j)

    def test_finite_support_not_renormalized(self):
        p,v=self.check_reference(np.array([90.,110.]),np.full((2,1),3.),support=.01)
        self.assertLess(v['profile_mass'][0],.85)
        np.testing.assert_allclose(v['jbar'][:,0],3*v['profile_mass'],rtol=1.e-15)
        np.testing.assert_array_equal(p.rates(0,1,v)['spontaneous'],[7.])

    def test_dark_and_outside_not_conflated(self):
        p=self.provider([90.,110.],np.zeros((2,2)),centers=(100.,120.))
        v=p.contract(0,2,radiation_generation=70)
        np.testing.assert_array_equal(v['status'],[2,3])
        np.testing.assert_array_equal(v['jbar'][0],[0.,0.])
        self.assertTrue(np.isnan(v['jbar'][1]).all())
        self.assertTrue(np.isnan(v['profile_mass'][1]))

    def test_owned_inputs_and_generation(self):
        nu=np.array([90.,110.]);j=np.ones((2,1));p=self.provider(nu,j)
        j[:]=9;nu[:]=1
        np.testing.assert_allclose(p.contract(0,1,radiation_generation=70)['jbar'],1,rtol=1.e-14)
        with self.assertRaises(ValueError):p.contract(0,1,radiation_generation=71)

    def test_late_bad_value_preserves_all_c_outputs(self):
        p=self.provider([90.,110.],np.ones((2,2)))
        j=np.array([[1.,1.],[1.,np.nan]]);center=np.array([100.])
        out=np.full((1,2),-7.);mass=np.array([-8.]);status=np.array([99],dtype='u1')
        rc=p.function(2,2,1,*[a.ctypes.data_as(p.ptr) for a in (p.frequency,j,center)],
            .01,.06,out.ctypes.data_as(p.ptr),mass.ctypes.data_as(p.ptr),
            status.ctypes.data_as(C.POINTER(C.c_uint8)))
        self.assertNotEqual(rc,0)
        np.testing.assert_array_equal(out,[[-7.,-7.]])
        np.testing.assert_array_equal(mass,[-8.]);np.testing.assert_array_equal(status,[99])


if __name__=='__main__':
    unittest.main()
