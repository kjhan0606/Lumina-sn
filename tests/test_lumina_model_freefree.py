import unittest

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_model_freefree import ModelFreeFree


class SyntheticGaunt:
    def gaunt(self,nu,t,charge):
        return 1.+.01*charge+1.e-6*np.asarray(t)[:,None]+1.e-17*np.asarray(nu)[None,:]


class FreeFreeTests(unittest.TestCase):
    def setUp(self):
        self.nu=np.geomspace(1.e12,3.e15,101);self.w=np.zeros_like(self.nu)
        self.w[:-1]+=.5*np.diff(self.nu);self.w[1:]+=.5*np.diff(self.nu)
        self.state=dict(columns=np.array([[20,1,0],[20,2,0],[20,3,-1]]),
            superlevel=np.array([[2.e6,3.e6],[5.e6,7.e6],[11.e6,13.e6]]),
            sl_index={(20,1,0):0,(20,2,0):1,(20,3,-1):2},
            temperature=np.array([8000.,18000.]),ne=np.array([1.e8,2.e8]))
        self.models={(20,2):{},(20,3):{}};self.kernels=SyntheticGaunt()

    def provider(self,J,evaluation=None):
        return ModelFreeFree(self.nu,self.w,J,self.kernels,self.models,self.state['columns'],
            radiation_generation=70,evaluation=evaluation,kernel_identity='synthetic-gaunt-only')

    def test_direct_and_cont_freq_thermal_integral_matches_existing_sources(self):
        j=np.ones((2,len(self.nu)))*1.e-6
        for evaluation in (None,self.nu*1.002):
            p=self.provider(j,evaluation);coeff=p.prepare(self.state['temperature'],radiation_generation=70,frequency_block=7)
            actual=p.contract(coeff,self.state,radiation_generation=70,material_generation=1)['terms']
            evalnu=self.nu if evaluation is None else evaluation
            source=S.free_free(evalnu,self.state,self.models,self.kernels)
            for name,chi,eta in (('all_positive_ions','chi','eta'),('native_upper_ions','native_chi','native_eta')):
                c,e=source[chi],source[eta]
                if evaluation is not None:c,e=S.rescale_continuum(self.nu,evaluation,self.state['temperature'],c,e)
                heat=4*np.pi*np.sum(c*j*self.w,axis=1);cool=4*np.pi*np.sum(e*self.w,axis=1)
                np.testing.assert_allclose(actual[name]['heating'],heat,rtol=5.e-14)
                np.testing.assert_allclose(actual[name]['cooling'],cool,rtol=5.e-14)

    def test_lte_balance_without_profile_or_frequency_renormalization(self):
        x=B.H*self.nu[None,:]/(B.KB*self.state['temperature'][:,None])
        j=2*B.H*self.nu[None,:]**3/B.C**2/np.expm1(x)
        p=self.provider(j,self.nu*1.002);c=p.prepare(self.state['temperature'],radiation_generation=70)
        values=p.contract(c,self.state,radiation_generation=70,material_generation=1)['terms']
        for term in values.values():
            np.testing.assert_allclose(term['heating'],term['cooling'],rtol=1.e-14)

    def test_reuse_for_ne_population_but_not_temperature_or_other_field(self):
        j=np.full((2,len(self.nu)),1.e-6);p=self.provider(j)
        coeff=p.prepare(self.state['temperature'],radiation_generation=70)
        a=p.contract(coeff,self.state,radiation_generation=70,material_generation=1)['terms']
        changed={**self.state,'ne':self.state['ne']*2}
        b=p.contract(coeff,changed,radiation_generation=70,material_generation=2)['terms']
        for key in a:np.testing.assert_array_equal(b[key]['heating'],a[key]['heating']*2)
        changed={**self.state,'superlevel':self.state['superlevel'].copy()};changed['superlevel'][0]*=2
        b=p.contract(coeff,changed,radiation_generation=70,material_generation=3)['terms']
        self.assertTrue(np.all(b['all_positive_ions']['heating']>a['all_positive_ions']['heating']))
        np.testing.assert_array_equal(b['native_upper_ions']['heating'],a['native_upper_ions']['heating'])
        for state in ({**self.state,'temperature':self.state['temperature']*1.01},
                      {**self.state,'ne':np.array([np.nan,1.])}):
            with self.assertRaises(ValueError):p.contract(coeff,state,radiation_generation=70,material_generation=4)
        with self.assertRaises(ValueError):self.provider(j*2).contract(coeff,self.state,radiation_generation=70,material_generation=5)

    def test_frequency_partition_and_owned_input(self):
        j=np.full((2,len(self.nu)),1.e-6);p=self.provider(j);j[:]=9.
        a=p.prepare(self.state['temperature'],radiation_generation=70,frequency_block=1)
        b=p.prepare(self.state['temperature'],radiation_generation=70,frequency_block=101)
        np.testing.assert_allclose(a.heating,b.heating,rtol=3.e-15)
        np.testing.assert_allclose(a.cooling,b.cooling,rtol=3.e-15)
        with self.assertRaises(ValueError):p.prepare(self.state['temperature'],radiation_generation=71)


if __name__=='__main__':unittest.main()
