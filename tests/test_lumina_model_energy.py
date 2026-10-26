import unittest

import numpy as np

from lumina_model_energy import ModelThermalEnergy
import cmfgen_model_continuum as B


class EnergyTests(unittest.TestCase):
    def setUp(self):
        rows=[dict(atomic_number=16,ion_number=charge,level_number=l,super_level=s)
              for charge,levels in ((1,((0,0),(1,1),(2,1))),(2,((0,0),(1,1)))) for l,s in levels]
        columns=np.array([[16,1,0],[16,1,1],[16,2,0],[16,2,1],[16,3,-1]])
        ions={(16,2):dict(edge_Hz=[20.,18.,16.],excitation_Hz=[0.,2.,4.],g=[1.,1.,3.]),
              (16,3):dict(edge_Hz=[40.,37.],excitation_Hz=[0.,3.],g=[1.,1.])}
        self.energy=ModelThermalEnergy(rows,columns,ions,{16:10*B.H})
        self.f=np.array([[1.,1.],[.75,.25],[.25,.75],[1.,1.],[1.,1.]])

    def test_static_and_trial_dependent_means_keep_terminal(self):
        native=self.energy.superlevel_energy(self.f,'native_static_raw')/B.H
        physical=self.energy.superlevel_energy(self.f,'physical_neutral')/B.H
        np.testing.assert_allclose(native[:,0],[0,3.5,20,23,60],rtol=1.e-14)
        np.testing.assert_allclose(physical,[[10,10],[12.5,13.5],[30,30],[33,33],[70,70]],rtol=1.e-14)
        aligned=self.energy.superlevel_energy(self.f,'native_static_aligned')/B.H
        np.testing.assert_allclose(aligned[:,0],[10,13.5,30,33,70],rtol=1.e-14)

    def test_same_sl_not_dropped_and_density_not_ne(self):
        factor=self.energy.line_scaling([0,1],[2,2],[4.,2.],[2.,10.],
                   enabled=True,ratio_limit=.5,density_limit=10.)
        np.testing.assert_allclose(factor,[[.875,1.],[1.,1.]],rtol=1.e-14)

    def test_physical_internal_energy_population_and_fraction_derivative(self):
        pop=np.ones((5,2));direction=np.zeros_like(pop);direction[0]=-1;direction[4]=1
        h=1.e-4
        derivative=(self.energy.internal_energy(pop+h*direction,self.f,'physical_neutral')-
                    self.energy.internal_energy(pop-h*direction,self.f,'physical_neutral'))/(2*h)
        np.testing.assert_allclose(derivative,12*B.H,rtol=1.e-10)
        df=np.zeros_like(self.f);df[1]=1;df[2]=-1
        derivative=(self.energy.internal_energy(pop,self.f+h*df,'physical_neutral')-
                    self.energy.internal_energy(pop,self.f-h*df,'physical_neutral'))/(2*h)
        np.testing.assert_allclose(derivative,-.4*B.H,rtol=1.e-10)

    def test_bad_fraction_and_implicit_ledger_rejected(self):
        f=self.f.copy();f[1,0]+=.1
        with self.assertRaises(ValueError):self.energy.superlevel_energy(f,'physical_neutral')
        with self.assertRaises(ValueError):self.energy.superlevel_energy(self.f,'unspecified')
        with self.assertRaises(ValueError):self.energy.internal_energy(np.zeros((5,2)),self.f,'native_static_raw')


if __name__=='__main__':unittest.main()
