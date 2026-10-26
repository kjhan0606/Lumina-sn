"""실제 native Fortran의1/2단계·시간·열정책 대조. full CMFGEN 실행은 아니다."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

import numpy as np

from prepare_cmfgen_instantaneous_decay import corrected_source,stage,ORIGINAL_SHA256
from lumina_model_energy import instantaneous_local_decay_heating


class NativeDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native=os.environ.get('CMFGEN_SOURCE_ROOT')
        if native is None:raise unittest.SkipTest('명시한 native 소스 경로 필요')
        cls.native=Path(native);cls.project=Path(__file__).resolve().parents[1]
        cls.tmp=tempfile.TemporaryDirectory(prefix='lumina-native-decay-');cls.root=Path(cls.tmp.name)
        cls.original=cls.native/'new_main/subs/do_species_decays_v2.f'
        cls.original_bytes=cls.original.read_bytes()
        cls.cases={};cls.real90=None;cls.C_owner90=None
        cls.staging=cls.root/'stage';cls.report=stage(cls.original,cls.staging)
        cls.binary={}
        for name,source in (('original',cls.staging/'do_species_decays_v2.original.f'),
                            ('corrected',cls.staging/'do_species_decays_v2.f')):
            build=cls.root/name;build.mkdir();binary=build/'decay'
            command=['gfortran','-O2','-ffixed-line-length-none','-I',str(build),'-J',str(build),
                str(cls.native/'tools/set_kind_module.f'),str(cls.native/'new_main/mod_subs/nuc_iso_mod.f'),
                str(source),str(cls.project/'tests/native_decay_zero_dt_driver.f90'),'-o',str(binary)]
            subprocess.run(command,cwd=build,check=True,capture_output=True,timeout=60)
            cls.binary[name]=binary

    @classmethod
    def tearDownClass(cls):
        assert cls.original.read_bytes()==cls.original_bytes
        destination=os.environ.get('LUMINA_NATIVE_DECAY_REPORT_DIR')
        if destination:
            out=Path(destination);out.mkdir()
            for name in ('do_species_decays_v2.original.f','do_species_decays_v2.f','stage.json'):
                shutil.copyfile(cls.staging/name,out/name)
            for name,binary in cls.binary.items():shutil.copyfile(binary,out/(name+'-decay'))
            np.savez(out/'cases.npz',**cls.cases)
            report=dict(stage=cls.report,conditions=len(cls.cases)//2,real_native90=cls.real90,C_owner90=cls.C_owner90,
                output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
                scope='직접시험 증거; 최종 판정은 unittest 종료 상태를 함께 확인',
                installed_in_native=False,new_full_native=0,new_transport=0,final_goal='NOT_MET')
            (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(evidence=str(out),conditions=report['conditions'],real_native90=cls.real90,C_owner90=cls.C_owner90)))
        cls.tmp.cleanup()

    def result(self,name,chain,dt,instant,population_file=None):
        out=self.root/(self.id().split('.')[-1]+f'-{name}-{chain}-{dt}-{instant}');out.mkdir()
        command=[str(self.binary[name]),str(chain),str(dt),'T' if instant else 'F']
        if population_file is not None:command.append(str(population_file))
        child=subprocess.run(command,
                             cwd=out,check=True,capture_output=True,timeout=10)
        (out/'stdout.txt').write_bytes(child.stdout);(out/'stderr.txt').write_bytes(child.stderr)
        value=np.fromfile(out/'result.bin',dtype='f8')
        nd=90 if population_file is not None else 3
        self.assertEqual(value.shape,(16*nd,));self.assertTrue(np.isfinite(value).all())
        value=value.reshape(16,nd)
        self.cases[f'{self.id().split(".")[-1]}-{name}-{chain}-{dt}-{instant}-{nd}']=value.copy()
        return value

    def test_all_finite_time_modes_and_zero_time_noninstant_preserved(self):
        for chain in (1,2):
            for dt,instant in ((0.,False),(60.,False),(60.,True),(86400.,False),(86400.,True)):
                original=self.result('original',chain,dt,instant)
                corrected=self.result('corrected',chain,dt,instant)
                np.testing.assert_array_equal(original,corrected)

    def test_zero_time_instantaneous_heat_not_isotope_evolution(self):
        ni=np.array([1.e6,2.e6,0.]);co=np.array([3.e6,0.,4.e6])
        channels=[('Ni',np.log(2)/(6.0749*86400),1.7182758*1.602176634e-6),
                  ('Co',np.log(2)/(77.233*86400),3.7488932*1.602176634e-6)]
        kinetic=[('Ni',channels[0][1],.03*1.602176634e-6),('Co',channels[1][1],.12*1.602176634e-6)]
        for chain in (1,2):
            original=self.result('original',chain,0.,True)
            corrected=self.result('corrected',chain,0.,True)
            np.testing.assert_array_equal(original[:2],np.zeros((2,3)))
            np.testing.assert_allclose(corrected[0],instantaneous_local_decay_heating({'Ni':ni,'Co':co},channels[:chain]),rtol=3.e-15,atol=0)
            np.testing.assert_allclose(corrected[1],instantaneous_local_decay_heating({'Ni':ni,'Co':co},kinetic[:chain]),rtol=3.e-15,atol=0)
            # old_pop·old_pop_decay·num_decays와 원소 총수는 그대로다. decay_lum만 순간출력을 갖는다.
            for row in (2,3,4,6,7,8,10,11,12,14,15):
                np.testing.assert_array_equal(corrected[row],original[row])

    def test_staging_does_not_accept_changed_or_overwrite_original(self):
        with self.assertRaises(ValueError):corrected_source(self.original_bytes+b'! changed\n')
        with self.assertRaises(FileExistsError):stage(self.original,self.staging)
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).hexdigest(),ORIGINAL_SHA256)

    def test_real_native90_current_isotopes_same_formula_no_state_evolution(self):
        run=os.environ.get('LUMINA_NATIVE90_MATERIAL')
        if run is None:self.skipTest('기존 native90 물질 경로 필요')
        from analyze_cmfgen_material_inventory import block,isotope_masses
        run=Path(run);hydro=run/'SN_HYDRO_FOR_NEXT_MODEL'
        self.assertEqual(hashlib.sha256((run/'NUC_DECAY_DATA').read_bytes()).hexdigest(),
                         'e38afd62b09d0288fe1793f439f0352b682c2cb518da7f592d94fe02ee931522')
        rho=block(hydro,'Density (g/cm^3)',90)
        masses=isotope_masses(run/'NUC_DECAY_DATA')
        # RD_SN_DATA는 반응행의 parent 질량이 아니라 ISO 목록의 질량을 사용한다.
        self.assertEqual(masses['COB',56],55.939839)
        populations=np.stack([rho*block(hydro,name+' 56 mass fraction',90)/masses[name,56]/1.66e-24
                              for name in ('NICK','COB','IRON')])
        path=self.root/'native90-isotopes.bin'
        with path.open('xb') as f:
            f.write(np.array([90],dtype='i4').tobytes());f.write(populations.astype('f8').tobytes())
        original=self.result('original',2,0.,True,path)
        corrected=self.result('corrected',2,0.,True,path)
        channels=[('Ni',np.log(2)/(6.0749*86400),1.7182758*1.602176634e-6),
                  ('Co',np.log(2)/(77.233*86400),3.7488932*1.602176634e-6)]
        expected=instantaneous_local_decay_heating({'Ni':populations[0],'Co':populations[1]},channels)
        np.testing.assert_array_equal(original[0],np.zeros(90))
        np.testing.assert_allclose(corrected[0],expected,rtol=3.e-15,atol=0)
        np.testing.assert_array_equal(corrected[1],np.zeros(90))
        for row in (2,3,4,6,7,8,10,11,12,14,15):np.testing.assert_array_equal(corrected[row],original[row])
        self.__class__.real90=dict(hydro_sha256=hashlib.sha256(hydro.read_bytes()).hexdigest(),
            points=90,original_heating_max=float(original[0].max()),corrected_heating_max=float(corrected[0].max()),
            maximum_relative_formula_error=float(np.max(abs(corrected[0]/expected-1))),
            isotope_mass_amu={name:masses[name,56] for name in ('NICK','COB','IRON')},
            isotope_state_and_decay_counts_unchanged=True,kinetic_energy_zero_for_actual_input=True)

    def test_same_C_owned_native90_numbers_without_mass_reconversion(self):
        bf=os.environ.get('LUMINA_NATIVE_BF90');run=os.environ.get('LUMINA_NATIVE90_MATERIAL')
        if bf is None or run is None:self.skipTest('동일 C 수밀도/핵종 판정 입력 필요')
        from lumina_native_heat_sources import instantaneous_profile,vector
        bf=Path(bf);run=Path(run)
        report=json.loads((bf/'result.json').read_text())
        snapshot_path=bf/'material-snapshot.npz'
        self.assertEqual(hashlib.sha256((bf/'result.json').read_bytes()).hexdigest(),
                         '482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6')
        self.assertEqual(hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),report['output_sha256'][snapshot_path.name])
        with np.load(snapshot_path) as f:columns=f['columns'];population=f['superlevel']
        radius=vector((run/'SN_HYDRO_FOR_NEXT_MODEL').read_text(),'Radius grid (10^10 cm)',90)*1.e10
        expected,identity=instantaneous_profile(run,columns,population,radius,19.48*86400)
        parents=np.stack([population[columns[:,0]==z].sum(axis=0) for z in (28,27,26)])
        path=self.root/'same-C-native90.bin'
        with path.open('xb') as f:f.write(np.array([90],dtype='i4').tobytes());f.write(parents.astype('f8').tobytes())
        original=self.result('original',2,0.,True,path);corrected=self.result('corrected',2,0.,True,path)
        np.testing.assert_array_equal(original[:2],np.zeros((2,90)))
        np.testing.assert_allclose(corrected[0],expected,rtol=3.e-15,atol=0)
        for row in (2,3,4,6,7,8,10,11,12,14,15):np.testing.assert_array_equal(corrected[row],original[row])
        self.__class__.C_owner90=dict(source_BF_result_sha256=hashlib.sha256((bf/'result.json').read_bytes()).hexdigest(),
            input_sha256=identity['input_sha256'],number_owner=identity['parent_number_owner'],
            points=90,corrected_heating_min=float(corrected[0].min()),corrected_heating_max=float(corrected[0].max()),
            maximum_relative_formula_error=float(np.max(abs(corrected[0]/expected-1))),
            new_mass_conversion=False,new_interpolation=False,isotope_state_unchanged=True)


if __name__=='__main__':unittest.main()
