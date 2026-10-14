"""실제 최종 파일 형식의 작은 합성 예제로 비교 판독을 확인한다."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("reader",Path(__file__).resolve().parents[1]/"scripts/read_cmf_common_material.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class ReadoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run,self.model,self.public = [self.root/n for n in ("run","model","public")]
        for p in (self.run/"work",self.model,self.public):
            p.mkdir(parents=True)
        self.work = self.run/"work"
        self.write(self.model/"geometry.csv","shell_id,v_inner,v_outer",[(0,1e8,3e8)])
        self.write(self.model/"density.csv","shell_id,rho",[(0,100*reader.AMU)])
        self.write(self.model/"abundances.csv","atomic_number,0",[(14,.5),(16,.5)])
        self.write(self.model/"atom_masses.csv","atomic_number,mass_amu",[(14,1),(16,1)])
        self.write(self.model/"levels.csv","atomic_number,ion_number",[(14,1),(14,1),(14,2),(16,1),(16,2)])
        self.write(self.work/"lumina_ion_pops.csv","shell_id,Z,stage,n_ion,population_generation",
                   [(0,14,1,20,13),(0,14,2,30,13),(0,16,1,10,13),(0,16,2,40,13)])
        self.nlte = [(14,1,0,0,8,13),(14,1,0,1,12,13),(14,2,0,2,30,13),
                     (16,1,0,3,10,13),(16,2,0,4,40,13)]
        self.write_nlte()
        (self.work/"cmfgen_final_publication.json").write_text(json.dumps(dict(complete=True,
            population_generation=13,required_population_generation=13)))
        self.obs = dict(material_generation=13,radiation_generation=8,outer_convergence_verified=False)
        self.write_obs()
        for name,fractions in (("si","0 0.4 0.6 0"),("s","0 0.2 0.8 0")):
            (self.public/f"ionfrac_{name}_toy06_cmfgen.txt").write_text(
                f"#TIME: 19.480\n#NVEL: 2\n#vel_mid[km/s] {name}0 {name}1 {name}2 {name}3\n"
                f"1000 {fractions}\n3000 {fractions}\n")

    def tearDown(self):
        self.temp.cleanup()

    def write(self,path,header,data):
        with path.open("w",newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header.split(","));writer.writerows(data)

    def write_nlte(self):
        self.write(self.work/"nlte_levels_final.csv","Z,ion,shell,global_idx,n_pop,population_generation",self.nlte)

    def write_obs(self):
        (self.work/"lumina_spectrum_common_obs.json").write_text(json.dumps(self.obs))

    def run_reader(self):
        return reader.analyze(self.run,self.model,self.public,self.root/"output")

    def test_full_identity_and_fraction(self):
        summary = self.run_reader()
        for m in summary["metrics"]:
            if m["compared_shells"]:
                self.assertAlmostEqual(m["mean_absolute_fraction_difference"],0)
        self.assertEqual(summary["goal_verdict"],"NOT_JUDGED")
        total = list(reader.rows(self.root/"output/element_totals.csv"))[0]
        self.assertAlmostEqual(float(total["available_nlte_fraction_sum"]),1)

    def test_no_renormalization_or_saha_substitution(self):
        self.nlte[0] = (14,1,0,0,3,13);self.write_nlte()
        self.run_reader()
        r = next(r for r in reader.rows(self.root/"output/ion_fractions.csv") if r["Z"]=="14" and r["stage"]=="1")
        self.assertAlmostEqual(float(r["nlte_fraction"]),.3)
        self.assertAlmostEqual(float(r["stage_fraction"]),.4)

    def test_missing_level_is_missing_not_zero(self):
        self.nlte.pop(0);self.write_nlte();self.run_reader()
        r = next(r for r in reader.rows(self.root/"output/ion_fractions.csv") if r["Z"]=="14" and r["stage"]=="1")
        self.assertEqual(r["nlte_fraction"],"")
        self.assertEqual(r["nlte_levels_present"],"1")

    def test_wrong_generation_rejected(self):
        self.obs["material_generation"] = 12;self.write_obs()
        with self.assertRaises(ValueError):self.run_reader()

    def test_duplicate_level_rejected(self):
        self.nlte.append(self.nlte[0]);self.write_nlte()
        with self.assertRaises(ValueError):self.run_reader()

    def test_no_time_or_velocity_extrapolation(self):
        p = self.public/"ionfrac_si_toy06_cmfgen.txt"
        with self.assertRaises(ValueError):reader.public_block(p,19.5)
        _,data = reader.public_block(p,19.48)
        self.assertIsNone(reader.interpolate(data,999,2))
        self.assertAlmostEqual(reader.interpolate([[1,2],[3,6]],2,1),4)

    def test_existing_output_is_preserved(self):
        self.run_reader()
        original = (self.root/"output/summary.json").read_bytes()
        with self.assertRaises(FileExistsError):self.run_reader()
        self.assertEqual(original,(self.root/"output/summary.json").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
