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

    def write_history(self):
        folder = self.work/"physics_comparison"
        folder.mkdir()
        self.obs["material_iteration"] = 1
        self.write_obs()
        for iteration,pg in ((0,11),(1,13)):
            prefix = f"physics_DET_iter{iteration:04d}"
            manifest = dict(schema="LUMINA_PHYSICS_COMPARISON_V1",transaction_status="COMMITTED",
                lane="DET",iteration=iteration,epoch_s=19.48*86400,n_shells=1,
                population_generation=pg,radiation_generation=6+iteration,
                atomic_model_sha256="a"*64,geometry_sha256="b"*64,
                shell_file=prefix+".shell.csv",ion_file=prefix+".ion.csv")
            (folder/(prefix+".manifest.json")).write_text(json.dumps(manifest))
            self.write(folder/manifest["shell_file"],"shell_id,T_e_K,n_e_cm3",
                       [(0,8000+2000*iteration,100+100*iteration)])
            header = "population_generation,shell_id"
            values = [pg,0]
            for z in (14,16):
                header += f",Z{z}_stage_q1_cm3,Z{z}_stage_q2_cm3,Z{z}_nlte_q1_cm3,Z{z}_nlte_q2_cm3"
                values += ([30-10*iteration,20+10*iteration,35-15*iteration,15+15*iteration]
                           if z==14 else [20-10*iteration,30+10*iteration,25-15*iteration,25+15*iteration])
            self.write(folder/manifest["ion_file"],header,[values])

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

    def test_iteration_changes_keep_values_and_no_verdict(self):
        self.write_history()
        summary = self.run_reader()
        history = summary["iteration_history"]
        self.assertEqual(history["snapshots"],2)
        self.assertEqual(history["convergence_verdict"],"NOT_JUDGED")
        measure = history["changes"][0]["measurements"]
        self.assertEqual(measure["T_e_K"]["worst"]["previous"],8000)
        self.assertEqual(measure["T_e_K"]["worst"]["current"],10000)
        self.assertAlmostEqual(measure["T_e_K"]["worst"]["value"],.2)
        self.assertEqual(measure["T_e_K"]["previous_current_units"],"K")
        self.assertAlmostEqual(measure["n_e_cm3"]["worst"]["value"],.5)
        self.assertAlmostEqual(measure["Z14_stage_q1_cm3"]["worst"]["value"],.2)
        self.assertAlmostEqual(measure["Z14_nlte_q1_cm3"]["worst"]["value"],.3)
        self.assertEqual(len([p for p in summary["input_sha256"] if "physics_comparison" in p]),6)

    def test_missing_history_does_not_claim_convergence(self):
        history = self.run_reader()["iteration_history"]
        self.assertEqual(history["status"],"INSUFFICIENT_HISTORY")
        self.assertEqual(history["changes"],[])

    def test_stale_final_history_rejected(self):
        self.write_history()
        self.obs["radiation_generation"] = 7
        self.write_obs()
        with self.assertRaisesRegex(ValueError,"마지막 반복"):
            self.run_reader()

    def test_duplicate_history_shell_rejected(self):
        self.write_history()
        path = self.work/"physics_comparison/physics_DET_iter0000.shell.csv"
        self.write(path,"shell_id,T_e_K,n_e_cm3",[(0,8000,100),(0,8000,100)])
        with self.assertRaisesRegex(ValueError,"중복 셸"):
            self.run_reader()

    def test_zero_element_history_is_not_a_zero_fraction(self):
        self.write_history()
        final = json.loads((self.work/"cmfgen_final_publication.json").read_text())
        history,_ = reader.iteration_changes(self.work,{0:{14:0,16:50}},final,self.obs,19.48)
        metric = history["changes"][0]["measurements"]["Z14_nlte_q1_cm3"]
        self.assertEqual(metric["compared_shells"],0)
        self.assertIsNone(metric["worst"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
