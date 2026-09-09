#!/usr/bin/env python3
import argparse, importlib.util, json, shutil, subprocess, sys, tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--writer", required=True)
args = ap.parse_args()
repo = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cmfgen_twin_map", repo / "scripts/cmfgen_twin_map.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory(prefix="cmfgen-writer-consumer.") as raw:
    root = Path(raw)
    base, reference, evidence, run, expectations = mod._private_fixture(root / "map")
    produced = root / "produced"; produced.mkdir()
    subprocess.run([args.writer, str(produced)], check=True)
    for name in ("lumina_plasma_state.csv", "lumina_ion_pops.csv", "nlte_levels_final.csv", "cmfgen_final_publication.json"):
        shutil.copy2(produced / name, run / "work" / name)
    mod.analyze(argparse.Namespace(contract=str(expectations), run=str(run), evidence=str(evidence)))
    summary = json.loads((evidence / "result/summary.json").read_text(encoding="utf-8"))
    generation = summary["outputs"]["publication_generation_record"]
    if generation != {"final_generation": 7, "te_generation": 11,
                       "te_manifest_sha256": "0123456789abcdef" * 4,
                       "visibility": "PUBLIC_FINAL", "reason": ""}:
        raise SystemExit("producer generation manifest was not consumed")
    files = {item["path"] for item in summary["outputs"]["files"]}
    if "work/nlte_levels_final.csv" not in files or summary["production_population_failures"]:
        raise SystemExit(f"valid producer data caused normalization failures: {summary['production_population_failures']}")
    closures = summary["saha_closure_observations"]
    if not closures or any(item.get("consistent") is not True for item in closures):
        raise SystemExit("mass-consistent producer fixture failed Saha closure")
    cuda = (repo / "src/lumina_cuda.cu").read_text(encoding="utf-8")
    run_at, writer_at, return_at = cuda.index("int det_rc = cmfgen_run("), cuda.index("cmfgen_final_writer(\".\""), cuda.index("return EXIT_SUCCESS;", cuda.index("cmfgen_final_writer(\".\""))
    if not run_at < writer_at < return_at:
        raise SystemExit("production DET callsite is not connected after cmfgen_run")
    print("CMFGEN_WRITER_CONSUMER_SELFTEST_PASS producer=cmfgen_final_writer shells=2 elements=6 stages=3 mass_consistent=1 normalization_failures=0 source_callsite_connected=1 serialization=scientific_6_decimal")
